import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.ingestion.market_api import PricePanel
from src.modeling import step1_pooled as S


def _long_panel(n_stocks=40, start='2015-01-31', periods=72, seed=0, signal=0.0):
    """Synthetic month-end panel with one rank feature and a label that ends 6 months later."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq='ME')
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    x = rng.uniform(-0.5, 0.5, len(index))
    fwd = signal * x + 0.2 * rng.standard_normal(len(index))
    label_end = index.get_level_values('date') + pd.DateOffset(months=6)
    return pd.DataFrame({'r_x': x, 'fwd': fwd, 'label_end': label_end,
                         'y': (fwd >= 0.1).astype(float)}, index=index)


def test_per_date_auc_skips_small_and_single_class_dates():
    rng = np.random.default_rng(0)
    rows = []
    for date, n, single in (('2024-01-31', 40, False), ('2024-02-29', 10, False), ('2024-03-28', 40, True)):
        for i in range(n):
            label = 1.0 if single else float(i % 3 == 0)
            rows.append((pd.Timestamp(date), f"T{i:02d}", rng.random(), label))
    df = pd.DataFrame(rows, columns=['date', 'ticker', 'p', 'y']).set_index(['date', 'ticker'])

    auc = S.per_date_auc(df, 'p', 'y', min_stocks=30)

    assert list(auc.index) == [pd.Timestamp('2024-01-31')]
    first = df.loc['2024-01-31']
    assert auc.iloc[0] == pytest.approx(roc_auc_score(first['y'], first['p']))


def test_block_bootstrap_is_deterministic_and_the_interval_contains_the_mean():
    values = np.random.default_rng(1).normal(0.56, 0.03, 90)
    first = S.block_bootstrap_ci(values, block=6, replications=500, seed=0)
    second = S.block_bootstrap_ci(values, block=6, replications=500, seed=0)
    other = S.block_bootstrap_ci(values, block=6, replications=500, seed=1)

    assert first == second
    assert first != other
    assert first['ci_lower'] < first['mean'] < first['ci_upper']
    assert first['mean'] == pytest.approx(values.mean()) and first['n'] == 90
    short = S.block_bootstrap_ci(values[:11], block=6, replications=500, seed=0)
    assert np.isnan(short['ci_lower']) and np.isnan(short['ci_upper'])


@pytest.mark.parametrize('ci,passed,reason', [
    ([0.52, 0.60], True, None),
    ([0.50, 0.58], False, S.GATE_LOW_AUC),      # at the threshold: fails
    ([0.47, 0.53], False, S.GATE_LOW_AUC),
    (None, False, S.GATE_INSUFFICIENT_DATES),
])
def test_gate_requires_the_lower_ci_bound_above_the_threshold(ci, passed, reason):
    validation = S.ValidationResult(years=[], overall={'auc_ci': ci}, per_date=pd.DataFrame())
    gate = S.evaluate_gate(validation, {'step1_min_validation_auc_lower': 0.5})
    assert gate['passed'] is passed and gate['reason'] == reason and gate['threshold'] == 0.5


@pytest.mark.parametrize('n_ranked,expected', [(370, 37), (375, 38), (374, 37), (5, 1), (4, 1), (1, 1), (0, 0)])
def test_top_share_size_rounds_half_up_with_at_least_one_stock(n_ranked, expected):
    assert S.n_up_for(n_ranked, 0.10) == expected


def test_ties_are_broken_by_ticker():
    probability = pd.Series({'BBB': 0.5, 'AAA': 0.5, 'CCC': 0.7, 'DDD': 0.1})
    assert S.rank_by_probability(probability).to_dict() == {'BBB': 3, 'AAA': 2, 'CCC': 1, 'DDD': 4}
    assert S.top_share_mask(probability, 0.5).tolist() == [False, True, True, False]


def test_nobody_is_up_when_the_gate_fails(step1_settings):
    panel = _long_panel(seed=2, signal=1.0)
    panel = panel.rename(columns={'r_x': 'r_ret_21'})
    settings = {**step1_settings, 'step1_features': ['ret_21'], 'step1_up_share': 0.1}
    model = S.fit_pooled_model(panel[['r_ret_21']], panel['y'], settings)
    cross_section = panel.xs(panel.index.get_level_values('date')[-1], level='date')

    passed = S.score_cross_section(model, cross_section, settings, gate_passed=True)
    failed = S.score_cross_section(model, cross_section, settings, gate_passed=False)

    assert (passed['predicted_class'] == 'UP').sum() == 4  # 10% of 40
    assert set(passed.loc[passed['predicted_class'] == 'UP', 'rank']) == {1, 2, 3, 4}
    assert (failed['predicted_class'] == 'NOT_UP').all()
    assert failed['rank'].tolist() == passed['rank'].tolist()
    assert passed['top_percent'].max() == pytest.approx(100.0)


def test_walk_forward_training_labels_end_before_the_first_test_date(step1_settings):
    panel = _long_panel(seed=3, signal=1.0)
    scoring_date = panel.index.get_level_values('date').max()
    result = S.walk_forward_validation(panel, ['r_x'], scoring_date, step1_settings)

    fitted = [year for year in result.years if year['status'] == 'ok']
    assert [year['year'] for year in result.years] == list(range(scoring_date.year - 3, scoring_date.year + 1))
    assert fitted
    dates = panel.index.get_level_values('date')
    for year in fitted:
        test_start = pd.Timestamp(year['test_start'])
        assert pd.Timestamp(year['max_label_end']) < test_start
        assert pd.Timestamp(year['train_last']) < test_start
        expected_rows = ((panel['label_end'] < test_start) & (dates < test_start)
                         & (dates >= test_start - pd.DateOffset(years=3))).sum()
        assert year['n_train'] == expected_rows
    # Test rows need a known label: the last 6 months are never scored
    assert result.per_date['date'].max() <= scoring_date - pd.DateOffset(months=6)


def test_a_year_without_training_rows_is_skipped_and_recorded(step1_settings):
    panel = _long_panel(start='2019-01-31', periods=30, seed=4)
    scoring_date = panel.index.get_level_values('date').max()
    result = S.walk_forward_validation(panel, ['r_x'], scoring_date, step1_settings)
    statuses = {year['year']: year['status'] for year in result.years}
    assert statuses[2018] == 'no_test_rows' and statuses[2019] == 'no_training_rows'
    assert statuses[2020] == 'ok'


def _special_universe(simulate_panel):
    prices = simulate_panel(n_stocks=36, n_days=1300, drift_sd=0.0008, seed=5)
    close, volume = prices.close.copy(), prices.volume.copy()
    close['PENNY.DE'] = 0.8
    volume['PENNY.DE'] = 1000.0
    close['NEW.DE'] = np.nan
    close.iloc[-100:, close.columns.get_loc('NEW.DE')] = 20.0
    volume['NEW.DE'] = 1000.0
    close['STALE.DE'] = close['S000.DE']
    close.iloc[-8:, close.columns.get_loc('STALE.DE')] = np.nan
    volume['STALE.DE'] = 1000.0
    close['NOVOL.DE'] = close['S001.DE']
    volume['NOVOL.DE'] = 0.0
    return PricePanel(close=close, volume=volume, missing=['GONE.DE'])


def test_ineligible_stocks_get_a_status_and_no_score(simulate_panel, step1_settings):
    result = S.execute_step1_pooled(_special_universe(simulate_panel), step1_settings)
    status = result.results['status']

    assert status['PENNY.DE'] == 'not_eligible: price_at_or_below_min'
    assert status['NEW.DE'] == 'not_eligible: insufficient_history'
    assert status['STALE.DE'] == 'not_eligible: stale_price'
    assert status['NOVOL.DE'] == 'not_eligible: no_traded_value'
    for ticker in ('PENNY.DE', 'NEW.DE', 'STALE.DE', 'NOVOL.DE'):
        entry = S.ticker_step1_model(result, ticker)
        assert entry['predicted_class'] == 'NOT_UP'
        assert entry['probability'] is None and entry['rank'] is None and entry['top_percent'] is None
        assert entry['feature_percentiles'] is None and entry['logit_contributions'] is None
    cross = result.summary['cross_section']
    assert cross['n_universe'] == 40 and cross['n_ranked'] == 36
    assert cross['not_eligible'] == {'stale_price': 1, 'insufficient_history': 1,
                                     'price_at_or_below_min': 1, 'no_traded_value': 1}
    assert cross['no_price_data'] == ['GONE.DE']
    assert result.results.loc['STALE.DE', 'latest_price'] == pytest.approx(
        _special_universe(simulate_panel).close['STALE.DE'].dropna().iloc[-1])


def test_payloads_and_summary_are_plain_json(simulate_panel, step1_settings):
    result = S.execute_step1_pooled(_special_universe(simulate_panel), step1_settings)

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            # Exact types: numpy scalars (np.float64 subclasses float) must have been converted
            assert value is None or type(value) in (str, bool, int, float), type(value)
            if isinstance(value, float):
                assert np.isfinite(value)

    walk(result.summary)
    json.dumps(result.summary, allow_nan=False)
    for ticker in result.results.index:
        entry = S.ticker_step1_model(result, ticker)
        walk(entry)
        json.dumps(entry, allow_nan=False)
        walk(S.ticker_feature_diagnostics(result, ticker))
    ranked = S.ticker_step1_model(result, 'S002.DE')
    assert isinstance(ranked['rank'], int) and isinstance(ranked['probability'], float)
    assert set(ranked['feature_percentiles']) == set(result.feature_names)
    assert ranked['scoring_date'] == result.summary['scoring_date']
    assert all(0 < pct <= 100 for pct in ranked['feature_percentiles'].values())
    # Contributions are coefficient times rank value
    coef = result.summary['coefficients']
    row = result.results.loc['S002.DE']
    for name, contribution in ranked['logit_contributions'].items():
        assert contribution == pytest.approx(coef[name] * row[f"r_{name}"], abs=1e-4)


def test_a_planted_cross_sectional_signal_is_found_and_passes_the_gate(simulate_panel, step1_settings):
    result = S.execute_step1_pooled(simulate_panel(drift_sd=0.0008, seed=11), step1_settings)
    validation = result.summary['validation']

    assert validation['auc_mean'] > 0.6
    assert validation['auc_ci'][0] > 0.5
    assert result.summary['gate']['passed'] is True
    n_up = result.summary['cross_section']['n_up']
    assert n_up == S.n_up_for(result.summary['cross_section']['n_ranked'], 0.10)
    assert len(result.up_tickers) == n_up
    up = result.results.loc[result.up_tickers]
    assert (up['status'] == 'ok').all() and up['rank'].max() == n_up


def test_pure_noise_is_not_mistaken_for_a_signal(simulate_panel, step1_settings):
    # Production-like validation length (8 years, about 90 test dates); with fewer dates the
    # block bootstrap has too few blocks to cover 0.5 reliably
    prices = simulate_panel(n_stocks=80, n_days=2850, drift_sd=0.0, seed=100, start='2012-01-02')
    result = S.execute_step1_pooled(prices, {**step1_settings, 'step1_validation_years': 8})
    low, high = result.summary['validation']['auc_ci']
    assert result.summary['validation']['n_dates'] > 80

    assert low < 0.5 < high
    assert result.summary['gate']['passed'] is False
    assert result.up_tickers == []
    ranked = result.results['rank'].notna()
    assert (result.results.loc[ranked, 'status'] == 'model_gate_failed').all()
