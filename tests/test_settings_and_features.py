import numpy as np
import pandas as pd
import pytest

from config.settings import load_settings
from src.ingestion.market_api import PricePanel
from src.processing import features as F


def test_load_settings_returns_independent_copies():
    first = load_settings()
    first['horizon_days'] = -1
    assert load_settings()['horizon_days'] != -1


def test_settings_contain_parameters_used_by_the_pipeline():
    from src.modeling.step1_pooled import SUMMARY_PARAMETER_KEYS
    from src.orchestration.steps import APPLIED_PARAMETER_KEYS

    settings = load_settings()
    for key in ('horizon_days', 'threshold', 'step1_features', 'step1_history_years', 'feature_warmup_years',
                'step1_l2_c', 'step1_solver_tol', 'step1_max_iter', 'step1_up_share', 'step1_min_price_eur',
                'step1_ffill_limit_days', 'step1_validation_years', 'step1_min_stocks_per_date',
                'step1_bootstrap_block_months', 'step1_bootstrap_replications', 'step1_bootstrap_seed',
                'step1_min_validation_auc_lower', 'price_download_batch_size', 'price_download_retries',
                'min_step2_score', 'min_price_eur', 'gemini_model', 'xetra_t7_url'):
        assert key in settings
    for removed in ('features_to_select', 'anova_k', 'quantiles', 'cv_splits', 'sfs_cv_splits',
                    'min_cv_train_rows', 'min_cv_accuracy', 'fred_publication_lag_days',
                    'fred_max_staleness_days', 'step2_history_years'):
        assert removed not in settings
    # Every payload and summary key must exist, or the export raises KeyError
    assert all(key in settings for key in APPLIED_PARAMETER_KEYS)
    assert all(key in settings for key in SUMMARY_PARAMETER_KEYS)
    # The eligibility price and the dashboard price filter are distinct parameters
    assert settings['step1_min_price_eur'] != settings['min_price_eur']


def test_default_feature_list_is_known_to_the_registry():
    names = load_settings()['step1_features']
    assert F.validate_feature_names(names) == names
    assert len(names) == 10


def test_unknown_or_duplicate_feature_names_raise():
    with pytest.raises(ValueError, match='Unknown'):
        F.validate_feature_names(['ret_21', 'not_a_feature'])
    with pytest.raises(ValueError, match='duplicates'):
        F.validate_feature_names(['ret_21', 'ret_21'])
    with pytest.raises(ValueError):
        F.validate_feature_names([])


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2020-01-01', periods=n)
    close = pd.Series(50 * np.exp(np.cumsum(0.02 * rng.standard_normal(n))), index=index)
    volume = pd.Series(rng.integers(100, 1000, n).astype(float), index=index)
    return close, volume


def test_each_feature_matches_a_hand_computation():
    close, volume = _series()
    volume.iloc[300] = np.nan  # a missing volume on a trading day counts as 0
    frame = F.stock_feature_frame(close, volume, list(F.FEATURE_FUNCTIONS), horizon_days=126)
    c, v = close.to_numpy(), volume.fillna(0.0).to_numpy()
    r = c[1:] / c[:-1] - 1  # r[i - 1] is the return of day i
    t = 350

    expected = {
        'ret_21': c[t] / c[t - 21] - 1,
        'ret_63': c[t] / c[t - 63] - 1,
        'ret_126': c[t] / c[t - 126] - 1,
        'mom_12_1': c[t - 21] / c[t - 252] - 1,
        'dist_sma50': c[t] / c[t - 49:t + 1].mean() - 1,
        'dist_sma200': c[t] / c[t - 199:t + 1].mean() - 1,
        'vol_21': np.std(r[t - 21:t], ddof=1),
        'vol_126': np.std(r[t - 126:t], ddof=1),
        'log_traded_value': np.log1p(np.mean(c[t - 125:t + 1] * v[t - 125:t + 1])),
        'drawdown_252': c[t] / c[t - 251:t + 1].max() - 1,
    }
    for name, value in expected.items():
        assert frame[name].iloc[t] == pytest.approx(value, rel=1e-10), name
    # Windows are not filled before they are complete
    assert frame['mom_12_1'].iloc[:252].isna().all() and frame['mom_12_1'].iloc[252:].notna().all()
    assert frame['drawdown_252'].iloc[:251].isna().all() and frame['drawdown_252'].iloc[251:].notna().all()
    assert frame['vol_21'].iloc[:21].isna().all() and pd.notna(frame['vol_21'].iloc[21])
    assert frame['log_traded_value'].iloc[:125].isna().all() and pd.notna(frame['log_traded_value'].iloc[125])


def test_labels_end_horizon_own_trading_days_later_and_the_last_rows_are_unknown():
    close, volume = _series(n=300)
    frame = F.stock_feature_frame(close, volume, ['ret_21'], horizon_days=126)
    assert frame['label_end'].iloc[10] == close.index[136]
    assert frame['fwd'].iloc[10] == pytest.approx(close.iloc[136] / close.iloc[10] - 1)
    assert frame['fwd'].iloc[-126:].isna().all() and frame['label_end'].iloc[-126:].isna().all()
    assert frame['fwd'].iloc[:-126].notna().all()

    panel = F.build_feature_panel(PricePanel(close.to_frame('A.DE'), volume.to_frame('A.DE')),
                                  ['ret_21'], {**load_settings(), 'step1_min_price_eur': 0.0})
    long = F.to_long_panel(panel, panel.calendar[-130:], load_settings())
    threshold = load_settings()['threshold']
    known = long['fwd'].notna()
    assert long.loc[known, 'y'].tolist() == (long.loc[known, 'fwd'] >= threshold).astype(float).tolist()
    assert long.loc[~known, 'y'].isna().all() and (~known).sum() == 126


def test_a_day_without_a_trade_carries_the_last_row_for_at_most_the_ffill_limit():
    index = pd.bdate_range('2020-01-01', periods=600)
    close = pd.DataFrame({'A.DE': np.linspace(10, 20, 600), 'B.DE': np.linspace(20, 10, 600)}, index=index)
    close.iloc[300:307, 0] = np.nan  # A does not trade on 7 consecutive days
    prices = PricePanel(close=close, volume=pd.DataFrame(100.0, index=index, columns=close.columns))
    settings = {**load_settings(), 'step1_ffill_limit_days': 5}
    panel = F.build_feature_panel(prices, ['ret_21'], settings)

    price = panel.price['A.DE']
    assert (price.iloc[300:305] == close.iloc[299, 0]).all()   # carried for 5 days
    assert price.iloc[305:307].isna().all()                     # then stale
    assert pd.notna(panel.label_end['A.DE'].iloc[299])
    assert (panel.label_end['A.DE'].iloc[300:305] == panel.label_end['A.DE'].iloc[299]).all()
    assert not panel.eligible['A.DE'].iloc[305:307].any()


def test_eligibility_requires_features_traded_value_and_a_minimum_price(simulate_panel):
    prices = simulate_panel(n_stocks=4, n_days=400, seed=1)
    close, volume = prices.close.copy(), prices.volume.copy()
    close['S001.DE'] = 0.9                      # at or below step1_min_price_eur
    volume['S002.DE'] = 0.0                     # no traded value
    close.iloc[:200, 3] = np.nan                # listed later: features missing
    settings = load_settings()
    panel = F.build_feature_panel(PricePanel(close, volume), settings['step1_features'], settings)

    last = panel.eligible.iloc[-1]
    assert last.to_dict() == {'S000.DE': True, 'S001.DE': False, 'S002.DE': False, 'S003.DE': False}
    # Eligible exactly once every feature is present
    first_full = panel.features['mom_12_1']['S000.DE'].first_valid_index()
    assert not panel.eligible['S000.DE'].loc[:first_full].iloc[:-1].any()
    assert panel.eligible['S000.DE'].loc[first_full]


def test_rank_within_date_matches_pandas_pct_rank_on_the_eligible_set():
    values = pd.DataFrame([[3.0, 1.0, 2.0, 2.0, 9.0], [1.0, np.nan, 5.0, 4.0, 0.0]],
                          index=pd.to_datetime(['2024-01-31', '2024-02-29']),
                          columns=['A', 'B', 'C', 'D', 'E'])
    eligible = pd.DataFrame([[True, True, True, True, False], [True, False, True, True, True]],
                            index=values.index, columns=values.columns)
    ranks = F.rank_within_date(values, eligible)
    for date in values.index:
        expected = values.loc[date][eligible.loc[date]].rank(pct=True, method='average') - 0.5
        pd.testing.assert_series_equal(ranks.loc[date][eligible.loc[date]], expected, check_names=False)
        assert ranks.loc[date][~eligible.loc[date]].isna().all()
    assert ranks.loc['2024-01-31', 'C'] == ranks.loc['2024-01-31', 'D'] == pytest.approx(0.625 - 0.5)
    assert ranks.loc['2024-01-31', 'A'] == pytest.approx(0.5)


def test_month_end_dates_are_the_last_calendar_date_of_each_month():
    calendar = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-27', '2024-02-28', '2024-03-01'])
    assert list(F.month_end_dates(calendar)) == list(pd.to_datetime(['2024-01-31', '2024-02-28', '2024-03-01']))


def test_features_and_ranks_at_a_date_do_not_change_when_later_prices_are_removed(simulate_panel):
    prices = simulate_panel(n_stocks=12, n_days=700, drift_sd=0.001, seed=3)
    settings = load_settings()
    names = settings['step1_features']
    cut = prices.close.index[500]
    truncated = PricePanel(prices.close.loc[:cut], prices.volume.loc[:cut])

    full_panel = F.build_feature_panel(prices, names, settings)
    cut_panel = F.build_feature_panel(truncated, names, settings)
    full = F.to_long_panel(full_panel, [cut], settings)
    part = F.to_long_panel(cut_panel, [cut], settings)

    columns = F.rank_columns(names) + names + ['price']
    pd.testing.assert_frame_equal(full[columns], part[columns])
    assert part['y'].isna().all() and full['y'].notna().all()
