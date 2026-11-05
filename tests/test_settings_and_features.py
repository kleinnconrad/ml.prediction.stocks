"""
Offline tests of the settings file and of the Step 1 features (src/processing/features.py):
hand-computed feature values, labels and their boundary, the union-calendar alignment,
the eligibility rules, the within-date ranks with the neutral rank of missing values, the
month-ends and point-in-time behavior. The candidate pool of the variable review is tested
in test_feature_registry.py.
"""
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from config.settings import load_settings
from src.ingestion.market_api import PricePanel
from src.modeling.variable_review import REVIEW_PARAMETER_KEYS
from src.processing import features as F
from tests.conftest import DEFAULT_STEP1_FEATURES

Settings = Dict[str, Any]
PanelFactory = Callable[..., PricePanel]


def test_load_settings_returns_independent_copies() -> None:
    first = load_settings()
    first['horizon_days'] = -1
    assert load_settings()['horizon_days'] != -1


def test_settings_contain_parameters_used_by_the_pipeline() -> None:
    from src.modeling.step1_pooled import SUMMARY_PARAMETER_KEYS
    from src.orchestration.steps import APPLIED_PARAMETER_KEYS

    settings = load_settings()
    for key in ('horizon_days', 'threshold', 'step1_features', 'step1_history_years', 'feature_warmup_years',
                'step1_l2_c', 'step1_solver_tol', 'step1_max_iter', 'step1_up_share', 'step1_min_price_eur',
                'step1_ffill_limit_days', 'step1_validation_years', 'step1_min_stocks_per_date',
                'step1_bootstrap_block_months', 'step1_bootstrap_replications', 'step1_bootstrap_seed',
                'step1_min_validation_auc_lower', 'price_download_batch_size', 'price_download_retries',
                'min_step2_score', 'min_price_eur', 'gemini_model', 'xetra_t7_url', 'step1_min_history_days',
                'step1_features_adopted', 'market_index_ticker'):
        assert key in settings
    assert all(key in settings for key in REVIEW_PARAMETER_KEYS)
    # The adoption date is a quoted string, so the line-based settings update can rewrite it
    assert isinstance(settings['step1_features_adopted'], str)
    pd.Timestamp(settings['step1_features_adopted'])
    for removed in ('features_to_select', 'anova_k', 'quantiles', 'cv_splits', 'sfs_cv_splits',
                    'min_cv_train_rows', 'min_cv_accuracy', 'fred_publication_lag_days',
                    'fred_max_staleness_days', 'step2_history_years'):
        assert removed not in settings
    # Every payload and summary key must exist, or the export raises KeyError
    assert all(key in settings for key in APPLIED_PARAMETER_KEYS)
    assert all(key in settings for key in SUMMARY_PARAMETER_KEYS)
    # The eligibility price and the dashboard price filter are separate parameters
    assert {'step1_min_price_eur', 'min_price_eur'} <= set(settings)


def test_default_feature_list_is_known_to_the_registry() -> None:
    names = load_settings()['step1_features']
    assert F.validate_feature_names(names) == names


def test_unknown_or_duplicate_feature_names_raise() -> None:
    with pytest.raises(ValueError, match='Unknown'):
        F.validate_feature_names(['ret_21', 'not_a_feature'])
    with pytest.raises(ValueError, match='duplicates'):
        F.validate_feature_names(['ret_21', 'ret_21'])
    with pytest.raises(ValueError):
        F.validate_feature_names([])


def _series(n: int = 400, seed: int = 0) -> Tuple[pd.Series, pd.Series]:
    """A random-walk close and an integer volume series on business days."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2020-01-01', periods=n)
    close = pd.Series(50 * np.exp(np.cumsum(0.02 * rng.standard_normal(n))), index=index)
    volume = pd.Series(rng.integers(100, 1000, n).astype(float), index=index)
    return close, volume


def test_each_production_feature_matches_a_hand_computation() -> None:
    close, volume = _series()
    volume.iloc[300] = np.nan  # a missing volume on a trading day counts as 0
    frame = F.stock_feature_frame(F.StockSeries.build(close, volume), DEFAULT_STEP1_FEATURES, horizon_days=126)
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
    assert frame['history_days'].tolist() == list(range(1, len(close) + 1))


def test_labels_end_horizon_own_trading_days_later_and_the_last_rows_are_unknown(step1_settings: Settings) -> None:
    horizon = step1_settings['horizon_days']
    close, volume = _series(n=300)
    frame = F.stock_feature_frame(F.StockSeries.build(close, volume), ['ret_21'], horizon_days=horizon)
    assert frame['label_end'].iloc[10] == close.index[10 + horizon]
    assert frame['fwd'].iloc[10] == pytest.approx(close.iloc[10 + horizon] / close.iloc[10] - 1)
    assert frame['fwd'].iloc[-horizon:].isna().all() and frame['label_end'].iloc[-horizon:].isna().all()
    assert frame['fwd'].iloc[:-horizon].notna().all()

    # A short history: the label boundary is tested on rows that are eligible after 126 own days
    panel = F.build_feature_panel(PricePanel(close.to_frame('A.DE'), volume.to_frame('A.DE')), ['ret_21'],
                                  {**step1_settings, 'step1_min_price_eur': 0.0, 'step1_min_history_days': 126})
    long = F.to_long_panel(panel, panel.calendar[-(horizon + 4):], step1_settings)
    known = long['fwd'].notna()
    assert long.loc[known, 'y'].tolist() == (long.loc[known, 'fwd'] >= step1_settings['threshold']).astype(float).tolist()
    assert long.loc[~known, 'y'].isna().all() and (~known).sum() == horizon


def test_a_forward_return_equal_to_the_threshold_is_a_positive_label(step1_settings: Settings) -> None:
    threshold = step1_settings['threshold']
    dates = pd.to_datetime(['2024-01-31', '2024-02-29'])
    tickers = ['A.DE', 'B.DE', 'C.DE']

    def wide(value: Any) -> pd.DataFrame:
        return pd.DataFrame(value, index=dates, columns=tickers)

    panel = F.FeaturePanel(
        feature_names=['ret_21'], features={'ret_21': wide([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])},
        traded_value=wide(10.0), fwd=wide([[threshold, np.nextafter(threshold, -1.0), np.nan],
                                           [threshold + 0.5, -0.2, threshold]]),
        label_end=wide(pd.Timestamp('2024-08-30')), price=wide(20.0), eligible=wide(True))

    long = F.to_long_panel(panel, dates, step1_settings)

    assert long.loc[(dates[0], 'A.DE'), 'y'] == 1.0   # exactly at the threshold: positive
    assert long.loc[(dates[0], 'B.DE'), 'y'] == 0.0   # just below it
    assert np.isnan(long.loc[(dates[0], 'C.DE'), 'y'])  # not known yet
    assert long.xs(dates[1], level='date')['y'].tolist() == [1.0, 0.0, 1.0]


def test_a_day_without_a_trade_carries_the_last_row_for_at_most_the_ffill_limit(step1_settings: Settings) -> None:
    index = pd.bdate_range('2020-01-01', periods=600)
    close = pd.DataFrame({'A.DE': np.linspace(10, 20, 600), 'B.DE': np.linspace(20, 10, 600)}, index=index)
    close.iloc[300:307, 0] = np.nan  # A does not trade on 7 consecutive days
    prices = PricePanel(close=close, volume=pd.DataFrame(100.0, index=index, columns=close.columns))
    settings = {**step1_settings, 'step1_ffill_limit_days': 5}
    panel = F.build_feature_panel(prices, ['ret_21'], settings)

    price = panel.price['A.DE']
    assert (price.iloc[300:305] == close.iloc[299, 0]).all()   # carried for 5 days
    assert price.iloc[305:307].isna().all()                     # then stale
    assert pd.notna(panel.label_end['A.DE'].iloc[299])
    assert (panel.label_end['A.DE'].iloc[300:305] == panel.label_end['A.DE'].iloc[299]).all()
    assert not panel.eligible['A.DE'].iloc[305:307].any()


def test_eligibility_requires_history_traded_value_and_a_price_above_the_minimum(
        simulate_panel: PanelFactory, step1_settings: Settings) -> None:
    prices = simulate_panel(n_stocks=4, n_days=400, seed=1)
    close, volume = prices.close.copy(), prices.volume.copy()
    close['S001.DE'] = step1_settings['step1_min_price_eur']  # exactly at the minimum: not eligible
    volume['S002.DE'] = 0.0                                    # no traded value
    close.iloc[:200, 3] = np.nan                               # listed later: 200 own trading days
    panel = F.build_feature_panel(PricePanel(close, volume), step1_settings['step1_features'], step1_settings)

    last = panel.eligible.iloc[-1]
    assert last.to_dict() == {'S000.DE': True, 'S001.DE': False, 'S002.DE': False, 'S003.DE': False}
    # The price is the only reason for S001.DE: its features and traded value are present
    assert panel.traded_value['S001.DE'].iloc[-1] > 0
    assert all(pd.notna(frame['S001.DE'].iloc[-1]) for frame in panel.features.values())
    # Eligible from the 253rd own trading day on, which is also the first day with every default feature
    first_full = panel.features['mom_12_1']['S000.DE'].first_valid_index()
    assert panel.calendar.get_loc(first_full) == step1_settings['step1_min_history_days'] - 1
    assert not panel.eligible['S000.DE'].loc[:first_full].iloc[:-1].any()
    assert panel.eligible['S000.DE'].loc[first_full]
    assert panel.history_days['S003.DE'].iloc[-1] == 200


def _old_eligibility(panel: F.FeaturePanel, min_price: float) -> pd.DataFrame:
    """The eligibility rule up to release 3.0.1: all features present, positive traded value, price above minimum."""
    ok = np.ones(panel.price.shape, dtype=bool)
    for frame in panel.features.values():
        ok &= frame.notna().to_numpy()
    value = panel.traded_value.to_numpy(dtype=float)
    ok &= np.isfinite(value) & (np.nan_to_num(value, nan=-1.0) > 0)
    ok &= np.nan_to_num(panel.price.to_numpy(dtype=float), nan=0.0) > min_price
    return pd.DataFrame(ok, index=panel.price.index, columns=panel.price.columns)


def test_the_history_rule_gives_the_old_eligible_set_for_the_default_features(
        simulate_panel: PanelFactory, step1_settings: Settings) -> None:
    prices = simulate_panel(n_stocks=12, n_days=900, seed=4)
    close, volume = prices.close.copy(), prices.volume.copy()
    close.iloc[:500, 1] = np.nan                        # listed later
    close.iloc[600:603, 2] = np.nan                     # three days without a trade (carried)
    close.iloc[700:720, 3] = np.nan                     # suspended: stale for a while
    volume.iloc[:, 4] = 0.0                             # never traded
    close.iloc[:, 5] = 0.9 + 0.2 * np.sin(np.arange(len(close)) / 30.0)  # around the minimum price
    panel = F.build_feature_panel(PricePanel(close, volume), DEFAULT_STEP1_FEATURES, step1_settings)

    old = _old_eligibility(panel, step1_settings['step1_min_price_eur'])
    pd.testing.assert_frame_equal(panel.eligible, old)
    assert 0 < panel.eligible.to_numpy().mean() < 1


def test_an_eligible_stock_without_a_feature_value_gets_the_neutral_rank(
        simulate_panel: PanelFactory, step1_settings: Settings) -> None:
    prices = simulate_panel(n_stocks=8, n_days=700, seed=9)
    # 400 own trading days: eligible, but the 504-day momentum reversal is still missing
    close = prices.close.copy()
    close.iloc[:300, 0] = np.nan
    names = ['ret_21', 'mom_24_12']
    panel = F.build_feature_panel(PricePanel(close, prices.volume), names, step1_settings)
    date = panel.calendar[-1]
    assert panel.eligible.loc[date].all()
    assert pd.isna(panel.features['mom_24_12'].loc[date, 'S000.DE'])

    long = F.to_long_panel(panel, [date], step1_settings).xs(date, level='date')
    assert long.loc['S000.DE', 'r_mom_24_12'] == F.NEUTRAL_RANK
    # The others are ranked among the seven stocks that have the value
    others = panel.features['mom_24_12'].loc[date].drop('S000.DE')
    expected = others.rank(pct=True, method='average') - 0.5
    pd.testing.assert_series_equal(long['r_mom_24_12'].drop('S000.DE'), expected, check_names=False)
    assert long['r_ret_21'].notna().all()

    # The same rule in the long-panel ranks of the variable review
    values = long[names].copy()
    values.index = pd.MultiIndex.from_product([[date], values.index], names=['date', 'ticker'])
    ranks = F.rank_long_panel(values)
    np.testing.assert_allclose(ranks['mom_24_12'].to_numpy(), long['r_mom_24_12'].to_numpy(), rtol=0, atol=1e-15)


def test_rank_within_date_matches_pandas_pct_rank_on_the_eligible_set() -> None:
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

    # An eligible cell without a value gets the neutral rank; the others are ranked without it
    eligible.loc['2024-02-29', 'B'] = True
    ranks = F.rank_within_date(values, eligible)
    assert ranks.loc['2024-02-29', 'B'] == F.NEUTRAL_RANK
    expected = values.loc['2024-02-29'].drop('B').rank(pct=True, method='average') - 0.5
    pd.testing.assert_series_equal(ranks.loc['2024-02-29'].drop('B'), expected, check_names=False)


def test_month_end_dates_are_the_last_calendar_date_of_each_month() -> None:
    calendar = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-27', '2024-02-28', '2024-03-01'])
    assert list(F.month_end_dates(calendar)) == list(pd.to_datetime(['2024-01-31', '2024-02-28', '2024-03-01']))


def test_features_and_ranks_at_a_date_do_not_change_when_later_prices_are_removed(
        simulate_panel: PanelFactory, step1_settings: Settings) -> None:
    prices = simulate_panel(n_stocks=12, n_days=700, drift_sd=0.001, seed=3)
    names = step1_settings['step1_features']
    cut = prices.close.index[500]
    truncated = PricePanel(prices.close.loc[:cut], prices.volume.loc[:cut])

    full_panel = F.build_feature_panel(prices, names, step1_settings)
    cut_panel = F.build_feature_panel(truncated, names, step1_settings)
    full = F.to_long_panel(full_panel, [cut], step1_settings)
    part = F.to_long_panel(cut_panel, [cut], step1_settings)

    columns = F.rank_columns(names) + names + ['price']
    pd.testing.assert_frame_equal(full[columns], part[columns])
    assert part['y'].isna().all() and full['y'].notna().all()
