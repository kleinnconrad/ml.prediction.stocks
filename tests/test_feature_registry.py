"""
Offline tests of the feature registry (src/processing/features.py): the size and families of
the candidate pool, that every candidate computes on synthetic OHLCV and market data, that
no candidate looks ahead, hand-computed values for at least one member of every family, the
ten production features against their formulas of release 3.0.1, and missing inputs.
"""
import math
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.processing import features as F
from tests.conftest import DEFAULT_STEP1_FEATURES, synthetic_stock

T = 700  # position (own trading day) of the hand computations


@pytest.fixture(scope='module')
def stock() -> Dict[str, pd.Series]:
    """A synthetic stock with OHLCV, trading gaps, zero-volume days and a market index."""
    return synthetic_stock(n_days=900, seed=1)


@pytest.fixture(scope='module')
def series(stock: Dict[str, pd.Series]) -> F.StockSeries:
    """The stock as feature input."""
    return F.StockSeries.build(stock['close'], volume=stock['volume'], open_=stock['open'], high=stock['high'],
                               low=stock['low'], market=stock['market'])


@pytest.fixture(scope='module')
def frame(series: F.StockSeries) -> pd.DataFrame:
    """Every registered feature of the stock."""
    return F.stock_feature_frame(series, list(F.FEATURE_FUNCTIONS), horizon_days=126)


def test_the_pool_has_200_to_300_named_candidates_in_families() -> None:
    assert 200 <= len(F.FEATURE_FUNCTIONS) <= 300
    assert set(F.FEATURE_FAMILIES) == set(F.FEATURE_FUNCTIONS)
    assert set(F.FEATURE_FAMILIES.values()) == {
        'returns', 'momentum', 'sma_distance', 'sma_crossover', 'sma_slope', 'highs_lows', 'volatility',
        'range_volatility', 'higher_moments', 'market', 'liquidity', 'trend_quality', 'overnight_intraday',
        'oscillators', 'seasonality'}
    assert set(DEFAULT_STEP1_FEATURES) <= set(F.FEATURE_FUNCTIONS)
    # The candidates of the E6 study are in the pool under their E6 names, except the log price:
    # a past adjusted close depends on the later splits (no point-in-time price level)
    e6 = ['ret_5', 'ret_10', 'ret_21', 'ret_42', 'ret_63', 'ret_126', 'ret_189', 'ret_252', 'mom_12_1', 'mom_6_1',
          'mom_12_6', 'dist_sma20', 'dist_sma50', 'dist_sma100', 'dist_sma200', 'sma50_sma200', 'sma200_chg_63',
          'drawdown_252', 'dist_low_252', 'range_pos_252', 'days_since_high_252', 'maxdd_252', 'vol_5', 'vol_21',
          'vol_63', 'vol_126', 'vol_252', 'downvol_126', 'vol_ratio_21_126', 'skew_63', 'skew_252', 'kurt_63',
          'kurt_252', 'max_ret_21', 'min_ret_21', 'beta_252', 'corr_252', 'beta_63', 'idio_vol_252', 'idio_vol_63',
          'downbeta_252', 'resid_mom_12_1', 'log_tv_21', 'log_traded_value', 'amihud_21', 'amihud_126',
          'zero_ret_126', 'zero_vol_126', 'vol_trend_21_126', 'seas_1y', 'id_252', 'autocorr_63']
    assert len(e6) == 52 and set(e6) <= set(F.FEATURE_FUNCTIONS)
    assert 'log_price' not in F.FEATURE_FUNCTIONS


def test_every_candidate_computes_and_has_values_after_its_warm_up(frame: pd.DataFrame) -> None:
    values = frame[list(F.FEATURE_FUNCTIONS)]
    assert np.isfinite(values.to_numpy()[~np.isnan(values.to_numpy())]).all()
    without_values = [name for name in F.FEATURE_FUNCTIONS if values[name].notna().sum() == 0]
    assert without_values == []
    # The longest windows need 505 own trading days (504-day look-back plus today)
    assert values.iloc[-1].notna().all()


def test_no_candidate_uses_data_after_its_date(stock: Dict[str, pd.Series], frame: pd.DataFrame) -> None:
    for position in (560, 700, len(stock['close']) - 2):
        date = stock['close'].index[position]
        cut = {name: values.loc[:date] for name, values in stock.items()}
        truncated = F.stock_feature_frame(
            F.StockSeries.build(cut['close'], volume=cut['volume'], open_=cut['open'], high=cut['high'],
                                low=cut['low'], market=cut['market']),
            list(F.FEATURE_FUNCTIONS), horizon_days=126)
        full_row = frame.loc[date, list(F.FEATURE_FUNCTIONS)].to_numpy(dtype=float)
        cut_row = truncated.loc[date, list(F.FEATURE_FUNCTIONS)].to_numpy(dtype=float)
        np.testing.assert_allclose(cut_row, full_row, rtol=1e-12, atol=0, equal_nan=True)


def _old_production_features(close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
    """The ten Step 1 features exactly as computed up to release 3.0.1 (close and volume on own days)."""
    c = close.dropna().astype(float)
    v = volume.reindex(c.index).fillna(0.0).astype(float)
    return {
        'ret_21': c.pct_change(21), 'ret_63': c.pct_change(63), 'ret_126': c.pct_change(126),
        'mom_12_1': c.shift(21) / c.shift(252) - 1.0,
        'dist_sma50': c / c.rolling(50).mean() - 1.0, 'dist_sma200': c / c.rolling(200).mean() - 1.0,
        'vol_21': c.pct_change().rolling(21).std(), 'vol_126': c.pct_change().rolling(126).std(),
        'log_traded_value': np.log1p((c * v).rolling(126).mean()),
        'drawdown_252': c / c.rolling(252).max() - 1.0,
    }


def test_the_production_features_are_unchanged(stock: Dict[str, pd.Series], frame: pd.DataFrame) -> None:
    old = _old_production_features(stock['close'], stock['volume'])
    for name in DEFAULT_STEP1_FEATURES:
        pd.testing.assert_series_equal(frame[name], old[name].replace([np.inf, -np.inf], np.nan),
                                       check_exact=True, check_names=False)
    # Also when only closes and volumes are known (the inputs of release 3.0)
    only = F.stock_feature_frame(F.StockSeries.build(stock['close'], stock['volume']), DEFAULT_STEP1_FEATURES, 126)
    pd.testing.assert_frame_equal(only[DEFAULT_STEP1_FEATURES], frame[DEFAULT_STEP1_FEATURES], check_exact=True)


def _sample_skew(x: np.ndarray) -> float:
    """Adjusted Fisher-Pearson skewness (pandas' rolling skew)."""
    n, d = x.size, x - x.mean()
    return math.sqrt(n * (n - 1)) / (n - 2) * (d ** 3).mean() / (d ** 2).mean() ** 1.5


def _wilder(values: np.ndarray, window: int) -> np.ndarray:
    """Wilder smoothing as a recursion: s[0] = x[0], s[i] = s[i-1] + (x[i] - s[i-1]) / window."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = out[i - 1] + (values[i] - out[i - 1]) / window
    return out


def test_one_member_of_every_family_matches_a_hand_computation(stock: Dict[str, pd.Series],
                                                               frame: pd.DataFrame) -> None:
    c = stock['close'].to_numpy()
    v = stock['volume'].to_numpy()
    o, h, lo = stock['open'].to_numpy(), stock['high'].to_numpy(), stock['low'].to_numpy()
    m = stock['market'].reindex(stock['close'].index).to_numpy()  # every own day is a market day
    r = np.concatenate([[np.nan], c[1:] / c[:-1] - 1.0])
    rm = np.concatenate([[np.nan], m[1:] / m[:-1] - 1.0])
    t = T
    window = slice(t - 62, t + 1)          # the last 63 days including t

    def sma(k: int, end: int = t) -> float:
        return c[end - k + 1:end + 1].mean()

    win = c[t - 62:t + 1]
    drawdowns = win / np.maximum.accumulate(win) - 1.0
    log_c = np.log(c[t - 62:t + 1])
    slope, intercept = np.polyfit(np.arange(63.0), log_c, 1)
    fitted = intercept + slope * np.arange(63.0)
    residual = log_c - fitted
    se = math.sqrt((residual @ residual) / 61 / ((np.arange(63.0) - 31.0) ** 2).sum())
    r2 = 1.0 - (residual @ residual) / ((log_c - log_c.mean()) ** 2).sum()
    tv = c * v
    illiquid = [abs(r[i]) / tv[i] for i in range(t - 20, t + 1) if tv[i] > 0]
    gain = _wilder(np.clip(np.diff(c), 0, None), 14)
    loss = _wilder(np.clip(-np.diff(c), 0, None), 14)
    months = stock['close'].index.to_period('M')
    month_close = stock['close'].groupby(months).last()
    period = months[t]
    lagged = month_close[period - 11] / month_close[period - 12] - 1.0

    expected = {
        'ret_84': c[t] / c[t - 84] - 1.0,                                             # returns
        'mom_12_3': c[t - 63] / c[t - 252] - 1.0,                                     # momentum
        'mom_24_12': c[t - 252] / c[t - 504] - 1.0,
        'dist_sma150': c[t] / sma(150) - 1.0,                                         # sma_distance
        'sma10_sma50': sma(10) / sma(50) - 1.0,                                       # sma_crossover
        'sma50_chg_21': sma(50) / sma(50, t - 21) - 1.0,                              # sma_slope
        'days_since_high_63': (62 - np.flatnonzero(win == win.max())[-1]) / 62.0,     # highs_lows
        'maxdd_63': drawdowns.min(),
        'ulcer_63': math.sqrt((drawdowns ** 2).mean()),
        'range_pos_126': (c[t] - c[t - 125:t + 1].min()) / (c[t - 125:t + 1].max() - c[t - 125:t + 1].min()),
        'downvol_63': math.sqrt((np.clip(r[window], None, 0) ** 2).mean()),         # volatility
        'vol_ratio_5_63': np.std(r[t - 4:t + 1], ddof=1) / np.std(r[window], ddof=1),
        'parkinson_21': math.sqrt((np.log(h[t - 20:t + 1] / lo[t - 20:t + 1]) ** 2).mean() / (4 * math.log(2))),
        'garman_klass_21': math.sqrt((0.5 * np.log(h[t - 20:t + 1] / lo[t - 20:t + 1]) ** 2
                                      - (2 * math.log(2) - 1) * np.log(c[t - 20:t + 1] / o[t - 20:t + 1]) ** 2).mean()),
        'hl_range_21': ((h[t - 20:t + 1] - lo[t - 20:t + 1]) / c[t - 20:t + 1]).mean(),  # range_volatility
        'skew_63': _sample_skew(r[window]),                                           # higher_moments
        'max_ret_21': r[t - 20:t + 1].max(),
        'beta_63': np.cov(r[window], rm[window])[0, 1] / np.var(rm[window], ddof=1),  # market
        'idio_vol_63': np.std(r[window], ddof=1) * math.sqrt(1 - np.corrcoef(r[window], rm[window])[0, 1] ** 2),
        'rel_ret_21': (c[t] / c[t - 21] - 1.0) - (m[t] / m[t - 21] - 1.0),
        'mkt_r2_63': np.corrcoef(r[window], rm[window])[0, 1] ** 2,
        'amihud_21': np.mean(illiquid),                                               # liquidity
        'zero_vol_63': float((v[window] <= 0).mean()),
        'vol_trend_5_63': v[t - 4:t + 1].mean() / v[window].mean(),
        'volume_cv_21': np.std(v[t - 20:t + 1], ddof=1) / v[t - 20:t + 1].mean(),
        'trend_r2_63': r2,                                                            # trend_quality
        'trend_t_63': slope / se,
        'up_days_21': float((r[t - 20:t + 1] > 0).mean()),
        'autocorr_63': np.corrcoef(r[window], r[t - 63:t])[0, 1],
        'overnight_21': (o[t - 20:t + 1] / c[t - 21:t] - 1.0).mean(),                # overnight_intraday
        'intraday_21': (c[t - 20:t + 1] / o[t - 20:t + 1] - 1.0).mean(),
        'rsi_14': 100.0 * gain[t - 1] / (gain[t - 1] + loss[t - 1]),                  # oscillators
        'stoch_k_14': 100.0 * (c[t] - lo[t - 13:t + 1].min()) / (h[t - 13:t + 1].max() - lo[t - 13:t + 1].min()),
        'williams_r_14': -100.0 * (h[t - 13:t + 1].max() - c[t]) / (h[t - 13:t + 1].max() - lo[t - 13:t + 1].min()),
        'boll_b_20': (c[t] - (sma(20) - 2 * np.std(c[t - 19:t + 1]))) / (4 * np.std(c[t - 19:t + 1])),
        'cmf_21': (((c - lo) - (h - c)) / (h - lo) * v)[t - 20:t + 1].sum() / v[t - 20:t + 1].sum(),
        'seas_1y': lagged,                                                            # seasonality
    }
    for name, value in expected.items():
        assert frame[name].iloc[t] == pytest.approx(value, rel=1e-8, abs=1e-12), name
    # downbeta needs at least 30 falling-market days in its window; checked on a 126-day window
    window126 = slice(t - 125, t + 1)
    falling = rm[window126] < 0
    assert falling.sum() >= 30
    slope = np.polyfit(rm[window126][falling], r[window126][falling], 1)[0]
    assert frame['downbeta_126'].iloc[t] == pytest.approx(slope, rel=1e-8)
    # MACD line: EMA(12) - EMA(26) of the close (recursive, alpha = 2 / (span + 1)), divided by the close
    ema = {span: _wilder(c, (span + 1) / 2.0) for span in (12, 26)}
    assert frame['macd_line'].iloc[t] == pytest.approx((ema[12][t] - ema[26][t]) / c[t], rel=1e-8)


def test_ohlc_and_market_features_are_missing_without_their_inputs(stock: Dict[str, pd.Series]) -> None:
    without = F.stock_feature_frame(F.StockSeries.build(stock['close'], stock['volume']),
                                    list(F.FEATURE_FUNCTIONS), horizon_days=126)
    needs_ohlc = {'parkinson_21', 'garman_klass_63', 'hl_range_5', 'overnight_21', 'intraday_63', 'stoch_k_14',
                  'williams_r_14', 'cci_20', 'mfi_14', 'cmf_21', 'adx_14'}
    needs_market = {name for name, family in F.FEATURE_FAMILIES.items() if family == 'market'}
    for name in needs_ohlc | needs_market:
        assert without[name].isna().all(), name
    others = [name for name in F.FEATURE_FUNCTIONS if name not in needs_ohlc | needs_market
              and not name.startswith(('parkinson', 'garman', 'hl_range', 'overnight', 'intraday', 'stoch',
                                       'williams', 'cci', 'mfi', 'cmf', 'adx'))]
    assert without[others].iloc[-1].notna().all()


def test_market_closes_are_carried_to_the_stock_days() -> None:
    days = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-05'])
    market = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-04']))
    series = F.StockSeries.build(pd.Series([10.0, 11.0, 12.0], index=days), market=market)
    # 2024-01-02 carries the close of 2024-01-01; 2024-01-05 that of 2024-01-04
    assert series.market.tolist() == [100.0, 110.0, 121.0]
    assert series.market_returns.iloc[1:].tolist() == pytest.approx([0.1, 0.1])
