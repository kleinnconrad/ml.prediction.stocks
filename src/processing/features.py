"""
Step 1 features of the pooled cross-sectional model and the candidate pool of the variable review.

Every feature is computed per stock on the stock's own trading days (the rows where the
stock has a close) from trailing windows only. A feature function receives a
`StockSeries`: the stock's close, volume, open, high and low on its own days and the
close of the market index (`market_index_ticker`) carried to those days. The registry
`FEATURE_FUNCTIONS` holds every named feature; `FEATURE_FAMILIES` maps each name to its
family. Step 1 computes only the features in `step1_features`; the quarterly variable
review (src/modeling/variable_review.py) computes the whole registry as its candidate pool.

The per-stock rows are aligned on the union calendar of all downloaded trading days: a day
without a trade carries the stock's last trading day for at most `step1_ffill_limit_days`
days. On every date, only eligible stocks are ranked: at least `step1_min_history_days`
own trading days up to the date, a positive 126-day traded value, a price above
`step1_min_price_eur` and a last trade within the carry limit. The rule does not depend on
the feature list. The model inputs are the within-date percentile ranks minus 0.5 over the
eligible stocks that have a value; an eligible stock without a value of a feature gets the
neutral rank 0.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..ingestion.market_api import PricePanel

logger = logging.getLogger(__name__)

# Eligibility inputs that are computed even when they are not model features
TRADED_VALUE_FEATURE = 'log_traded_value'
HISTORY_COLUMN = 'history_days'
RANK_PREFIX = 'r_'
# Model input of an eligible stock without a value of a feature (the median rank)
NEUTRAL_RANK = 0.0
# Absolute daily return below which a day counts as unchanged (zero-return share)
ZERO_RETURN_TOLERANCE = 1e-12
# Family of the features that need the market index
MARKET_FAMILY = 'market'


@dataclass
class StockSeries:
    """
    Inputs of the feature functions: one stock on its own trading days.

    Build instances with `StockSeries.build()`. Intermediate series that several features
    share (returns, moving averages, rolling extremes) are computed once and memoized.

    Attributes:
        close (pd.Series): Adjusted closes; the index holds the stock's own trading days.
        volume (pd.Series): Volumes on the same days; a missing volume counts as 0.
        open (pd.Series): Adjusted opens on the same days (NaN where unknown).
        high (pd.Series): Adjusted highs on the same days (NaN where unknown).
        low (pd.Series): Adjusted lows on the same days (NaN where unknown).
        market (pd.Series): Close of the market index on or before each day (NaN before
            the first index close or if the index is unavailable).
    """
    close: pd.Series
    volume: pd.Series
    open: pd.Series
    high: pd.Series
    low: pd.Series
    market: pd.Series
    _memo: Dict[Tuple[Any, ...], pd.Series] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def build(cls, close: pd.Series, volume: Optional[pd.Series] = None, open_: Optional[pd.Series] = None,
              high: Optional[pd.Series] = None, low: Optional[pd.Series] = None,
              market: Optional[pd.Series] = None) -> 'StockSeries':
        """
        Restricts the inputs to the stock's own trading days.

        Args:
            close (pd.Series): Adjusted closes; NaN rows (days without a trade) are dropped.
            volume (Optional[pd.Series]): Volumes; missing values count as 0.
            open_ (Optional[pd.Series]): Adjusted opens.
            high (Optional[pd.Series]): Adjusted highs.
            low (Optional[pd.Series]): Adjusted lows.
            market (Optional[pd.Series]): Closes of the market index on its own days.

        Returns:
            StockSeries: The aligned inputs.
        """
        c = close.dropna().astype(float)
        index = c.index

        def own(values: Optional[pd.Series]) -> pd.Series:
            if values is None:
                return pd.Series(np.nan, index=index, dtype=float)
            return values.reindex(index).astype(float)

        if market is not None and not market.dropna().empty:
            m = market.dropna().astype(float).sort_index()
            m = m[~m.index.duplicated(keep='last')]
            # The index close on or before each own day: a stock return and the market return
            # then cover the same interval
            m = m.reindex(index, method='ffill')
        else:
            m = pd.Series(np.nan, index=index, dtype=float)
        return cls(close=c, volume=own(volume).fillna(0.0), open=own(open_), high=own(high), low=own(low), market=m)

    def memo(self, key: Tuple[Any, ...], compute: Callable[[], pd.Series]) -> pd.Series:
        """
        Returns a memoized intermediate series, computing it on first use.

        Args:
            key (Tuple[Any, ...]): Identifier of the series.
            compute (Callable[[], pd.Series]): Computes the series.

        Returns:
            pd.Series: The series.
        """
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    @property
    def returns(self) -> pd.Series:
        """
        Daily returns between consecutive own trading days.

        Returns:
            pd.Series: close[t] / close[t - 1] - 1 (NaN on the first day).
        """
        return self.memo(('returns',), lambda: self.close.pct_change())

    @property
    def market_returns(self) -> pd.Series:
        """
        Market index returns over the same intervals as `returns`.

        Returns:
            pd.Series: The index return between consecutive own trading days (NaN where the
            stock return is NaN).
        """
        return self.memo(('market_returns',), lambda: self.market.pct_change().where(self.returns.notna()))

    @property
    def traded_value(self) -> pd.Series:
        """
        Daily traded value.

        Returns:
            pd.Series: close x volume.
        """
        return self.memo(('traded_value',), lambda: self.close * self.volume)

    def sma(self, window: int) -> pd.Series:
        """
        Simple moving average of the close.

        Args:
            window (int): Number of own trading days.

        Returns:
            pd.Series: The mean close of the last `window` days (NaN before `window` days).
        """
        return self.memo(('sma', window), lambda: self.close.rolling(window).mean())

    def rolling_max(self, window: int) -> pd.Series:
        """
        Highest close.

        Args:
            window (int): Number of own trading days.

        Returns:
            pd.Series: The highest close of the last `window` days (NaN before `window` days).
        """
        return self.memo(('max', window), lambda: self.close.rolling(window).max())

    def rolling_min(self, window: int) -> pd.Series:
        """
        Lowest close.

        Args:
            window (int): Number of own trading days.

        Returns:
            pd.Series: The lowest close of the last `window` days (NaN before `window` days).
        """
        return self.memo(('min', window), lambda: self.close.rolling(window).min())

    def volatility(self, window: int) -> pd.Series:
        """
        Sample standard deviation of the daily returns.

        Args:
            window (int): Number of daily returns.

        Returns:
            pd.Series: The standard deviation (ddof 1) of the last `window` returns.
        """
        return self.memo(('vol', window), lambda: self.returns.rolling(window).std())

    def market_correlation(self, window: int) -> pd.Series:
        """
        Correlation of the daily returns with the market returns.

        Args:
            window (int): Number of daily returns.

        Returns:
            pd.Series: The correlation over the last `window` returns.
        """
        return self.memo(('corr', window), lambda: self.returns.rolling(window).corr(self.market_returns))

    def market_beta(self, window: int) -> pd.Series:
        """
        Market beta of the daily returns.

        Args:
            window (int): Number of daily returns.

        Returns:
            pd.Series: The covariance with the market returns divided by their variance over
            the last `window` returns.
        """
        return self.memo(('beta', window), lambda: self.returns.rolling(window).cov(self.market_returns)
                         / self.market_returns.rolling(window).var())


FeatureFunction = Callable[[StockSeries], pd.Series]


def _series(values: np.ndarray, like: pd.Series) -> pd.Series:
    """Wraps an array in a float series with the index of `like`."""
    return pd.Series(values, index=like.index, dtype=float)


def _window_result(series: pd.Series, window: int, reduce: Callable[[np.ndarray], np.ndarray]) -> pd.Series:
    """
    Applies a reduction to every trailing window of `window` values.

    Args:
        series (pd.Series): Input values.
        window (int): Window length; the first window - 1 rows are NaN.
        reduce (Callable[[np.ndarray], np.ndarray]): Maps the (n_windows x window) matrix
            to one value per window. NaN inputs propagate to the windows that contain them.

    Returns:
        pd.Series: The reduced values.
    """
    out = np.full(len(series), np.nan)
    if len(series) >= window:
        with np.errstate(invalid='ignore', divide='ignore'):
            out[window - 1:] = reduce(sliding_window_view(series.to_numpy(dtype=float), window))
    return _series(out, series)


def _positive(values: pd.Series) -> pd.Series:
    """The values where they are positive, NaN elsewhere."""
    return values.where(values > 0)


# ----------------------------------------------------------------------------------------
# Returns and momentum
# ----------------------------------------------------------------------------------------
def _period_return(periods: int) -> FeatureFunction:
    """
    Family `returns`: the `periods`-day return, close[t] / close[t - periods] - 1.

    Args:
        periods (int): Look-back in own trading days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.close.pct_change(periods)
    return feature


def _skip_momentum(start_lag: int, end_lag: int) -> FeatureFunction:
    """
    Family `momentum`: return from `start_lag` to `end_lag` days ago, close[t - end_lag] / close[t - start_lag] - 1.

    The last `end_lag` days are skipped (e.g. 12-1 momentum: 252 to 21 days ago;
    24-12 momentum, the long-term reversal: 504 to 252 days ago).

    Args:
        start_lag (int): Start of the period in own trading days before t.
        end_lag (int): End of the period in own trading days before t.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.close.shift(end_lag) / s.close.shift(start_lag) - 1.0
    return feature


def momentum_12_1(s: StockSeries) -> pd.Series:
    """
    12-month momentum without the latest month: close[t-21] / close[t-252] - 1.

    Args:
        s (StockSeries): The stock on its own trading days.

    Returns:
        pd.Series: The feature.
    """
    return s.close.shift(21) / s.close.shift(252) - 1.0


# ----------------------------------------------------------------------------------------
# Moving averages
# ----------------------------------------------------------------------------------------
def _distance_to_sma(window: int) -> FeatureFunction:
    """
    Family `sma_distance`: relative distance to the `window`-day simple moving average, close / SMA - 1.

    Args:
        window (int): Window of the moving average.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.close / s.sma(window) - 1.0
    return feature


def _sma_crossover(fast: int, slow: int) -> FeatureFunction:
    """
    Family `sma_crossover`: ratio of two simple moving averages, SMA(fast) / SMA(slow) - 1.

    Args:
        fast (int): Window of the shorter average.
        slow (int): Window of the longer average.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.sma(fast) / s.sma(slow) - 1.0
    return feature


def _sma_slope(window: int, lag: int) -> FeatureFunction:
    """
    Family `sma_slope`: change of the `window`-day SMA over `lag` days, SMA[t] / SMA[t - lag] - 1.

    Args:
        window (int): Window of the moving average.
        lag (int): Look-back of the change in own trading days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        average = s.sma(window)
        return average / average.shift(lag) - 1.0
    return feature


# ----------------------------------------------------------------------------------------
# Highs, lows and drawdowns
# ----------------------------------------------------------------------------------------
def _distance_to_high(window: int) -> FeatureFunction:
    """
    Family `highs_lows`: distance to the `window`-day high, close / max(close) - 1 (0 at a new high).

    Args:
        window (int): Look-back of the high, including today.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.close / s.rolling_max(window) - 1.0
    return feature


def drawdown_252(s: StockSeries) -> pd.Series:
    """
    Relative distance of the close to its 252-day maximum (0 at a new high, negative below).

    Args:
        s (StockSeries): The stock on its own trading days.

    Returns:
        pd.Series: The feature.
    """
    return s.close / s.close.rolling(252).max() - 1.0


def _distance_to_low(window: int) -> FeatureFunction:
    """
    Family `highs_lows`: distance to the `window`-day low, close / min(close) - 1 (0 at a new low).

    Args:
        window (int): Look-back of the low, including today.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.close / s.rolling_min(window) - 1.0
    return feature


def _range_position(window: int) -> FeatureFunction:
    """
    Family `highs_lows`: position of the close in its `window`-day range, (close - low) / (high - low).

    Args:
        window (int): Look-back of the range; NaN if the range is zero.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        high, low = s.rolling_max(window), s.rolling_min(window)
        spread = high - low
        return ((s.close - low) / spread).where(spread > 0)
    return feature


def _days_since_extreme(window: int, highest: bool) -> FeatureFunction:
    """
    Family `highs_lows`: days since the `window`-day high (or low), divided by window - 1.

    0 means the extreme is today, 1 that it is the oldest day of the window; ties count
    the latest occurrence.

    Args:
        window (int): Look-back, including today.
        highest (bool): True for the high, False for the low.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        def reduce(windows: np.ndarray) -> np.ndarray:
            newest_first = windows[:, ::-1]
            position = np.argmax(newest_first, axis=1) if highest else np.argmin(newest_first, axis=1)
            return position / (window - 1.0)
        return _window_result(s.close, window, reduce)
    return feature


def _drawdowns_in_window(windows: np.ndarray) -> np.ndarray:
    """Drawdown of every close of each window from the running maximum since the window start."""
    return windows / np.maximum.accumulate(windows, axis=1) - 1.0


def _max_drawdown(window: int) -> FeatureFunction:
    """
    Family `highs_lows`: maximum drawdown within the last `window` days (0 or negative).

    The drawdown of a close is measured from the highest close since the start of the window.

    Args:
        window (int): Window length.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return _window_result(s.close, window, lambda w: _drawdowns_in_window(w).min(axis=1))
    return feature


def _ulcer_index(window: int) -> FeatureFunction:
    """
    Family `highs_lows`: Ulcer index, the root mean square drawdown within the last `window` days.

    The drawdown of a close is measured from the highest close since the start of the window.

    Args:
        window (int): Window length.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return _window_result(s.close, window, lambda w: np.sqrt((_drawdowns_in_window(w) ** 2).mean(axis=1)))
    return feature


# ----------------------------------------------------------------------------------------
# Volatility
# ----------------------------------------------------------------------------------------
def _volatility(window: int) -> FeatureFunction:
    """
    Family `volatility`: sample standard deviation of the daily returns over `window` days.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.volatility(window)
    return feature


def _semi_volatility(window: int, downside: bool) -> FeatureFunction:
    """
    Family `volatility`: downside (or upside) volatility, sqrt(mean(min(r, 0)^2)) (or max(r, 0)) over `window` days.

    Args:
        window (int): Number of daily returns.
        downside (bool): True for the negative returns, False for the positive ones.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        part = s.returns.clip(upper=0.0) if downside else s.returns.clip(lower=0.0)
        return np.sqrt((part ** 2).rolling(window).mean())
    return feature


def _volatility_ratio(short: int, long: int) -> FeatureFunction:
    """
    Family `volatility`: ratio of the `short`-day to the `long`-day volatility.

    Args:
        short (int): Window of the numerator.
        long (int): Window of the denominator.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.volatility(short) / s.volatility(long)
    return feature


def _volatility_of_volatility(window: int) -> FeatureFunction:
    """
    Family `volatility`: coefficient of variation of the 21-day volatility over `window` days.

    Args:
        window (int): Number of daily 21-day volatility values.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        vol = s.volatility(21)
        return vol.rolling(window).std() / _positive(vol.rolling(window).mean())
    return feature


# ----------------------------------------------------------------------------------------
# Range-based volatility (open, high, low)
# ----------------------------------------------------------------------------------------
def _log_high_low(s: StockSeries) -> pd.Series:
    """ln(high / low) per day; NaN for missing or inconsistent prices (high < low or not positive)."""
    def compute() -> pd.Series:
        valid = (s.high > 0) & (s.low > 0) & (s.high >= s.low)
        return np.log(s.high.where(valid) / s.low.where(valid))
    return s.memo(('log_hl',), compute)


def _parkinson(window: int) -> FeatureFunction:
    """
    Family `range_volatility`: Parkinson volatility, sqrt(mean(ln(H/L)^2) / (4 ln 2)) over `window` days.

    Args:
        window (int): Number of days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return np.sqrt((_log_high_low(s) ** 2).rolling(window).mean() / (4.0 * math.log(2.0)))
    return feature


def _garman_klass(window: int) -> FeatureFunction:
    """
    Family `range_volatility`: Garman-Klass volatility over `window` days.

    sqrt(mean(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2)); NaN if the mean is not positive.

    Args:
        window (int): Number of days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        valid = (s.open > 0) & (s.close > 0)
        log_co = np.log(s.close.where(valid) / s.open.where(valid))
        term = 0.5 * _log_high_low(s) ** 2 - (2.0 * math.log(2.0) - 1.0) * log_co ** 2
        return np.sqrt(_positive(term.rolling(window).mean()))
    return feature


def _high_low_range(window: int) -> FeatureFunction:
    """
    Family `range_volatility`: mean daily range relative to the close, mean((high - low) / close) over `window` days.

    Args:
        window (int): Number of days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        spread = (s.high - s.low).where(s.high >= s.low)
        return (spread / s.close).rolling(window).mean()
    return feature


# ----------------------------------------------------------------------------------------
# Higher moments and extreme returns
# ----------------------------------------------------------------------------------------
def _skewness(window: int) -> FeatureFunction:
    """
    Family `higher_moments`: sample skewness of the daily returns over `window` days.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.returns.rolling(window).skew()
    return feature


def _kurtosis(window: int) -> FeatureFunction:
    """
    Family `higher_moments`: sample excess kurtosis of the daily returns over `window` days.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.returns.rolling(window).kurt()
    return feature


def _extreme_return(window: int, highest: bool) -> FeatureFunction:
    """
    Family `higher_moments`: largest (or smallest) daily return over `window` days.

    Args:
        window (int): Number of daily returns.
        highest (bool): True for the maximum, False for the minimum.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        rolling = s.returns.rolling(window)
        return rolling.max() if highest else rolling.min()
    return feature


# ----------------------------------------------------------------------------------------
# Market-relative (market index carried to the stock's own days)
# ----------------------------------------------------------------------------------------
def _beta(window: int) -> FeatureFunction:
    """
    Family `market`: beta against the market index over `window` days, cov(r, r_m) / var(r_m).

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.market_beta(window)
    return feature


def _correlation(window: int) -> FeatureFunction:
    """
    Family `market`: correlation of the daily returns with the market returns over `window` days.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.market_correlation(window)
    return feature


def _idiosyncratic_volatility(window: int) -> FeatureFunction:
    """
    Family `market`: idiosyncratic volatility over `window` days, vol * sqrt(1 - corr^2).

    The residual standard deviation of the market model, expressed with the stock's
    volatility and its correlation with the market.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.volatility(window) * np.sqrt((1.0 - s.market_correlation(window) ** 2).clip(lower=0.0))
    return feature


def _conditional_beta(window: int, downside: bool) -> FeatureFunction:
    """
    Family `market`: beta over the days with a falling (or rising) market within `window` days.

    The slope of the stock return on the market return over the days with r_m < 0 (or
    r_m > 0); NaN with fewer than 30 such days or no variation of r_m.

    Args:
        window (int): Number of daily returns.
        downside (bool): True for the falling-market days, False for the rising ones.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        r, rm = s.returns, s.market_returns
        condition = rm < 0 if downside else rm > 0
        indicator = condition.astype(float).where(rm.notna() & r.notna())
        n_days = indicator.rolling(window).sum()
        sum_x = (indicator * rm).rolling(window).sum()
        sum_y = (indicator * r).rolling(window).sum()
        sum_xx = (indicator * rm * rm).rolling(window).sum()
        sum_xy = (indicator * rm * r).rolling(window).sum()
        denominator = sum_xx - sum_x * sum_x / n_days
        return ((sum_xy - sum_x * sum_y / n_days) / denominator).where((n_days >= 30) & (denominator > 0))
    return feature


def _market_r_squared(window: int) -> FeatureFunction:
    """
    Family `market`: R-squared of the market model over `window` days, corr(r, r_m)^2.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.market_correlation(window) ** 2
    return feature


def _residual_momentum(window: int, skip: int) -> FeatureFunction:
    """
    Family `market`: residual momentum, the market-model residual return from `window` to `skip` days ago.

    (sum of r - beta * sum of r_m) over the days t - window + 1 ... t - skip, divided by
    the idiosyncratic volatility times sqrt(window - skip); beta and the idiosyncratic
    volatility are estimated over the last `window` days.

    Args:
        window (int): Estimation window and start of the period (252 for 12-1).
        skip (int): Latest days left out (21 for 12-1).

    Returns:
        FeatureFunction: The feature function.
    """
    span = window - skip
    idiosyncratic = _idiosyncratic_volatility(window)

    def feature(s: StockSeries) -> pd.Series:
        sum_stock = s.returns.rolling(span).sum().shift(skip)
        sum_market = s.market_returns.rolling(span).sum().shift(skip)
        idio = idiosyncratic(s)
        return ((sum_stock - s.market_beta(window) * sum_market) / (idio * math.sqrt(span))).where(idio > 0)
    return feature


def _relative_return(periods: int) -> FeatureFunction:
    """
    Family `market`: relative strength, the `periods`-day return minus the market return over the same days.

    Args:
        periods (int): Look-back in own trading days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.close.pct_change(periods) - s.market.pct_change(periods)
    return feature


# ----------------------------------------------------------------------------------------
# Liquidity and volume
# ----------------------------------------------------------------------------------------
def _log_traded_value(window: int) -> FeatureFunction:
    """
    Family `liquidity`: log of one plus the mean daily traded value (close x volume) over `window` days.

    Args:
        window (int): Number of days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return np.log1p(s.traded_value.rolling(window).mean())
    return feature


def log_traded_value(s: StockSeries) -> pd.Series:
    """
    Log of one plus the mean daily traded value (close x volume) over 126 trading days.

    Also the eligibility input: a stock needs a positive value. A missing volume counts as 0.

    Args:
        s (StockSeries): The stock on its own trading days.

    Returns:
        pd.Series: The feature.
    """
    return np.log1p((s.close * s.volume).rolling(126).mean())


def _amihud(window: int) -> FeatureFunction:
    """
    Family `liquidity`: Amihud illiquidity, mean(|r| / traded value) over `window` days.

    Days without traded value are left out of the mean; the window needs `window` returns.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        r, value = s.returns, s.traded_value
        illiquidity = (r.abs() / value).where(value > 0).where(r.notna())
        return illiquidity.rolling(window, min_periods=1).mean().where(r.rolling(window).count() == window)
    return feature


def _zero_return_share(window: int) -> FeatureFunction:
    """
    Family `liquidity`: share of days with an unchanged close (|r| < 1e-12) within `window` days.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        r = s.returns
        return (r.abs() < ZERO_RETURN_TOLERANCE).astype(float).where(r.notna()).rolling(window).mean()
    return feature


def _zero_volume_share(window: int) -> FeatureFunction:
    """
    Family `liquidity`: share of days without volume within `window` days.

    Args:
        window (int): Number of days (counted on the days with a return).

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return (s.volume <= 0).astype(float).where(s.returns.notna()).rolling(window).mean()
    return feature


def _volume_trend(short: int, long: int) -> FeatureFunction:
    """
    Family `liquidity`: mean volume over `short` days divided by the mean volume over `long` days.

    Args:
        short (int): Window of the numerator.
        long (int): Window of the denominator; NaN if its mean is zero.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.volume.rolling(short).mean() / _positive(s.volume.rolling(long).mean())
    return feature


def _volume_variation(window: int) -> FeatureFunction:
    """
    Family `liquidity`: coefficient of variation of the daily volume over `window` days.

    Args:
        window (int): Number of days; NaN if the mean volume is zero.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        rolling = s.volume.rolling(window)
        return rolling.std() / _positive(rolling.mean())
    return feature


# ----------------------------------------------------------------------------------------
# Trend quality
# ----------------------------------------------------------------------------------------
def _log_price_trend(window: int, statistic: str) -> FeatureFunction:
    """
    Family `trend_quality`: R-squared or slope t-statistic of a linear trend of the log close over `window` days.

    The log close of the last `window` days is regressed on the day number.

    Args:
        window (int): Window length.
        statistic (str): 'r2' or 't'.

    Returns:
        FeatureFunction: The feature function.
    """
    time = np.arange(window, dtype=float) - (window - 1) / 2.0
    sum_tt = float(time @ time)

    def reduce(windows: np.ndarray) -> np.ndarray:
        slope = windows @ time / sum_tt
        centered = windows - windows.mean(axis=1, keepdims=True)
        total = (centered ** 2).sum(axis=1)
        explained = slope ** 2 * sum_tt
        if statistic == 'r2':
            return np.where(total > 0, explained / total, np.nan)
        residual = np.clip(total - explained, 0.0, None)
        standard_error = np.sqrt(residual / (window - 2) / sum_tt)
        return np.where(standard_error > 0, slope / standard_error, np.nan)

    def feature(s: StockSeries) -> pd.Series:
        return _window_result(np.log(_positive(s.close)), window, reduce)
    return feature


def _up_day_share(window: int) -> FeatureFunction:
    """
    Family `trend_quality`: share of days with a positive return within `window` days.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        r = s.returns
        return (r > 0).astype(float).where(r.notna()).rolling(window).mean()
    return feature


def _information_discreteness(window: int) -> FeatureFunction:
    """
    Family `trend_quality`: information discreteness ("frog in the pan", Da, Gurun and Warachka, 2014).

    sign(return over `window` days) x (share of down days - share of up days) within the
    window; low values mean that the return accrued in many small moves.

    Args:
        window (int): Number of daily returns.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        r = s.returns
        up = (r > 0).astype(float).where(r.notna()).rolling(window).mean()
        down = (r < 0).astype(float).where(r.notna()).rolling(window).mean()
        return np.sign(s.close.pct_change(window)) * (down - up)
    return feature


def _autocorrelation(window: int) -> FeatureFunction:
    """
    Family `trend_quality`: first-order autocorrelation of the daily returns over `window` days.

    Args:
        window (int): Number of daily return pairs.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return s.returns.rolling(window).corr(s.returns.shift(1))
    return feature


# ----------------------------------------------------------------------------------------
# Overnight and intraday returns (open)
# ----------------------------------------------------------------------------------------
def _overnight_return(window: int) -> FeatureFunction:
    """
    Family `overnight_intraday`: mean overnight return, mean(open[t] / close[t - 1] - 1) over `window` days.

    Args:
        window (int): Number of days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return (_positive(s.open) / s.close.shift(1) - 1.0).rolling(window).mean()
    return feature


def _intraday_return(window: int) -> FeatureFunction:
    """
    Family `overnight_intraday`: mean intraday return, mean(close / open - 1) over `window` days.

    Args:
        window (int): Number of days.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        return (s.close / _positive(s.open) - 1.0).rolling(window).mean()
    return feature


# ----------------------------------------------------------------------------------------
# Oscillators
# ----------------------------------------------------------------------------------------
def _wilder_mean(values: pd.Series, window: int) -> pd.Series:
    """Wilder's moving average: exponential smoothing with alpha = 1 / window, defined after `window` values."""
    return values.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _rsi(window: int) -> FeatureFunction:
    """
    Family `oscillators`: relative strength index with Wilder's smoothing over `window` days.

    RSI = 100 x average gain / (average gain + average loss) of the daily close changes.

    Args:
        window (int): Smoothing window.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        change = s.close.diff()
        gain = _wilder_mean(change.clip(lower=0.0), window)
        loss = _wilder_mean((-change).clip(lower=0.0), window)
        return 100.0 * gain / _positive(gain + loss)
    return feature


def _stochastic_k(s: StockSeries, window: int) -> pd.Series:
    """Stochastic %K: 100 x (close - lowest low) / (highest high - lowest low) over `window` days."""
    def compute() -> pd.Series:
        highest, lowest = s.high.rolling(window).max(), s.low.rolling(window).min()
        spread = highest - lowest
        return (100.0 * (s.close - lowest) / spread).where(spread > 0)
    return s.memo(('stoch_k', window), compute)


def _stochastic(window: int, smoothing: int = 1) -> FeatureFunction:
    """
    Family `oscillators`: stochastic %K over `window` days, or its `smoothing`-day mean (%D).

    Args:
        window (int): Look-back of the high and low.
        smoothing (int): 1 for %K; 3 for the usual %D.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        k = _stochastic_k(s, window)
        return k if smoothing == 1 else k.rolling(smoothing).mean()
    return feature


def _williams_r(window: int) -> FeatureFunction:
    """
    Family `oscillators`: Williams %R over `window` days, -100 x (highest high - close) / (highest high - lowest low).

    Args:
        window (int): Look-back of the high and low.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        highest, lowest = s.high.rolling(window).max(), s.low.rolling(window).min()
        spread = highest - lowest
        return (-100.0 * (highest - s.close) / spread).where(spread > 0)
    return feature


def _macd_lines(s: StockSeries) -> Tuple[pd.Series, pd.Series]:
    """MACD line (12-day EMA minus 26-day EMA of the close) and its 9-day EMA (signal line)."""
    def line() -> pd.Series:
        fast = s.close.ewm(span=12, adjust=False, min_periods=12).mean()
        slow = s.close.ewm(span=26, adjust=False, min_periods=26).mean()
        return fast - slow
    macd = s.memo(('macd',), line)
    signal = s.memo(('macd_signal',), lambda: macd.ewm(span=9, adjust=False, min_periods=9).mean())
    return macd, signal


def _macd(component: str) -> FeatureFunction:
    """
    Family `oscillators`: MACD (12/26/9) line, signal line or histogram, divided by the close.

    Args:
        component (str): 'line', 'signal' or 'hist' (line minus signal).

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        macd, signal = _macd_lines(s)
        value = {'line': macd, 'signal': signal, 'hist': macd - signal}[component]
        return value / _positive(s.close)
    return feature


def _bollinger(window: int, component: str) -> FeatureFunction:
    """
    Family `oscillators`: Bollinger %B or bandwidth of bands at SMA +/- 2 standard deviations over `window` days.

    %B = (close - lower band) / (upper - lower band); bandwidth = (upper - lower band) / SMA.
    The standard deviation of the closes is the population one (ddof 0).

    Args:
        window (int): Window of the average and the standard deviation.
        component (str): 'b' or 'width'.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        middle = s.sma(window)
        band = 2.0 * s.close.rolling(window).std(ddof=0)
        if component == 'b':
            return ((s.close - (middle - band)) / (2.0 * band)).where(band > 0)
        return 2.0 * band / _positive(middle)
    return feature


def _typical_price(s: StockSeries) -> pd.Series:
    """Typical price (high + low + close) / 3."""
    return s.memo(('typical',), lambda: (s.high + s.low + s.close) / 3.0)


def _cci(window: int) -> FeatureFunction:
    """
    Family `oscillators`: commodity channel index over `window` days.

    (typical price - its SMA) / (0.015 x mean absolute deviation of the typical price).

    Args:
        window (int): Window length.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        typical = _typical_price(s)
        average = typical.rolling(window).mean()
        deviation = _window_result(typical, window,
                                   lambda w: np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1))
        return (typical - average) / _positive(0.015 * deviation)
    return feature


def _mfi(window: int) -> FeatureFunction:
    """
    Family `oscillators`: money flow index over `window` days.

    100 x positive money flow / (positive + negative money flow), where the money flow of
    a day is typical price x volume, positive if the typical price rose.

    Args:
        window (int): Window length.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        typical = _typical_price(s)
        flow = typical * s.volume
        change = typical.diff()
        positive = flow.where(change > 0, 0.0).where(change.notna())
        negative = flow.where(change < 0, 0.0).where(change.notna())
        sum_positive, sum_negative = positive.rolling(window).sum(), negative.rolling(window).sum()
        return 100.0 * sum_positive / _positive(sum_positive + sum_negative)
    return feature


def _chaikin_money_flow(window: int) -> FeatureFunction:
    """
    Family `oscillators`: Chaikin money flow over `window` days.

    sum(((close - low) - (high - close)) / (high - low) x volume) / sum(volume); a day with
    high = low contributes 0.

    Args:
        window (int): Window length.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        spread = s.high - s.low
        multiplier = (((s.close - s.low) - (s.high - s.close)) / spread.where(spread > 0)).where(spread > 0, 0.0)
        multiplier = multiplier.where(spread >= 0)
        flow = multiplier * s.volume
        return flow.rolling(window).sum() / _positive(s.volume.rolling(window).sum())
    return feature


def _obv_slope(window: int) -> FeatureFunction:
    """
    Family `oscillators`: slope of the on-balance volume over `window` days, divided by the mean volume.

    The on-balance volume adds the day's volume on an up day and subtracts it on a down
    day; its least-squares slope per day over the window does not depend on its level.

    Args:
        window (int): Window length.

    Returns:
        FeatureFunction: The feature function.
    """
    time = np.arange(window, dtype=float) - (window - 1) / 2.0
    sum_tt = float(time @ time)

    def feature(s: StockSeries) -> pd.Series:
        obv = (np.sign(s.close.diff()).fillna(0.0) * s.volume).cumsum()
        slope = _window_result(obv, window, lambda w: (w - w[:, :1]) @ time / sum_tt)
        return slope / _positive(s.volume.rolling(window).mean())
    return feature


def _adx(window: int) -> FeatureFunction:
    """
    Family `oscillators`: average directional index with Wilder's smoothing over `window` days.

    Args:
        window (int): Smoothing window of the directional movements and of the index.

    Returns:
        FeatureFunction: The feature function.
    """
    def feature(s: StockSeries) -> pd.Series:
        up, down = s.high.diff(), -s.low.diff()
        known = up.notna() & down.notna()
        plus_dm = up.where((up > down) & (up > 0), 0.0).where(known)
        minus_dm = down.where((down > up) & (down > 0), 0.0).where(known)
        previous = s.close.shift(1)
        true_range = pd.concat([s.high - s.low, (s.high - previous).abs(), (s.low - previous).abs()],
                               axis=1).max(axis=1, skipna=False)
        smoothed_range = _positive(_wilder_mean(true_range, window))
        plus_di = 100.0 * _wilder_mean(plus_dm, window) / smoothed_range
        minus_di = 100.0 * _wilder_mean(minus_dm, window) / smoothed_range
        dx = 100.0 * (plus_di - minus_di).abs() / _positive(plus_di + minus_di)
        return _wilder_mean(dx, window)
    return feature


# ----------------------------------------------------------------------------------------
# Seasonality
# ----------------------------------------------------------------------------------------
def _same_month_return(years_back: int) -> FeatureFunction:
    """
    Family `seasonality`: return of the coming calendar month `years_back` years earlier.

    On a day of month p, the return over calendar month p + 1 - 12 x years_back (from the
    last close of the month before to the last close of that month; Heston and Sadka, 2008).

    Args:
        years_back (int): 1 for the previous year, 2 for two years ago.

    Returns:
        FeatureFunction: The feature function.
    """
    lag = 12 * years_back - 1

    def feature(s: StockSeries) -> pd.Series:
        months = s.close.index.to_period('M')
        month_close = s.close.groupby(months).last()
        month_close = month_close.reindex(pd.period_range(month_close.index.min(), month_close.index.max(), freq='M'))
        month_return = month_close / month_close.shift(1) - 1.0
        return _series(month_return.shift(lag).reindex(months).to_numpy(), s.close)
    return feature


# ----------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------
FEATURE_FUNCTIONS: Dict[str, FeatureFunction] = {}
FEATURE_FAMILIES: Dict[str, str] = {}


def _register(family: str, entries: Dict[str, FeatureFunction]) -> None:
    """
    Adds the features of one family to the registry.

    Args:
        family (str): Family name.
        entries (Dict[str, FeatureFunction]): Feature functions by name.

    Raises:
        ValueError: If a name is registered twice.
    """
    for name, function in entries.items():
        if name in FEATURE_FUNCTIONS:
            raise ValueError(f"Feature {name} is registered twice.")
        FEATURE_FUNCTIONS[name] = function
        FEATURE_FAMILIES[name] = family


# One function per feature name, f(StockSeries) on a stock's own trading days. The model
# reads its feature list from `step1_features` in config/settings.yaml; the variable review
# tests the candidates listed by family in `review_candidate_features` there, and a test
# requires that list to contain every registered feature. A price
# level (the log price of study E6) is not registered: a past adjusted close is divided by
# the ratios of all later splits, so its level at a past date depends on splits that
# happened after that date.
_register('returns', {f"ret_{k}": _period_return(k) for k in (1, 5, 10, 21, 42, 63, 84, 126, 189, 252)})
_register('momentum', {'mom_12_1': momentum_12_1, **{
    f"mom_{a}_{b}": _skip_momentum(21 * a, 21 * b)
    for a, b in ((12, 2), (12, 3), (9, 1), (6, 1), (3, 1), (12, 6), (24, 12))}})
_register('sma_distance', {f"dist_sma{k}": _distance_to_sma(k) for k in (10, 20, 50, 100, 150, 200, 250)})
_register('sma_crossover', {f"sma{a}_sma{b}": _sma_crossover(a, b) for a, b in (
    (5, 20), (10, 20), (10, 50), (20, 50), (20, 100), (50, 200), (20, 200), (100, 200))})
_register('sma_slope', {f"sma{k}_chg_{lag}": _sma_slope(k, lag) for k, lag in (
    (20, 5), (20, 21), (50, 21), (50, 63), (100, 21), (100, 63), (200, 21), (200, 63))})
_register('highs_lows', {
    **{f"dist_high_{k}": _distance_to_high(k) for k in (10, 21, 63, 126)},
    'drawdown_252': drawdown_252,
    'dist_high_504': _distance_to_high(504),
    **{f"dist_low_{k}": _distance_to_low(k) for k in (10, 21, 63, 126, 252, 504)},
    **{f"range_pos_{k}": _range_position(k) for k in (21, 63, 126, 252, 504)},
    **{f"days_since_high_{k}": _days_since_extreme(k, highest=True) for k in (63, 126, 252)},
    **{f"days_since_low_{k}": _days_since_extreme(k, highest=False) for k in (63, 126, 252)},
    **{f"maxdd_{k}": _max_drawdown(k) for k in (63, 126, 252)},
    **{f"ulcer_{k}": _ulcer_index(k) for k in (63, 126, 252)},
})
_register('volatility', {
    **{f"vol_{k}": _volatility(k) for k in (5, 10, 21, 42, 63, 126, 252)},
    **{f"downvol_{k}": _semi_volatility(k, downside=True) for k in (21, 63, 126, 252)},
    **{f"upvol_{k}": _semi_volatility(k, downside=False) for k in (21, 63, 126, 252)},
    **{f"vol_ratio_{a}_{b}": _volatility_ratio(a, b) for a, b in (
        (5, 21), (5, 63), (10, 63), (21, 63), (21, 126), (21, 252), (63, 252))},
    **{f"vov_{k}": _volatility_of_volatility(k) for k in (126, 252)},
})
_register('range_volatility', {
    **{f"parkinson_{k}": _parkinson(k) for k in (10, 21, 63, 126)},
    **{f"garman_klass_{k}": _garman_klass(k) for k in (10, 21, 63, 126)},
    **{f"hl_range_{k}": _high_low_range(k) for k in (5, 10, 21, 63, 126)},
})
_register('higher_moments', {
    **{f"skew_{k}": _skewness(k) for k in (21, 63, 126, 252)},
    **{f"kurt_{k}": _kurtosis(k) for k in (21, 63, 126, 252)},
    **{f"max_ret_{k}": _extreme_return(k, highest=True) for k in (5, 10, 21, 63, 126, 252)},
    **{f"min_ret_{k}": _extreme_return(k, highest=False) for k in (5, 10, 21, 63, 126, 252)},
})
_register(MARKET_FAMILY, {
    **{f"beta_{k}": _beta(k) for k in (63, 126, 252, 504)},
    **{f"corr_{k}": _correlation(k) for k in (63, 126, 252)},
    **{f"idio_vol_{k}": _idiosyncratic_volatility(k) for k in (21, 63, 126, 252)},
    **{f"downbeta_{k}": _conditional_beta(k, downside=True) for k in (126, 252)},
    **{f"upbeta_{k}": _conditional_beta(k, downside=False) for k in (126, 252)},
    **{f"mkt_r2_{k}": _market_r_squared(k) for k in (63, 126, 252)},
    'resid_mom_12_1': _residual_momentum(252, 21),
    'resid_mom_6_1': _residual_momentum(126, 21),
    **{f"rel_ret_{k}": _relative_return(k) for k in (5, 10, 21, 63, 126, 252)},
})
_register('liquidity', {
    **{f"log_tv_{k}": _log_traded_value(k) for k in (5, 10, 21, 63)},
    'log_traded_value': log_traded_value,
    'log_tv_252': _log_traded_value(252),
    **{f"amihud_{k}": _amihud(k) for k in (10, 21, 63, 126, 252)},
    **{f"zero_ret_{k}": _zero_return_share(k) for k in (21, 63, 126, 252)},
    **{f"zero_vol_{k}": _zero_volume_share(k) for k in (21, 63, 126, 252)},
    **{f"vol_trend_{a}_{b}": _volume_trend(a, b) for a, b in (
        (5, 21), (5, 63), (21, 63), (21, 126), (21, 252), (63, 252))},
    **{f"volume_cv_{k}": _volume_variation(k) for k in (21, 63, 126)},
})
_register('trend_quality', {
    **{f"trend_r2_{k}": _log_price_trend(k, 'r2') for k in (21, 63, 126, 252)},
    **{f"trend_t_{k}": _log_price_trend(k, 't') for k in (21, 63, 126, 252)},
    **{f"up_days_{k}": _up_day_share(k) for k in (10, 21, 63, 126, 252)},
    **{f"id_{k}": _information_discreteness(k) for k in (63, 126, 252)},
    **{f"autocorr_{k}": _autocorrelation(k) for k in (21, 63, 126, 252)},
})
_register('overnight_intraday', {
    **{f"overnight_{k}": _overnight_return(k) for k in (5, 21, 63, 126)},
    **{f"intraday_{k}": _intraday_return(k) for k in (5, 21, 63, 126)},
})
_register('oscillators', {
    **{f"rsi_{k}": _rsi(k) for k in (7, 14, 21, 28)},
    'stoch_k_14': _stochastic(14), 'stoch_k_28': _stochastic(28), 'stoch_d_14': _stochastic(14, smoothing=3),
    'williams_r_14': _williams_r(14),
    'macd_line': _macd('line'), 'macd_signal': _macd('signal'), 'macd_hist': _macd('hist'),
    **{f"boll_b_{k}": _bollinger(k, 'b') for k in (20, 50)},
    **{f"boll_width_{k}": _bollinger(k, 'width') for k in (20, 50)},
    **{f"cci_{k}": _cci(k) for k in (20, 50)},
    **{f"mfi_{k}": _mfi(k) for k in (14, 28)},
    **{f"cmf_{k}": _chaikin_money_flow(k) for k in (21, 63)},
    **{f"obv_slope_{k}": _obv_slope(k) for k in (21, 63, 126)},
    **{f"adx_{k}": _adx(k) for k in (14, 28)},
})
_register('seasonality', {'seas_1y': _same_month_return(1), 'seas_2y': _same_month_return(2)})


def validate_feature_names(feature_names: Sequence[str]) -> List[str]:
    """
    Checks a feature list against the registry.

    Args:
        feature_names (Sequence[str]): Requested feature names.

    Returns:
        List[str]: The names as a list.

    Raises:
        ValueError: If the list is empty, contains duplicates or unknown names.
    """
    names = list(feature_names)
    if not names:
        raise ValueError("The Step 1 feature list (step1_features) is empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"The Step 1 feature list contains duplicates: {names}")
    unknown = [name for name in names if name not in FEATURE_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown Step 1 features {unknown}; known features: {sorted(FEATURE_FUNCTIONS)}")
    return names


def check_market_index(price_panel: PricePanel, feature_names: Sequence[str]) -> None:
    """
    Stops a run that requests market-relative features without the market index.

    Without the index (a failed download of `market_index_ticker`), every market-relative
    feature would be missing, enter the model at the neutral rank and get no weight, so the
    run would continue with a silently reduced feature list.

    Args:
        price_panel (PricePanel): Downloaded prices and the market index.
        feature_names (Sequence[str]): Requested features.

    Raises:
        ValueError: If a feature of the family `market` is requested and the index has no close.
    """
    needs_index = [name for name in feature_names if FEATURE_FAMILIES.get(name) == MARKET_FAMILY]
    if needs_index and (price_panel.market is None or price_panel.market.dropna().empty):
        raise ValueError(f"The market index (market_index_ticker) has no prices, so the market-relative "
                         f"features {needs_index} cannot be computed. Run again when the index downloads.")


def rank_columns(feature_names: Sequence[str]) -> List[str]:
    """
    Returns the names of the rank columns of the long panel.

    Args:
        feature_names (Sequence[str]): Feature names.

    Returns:
        List[str]: 'r_<feature>' for every feature.
    """
    return [f"{RANK_PREFIX}{name}" for name in feature_names]


def stock_series(price_panel: PricePanel, ticker: str) -> StockSeries:
    """
    Extracts the feature inputs of one ticker from a price panel.

    Args:
        price_panel (PricePanel): Downloaded prices; open, high, low and the market index
            may be absent (their features are then missing).
        ticker (str): A column of `price_panel.close`.

    Returns:
        StockSeries: The ticker on its own trading days.
    """
    def column(frame: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        return frame[ticker] if frame is not None and ticker in frame.columns else None

    return StockSeries.build(price_panel.close[ticker], volume=column(price_panel.volume),
                             open_=column(price_panel.open), high=column(price_panel.high),
                             low=column(price_panel.low), market=price_panel.market)


def stock_feature_frame(series: StockSeries, feature_names: Sequence[str], horizon_days: int) -> pd.DataFrame:
    """
    Computes the features, forward return and label end date of one stock on its own trading days.

    Args:
        series (StockSeries): The stock on its own trading days.
        feature_names (Sequence[str]): Features to compute.
        horizon_days (int): Forward horizon in the stock's own trading days.

    Returns:
        pd.DataFrame: One row per trading day with the features, `log_traded_value` (needed
        for eligibility), `fwd` (close[t + horizon] / close[t] - 1), `label_end` (the date of
        that later close), `price` and `history_days` (own trading days up to the row).
        Missing and non-finite feature values are NaN.
    """
    names = validate_feature_names(feature_names)
    c = series.close
    columns: Dict[str, Any] = {}
    for name in dict.fromkeys(names + [TRADED_VALUE_FEATURE]):
        # A non-finite value (e.g. a return against a zero close) counts as missing
        columns[name] = FEATURE_FUNCTIONS[name](series).astype(float).replace([np.inf, -np.inf], np.nan)
    own_dates = pd.Series(c.index, index=c.index)
    columns['fwd'] = c.shift(-horizon_days) / c - 1.0
    columns['label_end'] = own_dates.shift(-horizon_days)
    columns['price'] = c
    columns[HISTORY_COLUMN] = np.arange(1, len(c) + 1, dtype=float)
    return pd.DataFrame(columns, index=c.index)


def eligible_rows(history_days: np.ndarray, traded_value: np.ndarray, price: np.ndarray,
                  min_price: float, min_history_days: int) -> np.ndarray:
    """
    The eligibility rule of Step 1 and the variable review; it does not depend on the features.

    A row is eligible if the stock has at least `min_history_days` own trading days up to
    the date, a finite and positive 126-day traded value (log_traded_value) and a price
    above `min_price`. A stale row (last trade older than the carry limit) has no price
    and is therefore not eligible.

    Args:
        history_days (np.ndarray): Own trading days up to each row (NaN if stale).
        traded_value (np.ndarray): log_traded_value of each row.
        price (np.ndarray): Close of each row (NaN if stale).
        min_price (float): Minimum price (exclusive), `step1_min_price_eur`.
        min_history_days (int): Minimum own trading days, `step1_min_history_days`.

    Returns:
        np.ndarray: Boolean mask with the shape of the inputs.
    """
    history = np.nan_to_num(np.asarray(history_days, dtype=float), nan=0.0)
    value = np.asarray(traded_value, dtype=float)
    ok = history >= min_history_days
    ok &= np.isfinite(value) & (np.nan_to_num(value, nan=-1.0) > 0)
    ok &= np.nan_to_num(np.asarray(price, dtype=float), nan=0.0) > min_price
    return ok


@dataclass
class FeaturePanel:
    """
    Wide (date x ticker) frames of the Step 1 inputs on the union calendar.

    Attributes:
        feature_names (List[str]): The model features.
        features (Dict[str, pd.DataFrame]): Raw feature values per feature.
        traded_value (pd.DataFrame): log_traded_value (eligibility input).
        fwd (pd.DataFrame): Forward return over `horizon_days` own trading days (NaN if unknown).
        label_end (pd.DataFrame): Date of the close that ends the forward return (NaT if unknown).
        price (pd.DataFrame): Close carried for at most `step1_ffill_limit_days` days (NaN
            once the last own trading day is older, i.e. a stale price).
        eligible (pd.DataFrame): Boolean eligibility mask.
        history_days (Optional[pd.DataFrame]): Own trading days up to each date (NaN if stale).
    """
    feature_names: List[str]
    features: Dict[str, pd.DataFrame]
    traded_value: pd.DataFrame
    fwd: pd.DataFrame
    label_end: pd.DataFrame
    price: pd.DataFrame
    eligible: pd.DataFrame
    history_days: Optional[pd.DataFrame] = None

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """The union calendar of all downloaded trading days."""
        return pd.DatetimeIndex(self.price.index)

    @property
    def tickers(self) -> List[str]:
        """The tickers of the panel."""
        return list(self.price.columns)


def eligibility_mask(history_days: pd.DataFrame, traded_value: pd.DataFrame, price: pd.DataFrame,
                     min_price: float, min_history_days: int) -> pd.DataFrame:
    """
    Marks the (date, ticker) cells that may be ranked (see eligible_rows()).

    Args:
        history_days (pd.DataFrame): Own trading days up to each date.
        traded_value (pd.DataFrame): log_traded_value.
        price (pd.DataFrame): Prices.
        min_price (float): Minimum price (exclusive), `step1_min_price_eur`.
        min_history_days (int): Minimum own trading days, `step1_min_history_days`.

    Returns:
        pd.DataFrame: Boolean mask with the index and columns of `price`.
    """
    ok = eligible_rows(history_days.to_numpy(dtype=float), traded_value.to_numpy(dtype=float),
                       price.to_numpy(dtype=float), min_price, min_history_days)
    return pd.DataFrame(ok, index=price.index, columns=price.columns)


def build_feature_panel(price_panel: PricePanel, feature_names: Sequence[str],
                        settings: Dict[str, Any]) -> FeaturePanel:
    """
    Builds the wide feature, label and eligibility frames of all stocks on the union calendar.

    Args:
        price_panel (PricePanel): Downloaded prices and volumes.
        feature_names (Sequence[str]): Model features (`step1_features`).
        settings (Dict[str, Any]): Loaded settings (horizon_days, step1_ffill_limit_days,
            step1_min_price_eur, step1_min_history_days).

    Returns:
        FeaturePanel: The aligned frames.

    Raises:
        ValueError: On unknown feature names, or market-relative features without the market index.
    """
    names = validate_feature_names(feature_names)
    check_market_index(price_panel, names)
    horizon = int(settings['horizon_days'])
    limit = int(settings['step1_ffill_limit_days'])
    close = price_panel.close.sort_index()
    calendar = pd.DatetimeIndex(close.dropna(how='all').index)
    tickers = sorted(close.columns)
    columns = list(dict.fromkeys(names + [TRADED_VALUE_FEATURE, 'fwd', 'label_end', 'price', HISTORY_COLUMN]))
    store: Dict[str, Dict[str, pd.Series]] = {col: {} for col in columns}
    for ticker in tickers:
        if close[ticker].dropna().empty:
            continue
        frame = stock_feature_frame(stock_series(price_panel, ticker), names, horizon)
        # A day without a trade carries the last own trading day for at most `limit` days;
        # a forward return is never filled beyond that
        frame = frame.reindex(calendar, method='ffill', limit=limit)
        for col in columns:
            store[col][ticker] = frame[col]

    def wide(col: str) -> pd.DataFrame:
        frame = pd.DataFrame(store[col], index=calendar)
        return frame.reindex(columns=[t for t in tickers if t in store[col]])

    features = {name: wide(name) for name in names}
    traded_value = (features[TRADED_VALUE_FEATURE] if TRADED_VALUE_FEATURE in features
                    else wide(TRADED_VALUE_FEATURE))
    price = wide('price')
    history_days = wide(HISTORY_COLUMN)
    eligible = eligibility_mask(history_days, traded_value, price, float(settings['step1_min_price_eur']),
                                int(settings['step1_min_history_days']))
    logger.info(f"Feature panel: {price.shape[1]} stocks on {len(calendar)} trading days; "
                f"{int(eligible.iloc[-1].sum()) if len(calendar) else 0} eligible on the last day.")
    return FeaturePanel(feature_names=names, features=features, traded_value=traded_value,
                        fwd=wide('fwd'), label_end=wide('label_end'), price=price, eligible=eligible,
                        history_days=history_days)


def rank_within_date(values: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile rank minus 0.5 of every eligible cell among the eligible cells of its date that have a value.

    Uses pandas rank(pct=True, method='average') semantics: ties share their average rank
    and the highest value of n ranked stocks gets 1 - 0.5 = 0.5. An eligible cell without
    a value gets the neutral rank 0 (NEUTRAL_RANK).

    Args:
        values (pd.DataFrame): Feature values, dates x tickers.
        eligible (pd.DataFrame): Boolean eligibility mask aligned with `values`.

    Returns:
        pd.DataFrame: The ranks; NaN outside the eligible cells.
    """
    mask = eligible.reindex(index=values.index, columns=values.columns, fill_value=False).astype(bool)
    masked = values.astype(float).where(mask)
    ranks = masked.rank(axis=1, method='average', pct=True) - 0.5
    return ranks.mask(mask & ranks.isna(), NEUTRAL_RANK)


def rank_long_panel(values: pd.DataFrame) -> pd.DataFrame:
    """
    Within-date ranks of the columns of a long panel that holds only eligible rows.

    Same semantics as rank_within_date(): percentile rank minus 0.5 among the rows of the
    date that have a value; rows without a value get the neutral rank 0.

    Args:
        values (pd.DataFrame): Raw feature values indexed by (date, ticker), eligible rows only.

    Returns:
        pd.DataFrame: The ranks with the index and columns of `values`.
    """
    ranks = values.astype(float).groupby(level='date').rank(method='average', pct=True) - 0.5
    return ranks.fillna(NEUTRAL_RANK)


def month_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Returns the last calendar date of every month present in the calendar.

    Args:
        calendar (pd.DatetimeIndex): Trading days.

    Returns:
        pd.DatetimeIndex: One date per calendar month, in order.
    """
    calendar = pd.DatetimeIndex(calendar).sort_values()
    if calendar.empty:
        return calendar
    dates = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(dates.groupby([calendar.year, calendar.month]).max().to_numpy())


def label_column(fwd: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binary target: 1.0 if the forward return reaches `threshold`, 0.0 below it, NaN if unknown.

    Args:
        fwd (np.ndarray): Forward returns.
        threshold (float): The `threshold` setting.

    Returns:
        np.ndarray: The labels.
    """
    fwd = np.asarray(fwd, dtype=float)
    return np.where(np.isnan(fwd), np.nan, (fwd >= threshold).astype(float))


def to_long_panel(feature_panel: FeaturePanel, dates: Sequence[pd.Timestamp],
                  settings: Dict[str, Any]) -> pd.DataFrame:
    """
    Builds the long panel of the eligible (date, ticker) rows of the given dates.

    Args:
        feature_panel (FeaturePanel): The aligned frames.
        dates (Sequence[pd.Timestamp]): Dates of the union calendar (e.g. the month-ends).
        settings (Dict[str, Any]): Loaded settings (threshold).

    Returns:
        pd.DataFrame: Indexed by (date, ticker) with the rank columns r_<feature> (0 for an
        eligible row without a value), the raw features, fwd, label_end, price and y (1.0 if
        fwd >= threshold, 0.0 below, NaN if the forward return is not known yet).
    """
    threshold = float(settings['threshold'])
    names = feature_panel.feature_names
    dates = pd.DatetimeIndex(dates)
    eligible = feature_panel.eligible.loc[dates]
    rows, cols = np.nonzero(eligible.to_numpy())
    index = pd.MultiIndex.from_arrays([dates[rows], eligible.columns[cols]], names=['date', 'ticker'])
    data: Dict[str, np.ndarray] = {}
    for name, rank_col in zip(names, rank_columns(names)):
        values = feature_panel.features[name].loc[dates]
        data[rank_col] = rank_within_date(values, eligible).to_numpy()[rows, cols]
    for name in names:
        data[name] = feature_panel.features[name].loc[dates].to_numpy(dtype=float)[rows, cols]
    fwd = feature_panel.fwd.loc[dates].to_numpy(dtype=float)[rows, cols]
    data['fwd'] = fwd
    data['label_end'] = feature_panel.label_end.loc[dates].to_numpy()[rows, cols]
    data['price'] = feature_panel.price.loc[dates].to_numpy(dtype=float)[rows, cols]
    data['y'] = label_column(fwd, threshold)
    panel = pd.DataFrame(data, index=index)
    panel['label_end'] = pd.to_datetime(panel['label_end'])
    return panel


def stock_panel_rows(series: StockSeries, feature_names: Sequence[str], calendar: pd.DatetimeIndex,
                     dates: pd.DatetimeIndex, settings: Dict[str, Any]) -> pd.DataFrame:
    """
    The eligible rows of one stock at the given dates, without building wide frames.

    The stock's own-day frame is carried onto the union calendar exactly as in
    build_feature_panel() and read at `dates`; only eligible rows are kept. The variable
    review builds its long panel of the whole candidate pool from these rows.

    Args:
        series (StockSeries): The stock on its own trading days.
        feature_names (Sequence[str]): Features to compute.
        calendar (pd.DatetimeIndex): Union calendar of all stocks.
        dates (pd.DatetimeIndex): Dates of the calendar to keep (e.g. the month-ends).
        settings (Dict[str, Any]): Loaded settings (horizon_days, step1_ffill_limit_days,
            step1_min_price_eur, step1_min_history_days).

    Returns:
        pd.DataFrame: Rows indexed by date with the raw features, fwd, label_end and price.
    """
    frame = stock_feature_frame(series, feature_names, int(settings['horizon_days']))
    frame = frame.reindex(calendar, method='ffill', limit=int(settings['step1_ffill_limit_days'])).loc[dates]
    ok = eligible_rows(frame[HISTORY_COLUMN].to_numpy(), frame[TRADED_VALUE_FEATURE].to_numpy(),
                       frame['price'].to_numpy(), float(settings['step1_min_price_eur']),
                       int(settings['step1_min_history_days']))
    keep = list(dict.fromkeys(list(feature_names) + ['fwd', 'label_end', 'price']))
    return frame.loc[ok, keep]
