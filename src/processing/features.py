"""
Step 1 features of the pooled cross-sectional model.

Every feature is computed per stock on the stock's own trading days (the rows where the
stock has a close), together with the forward return over `horizon_days`, the date that
return ends on (`label_end`) and the price. The per-stock rows are then aligned on the
union calendar of all downloaded trading days: a day without a trade carries the stock's
last trading day for at most `step1_ffill_limit_days` days.

On every date, only eligible stocks are ranked (all features present, a positive 126-day
traded value and a price above `step1_min_price_eur`). The model inputs are the
within-date percentile ranks minus 0.5, so every input lies in (-0.5, 0.5].
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from ..ingestion.market_api import PricePanel

logger = logging.getLogger(__name__)

FeatureFunction = Callable[[pd.Series, pd.Series], pd.Series]

# Eligibility input that is computed even when it is not a model feature
TRADED_VALUE_FEATURE = 'log_traded_value'
RANK_PREFIX = 'r_'


def _period_return(periods: int) -> FeatureFunction:
    """Returns a feature function for the `periods`-day return."""
    def feature(close: pd.Series, volume: pd.Series) -> pd.Series:
        return close.pct_change(periods)
    return feature


def _distance_to_sma(window: int) -> FeatureFunction:
    """Returns a feature function for the relative distance to the `window`-day simple moving average."""
    def feature(close: pd.Series, volume: pd.Series) -> pd.Series:
        return close / close.rolling(window).mean() - 1.0
    return feature


def _volatility(window: int) -> FeatureFunction:
    """Returns a feature function for the standard deviation of daily returns over `window` days."""
    def feature(close: pd.Series, volume: pd.Series) -> pd.Series:
        return close.pct_change().rolling(window).std()
    return feature


def momentum_12_1(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    12-month momentum without the most recent month: close[t-21] / close[t-252] - 1.

    Args:
        close (pd.Series): Closes on the stock's own trading days.
        volume (pd.Series): Volumes on the same days (unused).

    Returns:
        pd.Series: The feature.
    """
    return close.shift(21) / close.shift(252) - 1.0


def log_traded_value(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Log of one plus the mean daily traded value (close x volume) over 126 trading days.

    Args:
        close (pd.Series): Closes on the stock's own trading days.
        volume (pd.Series): Volumes on the same days; a missing volume counts as 0.

    Returns:
        pd.Series: The feature.
    """
    return np.log1p((close * volume).rolling(126).mean())


def drawdown_252(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Relative distance of the close to its 252-day maximum (0 at a new high, negative below).

    Args:
        close (pd.Series): Closes on the stock's own trading days.
        volume (pd.Series): Volumes on the same days (unused).

    Returns:
        pd.Series: The feature.
    """
    return close / close.rolling(252).max() - 1.0


# One function per feature name, f(close, volume) on a stock's own trading days. The model
# reads its feature list from `step1_features` in config/settings.yaml.
FEATURE_FUNCTIONS: Dict[str, FeatureFunction] = {
    'ret_21': _period_return(21),
    'ret_63': _period_return(63),
    'ret_126': _period_return(126),
    'mom_12_1': momentum_12_1,
    'dist_sma50': _distance_to_sma(50),
    'dist_sma200': _distance_to_sma(200),
    'vol_21': _volatility(21),
    'vol_126': _volatility(126),
    'log_traded_value': log_traded_value,
    'drawdown_252': drawdown_252,
}


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


def rank_columns(feature_names: Sequence[str]) -> List[str]:
    """
    Returns the names of the rank columns of the long panel.

    Args:
        feature_names (Sequence[str]): Feature names.

    Returns:
        List[str]: 'r_<feature>' for every feature.
    """
    return [f"{RANK_PREFIX}{name}" for name in feature_names]


def stock_feature_frame(close: pd.Series, volume: pd.Series, feature_names: Sequence[str],
                        horizon_days: int) -> pd.DataFrame:
    """
    Computes the features, forward return and label end date of one stock on its own trading days.

    Args:
        close (pd.Series): Adjusted closes of the stock; NaN rows (days without a trade) are dropped.
        volume (pd.Series): Volumes of the stock; a missing volume on a trading day counts as 0.
        feature_names (Sequence[str]): Features to compute.
        horizon_days (int): Forward horizon in the stock's own trading days.

    Returns:
        pd.DataFrame: One row per trading day with the features, `log_traded_value` (needed
        for eligibility), `fwd` (close[t + horizon] / close[t] - 1), `label_end` (the date of
        that later close), `price` and `last_trade` (the row's own date).
    """
    names = validate_feature_names(feature_names)
    c = close.dropna().astype(float)
    v = volume.reindex(c.index).fillna(0.0).astype(float)
    frame = pd.DataFrame(index=c.index)
    for name in dict.fromkeys(names + [TRADED_VALUE_FEATURE]):
        # A non-finite value (e.g. a return against a zero close) counts as missing
        frame[name] = FEATURE_FUNCTIONS[name](c, v).replace([np.inf, -np.inf], np.nan)
    own_dates = pd.Series(c.index, index=c.index)
    frame['fwd'] = c.shift(-horizon_days) / c - 1.0
    frame['label_end'] = own_dates.shift(-horizon_days)
    frame['price'] = c
    frame['last_trade'] = own_dates
    return frame


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
        price (pd.DataFrame): Close carried for at most `step1_ffill_limit_days` days.
        last_trade (pd.DataFrame): Date of the stock's own trading day a row comes from.
        eligible (pd.DataFrame): Boolean eligibility mask.
    """
    feature_names: List[str]
    features: Dict[str, pd.DataFrame]
    traded_value: pd.DataFrame
    fwd: pd.DataFrame
    label_end: pd.DataFrame
    price: pd.DataFrame
    last_trade: pd.DataFrame
    eligible: pd.DataFrame

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """The union calendar of all downloaded trading days."""
        return pd.DatetimeIndex(self.price.index)

    @property
    def tickers(self) -> List[str]:
        """The tickers of the panel."""
        return list(self.price.columns)


def eligibility_mask(features: Dict[str, pd.DataFrame], traded_value: pd.DataFrame,
                     price: pd.DataFrame, min_price: float) -> pd.DataFrame:
    """
    Marks the (date, ticker) cells that may be ranked.

    A cell is eligible if every feature is present, log_traded_value is finite and
    positive, and the price is above `min_price`.

    Args:
        features (Dict[str, pd.DataFrame]): Raw feature values per feature.
        traded_value (pd.DataFrame): log_traded_value.
        price (pd.DataFrame): Prices.
        min_price (float): Minimum price (exclusive), `step1_min_price_eur`.

    Returns:
        pd.DataFrame: Boolean mask with the index and columns of `price`.
    """
    ok = np.ones(price.shape, dtype=bool)
    for frame in features.values():
        ok &= frame.notna().to_numpy()
    tv = traded_value.to_numpy(dtype=float)
    ok &= np.isfinite(tv) & (np.nan_to_num(tv, nan=-1.0) > 0)
    ok &= np.nan_to_num(price.to_numpy(dtype=float), nan=0.0) > min_price
    return pd.DataFrame(ok, index=price.index, columns=price.columns)


def build_feature_panel(price_panel: PricePanel, feature_names: Sequence[str],
                        settings: Dict[str, Any]) -> FeaturePanel:
    """
    Builds the wide feature, label and eligibility frames of all stocks on the union calendar.

    Args:
        price_panel (PricePanel): Downloaded closes and volumes.
        feature_names (Sequence[str]): Model features (`step1_features`).
        settings (Dict[str, Any]): Loaded settings (horizon_days, step1_ffill_limit_days,
            step1_min_price_eur).

    Returns:
        FeaturePanel: The aligned frames.

    Raises:
        ValueError: On unknown feature names.
    """
    names = validate_feature_names(feature_names)
    horizon = int(settings['horizon_days'])
    limit = int(settings['step1_ffill_limit_days'])
    close = price_panel.close.sort_index()
    calendar = pd.DatetimeIndex(close.dropna(how='all').index)
    tickers = sorted(close.columns)
    columns = list(dict.fromkeys(names + [TRADED_VALUE_FEATURE, 'fwd', 'label_end', 'price', 'last_trade']))
    store: Dict[str, Dict[str, pd.Series]] = {col: {} for col in columns}
    empty_volume = pd.Series(dtype=float)
    for ticker in tickers:
        series = close[ticker].dropna()
        if series.empty:
            continue
        volume = price_panel.volume[ticker] if ticker in price_panel.volume.columns else empty_volume
        frame = stock_feature_frame(series, volume, names, horizon)
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
    eligible = eligibility_mask(features, traded_value, price, float(settings['step1_min_price_eur']))
    logger.info(f"Feature panel: {price.shape[1]} stocks on {len(calendar)} trading days; "
                f"{int(eligible.iloc[-1].sum()) if len(calendar) else 0} eligible on the last day.")
    return FeaturePanel(feature_names=names, features=features, traded_value=traded_value,
                        fwd=wide('fwd'), label_end=wide('label_end'), price=price,
                        last_trade=wide('last_trade'), eligible=eligible)


def rank_within_date(values: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile rank minus 0.5 of every eligible cell among the eligible cells of its date.

    Uses pandas rank(pct=True, method='average') semantics: ties share their average rank
    and the highest value of n eligible stocks gets 1 - 0.5 = 0.5.

    Args:
        values (pd.DataFrame): Feature values, dates x tickers.
        eligible (pd.DataFrame): Boolean eligibility mask aligned with `values`.

    Returns:
        pd.DataFrame: The ranks; NaN outside the eligible cells.
    """
    mask = eligible.reindex(index=values.index, columns=values.columns, fill_value=False).astype(bool)
    masked = values.astype(float).where(mask)
    return masked.rank(axis=1, method='average', pct=True) - 0.5


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


def to_long_panel(feature_panel: FeaturePanel, dates: Sequence[pd.Timestamp],
                  settings: Dict[str, Any]) -> pd.DataFrame:
    """
    Builds the long panel of the eligible (date, ticker) rows of the given dates.

    Args:
        feature_panel (FeaturePanel): The aligned frames.
        dates (Sequence[pd.Timestamp]): Dates of the union calendar (e.g. the month-ends).
        settings (Dict[str, Any]): Loaded settings (threshold).

    Returns:
        pd.DataFrame: Indexed by (date, ticker) with the rank columns r_<feature>, the raw
        features, fwd, label_end, price and y (1.0 if fwd >= threshold, 0.0 below, NaN if
        the forward return is not known yet).
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
    data['y'] = np.where(np.isnan(fwd), np.nan, (fwd >= threshold).astype(float))
    panel = pd.DataFrame(data, index=index)
    panel['label_end'] = pd.to_datetime(panel['label_end'])
    return panel
