"""
Batch download of the adjusted daily closes and volumes of the Step 1 universe.

One request covers `price_download_batch_size` tickers. A failed ticker does not make
yfinance raise: it comes back as an all-NaN column of an otherwise filled frame. Such
tickers are detected per column and re-downloaded, up to `price_download_retries` times.
"""
import datetime
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException

logger = logging.getLogger(__name__)

# Errors of a yfinance network call: its own exceptions, network errors (the requests and
# curl_cffi exceptions derive from OSError) and malformed responses
DOWNLOAD_ERRORS = (YFException, OSError, ValueError, KeyError)
BACKOFF_JITTER_SECONDS = (0.5, 2.0)


@dataclass
class PricePanel:
    """
    Adjusted daily closes and volumes of the Step 1 universe.

    Attributes:
        close (pd.DataFrame): Adjusted closes, dates x tickers. The index is the union of
            all downloaded trading days; a stock without a trade on a date is NaN there.
        volume (pd.DataFrame): Traded volumes with the same index and columns as `close`.
        missing (List[str]): Requested tickers without any price data after all retries.
    """
    close: pd.DataFrame
    volume: pd.DataFrame
    missing: List[str] = field(default_factory=list)


def _price_field(data: pd.DataFrame, name: str, batch: Sequence[str]) -> pd.DataFrame:
    """
    Extracts one price field ('Close' or 'Volume') of a yfinance download as dates x tickers.

    Args:
        data (pd.DataFrame): yf.download result (MultiIndex columns Price x Ticker).
        name (str): The price field.
        batch (Sequence[str]): Requested tickers; yfinance returns them upper-cased.

    Returns:
        pd.DataFrame: One column per returned ticker, named as requested.
    """
    if isinstance(data.columns, pd.MultiIndex):
        if name not in data.columns.get_level_values(0):
            return pd.DataFrame(index=data.index)
        frame = data[name]
    elif name in data.columns and len(batch) == 1:
        frame = data[[name]].set_axis([batch[0]], axis=1)
    else:
        return pd.DataFrame(index=data.index)
    requested = {ticker.upper(): ticker for ticker in batch}
    frame = frame.rename(columns=lambda col: requested.get(str(col).upper(), str(col)))
    return frame.loc[:, [col for col in frame.columns if col in batch]].astype(float)


def _download_batch(batch: Sequence[str], start: datetime.date,
                    today: datetime.date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads the closes and volumes of one batch and drops rows dated today or later.

    Args:
        batch (Sequence[str]): Tickers of the batch.
        start (datetime.date): First date to fetch.
        today (datetime.date): The current Europe/Berlin date (exclusive end).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (closes, volumes); empty frames if the request failed.
    """
    try:
        data = yf.download(list(batch), start=start, end=today, auto_adjust=True,
                           progress=False, threads=True)
    except DOWNLOAD_ERRORS as e:
        logger.warning(f"Price download of {len(batch)} tickers failed: {e}")
        return pd.DataFrame(), pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()

    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    data = data.set_axis(index, axis=0)
    # Only fully closed trading days are used; the current day's candle is still moving
    data = data[data.index < pd.Timestamp(today)]
    return _price_field(data, 'Close', batch), _price_field(data, 'Volume', batch)


def fetch_price_panel(tickers: Sequence[str], history_years: int, settings: Dict[str, Any]) -> PricePanel:
    """
    Downloads the adjusted daily closes and volumes of all tickers in batches.

    Tickers that come back without data are re-downloaded with exponential backoff and
    jitter. Rows dated on or after the current Europe/Berlin date are dropped, so the
    last row is the last closed trading day.

    Args:
        tickers (Sequence[str]): Yahoo Finance tickers (e.g. 'SAP.DE').
        history_years (int): Years of history to fetch (training window plus warm-up).
        settings (Dict[str, Any]): Loaded settings (price_download_batch_size, price_download_retries).

    Returns:
        PricePanel: Closes and volumes on the union calendar plus the tickers without data.
    """
    requested = sorted(dict.fromkeys(tickers))
    batch_size = max(1, int(settings['price_download_batch_size']))
    retries = int(settings['price_download_retries'])
    today = pd.Timestamp.now(tz='Europe/Berlin').date()
    start = today - datetime.timedelta(days=history_years * 365)
    logger.info(f"Downloading daily prices of {len(requested)} tickers from {start} in batches of {batch_size}.")

    closes: Dict[str, pd.Series] = {}
    volumes: Dict[str, pd.Series] = {}
    pending = list(requested)
    for attempt in range(retries + 1):
        if attempt > 0:
            wait = 2 ** (attempt - 1) + random.uniform(*BACKOFF_JITTER_SECONDS)
            logger.warning(f"Retry {attempt}/{retries} for {len(pending)} tickers without data "
                           f"in {wait:.1f} s: {pending}")
            time.sleep(wait)
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            close, volume = _download_batch(batch, start, today)
            for ticker in batch:
                if ticker in close.columns and close[ticker].notna().any():
                    closes[ticker] = close[ticker]
                    volumes[ticker] = (volume[ticker] if ticker in volume.columns
                                       else pd.Series(float('nan'), index=close.index))
        pending = [ticker for ticker in pending if ticker not in closes]
        if not pending:
            break

    if closes:
        close_panel = pd.DataFrame(closes).sort_index()
        close_panel = close_panel.loc[:, sorted(close_panel.columns)].dropna(how='all')
    else:
        close_panel = pd.DataFrame(index=pd.DatetimeIndex([]))
    volume_panel = pd.DataFrame(volumes).reindex(index=close_panel.index, columns=close_panel.columns)
    missing = sorted(set(requested) - set(close_panel.columns))
    if missing:
        logger.warning(f"No price data for {len(missing)} tickers after {retries} retries: {missing}")
    if not close_panel.empty:
        logger.info(f"Price panel: {close_panel.shape[1]} tickers, {close_panel.shape[0]} trading days "
                    f"from {close_panel.index[0].date()} to {close_panel.index[-1].date()}.")
    return PricePanel(close=close_panel, volume=volume_panel, missing=missing)
