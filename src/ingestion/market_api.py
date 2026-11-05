"""
Batch download of the adjusted daily prices and volumes of the Step 1 universe.

One request covers `price_download_batch_size` tickers. A failed ticker does not make
yfinance raise: it comes back as an all-NaN column of an otherwise filled frame. Such
tickers are detected per column and re-downloaded, up to `price_download_retries` times.

The market index `market_index_ticker` is requested with the batches. It is an input of
the market-relative features and never part of the ranked universe, so it is kept apart
from the stock columns and does not extend the union calendar of the stocks.

`save_price_panel()` and `load_price_panel()` store a downloaded panel as CSV files, for
repeated local runs on the same prices.
"""
import datetime
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException

logger = logging.getLogger(__name__)

# Errors of a yfinance network call: its own exceptions, network errors (the requests and
# curl_cffi exceptions derive from OSError) and malformed responses
DOWNLOAD_ERRORS = (YFException, OSError, ValueError, KeyError)
BACKOFF_JITTER_SECONDS = (0.5, 2.0)
# yfinance price fields and the PricePanel attribute that holds each of them
PRICE_FIELDS = {'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'}
CACHE_MARKET_FILE = 'market.csv'
CACHE_META_FILE = 'meta.json'


@dataclass
class PricePanel:
    """
    Adjusted daily prices and volumes of the Step 1 universe.

    Attributes:
        close (pd.DataFrame): Adjusted closes, dates x tickers. The index is the union of
            all downloaded trading days of the stocks; a stock without a trade on a date
            is NaN there.
        volume (pd.DataFrame): Traded volumes with the same index and columns as `close`.
        missing (List[str]): Requested tickers without any price data after all retries.
        open (Optional[pd.DataFrame]): Adjusted opens like `close` (None if not downloaded).
        high (Optional[pd.DataFrame]): Adjusted highs like `close` (None if not downloaded).
        low (Optional[pd.DataFrame]): Adjusted lows like `close` (None if not downloaded).
        market (Optional[pd.Series]): Closes of the market index on its own trading days
            (None if it was not requested or has no data).
    """
    close: pd.DataFrame
    volume: pd.DataFrame
    missing: List[str] = field(default_factory=list)
    open: Optional[pd.DataFrame] = None
    high: Optional[pd.DataFrame] = None
    low: Optional[pd.DataFrame] = None
    market: Optional[pd.Series] = None


def _price_field(data: pd.DataFrame, name: str, batch: Sequence[str]) -> pd.DataFrame:
    """
    Extracts one price field (e.g. 'Close' or 'Volume') of a yfinance download as dates x tickers.

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


def _download_batch(batch: Sequence[str], start: datetime.date, today: datetime.date) -> Dict[str, pd.DataFrame]:
    """
    Downloads the prices and volumes of one batch and drops rows dated today or later.

    Args:
        batch (Sequence[str]): Tickers of the batch.
        start (datetime.date): First date to fetch.
        today (datetime.date): The current Europe/Berlin date (exclusive end).

    Returns:
        Dict[str, pd.DataFrame]: One frame (dates x tickers) per yfinance field of
        PRICE_FIELDS; empty frames if the request failed.
    """
    try:
        data = yf.download(list(batch), start=start, end=today, auto_adjust=True,
                           progress=False, threads=True)
    except DOWNLOAD_ERRORS as e:
        logger.warning(f"Price download of {len(batch)} tickers failed: {e}")
        return {name: pd.DataFrame() for name in PRICE_FIELDS}
    if data is None or data.empty:
        return {name: pd.DataFrame() for name in PRICE_FIELDS}

    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    data = data.set_axis(index, axis=0)
    # Only fully closed trading days are used; the current day's candle is still moving
    data = data[data.index < pd.Timestamp(today)]
    return {name: _price_field(data, name, batch) for name in PRICE_FIELDS}


def fetch_price_panel(tickers: Sequence[str], history_years: int, settings: Dict[str, Any]) -> PricePanel:
    """
    Downloads the adjusted daily prices and volumes of all tickers in batches.

    Tickers that come back without data are re-downloaded with exponential backoff and
    jitter. Rows dated on or after the current Europe/Berlin date are dropped, so the
    last row is the last closed trading day. The market index (`market_index_ticker`, if
    set) is downloaded with the batches and returned separately as `market`.

    Args:
        tickers (Sequence[str]): Yahoo Finance tickers (e.g. 'SAP.DE').
        history_years (int): Years of history to fetch (training window plus warm-up).
        settings (Dict[str, Any]): Loaded settings (price_download_batch_size,
            price_download_retries and, optionally, market_index_ticker).

    Returns:
        PricePanel: Closes, volumes, opens, highs and lows on the union calendar of the
        stocks, the market index closes and the tickers without data.
    """
    index_ticker = settings.get('market_index_ticker') or None
    stocks = sorted(dict.fromkeys(ticker for ticker in tickers if ticker != index_ticker))
    requested = stocks + ([index_ticker] if index_ticker else [])
    batch_size = max(1, int(settings['price_download_batch_size']))
    retries = int(settings['price_download_retries'])
    today = pd.Timestamp.now(tz='Europe/Berlin').date()
    start = today - datetime.timedelta(days=history_years * 365)
    logger.info(f"Downloading daily prices of {len(stocks)} tickers"
                f"{f' and the index {index_ticker}' if index_ticker else ''} from {start} in batches of {batch_size}.")

    series: Dict[str, Dict[str, pd.Series]] = {name: {} for name in PRICE_FIELDS}
    pending = list(requested)
    for attempt in range(retries + 1):
        if attempt > 0:
            wait = 2 ** (attempt - 1) + random.uniform(*BACKOFF_JITTER_SECONDS)
            logger.warning(f"Retry {attempt}/{retries} for {len(pending)} tickers without data "
                           f"in {wait:.1f} s: {pending}")
            time.sleep(wait)
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            fields = _download_batch(batch, start, today)
            close = fields['Close']
            for ticker in batch:
                if ticker in close.columns and close[ticker].notna().any():
                    for name, frame in fields.items():
                        series[name][ticker] = (frame[ticker] if ticker in frame.columns
                                                else pd.Series(float('nan'), index=close.index))
        pending = [ticker for ticker in pending if ticker not in series['Close']]
        if not pending:
            break

    market: Optional[pd.Series] = None
    if index_ticker and index_ticker in series['Close']:
        market = series['Close'][index_ticker].dropna().sort_index().rename(index_ticker)
    elif index_ticker:
        logger.warning(f"No data for the market index {index_ticker}; the market-relative features are missing.")
    closes = {ticker: values for ticker, values in series['Close'].items() if ticker != index_ticker}
    if closes:
        close_panel = pd.DataFrame(closes).sort_index()
        close_panel = close_panel.loc[:, sorted(close_panel.columns)].dropna(how='all')
    else:
        close_panel = pd.DataFrame(index=pd.DatetimeIndex([]))

    def aligned(name: str) -> pd.DataFrame:
        frame = pd.DataFrame({ticker: values for ticker, values in series[name].items() if ticker in closes})
        return frame.reindex(index=close_panel.index, columns=close_panel.columns)

    missing = sorted(set(stocks) - set(close_panel.columns))
    if missing:
        logger.warning(f"No price data for {len(missing)} tickers after {retries} retries: {missing}")
    if not close_panel.empty:
        logger.info(f"Price panel: {close_panel.shape[1]} tickers, {close_panel.shape[0]} trading days "
                    f"from {close_panel.index[0].date()} to {close_panel.index[-1].date()}.")
    return PricePanel(close=close_panel, volume=aligned('Volume'), missing=missing, open=aligned('Open'),
                      high=aligned('High'), low=aligned('Low'), market=market)


def save_price_panel(panel: PricePanel, directory: Path) -> None:
    """
    Writes a price panel as CSV files (one per field, the market index and a meta file).

    Args:
        panel (PricePanel): The panel to store.
        directory (Path): Target directory; created if missing, existing files are replaced.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for attribute in PRICE_FIELDS.values():
        frame = getattr(panel, attribute)
        if frame is not None:
            frame.to_csv(directory / f"{attribute}.csv", index_label='Date')
    if panel.market is not None:
        panel.market.to_frame(str(panel.market.name or 'market')).to_csv(directory / CACHE_MARKET_FILE,
                                                                          index_label='Date')
    meta = {'missing': list(panel.missing), 'tickers': list(panel.close.columns),
            'market_index': None if panel.market is None else str(panel.market.name)}
    (directory / CACHE_META_FILE).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    logger.info(f"Saved the price panel ({panel.close.shape[1]} tickers) to {directory}.")


def load_price_panel(directory: Path) -> PricePanel:
    """
    Reads a price panel written by save_price_panel().

    Args:
        directory (Path): Directory of the CSV files.

    Returns:
        PricePanel: The stored panel; fields without a file are None.

    Raises:
        FileNotFoundError: If the directory holds no close.csv or meta.json.
    """
    directory = Path(directory)
    for required in ('close.csv', CACHE_META_FILE):
        if not (directory / required).exists():
            raise FileNotFoundError(f"{directory / required} is missing; {directory} is not a price cache.")

    def read(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame.index = pd.DatetimeIndex(frame.index).rename(None)
        return frame.astype(float)

    fields = {attribute: read(directory / f"{attribute}.csv") if (directory / f"{attribute}.csv").exists() else None
              for attribute in PRICE_FIELDS.values()}
    meta = json.loads((directory / CACHE_META_FILE).read_text(encoding='utf-8'))
    market = None
    if (directory / CACHE_MARKET_FILE).exists():
        market = read(directory / CACHE_MARKET_FILE).iloc[:, 0].dropna()
        market.name = meta.get('market_index') or market.name
    close = fields.pop('close')
    volume = fields.pop('volume')
    if volume is None:
        volume = pd.DataFrame(float('nan'), index=close.index, columns=close.columns)
    logger.info(f"Loaded the price panel ({close.shape[1]} tickers, {close.shape[0]} trading days) from {directory}.")
    return PricePanel(close=close, volume=volume, missing=list(meta.get('missing', [])), market=market, **fields)
