"""
Offline tests of the batch price download (src/ingestion/market_api.py) with a mocked
yfinance: batching, retries of the tickers without data, the Berlin-today cut and the
mapping of the yfinance column layouts to the requested tickers.
"""
from typing import Any, List, Sequence

import numpy as np
import pandas as pd
import pytest

from src.ingestion import market_api


def _yf_frame(tickers: Sequence[str], dates: Sequence[Any], missing: Sequence[str] = ()) -> pd.DataFrame:
    """A yf.download result: MultiIndex columns (Price, Ticker), NaN columns for failed tickers."""
    columns = pd.MultiIndex.from_product([['Close', 'High', 'Low', 'Open', 'Volume'], sorted(t.upper() for t in tickers)],
                                         names=['Price', 'Ticker'])
    data = pd.DataFrame(1.0, index=pd.DatetimeIndex(dates, name='Date'), columns=columns)
    for i, ticker in enumerate(sorted(tickers)):
        data[('Close', ticker.upper())] = 10.0 + i + np.arange(len(dates))
        data[('Volume', ticker.upper())] = 1000.0
        if ticker in missing:
            data.loc[:, (slice(None), ticker.upper())] = np.nan
    return data


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skips the backoff waits between download attempts."""
    monkeypatch.setattr(market_api.time, 'sleep', lambda seconds: None)


def test_batches_retries_only_missing_tickers_and_drops_today(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    today = pd.Timestamp.now(tz='Europe/Berlin').normalize().tz_localize(None)
    dates = pd.date_range(end=today, periods=6, freq='D')  # the last row is dated today
    calls: List[List[str]] = []
    flaky = {'BBB.DE'}

    def fake_download(tickers: Sequence[str], **kwargs: Any) -> pd.DataFrame:
        calls.append(list(tickers))
        missing = [t for t in tickers if t in flaky or t == 'GONE.DE']
        flaky.difference_update(tickers)  # BBB.DE succeeds on its second request
        return _yf_frame(tickers, dates, missing=missing)

    monkeypatch.setattr(market_api.yf, 'download', fake_download)
    settings = {'price_download_batch_size': 2, 'price_download_retries': 2}
    panel = market_api.fetch_price_panel(['CCC.DE', 'AAA.DE', 'BBB.DE', 'GONE.DE'], 12, settings)

    assert calls[:2] == [['AAA.DE', 'BBB.DE'], ['CCC.DE', 'GONE.DE']]
    assert calls[2] == ['BBB.DE', 'GONE.DE']           # only the tickers without data are retried
    assert calls[3:] == [['GONE.DE']]                   # until the retries are used up
    assert list(panel.close.columns) == ['AAA.DE', 'BBB.DE', 'CCC.DE']
    assert panel.missing == ['GONE.DE']
    assert panel.close.index.max() < today              # rows dated today are dropped
    assert len(panel.close) == 5
    assert panel.volume.shape == panel.close.shape and (panel.volume == 1000.0).all().all()


def test_a_failing_request_is_retried(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    dates = pd.bdate_range('2024-01-01', periods=5)
    attempts = {'n': 0}

    def fake_download(tickers: Sequence[str], **kwargs: Any) -> pd.DataFrame:
        attempts['n'] += 1
        if attempts['n'] == 1:
            raise OSError("connection reset")
        return _yf_frame(tickers, dates)

    monkeypatch.setattr(market_api.yf, 'download', fake_download)
    panel = market_api.fetch_price_panel(['AAA.DE'], 12, {'price_download_batch_size': 60, 'price_download_retries': 3})

    assert attempts['n'] == 2
    assert panel.missing == [] and panel.close['AAA.DE'].notna().all()


def test_no_data_at_all_gives_an_empty_panel(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    monkeypatch.setattr(market_api.yf, 'download', lambda tickers, **kwargs: pd.DataFrame())
    panel = market_api.fetch_price_panel(['AAA.DE'], 12, {'price_download_batch_size': 60, 'price_download_retries': 1})
    assert panel.close.empty and panel.missing == ['AAA.DE']


def test_a_single_ticker_frame_without_a_ticker_level_is_mapped_to_the_requested_name(
        monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    dates = pd.DatetimeIndex(pd.bdate_range('2024-01-01', periods=5), name='Date')
    # Flat columns, as yfinance can return them for a single ticker
    flat = pd.DataFrame({'Open': 1.0, 'Close': 10.0 + np.arange(5), 'Volume': 500.0}, index=dates)
    monkeypatch.setattr(market_api.yf, 'download', lambda tickers, **kwargs: flat)

    panel = market_api.fetch_price_panel(['SAP.DE'], 12, {'price_download_batch_size': 60, 'price_download_retries': 1})

    assert list(panel.close.columns) == ['SAP.DE'] and panel.missing == []
    assert panel.close['SAP.DE'].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert (panel.volume['SAP.DE'] == 500.0).all()
