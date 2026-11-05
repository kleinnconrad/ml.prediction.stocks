"""
Offline tests of the batch price download (src/ingestion/market_api.py) with a mocked
yfinance: batching, retries of the tickers without data, the Berlin-today cut, the mapping
of the yfinance column layouts to the requested tickers, the open, high and low fields, the
market index and the CSV price cache.
"""
from pathlib import Path
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
        data[('Open', ticker.upper())] = 9.5 + i + np.arange(len(dates))
        data[('High', ticker.upper())] = 10.5 + i + np.arange(len(dates))
        data[('Low', ticker.upper())] = 9.0 + i + np.arange(len(dates))
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
    for frame in (panel.open, panel.high, panel.low):
        assert frame.shape == panel.close.shape and list(frame.columns) == list(panel.close.columns)
    assert (panel.low <= panel.open).all().all() and (panel.high >= panel.close).all().all()
    assert panel.market is None  # no market_index_ticker in the settings


def test_the_market_index_is_downloaded_with_the_batches_but_not_ranked(
        monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    dates = pd.bdate_range('2024-01-01', periods=6)
    calls: List[List[str]] = []

    def fake_download(tickers: Sequence[str], **kwargs: Any) -> pd.DataFrame:
        calls.append(list(tickers))
        frame = _yf_frame(tickers, dates)
        if '^GDAXI' in tickers:
            # The index also has a row on a day without any stock trade
            extra = frame.iloc[[0]].set_axis(pd.DatetimeIndex([pd.Timestamp('2023-12-29')], name='Date'))
            extra.loc[:, (slice(None), [t.upper() for t in tickers if t != '^GDAXI'])] = np.nan
            frame = pd.concat([extra, frame])
        return frame

    monkeypatch.setattr(market_api.yf, 'download', fake_download)
    settings = {'price_download_batch_size': 2, 'price_download_retries': 1, 'market_index_ticker': '^GDAXI'}
    panel = market_api.fetch_price_panel(['BBB.DE', 'AAA.DE', '^GDAXI'], 12, settings)

    assert calls == [['AAA.DE', 'BBB.DE'], ['^GDAXI']]
    assert list(panel.close.columns) == ['AAA.DE', 'BBB.DE'] and panel.missing == []
    assert panel.close.index.equals(pd.DatetimeIndex(dates, name='Date'))  # the index day does not extend it
    assert panel.market is not None and panel.market.name == '^GDAXI'
    assert len(panel.market) == len(dates) + 1


def test_a_missing_market_index_is_reported_and_left_out(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    dates = pd.bdate_range('2024-01-01', periods=5)
    monkeypatch.setattr(market_api.yf, 'download',
                        lambda tickers, **kwargs: _yf_frame(tickers, dates, missing=['^GDAXI']))
    settings = {'price_download_batch_size': 60, 'price_download_retries': 1, 'market_index_ticker': '^GDAXI'}
    panel = market_api.fetch_price_panel(['AAA.DE'], 12, settings)
    assert panel.market is None and panel.missing == [] and list(panel.close.columns) == ['AAA.DE']


def test_the_price_cache_round_trip_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2020-01-01', periods=50)
    close = pd.DataFrame(rng.lognormal(3.0, 1.0, (50, 3)), index=dates, columns=['AAA.DE', 'BBB.DE', 'C1.DE'])
    close.iloc[3, 1] = np.nan
    panel = market_api.PricePanel(close=close, volume=close * 1000.0, missing=['GONE.DE'], open=close * 0.99,
                                  high=close * 1.01, low=close * 0.98,
                                  market=pd.Series(rng.lognormal(9.0, 0.1, 52), name='^GDAXI',
                                                   index=pd.bdate_range('2019-12-30', periods=52)))
    market_api.save_price_panel(panel, tmp_path / 'cache')
    loaded = market_api.load_price_panel(tmp_path / 'cache')

    for name in ('close', 'volume', 'open', 'high', 'low'):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(panel, name), check_exact=True,
                                      check_freq=False, check_index_type=False)
    pd.testing.assert_series_equal(loaded.market, panel.market, check_exact=True, check_freq=False,
                                   check_index_type=False)
    assert loaded.missing == ['GONE.DE']
    with pytest.raises(FileNotFoundError):
        market_api.load_price_panel(tmp_path / 'empty')


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
