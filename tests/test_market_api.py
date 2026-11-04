import numpy as np
import pandas as pd
import pytest

from src.ingestion import market_api


def _yf_frame(tickers, dates, missing=()):
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
def no_sleep(monkeypatch):
    monkeypatch.setattr(market_api.time, 'sleep', lambda seconds: None)


def test_batches_retries_only_missing_tickers_and_drops_today(monkeypatch, no_sleep):
    today = pd.Timestamp.now(tz='Europe/Berlin').normalize().tz_localize(None)
    dates = pd.date_range(end=today, periods=6, freq='D')  # the last row is dated today
    calls = []
    flaky = {'BBB.DE'}

    def fake_download(tickers, **kwargs):
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


def test_a_failing_request_is_retried(monkeypatch, no_sleep):
    dates = pd.bdate_range('2024-01-01', periods=5)
    attempts = {'n': 0}

    def fake_download(tickers, **kwargs):
        attempts['n'] += 1
        if attempts['n'] == 1:
            raise OSError("connection reset")
        return _yf_frame(tickers, dates)

    monkeypatch.setattr(market_api.yf, 'download', fake_download)
    panel = market_api.fetch_price_panel(['AAA.DE'], 12, {'price_download_batch_size': 60, 'price_download_retries': 3})

    assert attempts['n'] == 2
    assert panel.missing == [] and panel.close['AAA.DE'].notna().all()


def test_no_data_at_all_gives_an_empty_panel(monkeypatch, no_sleep):
    monkeypatch.setattr(market_api.yf, 'download', lambda tickers, **kwargs: pd.DataFrame())
    panel = market_api.fetch_price_panel(['AAA.DE'], 12, {'price_download_batch_size': 60, 'price_download_retries': 1})
    assert panel.close.empty and panel.missing == ['AAA.DE']
