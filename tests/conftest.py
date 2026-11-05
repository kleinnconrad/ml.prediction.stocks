"""
Shared fixtures: synthetic price panels (closes and volumes, or full OHLCV with a market
index), fast Step 1 settings and small variable review settings (offline, no network).
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from config.settings import load_settings
from src.ingestion.market_api import PricePanel

PanelFactory = Callable[..., PricePanel]

# The ten default Step 1 features of release 3.0.0
DEFAULT_STEP1_FEATURES: List[str] = ['ret_21', 'ret_63', 'ret_126', 'mom_12_1', 'dist_sma50', 'dist_sma200',
                                     'vol_21', 'vol_126', 'log_traded_value', 'drawdown_252']

# Parameters whose values the tests rely on. They are pinned here, so that changing them in
# config/settings.yaml (where CLAUDE.md requires them to be changed) does not break the tests.
PINNED_STEP1_SETTINGS: Dict[str, Any] = {
    'horizon_days': 126,
    'threshold': 0.10,
    'step1_features': DEFAULT_STEP1_FEATURES,
    'step1_l2_c': 0.1,
    'step1_solver_tol': 1.0e-8,
    'step1_max_iter': 1000,
    'step1_up_share': 0.10,
    'step1_min_price_eur': 1.0,
    'step1_min_history_days': 253,
    'step1_ffill_limit_days': 5,
    'step1_bootstrap_block_months': 6,
    'step1_bootstrap_seed': 0,
    'step1_min_validation_auc_lower': 0.5,
}


# Small variable review parameters for synthetic data (src/modeling/variable_review.py)
REVIEW_TEST_SETTINGS: Dict[str, Any] = {
    'step1_history_years': 4,
    'step1_min_stocks_per_date': 20,
    'step1_bootstrap_replications': 300,
    'review_validation_years': 3,
    'review_inner_splits': 2,
    'review_inner_validation_months': 12,
    'review_inner_min_train_months': 12,
    'review_anova_k': 3,
    'review_stepwise_min_gain': 0.002,
    'review_max_features': 4,
    'review_lasso_c_grid': [0.001, 0.01, 0.1, 1.0],
    'review_lasso_max_features': 10,
    'review_lasso_seed': 0,
    'review_fm_t_hurdle': 3.0,
    'review_fm_nw_lag': 6,
    'review_auc_hurdle': 0.015,
    'review_family_alpha': 0.05,
    'review_min_winning_years': 6,
    'review_fresh_check_min_dates': 6,
    'review_manual_sets': {},
    'review_candidate_features': 'all',
}


@pytest.fixture
def step1_settings() -> Dict[str, Any]:
    """Production settings with the pinned model parameters, a short history and few bootstrap replications."""
    settings = load_settings()
    settings.update({key: list(value) if isinstance(value, list) else value
                     for key, value in PINNED_STEP1_SETTINGS.items()})
    settings.update({
        'step1_history_years': 3,
        'step1_validation_years': 4,
        'step1_min_stocks_per_date': 10,
        'step1_bootstrap_replications': 300,
    })
    return settings


@pytest.fixture
def simulate_panel() -> PanelFactory:
    """
    Returns a factory for synthetic universes of geometric random walks.

    Every stock has its own constant daily drift (normal with sd `drift_sd`). With a
    positive `drift_sd`, past returns predict future returns across stocks; with 0, the
    returns are pure noise.
    """
    def factory(n_stocks: int = 40, n_days: int = 2300, drift_sd: float = 0.0, vol: float = 0.015,
                seed: int = 0, start: str = '2014-01-01', volume: Optional[float] = 1000.0) -> PricePanel:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start, periods=n_days)
        drifts = rng.normal(0.0, drift_sd, n_stocks) if drift_sd > 0 else np.zeros(n_stocks)
        returns = drifts[None, :] + vol * rng.standard_normal((n_days, n_stocks))
        tickers = [f"S{i:03d}.DE" for i in range(n_stocks)]
        close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)
        volumes = pd.DataFrame(volume, index=dates, columns=tickers)
        return PricePanel(close=close, volume=volumes, missing=[])
    return factory


def synthetic_stock(n_days: int = 900, seed: int = 0, gap_share: float = 0.03) -> Dict[str, pd.Series]:
    """
    One synthetic stock with consistent OHLCV and a market index on a slightly different calendar.

    The stock's returns load on the market returns plus noise; a share `gap_share` of the
    business days has no trade (no row), and some days have no volume.

    Args:
        n_days (int): Business days of the market calendar.
        seed (int): Random seed.
        gap_share (float): Share of market days without a trade of the stock.

    Returns:
        Dict[str, pd.Series]: close, volume, open, high and low on the stock's own days, and
        market (the index closes on all market days).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', periods=n_days)
    market_returns = 0.0002 + 0.01 * rng.standard_normal(n_days)
    returns = 0.0003 + 1.2 * market_returns + 0.015 * rng.standard_normal(n_days)
    close = pd.Series(40.0 * np.exp(np.cumsum(returns)), index=dates)
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(0.004 * rng.standard_normal(n_days))
    high = np.maximum(close, open_) * np.exp(np.abs(0.006 * rng.standard_normal(n_days)))
    low = np.minimum(close, open_) * np.exp(-np.abs(0.006 * rng.standard_normal(n_days)))
    volume = pd.Series(rng.integers(0, 3000, n_days).astype(float), index=dates)
    volume[rng.random(n_days) < 0.05] = 0.0
    own = rng.random(n_days) >= gap_share
    own[:2] = True
    market = pd.Series(12000.0 * np.exp(np.cumsum(market_returns)), index=dates)
    return {'close': close[own], 'volume': volume[own], 'open': open_[own], 'high': high[own], 'low': low[own],
            'market': market}


def ohlc_panel(n_stocks: int = 40, n_days: int = 1500, drift_sd: float = 0.0008, seed: int = 0,
               start: str = '2016-01-04', missing_days: int = 10) -> PricePanel:
    """
    A synthetic universe with opens, highs, lows, volumes and a market index.

    Every stock has its own constant daily drift (normal with sd `drift_sd`) and loads on a
    common market return; `missing_days` days without a trade are scattered per stock.

    Args:
        n_stocks (int): Number of stocks.
        n_days (int): Business days.
        drift_sd (float): Standard deviation of the daily drifts (0: pure noise).
        seed (int): Random seed.
        start (str): First business day.
        missing_days (int): Days without a trade per stock.

    Returns:
        PricePanel: The universe with the market index '^GDAXI'.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    market = 0.0002 + 0.01 * rng.standard_normal(n_days)
    drifts = rng.normal(0.0, drift_sd, n_stocks)
    betas = rng.uniform(0.5, 1.5, n_stocks)
    returns = drifts[None, :] + market[:, None] * betas[None, :] + 0.015 * rng.standard_normal((n_days, n_stocks))
    tickers = [f"S{i:03d}.DE" for i in range(n_stocks)]
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)
    open_ = close.shift(1).bfill() * np.exp(0.003 * rng.standard_normal(close.shape))
    high = np.maximum(close, open_) * np.exp(np.abs(0.005 * rng.standard_normal(close.shape)))
    low = np.minimum(close, open_) * np.exp(-np.abs(0.005 * rng.standard_normal(close.shape)))
    volume = pd.DataFrame(rng.integers(500, 5000, close.shape).astype(float), index=dates, columns=tickers)
    for j in range(n_stocks):
        gaps = rng.choice(np.arange(1, n_days - 1), size=missing_days, replace=False)
        for frame in (close, open_, high, low, volume):
            frame.iloc[gaps, j] = np.nan
    index = pd.Series(10000.0 * np.exp(np.cumsum(market)), index=dates, name='^GDAXI')
    return PricePanel(close=close, volume=volume, missing=[], open=open_, high=high, low=low, market=index)


@pytest.fixture
def simulate_ohlc_panel() -> PanelFactory:
    """Returns ohlc_panel(), the factory for synthetic universes with OHLCV and a market index."""
    return ohlc_panel
