"""
Shared fixtures: synthetic price panels and fast Step 1 settings (offline, no network).
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
    'step1_ffill_limit_days': 5,
    'step1_bootstrap_block_months': 6,
    'step1_bootstrap_seed': 0,
    'step1_min_validation_auc_lower': 0.5,
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
