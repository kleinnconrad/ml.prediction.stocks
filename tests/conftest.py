"""
Shared fixtures: synthetic price panels and fast Step 1 settings (offline, no network).
"""
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from config.settings import load_settings
from src.ingestion.market_api import PricePanel

PanelFactory = Callable[..., PricePanel]


@pytest.fixture
def step1_settings() -> Dict[str, Any]:
    """Production settings with a short history and few bootstrap replications."""
    settings = load_settings()
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
