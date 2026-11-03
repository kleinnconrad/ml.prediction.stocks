"""
Per-ticker execution of Step 1 and Step 2, shared by the local CLI (batch_runner.py)
and the GitHub Actions scripts (scripts/02_run_step1.py, scripts/03_run_step2.py).
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

from config.settings import load_settings
from ..ingestion.company_profile import fetch_company_profile
from ..ingestion.funds_api import fetch_fundamentals
from ..ingestion.market_api import fetch_step1_data
from ..modeling.step1_macro import execute_step1
from ..modeling.step2_funds import execute_step2
from ..processing.features import engineer_features
from .json_exporter import (export_feature_diagnostics_json, export_prediction_json,
                            load_feature_diagnostics_json, load_prediction_json)

logger = logging.getLogger(__name__)

APPLIED_PARAMETER_KEYS = (
    'horizon_days', 'threshold', 'step1_history_years', 'feature_warmup_years',
    'features_to_select', 'anova_k', 'cv_splits', 'min_cv_accuracy',
    'min_step2_score', 'min_step2_applicable_rules',
)


def history_window(settings: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns the Step 1 training window and the history to fetch including feature warm-up.

    Args:
        settings (Dict[str, Any]): Loaded settings.

    Returns:
        Tuple[int, int]: (training years, years to fetch).
    """
    history_years = int(settings['step1_history_years'])
    return history_years, history_years + int(settings['feature_warmup_years'])


def fetch_market_snapshot(ticker: str, latest_price: Optional[float]) -> Dict[str, Optional[float]]:
    """
    Fetches the display-only P/E ratio, beta and average daily traded value (EUR millions).

    Args:
        ticker (str): The stock ticker.
        latest_price (Optional[float]): Last closing price used for the liquidity estimate.

    Returns:
        Dict[str, Optional[float]]: pe_ratio, beta and liquidity (None where unavailable).
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning(f"Failed to fetch info for {ticker}: {e}")
        return {"pe_ratio": None, "beta": None, "liquidity": None}
    avg_vol = info.get('averageVolume')
    liquidity = (avg_vol * latest_price) / 1_000_000 if avg_vol is not None and latest_price is not None else None
    return {"pe_ratio": info.get('trailingPE'), "beta": info.get('beta'), "liquidity": liquidity}


def run_step1_for_ticker(ticker: str, company_name: str, macro_df: pd.DataFrame,
                         settings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Runs Step 1 for one ticker and exports its prediction payload and diagnostics.

    Args:
        ticker (str): The stock ticker (e.g. 'SAP.DE').
        company_name (str): Company name from the Xetra T7 list.
        macro_df (pd.DataFrame): Pre-cached global macro feature matrix.
        settings (Optional[Dict[str, Any]]): Loaded settings (loaded if omitted).

    Returns:
        Optional[Dict[str, Any]]: The exported payload (final_prediction 'UP' or 'NOT_UP'),
        or None if no market data is available.
    """
    settings = settings or load_settings()
    history_years, fetch_years = history_window(settings)

    merged_df = fetch_step1_data(ticker, macro_df, history_years=fetch_years)
    if merged_df.empty:
        logger.warning(f"No market data for {ticker}. Skipping.")
        return None

    features_df = engineer_features(
        merged_df,
        horizon_days=int(settings['horizon_days']),
        threshold=float(settings['threshold']),
        history_years=history_years,
    )
    metrics = execute_step1(features_df, ticker=ticker)
    feature_diagnostics = metrics.pop('feature_diagnostics', {})

    latest_price = float(merged_df['Close'].iloc[-1]) if 'Close' in merged_df.columns else None
    profile = fetch_company_profile(ticker, company_name, str(settings['gemini_model']))

    payload = {
        "stock_name": ticker,
        "company_name": profile.get("full_name", company_name),
        "company_description": profile.get("description", "No description available."),
        "latest_price": latest_price,
        **fetch_market_snapshot(ticker, latest_price),
        "prediction_date": pd.Timestamp.now(tz='Europe/Berlin').date().isoformat(),
        "applied_parameters": {key: settings[key] for key in APPLIED_PARAMETER_KEYS},
        "step1_model": metrics,
        "final_prediction": "UP" if metrics.get('predicted_class') == 'UP' else "NOT_UP",
    }

    export_feature_diagnostics_json(ticker, {"step1_macro": feature_diagnostics})
    export_prediction_json(ticker, payload)
    return payload


def run_step2_for_ticker(ticker: str) -> Dict[str, Any]:
    """
    Runs Step 2 for a ticker that passed Step 1 and updates its exported payload.

    The Step 1 payload and diagnostics are loaded and extended, never replaced:
    final_prediction becomes 'UP_FINAL_BUY' if the fundamental ruleset passes and stays
    'UP' otherwise (including when no fundamentals are available).

    Args:
        ticker (str): The stock ticker.

    Returns:
        Dict[str, Any]: The updated payload.
    """
    payload = load_prediction_json(ticker) or {"stock_name": ticker}

    funds_df = fetch_fundamentals(ticker)
    if funds_df.empty:
        logger.warning(f"No fundamental data for {ticker}. Keeping the Step 1 result.")
        payload["final_prediction"] = "UP"
        export_prediction_json(ticker, payload)
        return payload

    mat_dir = os.path.join('outputs', 'matrices')
    os.makedirs(mat_dir, exist_ok=True)
    funds_df.to_csv(os.path.join(mat_dir, f"{ticker}_step2_funds.csv"))

    metrics, latest_pred_class = execute_step2(funds_df)

    diagnostics = load_feature_diagnostics_json(ticker)
    diagnostics["step2_funds"] = metrics.get('feature_diagnostics', {})
    export_feature_diagnostics_json(ticker, diagnostics)

    payload["step2_model"] = metrics
    payload["final_prediction"] = "UP_FINAL_BUY" if latest_pred_class == "UP" else "UP"
    export_prediction_json(ticker, payload)

    if latest_pred_class == "UP":
        logger.info(f"{ticker} passed Step 2 and is a buy candidate.")
    return payload
