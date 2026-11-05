"""
Execution of Step 1 and Step 2, shared by the local CLI (batch_runner.py) and the GitHub
Actions scripts (scripts/02_run_step1.py, scripts/03_run_step2.py).

Step 1 runs once for the whole universe: one pooled model ranks all stocks, and one
prediction payload is exported per ticker with price data. Step 2 runs per UP ticker and
extends that payload.
"""
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config.settings import load_settings
from ..ingestion.company_profile import fetch_company_profile
from ..ingestion.funds_api import fetch_fundamentals
from ..ingestion.market_api import fetch_price_panel
from ..modeling.step1_pooled import (Step1Result, execute_step1_pooled, ticker_feature_diagnostics,
                                     ticker_step1_model)
from ..modeling.step2_funds import execute_step2
from .json_exporter import (export_feature_diagnostics_json, export_model_summary, export_prediction_json,
                            load_feature_diagnostics_json, load_prediction_json)

logger = logging.getLogger(__name__)

APPLIED_PARAMETER_KEYS = (
    'horizon_days', 'threshold', 'step1_history_years', 'feature_warmup_years', 'step1_features',
    'step1_l2_c', 'step1_up_share', 'step1_min_price_eur', 'step1_validation_years',
    'step1_min_validation_auc_lower', 'min_step2_score', 'min_step2_applicable_rules',
)

QUALIFIED_TICKERS_PATH = Path('data') / 'state' / 'qualified_tickers.csv'
STEP1_PASSED_PATH = Path('data') / 'state' / 'step1_passed.csv'
DEFAULT_COMPANY_NAME = 'Unknown Company'
# Random pause before each ticker's Yahoo Finance .info request, against rate limiting
ANTI_JITTER_SECONDS = (0.5, 2.5)


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


def apply_anti_jitter() -> None:
    """Sleeps for a random interval so consecutive Yahoo Finance requests are spread out."""
    time.sleep(random.uniform(*ANTI_JITTER_SECONDS))


def load_qualified_universe(path: Path = QUALIFIED_TICKERS_PATH) -> List[Dict[str, str]]:
    """
    Reads the qualified universe written by scripts/01_run_t7.py.

    Args:
        path (Path): Path of qualified_tickers.csv (columns Ticker and, optionally, Company).

    Returns:
        List[Dict[str, str]]: {'Ticker': ..., 'Company': ...} per ticker, sorted by ticker.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no Ticker column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run scripts/01_run_t7.py (workflow "
                                "'1. T7 Download (Initialization)') to create and commit the qualified ticker list.")
    frame = pd.read_csv(path)
    if 'Ticker' not in frame.columns:
        raise ValueError(f"{path} has no 'Ticker' column.")
    if 'Company' not in frame.columns:
        frame['Company'] = DEFAULT_COMPANY_NAME
    frame['Company'] = frame['Company'].fillna(DEFAULT_COMPANY_NAME).astype(str)
    frame = frame.dropna(subset=['Ticker']).drop_duplicates(subset='Ticker').sort_values('Ticker')
    return [{'Ticker': str(row.Ticker), 'Company': row.Company} for row in frame.itertuples(index=False)]


def write_step1_passed(path: Path, tickers: Iterable[str]) -> None:
    """
    Writes the Step 1 UP tickers for Step 2 (column Ticker; header only if none is UP).

    Args:
        path (Path): Target file (data/state/step1_passed.csv).
        tickers (Iterable[str]): The UP tickers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Ticker': sorted(set(tickers))}).to_csv(path, index=False)


def load_step1_passed(path: Path = STEP1_PASSED_PATH) -> List[str]:
    """
    Reads the Step 1 UP tickers. A header-only file means that no stock is UP.

    Args:
        path (Path): Path of step1_passed.csv.

    Returns:
        List[str]: The UP tickers, de-duplicated and sorted.

    Raises:
        FileNotFoundError: If the file does not exist (Step 1 did not run or its artifact is missing).
        ValueError: If the file is empty or has no Ticker column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Did the Step 1 job run and upload the step1-state artifact?")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"{path} is empty; Step 1 always writes at least the header 'Ticker'.") from e
    if 'Ticker' not in frame.columns:
        raise ValueError(f"{path} has no 'Ticker' column.")
    return sorted(set(frame['Ticker'].dropna().astype(str)))


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


def build_step1_payload(ticker: str, company_name: str, step1_model: Dict[str, Any],
                        latest_price: Optional[float], settings: Dict[str, Any],
                        profile: Dict[str, Any], snapshot: Dict[str, Optional[float]],
                        prediction_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds the prediction payload of one ticker after Step 1.

    Args:
        ticker (str): The stock ticker.
        company_name (str): Company name from the Xetra T7 list.
        step1_model (Dict[str, Any]): The ticker's `step1_model` entry (ticker_step1_model()).
        latest_price (Optional[float]): Last close in the price panel.
        settings (Dict[str, Any]): Loaded settings.
        profile (Dict[str, Any]): Company profile (full_name, description).
        snapshot (Dict[str, Optional[float]]): pe_ratio, beta and liquidity.
        prediction_date (Optional[str]): ISO date; defaults to today in Europe/Berlin.

    Returns:
        Dict[str, Any]: The payload; final_prediction is 'UP' or 'NOT_UP'.
    """
    return {
        "stock_name": ticker,
        "company_name": profile.get("full_name", company_name),
        "company_description": profile.get("description", "No description available."),
        "latest_price": latest_price,
        "pe_ratio": snapshot.get("pe_ratio"),
        "beta": snapshot.get("beta"),
        "liquidity": snapshot.get("liquidity"),
        "prediction_date": prediction_date or pd.Timestamp.now(tz='Europe/Berlin').date().isoformat(),
        "applied_parameters": {key: settings[key] for key in APPLIED_PARAMETER_KEYS},
        "step1_model": step1_model,
        "final_prediction": "UP" if step1_model.get('predicted_class') == 'UP' else "NOT_UP",
    }


@dataclass
class Step1Run:
    """
    Outcome of one pooled Step 1 run.

    Attributes:
        result (Step1Result): Model result for every ticker with price data.
        exported (List[str]): Tickers whose payload was exported in this run.
        up_tickers (List[str]): Exported tickers predicted UP, sorted (the Step 2 input).
        missing (List[str]): Tickers without price data (no payload).
        timings (Dict[str, float]): Runtime in seconds per stage.
    """
    result: Step1Result
    exported: List[str] = field(default_factory=list)
    up_tickers: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> Dict[str, Any]:
        """The model summary (report key `step1_model`)."""
        return self.result.summary


def run_step1_pooled(universe: Sequence[Dict[str, str]], settings: Optional[Dict[str, Any]] = None,
                     only_tickers: Optional[Iterable[str]] = None) -> Step1Run:
    """
    Runs Step 1 for the whole universe and exports the per-ticker payloads.

    Downloads the prices of all tickers, fits and validates the pooled model, ranks the
    eligible stocks of the last closed trading day, writes the model summary to
    outputs/diagnostics/ and exports one payload (with company profile and market
    snapshot) and one feature diagnostics file per ticker with price data.

    Args:
        universe (Sequence[Dict[str, str]]): {'Ticker': ..., 'Company': ...} per ticker.
        settings (Optional[Dict[str, Any]]): Loaded settings (loaded if omitted).
        only_tickers (Optional[Iterable[str]]): Export only these tickers' payloads. The
            model is still fitted and ranked on the whole universe.

    Returns:
        Step1Run: The model result, the exported and UP tickers and the stage runtimes.
    """
    settings = settings or load_settings()
    companies = {row['Ticker']: (row.get('Company') or DEFAULT_COMPANY_NAME) for row in universe}
    _, fetch_years = history_window(settings)
    timings: Dict[str, float] = {}

    started = time.monotonic()
    price_panel = fetch_price_panel(sorted(companies), fetch_years, settings)
    timings['download_s'] = round(time.monotonic() - started, 1)

    started = time.monotonic()
    result = execute_step1_pooled(price_panel, settings)
    export_model_summary(result.summary, result.validation.per_date)
    timings['model_s'] = round(time.monotonic() - started, 1)

    wanted = None if only_tickers is None else set(only_tickers)
    targets = [ticker for ticker in result.results.index if wanted is None or ticker in wanted]
    started = time.monotonic()
    exported: List[str] = []
    for ticker in tqdm(targets, desc="Step 1 - export payloads"):
        apply_anti_jitter()
        latest_price = result.results.at[ticker, 'latest_price']
        latest_price = float(latest_price) if pd.notna(latest_price) else None
        company_name = companies.get(ticker, DEFAULT_COMPANY_NAME)
        payload = build_step1_payload(
            ticker, company_name, ticker_step1_model(result, ticker), latest_price, settings,
            profile=fetch_company_profile(ticker, company_name, str(settings['gemini_model'])),
            snapshot=fetch_market_snapshot(ticker, latest_price),
        )
        try:
            export_feature_diagnostics_json(ticker, {"step1_pooled": ticker_feature_diagnostics(result, ticker)})
            export_prediction_json(ticker, payload)
        except (TypeError, ValueError, OSError):
            logger.exception(f"Failed to export the Step 1 payload of {ticker}")
            continue
        exported.append(ticker)
    timings['export_s'] = round(time.monotonic() - started, 1)

    exported_set = set(exported)
    up_tickers = [ticker for ticker in result.up_tickers if ticker in exported_set]
    logger.info(f"Step 1 exported {len(exported)} payloads; {len(up_tickers)} UP. Runtime per stage: {timings}")
    return Step1Run(result=result, exported=exported, up_tickers=up_tickers,
                    missing=list(price_panel.missing), timings=timings)


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
