"""
Local execution flow behind src/main.py: the same Step 1, Step 2 and report code as the
GitHub Actions scripts, run sequentially in one process.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from config.settings import load_settings
from ..ingestion.xetra_t7 import fetch_xetra_t7
from ..processing.qualifier import filter_qualified_tickers
from .json_exporter import PREDICTIONS_DIR
from .report import consolidate_predictions, publish_report
from .steps import QUALIFIED_TICKERS_PATH, load_qualified_universe, run_step1_pooled, run_step2_for_ticker

logger = logging.getLogger(__name__)


def describe_step1_result(ticker: str, row: Dict[str, Any], n_ranked: int) -> str:
    """
    Formats one ticker's Step 1 result for the log.

    Args:
        ticker (str): The stock ticker.
        row (Dict[str, Any]): The ticker's row of Step1Result.results.
        n_ranked (int): Number of ranked stocks on the scoring date.

    Returns:
        str: e.g. "SAP.DE: rank 12 of 371, probability 0.41, UP (ok)", where the bracket
        holds the Step 1 status.
    """
    rank = row.get('rank')
    if rank is None or rank != rank:  # NaN: not ranked
        return f"{ticker}: not ranked ({row.get('status')}), {row.get('predicted_class')}"
    return (f"{ticker}: rank {int(rank)} of {n_ranked}, probability {float(row['probability']):.2f}, "
            f"{row.get('predicted_class')} ({row.get('status')})")


def run_single(ticker: str) -> bool:
    """
    Runs the pipeline for one ticker.

    A pooled model cannot be fitted on one stock: Step 1 is fitted and ranked on the
    universe in data/state/qualified_tickers.csv (plus the ticker if it is not listed),
    but only the requested ticker's payload is exported. Step 2 runs if the ticker is UP.

    Args:
        ticker (str): The stock ticker (e.g. 'SAP.DE').

    Returns:
        bool: True if the ticker is a final buy candidate.

    Raises:
        FileNotFoundError: If data/state/qualified_tickers.csv does not exist.
    """
    ticker = ticker.strip().upper()  # Yahoo Finance tickers are case-insensitive; the T7 list is upper case
    settings = load_settings()
    universe = load_qualified_universe(QUALIFIED_TICKERS_PATH)
    if ticker not in {row['Ticker'] for row in universe}:
        logger.info(f"{ticker} is not in {QUALIFIED_TICKERS_PATH}; it is added to the ranked universe for this run.")
        universe.append({'Ticker': ticker, 'Company': ticker})
    logger.info(f"Running Step 1 on {len(universe)} tickers and exporting {ticker}.")

    step1 = run_step1_pooled(universe, settings, only_tickers=[ticker])
    results = step1.result.results
    if ticker not in results.index:
        logger.warning(f"{ticker}: no price data. No payload was exported.")
        return False
    logger.info(describe_step1_result(ticker, results.loc[ticker].to_dict(),
                                      int(step1.summary['cross_section']['n_ranked'])))
    if ticker not in step1.up_tickers:
        return False
    payload = run_step2_for_ticker(ticker)
    return payload['final_prediction'] == 'UP_FINAL_BUY'


def run_batch() -> None:
    """
    Runs the local batch: T7 download and qualification, the pooled Step 1 for all
    qualified tickers, Step 2 for every UP ticker, then consolidation, publication (with
    the model summary) and the uplift evaluation.
    """
    logger.info("Starting local batch runner.")
    settings = load_settings()
    raw_df = fetch_xetra_t7(settings['xetra_t7_url'])
    qualified_tickers = filter_qualified_tickers(raw_df)
    if not qualified_tickers:
        logger.warning("No qualified tickers found. Exiting.")
        return

    step1 = run_step1_pooled(qualified_tickers, settings)

    buy_candidates: List[str] = []
    for ticker in tqdm(step1.up_tickers, desc="Step 2"):
        try:
            if run_step2_for_ticker(ticker)['final_prediction'] == 'UP_FINAL_BUY':
                buy_candidates.append(ticker)
        except Exception:  # one failing ticker must not stop the batch (as in scripts/03_run_step2.py)
            logger.exception(f"Failed {ticker} in Step 2")
    logger.info(f"Local batch run complete. {len(buy_candidates)} buy candidates found.")

    # Only this run's payloads: outputs/ may still hold files of tickers that left the universe
    prediction_files = [str(Path(PREDICTIONS_DIR) / f"{ticker}_prediction.json") for ticker in step1.exported]
    predictions = consolidate_predictions(prediction_files)
    publish_report(predictions, settings, os.path.join('data', 'processed'), step1_model_summary=step1.summary)

    try:
        from ..processing.uplift_evaluator import evaluate_uplift
        evaluate_uplift()
    except Exception as e:
        logger.error(f"Failed to run uplift evaluation: {e}")
