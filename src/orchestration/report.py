"""
Consolidation of per-ticker prediction payloads into the batch report, the buy-signal
list and the dated history archive. Used by scripts/04_consolidate.py (GitHub Actions)
and by the local batch runner.
"""
import glob
import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..ingestion.company_profile import FALLBACK_DESCRIPTIONS
from .json_exporter import write_json_atomic

logger = logging.getLogger(__name__)

VALID_PREDICTIONS = ("UP", "NOT_UP", "UP_FINAL_BUY")
REPORT_FILE_PATTERN = re.compile(r"report_(\d{4}-\d{2}-\d{2})\.json$")
PROFILE_CACHE_NAME = 'company_profiles_cache.json'


def find_prediction_files(pattern: str) -> List[str]:
    """
    Returns the prediction payloads matching a glob in which `**` matches any number of
    directories.
    
    The folder layout of downloaded GitHub Actions artifacts depends on the uploaded
    paths (their common parent becomes the artifact root), so the pipeline searches
    recursively, e.g. `step2_artifacts/**/predictions/*.json`.
    """
    return sorted(glob.glob(pattern, recursive=True))


def _preference(payload: Dict[str, Any]) -> tuple:
    # A payload that went through Step 2 is more complete; a buy beats a Step 2 rejection
    return ('step2_model' in payload, payload.get('final_prediction') == 'UP_FINAL_BUY')


def consolidate_predictions(prediction_files: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Merges prediction payloads from all shards into one payload per ticker.

    Every Step 2 shard artifact contains the Step 1 payloads of all tickers but the
    Step 2 results of its own shard only, so the most complete payload is kept.

    Args:
        prediction_files (Iterable[str]): Paths of *_prediction.json files.

    Returns:
        List[Dict[str, Any]]: One payload per ticker, sorted by ticker.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(prediction_files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read {path}: {e}")
            continue
        ticker = payload.get('stock_name')
        if not ticker:
            continue
        if payload.get('final_prediction') not in VALID_PREDICTIONS:
            payload['final_prediction'] = 'NOT_UP'
        current = merged.get(ticker)
        if current is None or _preference(payload) > _preference(current):
            merged[ticker] = payload
    return [merged[ticker] for ticker in sorted(merged)]


def build_report(predictions: List[Dict[str, Any]], settings: Dict[str, Any],
                 now: Optional[datetime] = None,
                 step1_model_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Builds the batch report published to the dashboard.

    Args:
        predictions (List[Dict[str, Any]]): Consolidated payloads.
        settings (Dict[str, Any]): Settings used for the run.
        now (Optional[datetime]): Execution time (UTC); defaults to the current time.
        step1_model_summary (Optional[Dict[str, Any]]): Summary of the pooled Step 1 model
            (validation, gate, coefficients), written as the top-level key `step1_model`.
            The key is omitted if no summary is given.

    Returns:
        Dict[str, Any]: The report.
    """
    now = now or datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "execution_date": now.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        "parameters": settings,
    }
    if step1_model_summary is not None:
        report["step1_model"] = step1_model_summary
    report["predictions"] = predictions
    return report


def load_model_summary(pattern: str) -> Optional[Dict[str, Any]]:
    """
    Loads the Step 1 model summary from the downloaded artifacts.

    Every Step 2 shard artifact carries an identical copy, so the first file in sorted
    order is used.

    Args:
        pattern (str): Glob in which `**` matches any number of directories.

    Returns:
        Optional[Dict[str, Any]]: The summary, or None (with a warning) if no file matches
        or the file cannot be read.
    """
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        logger.warning(f"No Step 1 model summary matches {pattern}; the report is published without 'step1_model'.")
        return None
    try:
        with open(paths[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read the Step 1 model summary {paths[0]}: {e}")
        return None


def write_final_buy_signals(predictions: List[Dict[str, Any]], path: str) -> List[str]:
    """
    Writes the tickers that passed both steps to a CSV with a `Ticker` column.

    Returns:
        List[str]: The buy candidates.
    """
    buys = sorted(p['stock_name'] for p in predictions if p.get('final_prediction') == 'UP_FINAL_BUY')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pd.DataFrame({'Ticker': buys}).to_csv(path, index=False)
    return buys


def archive_report(report: Dict[str, Any], history_dir: str, report_date: date) -> str:
    """
    Stores the report as history/YYYY-MM/report_YYYY-MM-DD.json.

    Returns:
        str: Path of the archived report.
    """
    path = os.path.join(history_dir, report_date.strftime('%Y-%m'), f"report_{report_date.isoformat()}.json")
    write_json_atomic(path, report)
    return path


def report_date_from_path(path: str) -> Optional[date]:
    """Returns the date encoded in a report_YYYY-MM-DD.json file name, or None."""
    match = REPORT_FILE_PATTERN.search(os.path.basename(path))
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def prune_history(history_dir: str, today: date, retention_days: int) -> List[str]:
    """
    Deletes archived reports older than the retention window.

    The age is taken from the date in the file name. File modification times are not
    usable: every file of a fresh git checkout has the checkout time.

    Args:
        history_dir (str): The history directory.
        today (date): Reference date.
        retention_days (int): Reports dated before today - retention_days are removed.

    Returns:
        List[str]: Removed files.
    """
    cutoff = today - timedelta(days=retention_days)
    removed = []
    for path in glob.glob(os.path.join(history_dir, '*', 'report_*.json')):
        report_date = report_date_from_path(path)
        if report_date is not None and report_date < cutoff:
            os.remove(path)
            removed.append(path)
    for month_dir in glob.glob(os.path.join(history_dir, '*')):
        if os.path.isdir(month_dir) and not os.listdir(month_dir):
            os.rmdir(month_dir)
    if removed:
        logger.info(f"Pruned {len(removed)} archived reports older than {cutoff}.")
    return removed


def update_profile_cache(predictions: List[Dict[str, Any]], cache_path: str) -> int:
    """
    Adds the company profiles of the consolidated predictions to the profile cache.
    
    The cache is committed by the pipeline, so the next run reads descriptions from it
    instead of requesting all of them from the Gemini API again.
    
    Args:
        predictions (List[Dict[str, Any]]): Consolidated payloads.
        cache_path (str): Path of company_profiles_cache.json.
        
    Returns:
        int: Number of cached profiles.
    """
    cache: Dict[str, Any] = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    for payload in predictions:
        description = payload.get('company_description')
        if payload.get('stock_name') and description and description not in FALLBACK_DESCRIPTIONS:
            cache[payload['stock_name']] = {
                'full_name': payload.get('company_name'),
                'description': description,
            }
    write_json_atomic(cache_path, dict(sorted(cache.items())))
    return len(cache)


def publish_report(predictions: List[Dict[str, Any]], settings: Dict[str, Any], out_dir: str,
                   now: Optional[datetime] = None,
                   step1_model_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Writes full_batch_report.json and final_buy_signals.csv to `out_dir`, archives the
    report under `out_dir`/history, prunes reports older than `history_retention_days`
    and adds the company profiles to `out_dir`/company_profiles_cache.json.

    Args:
        predictions (List[Dict[str, Any]]): Consolidated payloads.
        settings (Dict[str, Any]): Settings used for the run.
        out_dir (str): Output directory (data/processed in production).
        now (Optional[datetime]): Execution time (UTC); defaults to the current time.
        step1_model_summary (Optional[Dict[str, Any]]): Step 1 model summary, published as
            the report key `step1_model` (omitted if None).

    Returns:
        Dict[str, Any]: The published report.
    """
    now = now or datetime.now(timezone.utc)
    report = build_report(predictions, settings, now, step1_model_summary)
    write_json_atomic(os.path.join(out_dir, 'full_batch_report.json'), report)
    buys = write_final_buy_signals(predictions, os.path.join(out_dir, 'final_buy_signals.csv'))
    history_dir = os.path.join(out_dir, 'history')
    archive_report(report, history_dir, now.date())
    prune_history(history_dir, now.date(), int(settings['history_retention_days']))
    update_profile_cache(predictions, os.path.join(out_dir, PROFILE_CACHE_NAME))
    logger.info(f"Published report with {len(predictions)} predictions and {len(buys)} buy candidates to {out_dir}.")
    return report
