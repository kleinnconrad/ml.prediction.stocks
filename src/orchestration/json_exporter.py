"""
JSON export of the run outputs under outputs/: the per-ticker prediction payloads, the
per-ticker feature diagnostics and the Step 1 model summary with its per-date validation
table. Every JSON file is written atomically, with non-finite numbers as null.
"""
import datetime
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = os.path.join('outputs', 'predictions')
DIAGNOSTICS_DIR = os.path.join('outputs', 'diagnostics')
# Model-level Step 1 outputs; they stay under outputs/ so the CI artifact root remains outputs/
MODEL_SUMMARY_PATH = Path(DIAGNOSTICS_DIR) / 'step1_model_summary.json'
VALIDATION_PER_DATE_PATH = Path(DIAGNOSTICS_DIR) / 'step1_validation_per_date.csv'


def to_json_safe(value: Any) -> Any:
    """
    Converts numpy types to native Python types, non-finite floats to None and dates to ISO text.

    Python's json module writes NaN and Infinity as bare tokens, which are not valid JSON
    and break JSON.parse in the dashboard. Dates come from settings such as an unquoted
    `step1_features_adopted`, which YAML reads as a date.

    Args:
        value (Any): A JSON-like structure.

    Returns:
        Any: The same structure containing only JSON-serializable values.
    """
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NaT:
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(file_path: str, data: Any) -> None:
    """
    Serializes `data` first and then replaces `file_path` atomically.

    A serialization error therefore never truncates an existing file.

    Args:
        file_path (str): Target file.
        data (Any): JSON-like structure.
    """
    text = json.dumps(to_json_safe(data), indent=2, allow_nan=False)
    directory = os.path.dirname(file_path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp_path, file_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_json(file_path: str) -> Optional[Any]:
    """Reads a JSON file; returns None if the file does not exist."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_prediction_json(ticker: str, payload_data: Dict[str, Any]) -> None:
    """
    Exports the final prediction and model parameters to a strict JSON schema.

    Args:
        ticker (str): The stock ticker.
        payload_data (Dict[str, Any]): The formatted payload data.

    Raises:
        TypeError, ValueError: If the payload cannot be serialized; an existing file is kept.
    """
    file_path = os.path.join(PREDICTIONS_DIR, f"{ticker}_prediction.json")
    try:
        write_json_atomic(file_path, payload_data)
        logger.info(f"Successfully exported JSON prediction for {ticker} to {file_path}")
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize JSON for {ticker}: {e}")
        raise


def load_prediction_json(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Loads a previously exported prediction payload.

    Returns:
        Optional[Dict[str, Any]]: The payload, or None if it does not exist.
    """
    return _read_json(os.path.join(PREDICTIONS_DIR, f"{ticker}_prediction.json"))


def export_feature_diagnostics_json(ticker: str, diagnostics_data: Dict[str, Any]) -> None:
    """
    Exports the per-ticker feature diagnostics to a JSON file.

    Args:
        ticker (str): The stock ticker.
        diagnostics_data (Dict[str, Any]): Diagnostics keyed by step ("step1_pooled", "step2_funds").
    """
    file_path = os.path.join(DIAGNOSTICS_DIR, f"{ticker}_feature_diagnostics.json")
    write_json_atomic(file_path, diagnostics_data)
    logger.info(f"Successfully exported feature diagnostics for {ticker} to {file_path}")


def load_feature_diagnostics_json(ticker: str) -> Dict[str, Any]:
    """
    Loads previously exported feature diagnostics.

    Returns:
        Dict[str, Any]: The diagnostics, or an empty dict if none exist.
    """
    return _read_json(os.path.join(DIAGNOSTICS_DIR, f"{ticker}_feature_diagnostics.json")) or {}


def export_model_summary(summary: Dict[str, Any], per_date: Optional[pd.DataFrame] = None) -> Path:
    """
    Exports the Step 1 model summary and, optionally, the per-date validation table.

    Args:
        summary (Dict[str, Any]): The model summary (report key `step1_model`).
        per_date (Optional[pd.DataFrame]): Per-date validation metrics; dates are written as ISO strings.

    Returns:
        Path: Path of the summary file.
    """
    write_json_atomic(str(MODEL_SUMMARY_PATH), summary)
    if per_date is not None:
        table = per_date.copy()
        if 'date' in table.columns:
            table['date'] = pd.to_datetime(table['date']).dt.strftime('%Y-%m-%d')
        VALIDATION_PER_DATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(VALIDATION_PER_DATE_PATH, index=False)
    logger.info(f"Exported the Step 1 model summary to {MODEL_SUMMARY_PATH}")
    return MODEL_SUMMARY_PATH
