"""
Offline tests of the report consolidation and publication (src/orchestration/report.py):
merging of the shard payloads, the published files, the history archive and its pruning,
the profile cache and the Step 1 model summary key.
"""
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.orchestration.report import (consolidate_predictions, prune_history, publish_report,
                                      report_date_from_path)


def _write(path: Path, payload: Any) -> str:
    """Writes `payload` as JSON (creating the folders) and returns the path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return str(path)


def test_consolidation_keeps_the_most_complete_payload_per_ticker(tmp_path: Path) -> None:
    files = [
        # Shard 0 artifact: Step 2 ran for AAA, BBB only has its Step 1 payload
        _write(tmp_path / 's0' / 'AAA.json', {'stock_name': 'AAA', 'final_prediction': 'UP_FINAL_BUY', 'step2_model': {}}),
        _write(tmp_path / 's0' / 'BBB.json', {'stock_name': 'BBB', 'final_prediction': 'UP'}),
        # Shard 1 artifact: Step 2 ran for BBB
        _write(tmp_path / 's1' / 'AAA.json', {'stock_name': 'AAA', 'final_prediction': 'UP'}),
        _write(tmp_path / 's1' / 'BBB.json', {'stock_name': 'BBB', 'final_prediction': 'UP', 'step2_model': {'x': 1}}),
        _write(tmp_path / 's1' / 'CCC.json', {'stock_name': 'CCC', 'final_prediction': 'PENDING'}),
    ]
    merged = {p['stock_name']: p for p in consolidate_predictions(files)}
    assert merged['AAA']['final_prediction'] == 'UP_FINAL_BUY'
    assert merged['BBB']['step2_model'] == {'x': 1}
    assert merged['CCC']['final_prediction'] == 'NOT_UP'


def test_history_is_pruned_by_report_date_not_file_age(tmp_path: Path) -> None:
    old = _write(tmp_path / '2026-01' / 'report_2026-01-05.json', {})
    kept = _write(tmp_path / '2026-06' / 'report_2026-06-01.json', {})
    removed = prune_history(str(tmp_path), today=date(2026, 10, 9), retention_days=200)
    assert removed == [old]
    assert not (tmp_path / '2026-01').exists()
    assert (tmp_path / '2026-06' / 'report_2026-06-01.json').exists()
    assert report_date_from_path(kept) == date(2026, 6, 1)


def test_publish_writes_report_buy_list_and_archive(tmp_path: Path) -> None:
    predictions = [{'stock_name': 'AAA', 'final_prediction': 'UP_FINAL_BUY'},
                   {'stock_name': 'BBB', 'final_prediction': 'NOT_UP'}]
    now = datetime(2026, 10, 9, 1, 30, tzinfo=timezone.utc)
    publish_report(predictions, {'history_retention_days': 200}, str(tmp_path), now=now)

    report = json.loads((tmp_path / 'full_batch_report.json').read_text())
    assert report['execution_date'] == '2026-10-09T01:30:00.000000Z'
    assert pd.read_csv(tmp_path / 'final_buy_signals.csv')['Ticker'].tolist() == ['AAA']
    assert (tmp_path / 'history' / '2026-10' / 'report_2026-10-09.json').exists()


def test_profile_cache_keeps_generated_descriptions_only(tmp_path: Path) -> None:
    from src.orchestration.report import update_profile_cache

    cache = tmp_path / 'company_profiles_cache.json'
    cache.write_text(json.dumps({'OLD': {'full_name': 'Old AG', 'description': 'Kept.'}}))
    predictions = [
        {'stock_name': 'AAA', 'company_name': 'A AG', 'company_description': 'Makes things.'},
        {'stock_name': 'BBB', 'company_name': 'B AG', 'company_description': 'Error fetching description.'},
    ]
    assert update_profile_cache(predictions, str(cache)) == 2
    assert set(json.loads(cache.read_text())) == {'AAA', 'OLD'}


def test_prediction_files_are_found_in_every_artifact_layout(tmp_path: Path) -> None:
    from src.orchestration.report import find_prediction_files

    # Artifact root = common parent of the uploaded paths (outputs/) ...
    a = _write(tmp_path / 'step2_artifacts' / 'step2-final-shard-0' / 'predictions' / 'AAA.DE_prediction.json', {})
    # ... or the repository root when files outside outputs/ are uploaded as well
    b = _write(tmp_path / 'step2_artifacts' / 'step2-final-shard-1' / 'outputs' / 'predictions' / 'BBB.DE_prediction.json', {})

    found = find_prediction_files(str(tmp_path / 'step2_artifacts' / '**' / 'predictions' / '*.json'))

    assert found == sorted([a, b])


def test_the_model_summary_is_published_as_the_step1_model_key(tmp_path: Path) -> None:
    from src.orchestration.report import build_report

    summary = {'model': 'pooled_cross_sectional_logit', 'gate': {'passed': True}}
    now = datetime(2026, 11, 4, 1, 30, tzinfo=timezone.utc)
    assert 'step1_model' not in build_report([], {}, now)
    publish_report([{'stock_name': 'AAA', 'final_prediction': 'UP'}], {'history_retention_days': 200},
                   str(tmp_path), now=now, step1_model_summary=summary)

    report = json.loads((tmp_path / 'full_batch_report.json').read_text())
    assert report['step1_model'] == summary
    assert list(report) == ['execution_date', 'parameters', 'step1_model', 'predictions']
    archived = json.loads((tmp_path / 'history' / '2026-11' / 'report_2026-11-04.json').read_text())
    assert archived['step1_model'] == summary


def test_the_first_model_summary_of_the_artifacts_is_used(tmp_path: Path) -> None:
    from src.orchestration.report import load_model_summary

    pattern = str(tmp_path / 'step2_artifacts' / '**' / 'diagnostics' / 'step1_model_summary.json')
    assert load_model_summary(pattern) is None
    _write(tmp_path / 'step2_artifacts' / 'step2-final-shard-1' / 'diagnostics' / 'step1_model_summary.json', {'shard': 1})
    _write(tmp_path / 'step2_artifacts' / 'step2-final-shard-0' / 'diagnostics' / 'step1_model_summary.json', {'shard': 0})
    assert load_model_summary(pattern) == {'shard': 0}


def test_an_unreadable_model_summary_is_skipped_with_a_warning(tmp_path: Path,
                                                               caplog: pytest.LogCaptureFixture) -> None:
    from src.orchestration.report import load_model_summary

    path = tmp_path / 'diagnostics' / 'step1_model_summary.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "pooled_cross_sectional_logit", ')  # truncated file

    with caplog.at_level(logging.WARNING, logger='src.orchestration.report'):
        assert load_model_summary(str(path)) is None
    assert 'Failed to read the Step 1 model summary' in caplog.text
