"""
Offline tests of the orchestration (src/orchestration/steps.py, batch_runner.py and
json_exporter.py): JSON export, the Step 1 run with mocked network calls, the Step 2
extension of the payload and the state files passed between the steps.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.ingestion.market_api import PricePanel
from src.orchestration import json_exporter, steps

Settings = Dict[str, Any]
PanelFactory = Callable[..., PricePanel]


@pytest.fixture
def in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Runs the test in an empty working directory (outputs/ and data/ are written there)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_non_finite_values_are_written_as_null(in_tmp: Path) -> None:
    json_exporter.export_prediction_json('ABC.DE', {
        'top_percent': float('inf'), 'probability': np.float64('nan'), 'n': np.int64(3)})
    text = (in_tmp / 'outputs' / 'predictions' / 'ABC.DE_prediction.json').read_text()
    assert json.loads(text) == {'top_percent': None, 'probability': None, 'n': 3}


def test_failed_serialization_keeps_the_existing_file(in_tmp: Path) -> None:
    json_exporter.export_prediction_json('ABC.DE', {'final_prediction': 'UP'})
    with pytest.raises(TypeError):
        json_exporter.export_prediction_json('ABC.DE', {'bad': object()})
    assert json_exporter.load_prediction_json('ABC.DE') == {'final_prediction': 'UP'}


def test_step2_extends_the_step1_payload_and_diagnostics(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    json_exporter.export_prediction_json('ABC.DE', {
        'stock_name': 'ABC.DE', 'step1_model': {'predicted_class': 'UP'}, 'final_prediction': 'UP'})
    json_exporter.export_feature_diagnostics_json('ABC.DE', {'step1_pooled': {'features': ['ret_21']}})

    statement = {'Total Revenue': 100.0, 'Net Income': 10.0, 'Operating Cash Flow': 15.0,
                 'Free Cash Flow': 8.0, 'Operating Income': 14.0, 'Current Assets': 60.0,
                 'Current Liabilities': 40.0, 'Total Debt': 50.0, 'Stockholders Equity': 200.0}
    funds = pd.DataFrame([statement] * 5, index=pd.date_range('2025-03-31', periods=5, freq='QE'))
    funds['Total Revenue'] = [100, 101, 102, 103, 120.0]
    monkeypatch.setattr(steps, 'fetch_fundamentals', lambda ticker: funds)

    payload = steps.run_step2_for_ticker('ABC.DE')

    assert payload['step1_model'] == {'predicted_class': 'UP'}
    assert payload['final_prediction'] in ('UP', 'UP_FINAL_BUY')
    assert 'feature_diagnostics' in payload['step2_model']
    diagnostics = json_exporter.load_feature_diagnostics_json('ABC.DE')
    assert set(diagnostics) == {'step1_pooled', 'step2_funds'}


def test_step2_without_fundamentals_keeps_step1_result(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    json_exporter.export_prediction_json('ABC.DE', {'stock_name': 'ABC.DE', 'final_prediction': 'UP'})
    monkeypatch.setattr(steps, 'fetch_fundamentals', lambda ticker: pd.DataFrame())
    assert steps.run_step2_for_ticker('ABC.DE')['final_prediction'] == 'UP'


def test_fundamentals_cache_age_comes_from_the_manifest(in_tmp: Path) -> None:
    from datetime import date
    from src.ingestion import funds_api

    assert funds_api.cache_age_days() is None
    funds_api.write_cache_manifest(date(2026, 9, 22), 406)
    assert funds_api.cache_age_days(today=date(2026, 10, 9)) == 17


def _patch_step1_network(monkeypatch: pytest.MonkeyPatch, panel: PricePanel) -> None:
    """Replaces the price download, the company profiles, the market snapshot and the jitter."""
    monkeypatch.setattr(steps, 'fetch_price_panel', lambda tickers, years, settings: panel)
    monkeypatch.setattr(steps, 'fetch_company_profile',
                        lambda ticker, name, model: {'full_name': name, 'description': f'{name} makes things.'})
    monkeypatch.setattr(steps, 'fetch_market_snapshot',
                        lambda ticker, price: {'pe_ratio': 12.0, 'beta': 1.1, 'liquidity': 2.5})
    monkeypatch.setattr(steps, 'apply_anti_jitter', lambda: None)


def test_pooled_step1_exports_every_ticker_with_prices_and_the_model_summary(
        in_tmp: Path, monkeypatch: pytest.MonkeyPatch, simulate_panel: PanelFactory, step1_settings: Settings) -> None:
    panel = simulate_panel(n_stocks=30, n_days=1500, drift_sd=0.0012, seed=7)
    panel.missing = ['GONE.DE']
    _patch_step1_network(monkeypatch, panel)
    universe = [{'Ticker': t, 'Company': f'{t} AG'} for t in list(panel.close.columns) + ['GONE.DE']]

    run = steps.run_step1_pooled(universe, step1_settings)

    assert run.exported == sorted(panel.close.columns)
    assert not (in_tmp / 'outputs' / 'predictions' / 'GONE.DE_prediction.json').exists()
    assert run.up_tickers == run.result.up_tickers
    summary = json.loads((in_tmp / 'outputs' / 'diagnostics' / 'step1_model_summary.json').read_text())
    assert summary == json.loads(json.dumps(run.summary))
    assert summary['model'] == 'pooled_cross_sectional_logit'
    assert (in_tmp / 'outputs' / 'diagnostics' / 'step1_validation_per_date.csv').exists()

    payload = json_exporter.load_prediction_json(run.exported[0])
    assert set(payload) == {'stock_name', 'company_name', 'company_description', 'latest_price', 'pe_ratio',
                            'beta', 'liquidity', 'prediction_date', 'applied_parameters', 'step1_model',
                            'final_prediction'}
    assert set(payload['applied_parameters']) == set(steps.APPLIED_PARAMETER_KEYS)
    assert payload['latest_price'] == pytest.approx(float(panel.close[run.exported[0]].iloc[-1]))
    for ticker in run.exported:
        payload = json_exporter.load_prediction_json(ticker)
        expected = 'UP' if ticker in run.up_tickers else 'NOT_UP'
        assert payload['final_prediction'] == payload['step1_model']['predicted_class'] == expected
        assert 'validation' not in payload['step1_model'] and 'coefficients' not in payload['step1_model']
    diagnostics = json_exporter.load_feature_diagnostics_json(run.exported[0])
    assert set(diagnostics) == {'step1_pooled'}
    assert set(diagnostics['step1_pooled']) == {'features', 'feature_values', 'feature_percentiles'}


def test_only_the_requested_ticker_is_exported_but_the_whole_universe_is_ranked(
        in_tmp: Path, monkeypatch: pytest.MonkeyPatch, simulate_panel: PanelFactory, step1_settings: Settings) -> None:
    panel = simulate_panel(n_stocks=30, n_days=1500, drift_sd=0.0012, seed=7)
    _patch_step1_network(monkeypatch, panel)
    universe = [{'Ticker': t, 'Company': t} for t in panel.close.columns]

    run = steps.run_step1_pooled(universe, step1_settings, only_tickers=['S005.DE'])

    assert run.exported == ['S005.DE']
    assert sorted(p.name for p in (in_tmp / 'outputs' / 'predictions').iterdir()) == ['S005.DE_prediction.json']
    cross = run.summary['cross_section']
    assert cross['n_universe'] == 30 and cross['n_ranked'] == int(run.result.results['rank'].notna().sum()) > 20
    assert run.up_tickers in ([], ['S005.DE'])


def test_step1_payload_maps_the_predicted_class_to_the_final_prediction() -> None:
    settings = steps.load_settings()
    entry = {'model': 'pooled_cross_sectional_logit', 'status': 'ok', 'predicted_class': 'UP'}
    payload = steps.build_step1_payload('ABC.DE', 'ABC AG', entry, 12.5, settings,
                                        profile={'full_name': 'ABC AG', 'description': 'Makes things.'},
                                        snapshot={'pe_ratio': None, 'beta': 0.9, 'liquidity': 1.0},
                                        prediction_date='2026-11-04')
    assert payload['final_prediction'] == 'UP' and payload['prediction_date'] == '2026-11-04'
    assert payload['step1_model'] is entry
    entry_down = {**entry, 'status': 'model_gate_failed', 'predicted_class': 'NOT_UP'}
    assert steps.build_step1_payload('ABC.DE', 'ABC AG', entry_down, None, settings, {}, {})['final_prediction'] == 'NOT_UP'


def test_step1_passed_file_round_trip_and_header_only_means_no_up_stock(in_tmp: Path) -> None:
    path = in_tmp / 'data' / 'state' / 'step1_passed.csv'
    with pytest.raises(FileNotFoundError):
        steps.load_step1_passed(path)

    steps.write_step1_passed(path, [])
    assert path.read_text().strip() == 'Ticker'
    assert steps.load_step1_passed(path) == []

    steps.write_step1_passed(path, ['BBB.DE', 'AAA.DE', 'BBB.DE'])
    assert steps.load_step1_passed(path) == ['AAA.DE', 'BBB.DE']

    path.write_text('')
    with pytest.raises(ValueError):
        steps.load_step1_passed(path)


def test_qualified_universe_defaults_the_company_name(in_tmp: Path) -> None:
    path = in_tmp / 'qualified_tickers.csv'
    with pytest.raises(FileNotFoundError, match='01_run_t7'):
        steps.load_qualified_universe(path)
    path.write_text('Ticker\nBBB.DE\nAAA.DE\n')
    assert steps.load_qualified_universe(path) == [{'Ticker': 'AAA.DE', 'Company': 'Unknown Company'},
                                                  {'Ticker': 'BBB.DE', 'Company': 'Unknown Company'}]


def test_single_ticker_mode_ranks_the_qualified_universe(in_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.orchestration import batch_runner

    with pytest.raises(FileNotFoundError):
        batch_runner.run_single('SAP.DE')

    state = in_tmp / 'data' / 'state'
    state.mkdir(parents=True)
    (state / 'qualified_tickers.csv').write_text('Ticker,Company\nAAA.DE,A AG\n')
    captured: Dict[str, Any] = {}

    def fake_run(universe: List[Dict[str, str]], settings: Settings,
                 only_tickers: Optional[List[str]] = None) -> None:
        captured['universe'], captured['only'] = universe, only_tickers
        raise RuntimeError('stop after capturing the arguments')

    monkeypatch.setattr(batch_runner, 'run_step1_pooled', fake_run)
    with pytest.raises(RuntimeError):
        batch_runner.run_single('sap.de')
    assert [row['Ticker'] for row in captured['universe']] == ['AAA.DE', 'SAP.DE']
    assert captured['only'] == ['SAP.DE']


def test_rank_log_line_describes_ranked_and_unranked_tickers() -> None:
    from src.orchestration.batch_runner import describe_step1_result

    ranked = {'rank': 12.0, 'probability': 0.4123, 'predicted_class': 'UP', 'status': 'ok'}
    assert describe_step1_result('SAP.DE', ranked, 371) == 'SAP.DE: rank 12 of 371, probability 0.41, UP (ok)'
    unranked = {'rank': float('nan'), 'probability': float('nan'), 'predicted_class': 'NOT_UP',
                'status': 'not_eligible: insufficient_history'}
    assert describe_step1_result('NEW.DE', unranked, 371) == \
        'NEW.DE: not ranked (not_eligible: insufficient_history), NOT_UP'
