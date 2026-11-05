"""
Offline tests of the workflow script scripts/05_review_variables.py: argument parsing, the
GitHub Actions output file and two complete runs on a synthetic price cache, one that keeps
the feature set and one that switches it (settings update, pull request and commit texts).
"""
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from config.settings import load_settings
from src.ingestion.market_api import save_price_panel
from tests.conftest import PINNED_STEP1_SETTINGS, REVIEW_TEST_SETTINGS, ohlc_panel

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts' / '05_review_variables.py'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Imports the script as a module; the thread variables it sets are restored afterwards."""
    for name in THREAD_VARIABLES:
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location('review_variables_script', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arguments_and_their_defaults(script: ModuleType) -> None:
    defaults = script.parse_args([])
    assert defaults.n_jobs == (os.cpu_count() or 1) and defaults.candidates is None
    assert defaults.price_cache is None and defaults.review_date is None and defaults.pr_body is None
    assert defaults.output_dir == Path('data') / 'processed' / 'variable_reviews'
    args = script.parse_args(['--n-jobs', '2', '--candidates', 'ret_21, vol_21', '--price-cache', 'cache',
                              '--review-date', '2027-02-01', '--pr-body', 'pr.md', '--commit-body', 'commit.txt',
                              '--output-dir', 'reports'])
    assert args.n_jobs == 2 and args.candidates == ['ret_21', 'vol_21'] and args.price_cache == Path('cache')
    assert args.review_date == pd.Timestamp('2027-02-01') and args.pr_body == Path('pr.md')
    assert args.commit_body == Path('commit.txt') and args.output_dir == Path('reports')
    for invalid in (['--n-jobs', '0'], ['--review-date', 'tomorrow'], ['--candidates', ' , ']):
        with pytest.raises(SystemExit):
            script.parse_args(invalid)


def test_outputs_are_appended_to_the_github_output_file(script: ModuleType, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / 'github_output'
    output.write_text('earlier=1\n')
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    assert script.write_github_output({'decision': 'switch', 'procedure': 'lasso'})
    assert output.read_text() == 'earlier=1\ndecision=switch\nprocedure=lasso\n'
    with pytest.raises(ValueError):
        script.write_github_output({'decision': 'keep\nprocedure=x'})
    monkeypatch.delenv('GITHUB_OUTPUT')
    assert script.write_github_output({'decision': 'keep'}) is False


def _prepare_run(script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 overrides: Dict[str, Any]) -> Dict[str, Path]:
    """A working directory with the universe, a price cache, a settings copy and an output file."""
    prices = ohlc_panel(n_stocks=40, n_days=1800, seed=5)
    work = tmp_path / 'work'
    (work / 'data' / 'state').mkdir(parents=True)
    pd.DataFrame({'Ticker': list(prices.close.columns)}).to_csv(work / 'data' / 'state' / 'qualified_tickers.csv',
                                                               index=False)
    save_price_panel(prices, tmp_path / 'cache')
    settings_path = tmp_path / 'settings.yaml'
    settings_path.write_text((REPO / 'config' / 'settings.yaml').read_text(encoding='utf-8'), encoding='utf-8')
    settings = {**load_settings(), **PINNED_STEP1_SETTINGS, **REVIEW_TEST_SETTINGS, **overrides}
    monkeypatch.setattr(script, 'load_settings', lambda: settings)
    monkeypatch.setattr(script, 'SETTINGS_PATH', settings_path)
    monkeypatch.setenv('GITHUB_OUTPUT', str(tmp_path / 'github_output'))
    monkeypatch.chdir(work)
    return {'work': work, 'cache': tmp_path / 'cache', 'settings': settings_path, 'output': tmp_path / 'github_output',
            'pr': tmp_path / 'pr.md', 'commit': tmp_path / 'commit.txt'}


def _run(script: ModuleType, paths: Dict[str, Path], review_date: str = '2026-11-05') -> Dict[str, Any]:
    """Runs main() on the prepared directory and returns the JSON report."""
    code = script.main(['--price-cache', str(paths['cache']), '--candidates', 'ret_21,ret_63,ret_252,mom_12_1,vol_21',
                        '--n-jobs', '1', '--review-date', review_date, '--pr-body', str(paths['pr']),
                        '--commit-body', str(paths['commit'])])
    assert code == 0
    folder = paths['work'] / 'data' / 'processed' / 'variable_reviews'
    assert (folder / f'review_{review_date}.md').read_text().startswith(f'# Variable Review {review_date}')
    return json.loads((folder / f'review_{review_date}.json').read_text())


def test_a_review_that_keeps_the_features_leaves_the_settings_unchanged(
        script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _prepare_run(script, tmp_path, monkeypatch, {})
    before = paths['settings'].read_text()
    report = _run(script, paths)

    assert report['decision'] == 'keep'
    assert paths['settings'].read_text() == before
    assert not paths['pr'].exists() and not paths['commit'].exists()
    assert paths['output'].read_text() == 'decision=keep\nreview_date=2026-11-05\n'
    assert report['runtime_s']['download_s'] >= 0 and 'total_s' in report['runtime_s']
    assert report['correction']['m_cum'] == 4 and report['correction']['earlier_reports'] == []


def test_a_switch_updates_the_settings_and_writes_the_pull_request_texts(
        script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The simulated switch path of the spec (no AUC hurdle, no stability condition, family
    # alpha 1) with a weak champion
    paths = _prepare_run(script, tmp_path, monkeypatch, {'review_auc_hurdle': -1.0, 'review_min_winning_years': 0,
                                                         'review_family_alpha': 1.0, 'step1_features': ['ret_21']})
    # An earlier review after the adoption date counts toward m_cum
    folder = paths['work'] / 'data' / 'processed' / 'variable_reviews'
    folder.mkdir(parents=True)
    (folder / 'review_2026-11-10.json').write_text(json.dumps({'review_date': '2026-11-10',
                                                               'challengers': ['stepwise', 'lasso']}))
    before_text = paths['settings'].read_text()
    before = yaml.safe_load(before_text)
    report = _run(script, paths, review_date='2026-11-20')

    assert report['correction']['m_cum'] == 6 and report['correction']['earlier_reports'] == ['review_2026-11-10.json']
    assert report['correction']['ci_level'] == pytest.approx(1 - 1.0 / 6)

    assert report['decision'] == 'switch' and report['winner'] in report['challengers']
    after_text = paths['settings'].read_text()
    after = yaml.safe_load(after_text)
    assert before['step1_features_adopted'] != '2026-11-20'
    assert after['step1_features'] == report['adopted_features']
    assert after['step1_features_adopted'] == '2026-11-20'
    assert {k: v for k, v in after.items() if not k.startswith('step1_features')} == \
           {k: v for k, v in before.items() if not k.startswith('step1_features')}
    changed = [new for old, new in zip(before_text.splitlines(), after_text.splitlines()) if old != new]
    assert [line.split(':')[0] for line in changed] == ['step1_features', 'step1_features_adopted']
    assert paths['output'].read_text() == (f"decision=switch\nreview_date=2026-11-20\n"
                                           f"procedure={report['winner']}\n")
    body = paths['pr'].read_text()
    assert 'GITHUB_TOKEN' in body and report['winner'] in body
    assert 'data/processed/variable_reviews/review_2026-11-20.md' in body
    commit = paths['commit'].read_text()
    assert commit.startswith(f"The variable review of 2026-11-20 qualifies {report['winner']}")
    assert max(len(line) for line in commit.splitlines()) <= 72  # wrapped for git log
