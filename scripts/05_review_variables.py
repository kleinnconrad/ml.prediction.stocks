"""
GitHub Actions entry point of the quarterly variable review (workflow
`5. Quarterly Variable Review`, .github/workflows/05_review_variables.yml).

Downloads the daily prices of the qualified universe and the market index, runs the
champion-challenger review of src/modeling/variable_review.py and writes
data/processed/variable_reviews/review_YYYY-MM-DD.json and .md. The earlier reports in that
directory set the multiple-testing correction (m_cum) of the switch rule. If the rule holds,
it updates `step1_features` and `step1_features_adopted` in config/settings.yaml and
writes the pull request body (--pr-body) and the commit message body (--commit-body). The
decision, the review date and, for a switch, the winning procedure are appended to
$GITHUB_OUTPUT when that variable is set.

Usage (from the repository root):
    uv run --locked python scripts/05_review_variables.py [--n-jobs N] [--candidates a,b,...]
        [--price-cache DIR] [--review-date YYYY-MM-DD] [--pr-body FILE] [--commit-body FILE]
        [--output-dir DIR]
"""
import os
import sys

# One BLAS thread per process: the review runs its own worker processes (--n-jobs)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config.settings import SETTINGS_PATH, load_settings
from src.ingestion.market_api import PricePanel, fetch_price_panel, load_price_panel, save_price_panel
from src.modeling import variable_review as review
from src.orchestration.json_exporter import write_json_atomic
from src.orchestration.logging_setup import configure_logging
from src.orchestration.steps import QUALIFIED_TICKERS_PATH, history_window, load_qualified_universe

logger = logging.getLogger(__name__)

REVIEW_DIR = Path('data') / 'processed' / 'variable_reviews'


def _feature_list(value: str) -> List[str]:
    """Parses a comma-separated list of feature names."""
    names = [name.strip() for name in value.split(',') if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected a comma-separated list of feature names")
    return names


def _review_date(value: str) -> pd.Timestamp:
    """Parses an ISO date."""
    try:
        return pd.Timestamp(value).normalize()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a date: {value!r}") from e


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parses the command line.

    Args:
        argv (Optional[Sequence[str]]): Arguments (default: sys.argv[1:]).

    Returns:
        argparse.Namespace: n_jobs, candidates, price_cache, review_date, pr_body,
        commit_body and output_dir.
    """
    parser = argparse.ArgumentParser(description="Quarterly champion-challenger review of the Step 1 features.")
    parser.add_argument('--n-jobs', type=int, default=os.cpu_count() or 1,
                        help="worker processes (default: the number of CPUs)")
    parser.add_argument('--candidates', type=_feature_list, default=None,
                        help="comma-separated feature names that replace the candidate pool (local testing)")
    parser.add_argument('--price-cache', type=Path, default=None,
                        help="directory of a CSV price cache: read if it exists, otherwise downloaded and written")
    parser.add_argument('--review-date', type=_review_date, default=None,
                        help="review date YYYY-MM-DD (default: today in Europe/Berlin); later prices are not used")
    parser.add_argument('--pr-body', type=Path, default=None,
                        help="file for the pull request body, written if the decision is switch")
    parser.add_argument('--commit-body', type=Path, default=None,
                        help="file for the commit message body, written if the decision is switch")
    parser.add_argument('--output-dir', type=Path, default=REVIEW_DIR,
                        help=f"directory of the review reports; its earlier reports set m_cum "
                             f"(default: {REVIEW_DIR.as_posix()})")
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1")
    return args


def load_prices(tickers: Sequence[str], settings: Dict[str, Any], cache: Optional[Path]) -> PricePanel:
    """
    Reads the prices from the cache, or downloads them (and fills the cache if one is given).

    Args:
        tickers (Sequence[str]): The qualified tickers.
        settings (Dict[str, Any]): Loaded settings.
        cache (Optional[Path]): Price cache directory.

    Returns:
        PricePanel: Prices, volumes and the market index.
    """
    if cache is not None and (cache / 'close.csv').exists():
        panel = load_price_panel(cache)
        unknown = sorted(set(tickers) - set(panel.close.columns) - set(panel.missing))
        if unknown:
            logger.warning(f"The price cache {cache} has no entry for {len(unknown)} tickers of the universe "
                           f"(delete it to download again): {unknown}")
        return panel
    _, fetch_years = history_window(settings)
    panel = fetch_price_panel(tickers, fetch_years, settings)
    if cache is not None:
        save_price_panel(panel, cache)
    return panel


def write_github_output(values: Dict[str, str], path: Optional[str] = None) -> bool:
    """
    Appends key=value lines to the GitHub Actions output file.

    Args:
        values (Dict[str, str]): Single-line output values.
        path (Optional[str]): Output file (default: the GITHUB_OUTPUT environment variable).

    Returns:
        bool: True if the values were written, False if no output file is set.

    Raises:
        ValueError: If a value contains a line break.
    """
    target = path or os.environ.get('GITHUB_OUTPUT')
    if not target:
        return False
    lines = []
    for key, value in values.items():
        if '\n' in str(value) or '\r' in str(value):
            raise ValueError(f"The output {key} must be a single line.")
        lines.append(f"{key}={value}\n")
    with open(target, 'a', encoding='utf-8') as handle:
        handle.writelines(lines)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Runs the review, writes the reports and, on a switch, the settings update and the pull request texts.

    Args:
        argv (Optional[Sequence[str]]): Command line arguments.

    Returns:
        int: Exit code 0.
    """
    configure_logging()
    args = parse_args(argv)
    started = time.monotonic()
    settings = load_settings()
    review_date = args.review_date or pd.Timestamp(pd.Timestamp.now(tz='Europe/Berlin').date())
    stamp = review_date.date().isoformat()
    universe = load_qualified_universe(QUALIFIED_TICKERS_PATH)
    logger.info(f"Variable review {stamp}: {len(universe)} qualified tickers, {args.n_jobs} worker processes.")

    download_started = time.monotonic()
    prices = load_prices([row['Ticker'] for row in universe], settings, args.price_cache)
    timings: Dict[str, Any] = {'download_s': round(time.monotonic() - download_started, 1)}
    result = review.run_variable_review(prices, settings, review_date, args.candidates, args.n_jobs,
                                        report_dir=args.output_dir)
    timings.update(result.timings)
    timings['total_s'] = round(time.monotonic() - started, 1)
    report = review.build_report(result, settings, timings, review.peak_memory_mb())

    json_path = args.output_dir / f"review_{stamp}.json"
    markdown_path = args.output_dir / f"review_{stamp}.md"
    write_json_atomic(str(json_path), report)
    markdown_path.write_text(review.render_markdown(report), encoding='utf-8')
    logger.info(f"Wrote {json_path} and {markdown_path}.")

    outputs = {'decision': report['decision'], 'review_date': stamp}
    if report['decision'] == review.DECISION_SWITCH:
        settings_text = SETTINGS_PATH.read_text(encoding='utf-8')
        SETTINGS_PATH.write_text(review.update_settings_text(settings_text, report['adopted_features'], stamp),
                                 encoding='utf-8')
        logger.info(f"Switch to {report['winner']}: updated step1_features and step1_features_adopted "
                    f"in {SETTINGS_PATH}.")
        if args.pr_body:
            args.pr_body.write_text(review.render_pr_body(report, markdown_path.as_posix()), encoding='utf-8')
        if args.commit_body:
            args.commit_body.write_text(review.render_commit_body(report, markdown_path.as_posix()), encoding='utf-8')
        outputs['procedure'] = report['winner']
    if write_github_output(outputs):
        logger.info(f"GitHub Actions outputs: {outputs}")
    logger.info(f"Variable review finished in {timings['total_s']} s: decision {report['decision']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
