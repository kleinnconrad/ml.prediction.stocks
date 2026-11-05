"""
GitHub Actions entry point of Step 2 (job `step2-matrix`, one shard per runner).

Reads the Step 1 UP tickers from data/state/step1_passed.csv (written by
scripts/02_run_step1.py; header only if no stock is UP), takes this runner's shard and
extends each ticker's prediction payload with the result of the fundamental ruleset.
"""
import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import argparse
from tqdm import tqdm
from src.orchestration.logging_setup import configure_logging
from src.orchestration.steps import STEP1_PASSED_PATH, apply_anti_jitter, load_step1_passed, run_step2_for_ticker

logger = logging.getLogger(__name__)


def main() -> None:
    """Runs Step 2 for this runner's shard of the Step 1 UP tickers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Matrix shard index")
    parser.add_argument("--total", type=int, default=1, help="Total number of runners")
    args = parser.parse_args()
    configure_logging()

    os.makedirs('data/processed', exist_ok=True)

    # 1. UP tickers of the Step 1 job (de-duplicated and sorted, so the sharding is deterministic);
    # a header-only file means that no stock is UP
    all_passed_tickers = load_step1_passed(STEP1_PASSED_PATH)

    # 2. Shard for Distributed Execution
    shard_tickers = all_passed_tickers[args.shard :: args.total]
    logger.info(f"[Runner {args.shard}/{args.total}] Processing {len(shard_tickers)} tickers for Step 2.")

    final_buy_candidates = []

    for ticker in tqdm(shard_tickers, desc=f"Step 2 - Shard {args.shard}"):
        apply_anti_jitter()
        try:
            payload = run_step2_for_ticker(ticker)
            if payload['final_prediction'] == 'UP_FINAL_BUY':
                final_buy_candidates.append(ticker)
        except Exception:
            logger.exception(f"Failed {ticker} in Step 2")

    # The final buy list is built from the consolidated payloads by scripts/04_consolidate.py
    logger.info(f"[Runner {args.shard}] Found {len(final_buy_candidates)} Final Buy Candidates.")

if __name__ == "__main__":
    main()
