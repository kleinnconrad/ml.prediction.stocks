"""
GitHub Actions entry point of Step 1 (job `step1`, one runner).

Ranks all qualified tickers with the pooled cross-sectional model, exports one prediction
payload per ticker with price data to outputs/predictions/, writes the model summary to
outputs/diagnostics/step1_model_summary.json and the UP tickers to
data/state/step1_passed.csv (header only if none is UP) for Step 2.
"""
import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from config.settings import load_settings
from src.orchestration.logging_setup import configure_logging
from src.orchestration.steps import (QUALIFIED_TICKERS_PATH, STEP1_PASSED_PATH, load_qualified_universe,
                                     run_step1_pooled, write_step1_passed)

logger = logging.getLogger(__name__)


def main() -> None:
    """Runs the pooled Step 1 for the qualified universe and saves the state for Step 2."""
    configure_logging()
    universe = load_qualified_universe(QUALIFIED_TICKERS_PATH)
    logger.info(f"Step 1: ranking {len(universe)} qualified tickers with the pooled model.")

    step1 = run_step1_pooled(universe, load_settings())

    write_step1_passed(STEP1_PASSED_PATH, step1.up_tickers)
    gate = step1.summary['gate']
    logger.info(f"Saved {len(step1.up_tickers)} UP tickers to {STEP1_PASSED_PATH} "
                f"(model gate {'passed' if gate['passed'] else 'failed: ' + str(gate['reason'])}).")


if __name__ == "__main__":
    main()
