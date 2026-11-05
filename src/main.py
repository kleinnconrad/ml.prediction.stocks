"""
Local CLI of the Xetra Two-Step Stock Prediction Engine.

Without arguments it runs the full local batch (T7 download, Step 1 for the qualified
universe, Step 2, report and uplift evaluation). With --ticker it ranks the qualified
universe with the pooled Step 1 model and exports only that ticker's payload.
"""
import logging
import sys
import os
import argparse
import tomllib
from pathlib import Path

# Add project root to sys.path to allow execution as a script from any directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from src.orchestration.batch_runner import run_batch, run_single
from src.orchestration.logging_setup import configure_logging

load_dotenv()

def read_version() -> str:
    """Returns the project version maintained by release-please in pyproject.toml."""
    with open(Path(__file__).resolve().parents[1] / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)['project']['version']

def main() -> None:
    """Parses the arguments and runs the batch or the single-ticker mode; exits with 1 on an error."""
    parser = argparse.ArgumentParser(description="Xetra Two-Step Stock Prediction Engine")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Rank the qualified universe with the pooled Step 1 model and export only this "
                             "ticker's payload (e.g., SAP.DE); Step 2 runs if it is UP")
    args = parser.parse_args()

    configure_logging(log_file=os.path.join('logs', 'xetra_predictor.log'))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Xetra Two-Step Stock Prediction Engine (v{read_version()})")
    
    try:
        if args.ticker:
            logger.info(f"Single Ticker Mode activated for: {args.ticker}")
            run_single(args.ticker)
        else:
            logger.info("Batch Mode activated.")
            run_batch()
            
        logger.info("Engine run completed successfully.")
    except Exception as e:
        logger.error(f"Engine run failed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
