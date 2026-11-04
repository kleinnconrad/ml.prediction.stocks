import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import argparse
from config.settings import load_settings
from src.orchestration.logging_setup import configure_logging
from src.orchestration.report import consolidate_predictions, find_prediction_files, load_model_summary, publish_report

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Consolidates the shard outputs into the published batch report.")
    parser.add_argument("--predictions-glob", default="step2_artifacts/**/predictions/*.json",
                        help="Glob of the prediction payloads to consolidate (** matches any directories)")
    parser.add_argument("--model-summary-glob", default="step2_artifacts/**/diagnostics/step1_model_summary.json",
                        help="Glob of the Step 1 model summary; the first match in sorted order is published "
                             "as the report key step1_model")
    parser.add_argument("--out-dir", default=os.path.join("data", "processed"),
                        help="Directory for full_batch_report.json, final_buy_signals.csv and history/")
    args = parser.parse_args()
    configure_logging()

    prediction_files = find_prediction_files(args.predictions_glob)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction payloads match {args.predictions_glob}")

    predictions = consolidate_predictions(prediction_files)
    publish_report(predictions, load_settings(), args.out_dir,
                   step1_model_summary=load_model_summary(args.model_summary_glob))

if __name__ == "__main__":
    main()
