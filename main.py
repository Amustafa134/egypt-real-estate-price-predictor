"""
Run the full Egypt Real Estate Price Predictor pipeline with one command.

Steps:
    1. Clean the raw listings        (src/cleaning/clean.py)
    2. Exploratory analysis + plots  (src/analysis/eda.py)
    3. Train and compare models      (src/modeling/train.py)
    4. Permutation feature importance (src/evaluation/importance.py)

Run from the repo root:
    python main.py
    python main.py --skip-eda     # skip the plots step
"""

import argparse
import os
import runpy
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "egypt_real_estate_listings.csv"

STEPS = [
    ("Cleaning", "src.cleaning.clean"),
    ("EDA", "src.analysis.eda"),
    ("Model comparison", "src.modeling.train"),
    ("Feature importance", "src.evaluation.importance"),
]


def run_step(title: str, module: str) -> None:
    """Run a module exactly as 'python -m <module>' would."""
    print("\n" + "=" * 70)
    print(f"  {title}  ({module})")
    print("=" * 70, flush=True)
    start = time.perf_counter()
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    print(f"\n  [{title} finished in {time.perf_counter() - start:.1f}s]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument(
        "--skip-eda", action="store_true", help="skip the EDA plots step"
    )
    args = parser.parse_args()

    # Scripts use paths relative to the repo root, so always run from there.
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if not RAW_PATH.exists():
        sys.exit(
            f"Raw data not found at {RAW_PATH.relative_to(REPO_ROOT)}.\n"
            "Download it from Kaggle ('Egyptian Real Estate Listings' by "
            "hassankhaled21) and place it there, or run "
            "'python -m src.data.download'."
        )

    total_start = time.perf_counter()
    for title, module in STEPS:
        if args.skip_eda and module == "src.analysis.eda":
            print(f"\n(skipping {title})")
            continue
        run_step(title, module)

    print(f"\nPipeline finished in {time.perf_counter() - total_start:.1f}s.")
    print("Results: reports/model_comparison.csv, reports/feature_importance.csv,")
    print("         reports/figures/")


if __name__ == "__main__":
    main()
