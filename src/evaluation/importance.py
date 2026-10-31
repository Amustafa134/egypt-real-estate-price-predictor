"""
Permutation feature importance for the Egypt Real Estate Price Predictor.

Uses the best setup from the model comparison: Random Forest with the
compound feature. Each ORIGINAL input column (e.g. "city", not each one-hot
column) is shuffled on the test set, and we measure how much the test RMSE
(log scale) gets worse. A bigger increase means the model relies on that
column more.

Caveat: size, bedrooms and bathrooms are correlated. When one is shuffled,
the others still carry part of the same information, so permutation
importance can understate each of them individually.

Run from the repo root:
    python -m src.evaluation.importance
"""

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.engineer import (
    RANDOM_STATE,
    REPO_ROOT,
    build_features,
    load_clean_data,
    split_data,
)
from src.modeling.train import build_pipeline, get_models

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
IMPORTANCE_PNG = FIGURES_DIR / "05_feature_importance.png"

MODEL_NAME = "random_forest"
INCLUDE_COMPOUND = True
N_REPEATS = 10


def compute_importance(pipeline, X_test, y_test) -> pd.DataFrame:
    """Permutation importance on the test set, per original input column."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "rmse_increase_mean": result.importances_mean,
            "rmse_increase_std": result.importances_std,
        }
    )
    return table.sort_values("rmse_increase_mean", ascending=False).reset_index(
        drop=True
    )


def plot_importance(table: pd.DataFrame, path) -> None:
    """Horizontal bar chart, most important at the top."""
    ordered = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        ordered["feature"],
        ordered["rmse_increase_mean"],
        xerr=ordered["rmse_increase_std"],
        color="#4C72B0",
    )
    ax.set_xlabel("Increase in test RMSE (log price) when the column is shuffled")
    ax.set_title("What drives the price prediction (Random Forest, with compound)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_clean_data()
    X, y = build_features(df, include_compound=INCLUDE_COMPOUND)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"=== Fitting {MODEL_NAME} (with compound) ===", flush=True)
    pipeline = build_pipeline(
        get_models()[MODEL_NAME], include_compound=INCLUDE_COMPOUND
    )
    pipeline.fit(X_train, y_train)

    print(f"=== Permutation importance ({N_REPEATS} repeats on test set) ===",
          flush=True)
    table = compute_importance(pipeline, X_test, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(IMPORTANCE_CSV, index=False)
    plot_importance(table, IMPORTANCE_PNG)

    view = table.copy()
    view["rmse_increase"] = (
        view["rmse_increase_mean"].map("{:.3f}".format)
        + " +/- "
        + view["rmse_increase_std"].map("{:.3f}".format)
    )
    print(view[["feature", "rmse_increase"]].to_string(index=False))

    print(f"\nSaved {IMPORTANCE_CSV.relative_to(REPO_ROOT)}")
    print(f"Saved {IMPORTANCE_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
    