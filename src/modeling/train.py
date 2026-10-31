"""
Train and compare models for the Egypt Real Estate Price Predictor.

Models:
    - median_baseline: always predicts the training median (the floor to beat)
    - ridge:           linear model
    - random_forest:   bagged trees
    - hist_gradient_boosting: scikit-learn's gradient boosting

Feature sets:
    - city_level:    governorate + city
    - with_compound: adds the compound / area name

Each model is wrapped in a Pipeline with the shared preprocessor, so all
imputation and encoding is learned from training folds only.

Run from the repo root:
    python -m src.modeling.train
"""

import time

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import regression_metrics
from src.features.engineer import (
    RANDOM_STATE,
    REPO_ROOT,
    build_features,
    build_preprocessor,
    load_clean_data,
    split_data,
)

REPORTS_DIR = REPO_ROOT / "reports"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
CV_FOLDS = 5

# feature set name -> include_compound
FEATURE_SETS = {
    "city_level": False,
    "with_compound": True,
}


def get_models() -> dict:
    """Model name -> unfitted estimator."""
    return {
        "median_baseline": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model, include_compound: bool = False) -> Pipeline:
    """Shared preprocessing + the given model."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(include_compound=include_compound)),
            ("model", model),
        ]
    )


def cross_validate_rmse(pipeline, X_train, y_train) -> tuple[float, float]:
    """5-fold CV RMSE on the log scale, training data only. Returns (mean, std)."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    rmse = -scores
    return float(rmse.mean()), float(rmse.std())


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """For each feature set and model: cross-validate, fit on train, score on test."""
    rows = []
    for feature_set, include_compound in FEATURE_SETS.items():
        X, y = build_features(df, include_compound=include_compound)
        X_train, X_test, y_train, y_test = split_data(X, y)

        for name, model in get_models().items():
            # The baseline ignores features, so run it only once.
            if name == "median_baseline" and include_compound:
                continue

            print(f"  [{feature_set}] training {name} ...", flush=True)
            start = time.perf_counter()

            pipeline = build_pipeline(model, include_compound=include_compound)
            cv_mean, cv_std = cross_validate_rmse(pipeline, X_train, y_train)

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)

            seconds = time.perf_counter() - start
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": name,
                    "cv_rmse_log_mean": cv_mean,
                    "cv_rmse_log_std": cv_std,
                    **{f"test_{k}": v for k, v in metrics.items()},
                    "seconds": seconds,
                }
            )
            print(f"    done in {seconds:.1f}s", flush=True)

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    """Readable table: log-scale scores plus business-friendly EGP numbers."""
    view = pd.DataFrame(
        {
            "features": results["feature_set"],
            "model": results["model"],
            "cv_rmse_log": results["cv_rmse_log_mean"].map("{:.3f}".format)
            + " +/- "
            + results["cv_rmse_log_std"].map("{:.3f}".format),
            "test_rmse_log": results["test_rmse_log"].map("{:.3f}".format),
            "test_r2_log": results["test_r2_log"].map("{:.3f}".format),
            "test_mae_M_egp": (results["test_mae_egp"] / 1e6).map("{:.2f}".format),
            "median_pct_err": (results["test_median_abs_pct_error"] * 100).map(
                "{:.1f}%".format
            ),
            "within_20pct": (results["test_within_20pct"] * 100).map(
                "{:.1f}%".format
            ),
            "seconds": results["seconds"].map("{:.0f}".format),
        }
    )
    print(view.to_string(index=False))


def main() -> None:
    df = load_clean_data()

    print(f"=== Training models on {len(FEATURE_SETS)} feature sets "
          f"({CV_FOLDS}-fold CV + test) ===")
    results = compare_models(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(COMPARISON_PATH, index=False)

    print("\n=== Model comparison ===")
    print_summary(results)
    print(f"\nSaved full results to {COMPARISON_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
    