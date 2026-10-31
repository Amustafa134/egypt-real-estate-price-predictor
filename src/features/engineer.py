"""
Feature engineering for the Egypt Real Estate Price Predictor.

Builds the model inputs (X) and target (y = natural log of price) from the
cleaned listings, and defines a scikit-learn preprocessor that is fitted on
the training data only.

Run from the repo root:
    python -m src.features.engineer
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = REPO_ROOT / "data" / "processed" / "listings_clean.csv"

NUMERIC_FEATURES = [
    "log_size",
    "bedrooms_count",
    "bathrooms",
    "has_maid_room",
    "has_installment_option",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "type",
    "governorate",
    "city",
]

TARGET = "log_price"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Categories seen fewer than this many times in TRAINING data are grouped
# into one "infrequent" column. Unseen categories at test time go there too.
MIN_CATEGORY_FREQUENCY = 100


def load_clean_data(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned listings produced by src/cleaning/clean.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python src\\cleaning\\clean.py' first."
        )
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y).

    y is log(price). Models predict on the log scale; errors are converted
    back to EGP during evaluation.

    price_per_sqm is deliberately NOT a feature: it is computed from price,
    so using it would leak the target into the inputs.
    """
    data = df.copy()

    data["log_size"] = np.log(data["size_sqm"])
    data[TARGET] = np.log(data["price"])

    # Boolean flags -> 0/1 numbers
    data["has_maid_room"] = data["has_maid_room"].astype(float)
    data["has_installment_option"] = data["has_installment_option"].astype(float)

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric: fill missing with the training median, then scale
             (scaling matters for the linear model, harmless for trees).
    Categorical: fill missing with "Missing", then one-hot encode,
             grouping rare categories using training data only.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    80/20 random split with a fixed seed.

    Known limitation: listings from the same compound can land on both
    sides of the split, which may make the test score slightly optimistic.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def main() -> None:
    df = load_clean_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print("=== Feature pipeline check ===")
    print(f"  rows total:  {len(X)}")
    print(f"  train rows:  {len(X_train)}")
    print(f"  test rows:   {len(X_test)}")
    print(f"  input columns:  {X.shape[1]}")
    print(f"  output columns: {X_train_t.shape[1]}")
    print(f"  test matrix shape: {X_test_t.shape}")
    print(f"  missing values after transform: {int(np.isnan(X_train_t).sum())}")

    print("\n=== Missing values before transform (train) ===")
    print(X_train.isna().sum().to_string())

    print("\n=== Output feature names ===")
    for name in preprocessor.get_feature_names_out():
        print(f"  {name}")


if __name__ == "__main__":
    main()
    