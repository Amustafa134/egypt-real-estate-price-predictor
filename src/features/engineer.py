"""
Feature engineering for the Egypt Real Estate Price Predictor.

Builds the model inputs (X) and target (y = natural log of price) from the
cleaned listings, and defines a scikit-learn preprocessor that is fitted on
the training data only.

Two feature sets are supported:
    - city level (include_compound=False): governorate + city
    - with compound (include_compound=True): adds the compound / area name,
      taken from the first part of the location string

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

COMPOUND_FEATURE = "compound"

TARGET = "log_price"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Categories seen fewer than this many times in TRAINING data are grouped
# into one "infrequent" column. Unseen categories at test time go there too.
MIN_CATEGORY_FREQUENCY = 100

# Compounds are much more numerous, so a lower threshold keeps more of them.
MIN_COMPOUND_FREQUENCY = 20


def load_clean_data(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned listings produced by src/cleaning/clean.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python src\\cleaning\\clean.py' first."
        )
    return pd.read_csv(path)


def extract_compound(location) -> str | None:
    """
    Take the compound / area name from a location string.

    Locations run from most specific to least specific, for example:
        "Taj City, 5th Settlement Compounds, The 5th Settlement, New Cairo City, Cairo"
        "Azha North, Ras Al Hekma, North Coast"
    The last part is the governorate and the one before it is the city.
    With 3 or more parts, the first part is the compound or area.
    With fewer parts there is nothing below city, so return None.
    """
    if not isinstance(location, str):
        return None
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if len(parts) >= 3:
        return parts[0]
    return None


def get_categorical_features(include_compound: bool) -> list[str]:
    """Categorical columns for the chosen feature set."""
    if include_compound:
        return CATEGORICAL_FEATURES + [COMPOUND_FEATURE]
    return list(CATEGORICAL_FEATURES)


def build_features(
    df: pd.DataFrame, include_compound: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
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

    if include_compound:
        data[COMPOUND_FEATURE] = data["location"].apply(extract_compound)

    X = data[NUMERIC_FEATURES + get_categorical_features(include_compound)]
    y = data[TARGET]
    return X, y


def _categorical_pipeline(min_frequency: int) -> Pipeline:
    """Fill missing with "Missing", then one-hot encode with rare grouping."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    min_frequency=min_frequency,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_preprocessor(include_compound: bool = False) -> ColumnTransformer:
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

    transformers = [
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", _categorical_pipeline(MIN_CATEGORY_FREQUENCY), CATEGORICAL_FEATURES),
    ]
    if include_compound:
        transformers.append(
            (
                "compound",
                _categorical_pipeline(MIN_COMPOUND_FREQUENCY),
                [COMPOUND_FEATURE],
            )
        )

    return ColumnTransformer(transformers=transformers)


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    80/20 random split with a fixed seed.

    Known limitation: listings from the same compound can land on both
    sides of the split. This measures accuracy for compounds the model has
    already seen, not for brand-new compounds.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def main() -> None:
    df = load_clean_data()

    for include_compound in (False, True):
        label = "with compound" if include_compound else "city level"
        X, y = build_features(df, include_compound=include_compound)
        X_train, X_test, y_train, y_test = split_data(X, y)

        preprocessor = build_preprocessor(include_compound=include_compound)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        print(f"=== Feature pipeline check: {label} ===")
        print(f"  train rows:  {len(X_train)}")
        print(f"  test rows:   {len(X_test)}")
        print(f"  input columns:  {X.shape[1]}")
        print(f"  output columns: {X_train_t.shape[1]}")
        print(f"  test matrix shape: {X_test_t.shape}")
        print(f"  missing values after transform: {int(np.isnan(X_train_t).sum())}")

        if include_compound:
            compounds = X_train[COMPOUND_FEATURE]
            counts = compounds.value_counts()
            kept = counts[counts >= MIN_COMPOUND_FREQUENCY]
            coverage = kept.sum() / len(compounds)
            print(f"  compound missing (train): {int(compounds.isna().sum())}")
            print(f"  unique compounds (train): {compounds.nunique()}")
            print(
                f"  compounds kept (>= {MIN_COMPOUND_FREQUENCY} listings): "
                f"{len(kept)}, covering {coverage:.1%} of train rows"
            )
            print("  top 10 compounds:")
            print(counts.head(10).to_string())
        print()


if __name__ == "__main__":
    main()
    