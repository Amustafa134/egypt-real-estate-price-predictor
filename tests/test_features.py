"""Tests for src/features/engineer.py. Synthetic data only, no file I/O."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    COMPOUND_FEATURE,
    build_features,
    build_preprocessor,
    extract_compound,
    split_data,
)


def _clean_listings(n: int = 200) -> pd.DataFrame:
    """
    Synthetic cleaned listings. "New Cairo City" is frequent (well above the
    100-listing threshold); "Rare City" appears only 5 times.
    """
    rng = np.random.default_rng(0)
    cities = ["New Cairo City"] * (n - 5) + ["Rare City"] * 5
    bathrooms = rng.integers(1, 5, size=n).astype(float)
    bathrooms[:3] = np.nan  # some missing values to impute
    return pd.DataFrame(
        {
            "price": rng.uniform(1e6, 2e7, size=n),
            "size_sqm": rng.uniform(50, 400, size=n),
            "bedrooms_count": rng.integers(0, 6, size=n).astype(float),
            "has_maid_room": rng.integers(0, 2, size=n).astype(bool),
            "bathrooms": bathrooms,
            "has_installment_option": rng.integers(0, 2, size=n).astype(bool),
            "payment_method": ["Cash"] * n,
            "type": ["Apartment"] * n,
            "governorate": ["Cairo"] * n,
            "city": cities,
            "location": ["Taj City, The 5th Settlement, New Cairo City, Cairo"] * n,
        }
    )


# ---------- extract_compound ----------

def test_extract_compound_takes_first_part():
    loc = "Taj City, 5th Settlement Compounds, The 5th Settlement, New Cairo City, Cairo"
    assert extract_compound(loc) == "Taj City"


def test_extract_compound_three_parts():
    assert extract_compound("Azha North, Ras Al Hekma, North Coast") == "Azha North"


def test_extract_compound_two_parts_returns_none():
    assert extract_compound("Hurghada, Red Sea") is None


def test_extract_compound_missing_returns_none():
    assert extract_compound(np.nan) is None


# ---------- build_features ----------

def test_target_is_log_price():
    df = _clean_listings()
    _, y = build_features(df)
    np.testing.assert_allclose(y.to_numpy(), np.log(df["price"].to_numpy()))


def test_no_target_leakage_in_features():
    X, _ = build_features(_clean_listings(), include_compound=True)
    assert "price" not in X.columns
    assert "log_price" not in X.columns
    assert "price_per_sqm" not in X.columns


def test_compound_only_when_requested():
    X_city, _ = build_features(_clean_listings(), include_compound=False)
    X_comp, _ = build_features(_clean_listings(), include_compound=True)
    assert COMPOUND_FEATURE not in X_city.columns
    assert COMPOUND_FEATURE in X_comp.columns


def test_log_size_matches_size():
    df = _clean_listings()
    X, _ = build_features(df)
    np.testing.assert_allclose(
        X["log_size"].to_numpy(), np.log(df["size_sqm"].to_numpy())
    )


# ---------- build_preprocessor ----------

def test_preprocessor_fills_missing_values():
    X, _ = build_features(_clean_listings())
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_rare_city_is_grouped_as_infrequent():
    X, _ = build_features(_clean_listings())
    pre = build_preprocessor().fit(X)
    names = list(pre.get_feature_names_out())
    assert "cat__city_New Cairo City" in names
    assert "cat__city_Rare City" not in names
    assert "cat__city_infrequent_sklearn" in names


def test_unseen_city_at_test_time_does_not_crash():
    X, _ = build_features(_clean_listings())
    pre = build_preprocessor().fit(X)

    new_row = X.iloc[[0]].copy()
    new_row["city"] = "Brand New City"
    out = pre.transform(new_row)

    names = list(pre.get_feature_names_out())
    infrequent_col = names.index("cat__city_infrequent_sklearn")
    assert out[0, infrequent_col] == 1.0


def test_preprocessor_with_compound_runs():
    X, _ = build_features(_clean_listings(), include_compound=True)
    out = build_preprocessor(include_compound=True).fit_transform(X)
    assert out.shape[0] == len(X)
    assert not np.isnan(out).any()


# ---------- split_data ----------

def test_split_sizes():
    X, y = build_features(_clean_listings(200))
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(y_train) == 160
    assert len(y_test) == 40


def test_split_is_reproducible():
    X, y = build_features(_clean_listings())
    first = split_data(X, y)[1].index.tolist()
    second = split_data(X, y)[1].index.tolist()
    assert first == second


def test_split_has_no_overlap():
    X, y = build_features(_clean_listings())
    X_train, X_test, _, _ = split_data(X, y)
    assert set(X_train.index).isdisjoint(set(X_test.index))
