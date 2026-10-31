"""Tests for src/cleaning/clean.py. Synthetic data only, no file I/O."""

import numpy as np
import pandas as pd

from src.cleaning.clean import (
    clean_dataset,
    clean_price,
    extract_size_sqm,
    parse_bedrooms,
    parse_location,
)


# ---------- clean_price ----------

def test_clean_price_removes_commas():
    result = clean_price(pd.Series(["8,000,000", "650,000"]))
    assert result.tolist() == [8_000_000, 650_000]


def test_clean_price_non_numeric_becomes_nan_not_zero():
    result = clean_price(pd.Series(["Ask for price"]))
    assert pd.isna(result.iloc[0])


# ---------- extract_size_sqm ----------

def test_extract_size_sqm_takes_sqm_value():
    result = extract_size_sqm(pd.Series(["732 sqft / 68 sqm"]))
    assert result.iloc[0] == 68.0


def test_extract_size_sqm_handles_commas():
    result = extract_size_sqm(pd.Series(["12,917 sqft / 1,200 sqm"]))
    assert result.iloc[0] == 1200.0


def test_extract_size_sqm_missing_or_no_sqm_gives_missing():
    result = extract_size_sqm(pd.Series([np.nan, "732 sqft"]))
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])


# ---------- parse_bedrooms ----------

def test_parse_bedrooms_counts():
    result = parse_bedrooms(pd.Series(["studio", "3+ Maid", "7++ Maid", "2"]))
    assert result["bedrooms_count"].tolist() == [0, 3, 7, 2]


def test_parse_bedrooms_maid_flag():
    result = parse_bedrooms(pd.Series(["studio", "3+ Maid", "2", np.nan]))
    assert result["has_maid_room"].tolist() == [False, True, False, False]


def test_parse_bedrooms_missing_count_stays_missing():
    result = parse_bedrooms(pd.Series([np.nan]))
    assert pd.isna(result["bedrooms_count"].iloc[0])


# ---------- parse_location ----------

def test_parse_location_full_string():
    result = parse_location(
        pd.Series(["Swan Lake Gouna, Al Gouna, Hurghada, Red Sea"])
    )
    assert result["governorate"].iloc[0] == "Red Sea"
    assert result["city"].iloc[0] == "Hurghada"


def test_parse_location_short_string():
    result = parse_location(pd.Series(["Granville, New Capital City, Cairo"]))
    assert result["governorate"].iloc[0] == "Cairo"
    assert result["city"].iloc[0] == "New Capital City"


def test_parse_location_single_segment_has_no_city():
    result = parse_location(pd.Series(["Cairo"]))
    assert result["governorate"].iloc[0] == "Cairo"
    assert result["city"].iloc[0] is None


# ---------- clean_dataset ----------

def _raw_listings() -> pd.DataFrame:
    """Five raw rows: 2 valid, 1 duplicate, 1 missing price, 1 implausible."""
    base = {
        "description": "Nice flat",
        "location": "Taj City, The 5th Settlement, New Cairo City, Cairo",
        "type": "Apartment",
        "bedrooms": "3+ Maid",
        "bathrooms": "2",
        "available_from": "2024-01-01",
        "payment_method": "Cash",
        "down_payment": np.nan,
    }
    rows = [
        # valid: 50,000 EGP per sqm
        {**base, "url": "u1", "price": "5,000,000", "size": "1,076 sqft / 100 sqm"},
        # exact duplicate of u1 except the URL -> removed
        {**base, "url": "u2", "price": "5,000,000", "size": "1,076 sqft / 100 sqm"},
        # missing price -> removed
        {**base, "url": "u3", "price": np.nan, "size": "1,615 sqft / 150 sqm"},
        # 1,000 EGP per sqm (below the 2,000 minimum) -> removed
        {**base, "url": "u4", "price": "100,000", "size": "1,076 sqft / 100 sqm"},
        # valid, with an installment plan and missing payment method
        {
            **base,
            "url": "u5",
            "price": "9,000,000",
            "size": "2,153 sqft / 200 sqm",
            "payment_method": np.nan,
            "down_payment": "900,000",
        },
    ]
    return pd.DataFrame(rows)


def test_clean_dataset_cleaning_log():
    cleaned = clean_dataset(_raw_listings())
    assert cleaned.attrs["cleaning_log"] == {
        "raw_rows": 5,
        "duplicates_removed": 1,
        "missing_price_or_size_removed": 1,
        "implausible_price_per_sqm_removed": 1,
        "final_rows": 2,
    }


def test_clean_dataset_keeps_expected_rows():
    cleaned = clean_dataset(_raw_listings())
    assert cleaned["url"].tolist() == ["u1", "u5"]


def test_clean_dataset_installment_flag_and_unknown_payment():
    cleaned = clean_dataset(_raw_listings()).set_index("url")
    assert not cleaned.loc["u1", "has_installment_option"]
    assert cleaned.loc["u5", "has_installment_option"]
    assert cleaned.loc["u5", "payment_method"] == "Unknown"


def test_clean_dataset_does_not_modify_input():
    raw = _raw_listings()
    before = raw.copy()
    clean_dataset(raw)
    pd.testing.assert_frame_equal(raw, before)
