"""
Cleans the raw Egyptian real estate listings data.

Every step here is a judgment call backed by what the data actually looks like
(see src/analysis/eda.py), not a blind .fillna():

- price: stored as a comma-separated string ("8,000,000")
- size: compound string ("732 sqft / 68 sqm") -- we extract the sqm value,
  since square meters is the standard unit in the Egyptian market
- bedrooms: messy categorical data ("3+ Maid", "studio", "7++ Maid") split into
  a bedroom count and a separate maid's-room flag
- location: hierarchical "Compound, Area, City, Governorate" string. We keep two
  levels: governorate (coarse) and city (fine, e.g. "New Cairo City"). The full
  string has 1,535 unique values -- too sparse to use directly.
- down_payment: 73% missing -- a missing value almost certainly means "cash
  only", so it becomes a has_installment_option flag instead of an imputation.
- exact duplicate listings (same content, different URL) are removed BEFORE any
  train/test split, otherwise copies land on both sides and inflate test scores.
- listings with an implausible price per sqm are removed as likely entry errors.
- available_from is intentionally NOT used: its values all fall in a two-month
  window, so it is a listing/scrape date, not a delivery date.
"""

import re

import pandas as pd

# Fixed plausibility bounds (EGP per sqm), chosen from the observed distribution:
# ~21 listings fall below 2,000 and ~23 above 500,000, far outside the 1st-99th
# percentile range (~8.5k - 268k). Fixed constants (not percentiles computed on
# the whole dataset) so no information from future test rows leaks into cleaning.
MIN_PRICE_PER_SQM = 2_000
MAX_PRICE_PER_SQM = 500_000


def clean_price(series: pd.Series) -> pd.Series:
    """'8,000,000' -> 8000000.0. Non-numeric values become NaN, not 0."""
    return pd.to_numeric(series.str.replace(",", "", regex=False), errors="coerce")


def extract_size_sqm(series: pd.Series) -> pd.Series:
    """'732 sqft / 68 sqm' -> 68.0 (sqm is the standard unit in this market)."""
    def _extract(val):
        if pd.isna(val):
            return None
        match = re.search(r"([\d,]+)\s*sqm", str(val))
        if match:
            return float(match.group(1).replace(",", ""))
        return None
    return series.apply(_extract)


def parse_bedrooms(series: pd.Series) -> pd.DataFrame:
    """
    Splits the messy 'bedrooms' column into two honest, separate signals:
      - bedrooms_count: the base number (studio -> 0, '3+ Maid' -> 3, '7++' -> 7)
      - has_maid_room: whether a maid's room was mentioned (a real price factor
        in the Egyptian market, kept as its own feature)
    """
    def _count(val):
        if pd.isna(val):
            return None
        val = str(val).strip().lower()
        if "studio" in val:
            return 0
        match = re.search(r"(\d+)", val)
        return int(match.group(1)) if match else None

    def _has_maid(val):
        if pd.isna(val):
            return False
        return "maid" in str(val).lower()

    return pd.DataFrame({
        "bedrooms_count": series.apply(_count),
        "has_maid_room": series.apply(_has_maid),
    })


def parse_location(series: pd.Series) -> pd.DataFrame:
    """
    'Swan Lake Gouna, Al Gouna, Hurghada, Red Sea' (most specific first) ->
        governorate = 'Red Sea'   (last segment)
        city        = 'Hurghada'  (second-to-last segment)

    Governorate alone is too coarse ("Cairo" covers both New Cairo and Shubra);
    city has ~70 unique values, with 18 of them covering ~95% of listings.
    Rare cities are grouped later, inside the modeling pipeline.
    """
    def _segments(val):
        if pd.isna(val):
            return []
        return [p.strip() for p in str(val).split(",")]

    segs = series.apply(_segments)
    return pd.DataFrame({
        "governorate": segs.apply(lambda p: p[-1] if len(p) >= 1 else None),
        "city": segs.apply(lambda p: p[-2] if len(p) >= 2 else None),
    })


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all cleaning steps and returns a model-ready dataframe.
    A summary of how many rows each step removed is stored in
    df.attrs["cleaning_log"] so the pipeline can report it.
    """
    df = df.copy()
    log = {"raw_rows": len(df)}

    # 1. Exact duplicates (everything identical except the URL)
    dedupe_cols = [c for c in df.columns if c != "url"]
    df = df.drop_duplicates(subset=dedupe_cols)
    log["duplicates_removed"] = log["raw_rows"] - len(df)

    # 2. Parse messy columns
    df["price"] = clean_price(df["price"])
    df["size_sqm"] = extract_size_sqm(df["size"])

    bedroom_cols = parse_bedrooms(df["bedrooms"])
    df["bedrooms_count"] = bedroom_cols["bedrooms_count"]
    df["has_maid_room"] = bedroom_cols["has_maid_room"]

    location_cols = parse_location(df["location"])
    df["governorate"] = location_cols["governorate"]
    df["city"] = location_cols["city"]

    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")

    # Missing down_payment almost certainly means "cash only", not "unknown".
    df["has_installment_option"] = df["down_payment"].notna()

    # Missing payment_method is kept as its own explicit category.
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    # 3. Rows we cannot train or evaluate on
    before = len(df)
    df = df.dropna(subset=["price", "size_sqm"])
    log["missing_price_or_size_removed"] = before - len(df)

    # 4. Implausible price per sqm (likely entry errors)
    before = len(df)
    price_per_sqm = df["price"] / df["size_sqm"]
    df = df[price_per_sqm.between(MIN_PRICE_PER_SQM, MAX_PRICE_PER_SQM)]
    log["implausible_price_per_sqm_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    log["final_rows"] = len(df)
    df.attrs["cleaning_log"] = log
    return df


if __name__ == "__main__":
    from pathlib import Path

    raw = pd.read_csv("data/raw/egypt_real_estate_listings.csv")
    cleaned = clean_dataset(raw)

    print("=== Cleaning report ===")
    for step, count in cleaned.attrs["cleaning_log"].items():
        print(f"  {step}: {count}")

    print(f"\nPrice stats (EGP):\n{cleaned['price'].describe().round(0)}")
    print(f"\nBedrooms count missing: {cleaned['bedrooms_count'].isna().sum()}")
    print(f"Bathrooms missing: {cleaned['bathrooms'].isna().sum()}")
    print(f"\nTop governorates:\n{cleaned['governorate'].value_counts().head(8)}")
    print(f"\nTop cities:\n{cleaned['city'].value_counts().head(8)}")

    out_path = Path("data/processed/listings_clean.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"\nSaved cleaned data to {out_path}")
    