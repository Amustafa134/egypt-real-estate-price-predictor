"""
Exploratory data analysis: prints key statistics and saves plots to
reports/figures/. This is analysis only -- anything the model depends on lives
in src/cleaning and src/features, not here.

NOTE: price_per_sqm is used here for EDA ONLY (comparing locations). It is
computed from price, the target we predict, so it must never be a model input.

Run from the repo root:
    python -m src.analysis.eda
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cleaning.clean import clean_dataset

RAW_PATH = "data/raw/egypt_real_estate_listings.csv"
FIGURES_DIR = Path("reports/figures")
MIN_GROUP_SIZE = 100  # only plot groups with enough listings to be meaningful

sns.set_theme(style="whitegrid", context="notebook")


def save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_price_distribution(df: pd.DataFrame):
    """Raw vs log price side by side: shows why the target is log-transformed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    cap = df["price"].quantile(0.99)
    axes[0].hist(df["price"] / 1e6, bins=60, range=(0, cap / 1e6), color="#4C72B0")
    axes[0].set_title(f"Raw price (skew = {df['price'].skew():.1f})")
    axes[0].set_xlabel("Price (million EGP; top 1% of listings not shown)")
    axes[0].set_ylabel("Listings")

    log_price = np.log10(df["price"])
    axes[1].hist(log_price, bins=60, color="#55A868")
    axes[1].set_title(f"log10(price) (skew = {np.log(df['price']).skew():.2f})")
    axes[1].set_xlabel("log10(price in EGP)")

    fig.suptitle("Price is heavily right-skewed; its log is near-symmetric")
    save_fig(fig, "01_price_distribution.png")


def plot_price_per_sqm_by_city(df: pd.DataFrame):
    """Median price per sqm by city (EDA only -- never a model feature)."""
    d = df.assign(price_per_sqm=df["price"] / df["size_sqm"])
    stats = d.groupby("city")["price_per_sqm"].agg(["median", "count"])
    stats = stats[stats["count"] >= MIN_GROUP_SIZE].sort_values("median")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(stats.index, stats["median"] / 1000, color="#4C72B0")
    ax.set_xlabel("Median price per sqm (thousand EGP)")
    ax.set_title(f"Where the market is expensive (cities with >= {MIN_GROUP_SIZE} listings)")
    save_fig(fig, "02_median_price_per_sqm_by_city.png")


def plot_price_by_type(df: pd.DataFrame):
    counts = df["type"].value_counts()
    keep = counts[counts >= MIN_GROUP_SIZE].index
    d = df[df["type"].isin(keep)].assign(log10_price=np.log10(df["price"]))
    order = d.groupby("type")["log10_price"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=d, y="type", x="log10_price", order=order, color="#8172B2", ax=ax)
    ax.set_xlabel("log10(price in EGP)")
    ax.set_ylabel("")
    ax.set_title(f"Price by property type (types with >= {MIN_GROUP_SIZE} listings)")
    save_fig(fig, "03_price_by_type.png")


def plot_size_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    hb = ax.hexbin(
        np.log10(df["size_sqm"]), np.log10(df["price"]),
        gridsize=45, cmap="viridis", mincnt=1, bins="log",
    )
    fig.colorbar(hb, ax=ax, label="listings (log scale)")
    ax.set_xlabel("log10(size in sqm)")
    ax.set_ylabel("log10(price in EGP)")
    ax.set_title("Size vs price: strong relationship, wide spread")
    save_fig(fig, "04_size_vs_price.png")


def print_summary(df: pd.DataFrame):
    print("\n=== Cleaning report ===")
    for step, count in df.attrs["cleaning_log"].items():
        print(f"  {step}: {count}")

    d = df.assign(log_price=np.log(df["price"]), log_size=np.log(df["size_sqm"]))
    numeric = ["log_size", "bedrooms_count", "bathrooms", "has_maid_room",
               "has_installment_option"]
    corr = d[numeric + ["log_price"]].astype(float).corr()["log_price"].drop("log_price")
    print("\n=== Correlation with log(price) ===")
    print(corr.sort_values(ascending=False).round(3).to_string())

    print("\n=== Median price per sqm by governorate (n >= 100) ===")
    pps = (df["price"] / df["size_sqm"]).groupby(df["governorate"]).agg(["median", "count"])
    print(pps[pps["count"] >= 100].sort_values("median", ascending=False).round(0).to_string())


def main():
    df = clean_dataset(pd.read_csv(RAW_PATH))
    print_summary(df)
    print("\n=== Saving figures ===")
    plot_price_distribution(df)
    plot_price_per_sqm_by_city(df)
    plot_price_by_type(df)
    plot_size_vs_price(df)


if __name__ == "__main__":
    main()
    