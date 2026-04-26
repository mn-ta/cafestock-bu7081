"""Data loading and pre-processing for the cafe sales dataset.

Source: Maven Roasters transactional data (Maven Analytics, 2023), 149,116
transactions across three NYC stores between January and June 2023.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


PERISHABLE_CATEGORIES = ("Bakery",)


@st.cache_data(show_spinner=False)
def load_transactions(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    df["hour"] = df["transaction_date"].dt.hour
    hour_from_time = pd.to_datetime(
        df["transaction_time"].astype(str), format="%H:%M:%S", errors="coerce"
    ).dt.hour
    df["hour"] = hour_from_time.fillna(df["hour"]).astype("Int64")
    df["day_of_week"] = df["transaction_date"].dt.day_name()
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    return df


@st.cache_data(show_spinner=False)
def daily_demand(
    df: pd.DataFrame,
    store: str | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    sub = df.copy()
    if store and store != "All stores":
        sub = sub[sub["store_location"] == store]
    if category and category != "All categories":
        sub = sub[sub["product_category"] == category]
    grouped = (
        sub.groupby(["transaction_date", "product_detail"])["transaction_qty"]
        .sum()
        .reset_index()
    )
    grouped["day_of_week"] = grouped["transaction_date"].dt.day_name()
    return grouped


@st.cache_data(show_spinner=False)
def perishable_summary(df: pd.DataFrame, store: str) -> pd.DataFrame:
    sub = df[df["product_category"].isin(PERISHABLE_CATEGORIES)]
    if store != "All stores":
        sub = sub[sub["store_location"] == store]
    daily = (
        sub.groupby(["transaction_date", "product_detail"])["transaction_qty"]
        .sum()
        .reset_index()
    )
    summary = (
        daily.groupby("product_detail")["transaction_qty"]
        .agg(["mean", "std", "min", "median", "max", "count"])
        .round(2)
        .rename(
            columns={
                "mean": "avg_daily_demand",
                "std": "demand_std",
                "min": "min_daily",
                "median": "median_daily",
                "max": "peak_daily",
                "count": "active_days",
            }
        )
    )
    price_lookup = (
        sub.groupby("product_detail")["unit_price"].mean().round(2)
    )
    summary["unit_price"] = price_lookup
    summary["coef_variation"] = (
        summary["demand_std"] / summary["avg_daily_demand"]
    ).round(2)
    return summary.sort_values("avg_daily_demand", ascending=False)


def get_demand_series(
    df: pd.DataFrame,
    product: str,
    store: str,
) -> np.ndarray:
    sub = df[df["product_detail"] == product]
    if store != "All stores":
        sub = sub[sub["store_location"] == store]
    daily = sub.groupby("transaction_date")["transaction_qty"].sum()
    full_index = pd.date_range(
        sub["transaction_date"].min(), sub["transaction_date"].max(), freq="D"
    )
    daily = daily.reindex(full_index, fill_value=0)
    return daily.values.astype(float)
