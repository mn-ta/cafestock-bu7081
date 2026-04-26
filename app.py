"""CafeStock - inventory decision-support prototype for independent cafes.

Built for the BU7081 assessment portfolio. Uses Monte Carlo simulation to
recommend daily order quantities for perishable bakery items, trading off
the cost of waste against the cost of stockouts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (
    PERISHABLE_CATEGORIES,
    daily_demand,
    get_demand_series,
    load_transactions,
    perishable_summary,
)
from src.simulation import (
    CostInputs,
    annualised_savings,
    newsvendor_quantity,
    simulate_for_quantity,
    status_quo_quantity,
    sweep_order_quantities,
)

DATA_PATH = Path(__file__).parent / "data" / "coffee_shop_sales.csv"
GBP_PER_USD = 0.79  # rough conversion so figures are plausible for a UK cafe

st.set_page_config(
    page_title="CafeStock - Inventory Decision Support",
    page_icon="☕",
    layout="wide",
)


def header():
    st.title("CafeStock")
    st.caption(
        "Monte Carlo decision support for daily perishable ordering in "
        "independent cafes. Built on 149,116 real coffee-shop transactions "
        "from the Maven Roasters dataset (Maven Analytics, 2023)."
    )


@st.cache_data(show_spinner="Loading transactions...")
def get_data():
    return load_transactions(DATA_PATH)


def page_overview(df: pd.DataFrame):
    st.subheader("1. Business overview")
    st.markdown(
        "Independent cafes run on margins of around 10-15 percent, and a "
        "single morning with too few croissants or a Friday afternoon with "
        "20 unsold scones can wipe out a day's profit. CafeStock turns the "
        "cafe's own till data into a daily order recommendation that is "
        "calibrated to the cost of waste, the cost of a missed sale, and "
        "the natural variability of demand."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Transactions analysed", f"{len(df):,}")
    col_b.metric("Stores", df["store_location"].nunique())
    col_c.metric(
        "Date range",
        f"{df['transaction_date'].min().date()}",
        f"to {df['transaction_date'].max().date()}",
    )
    col_d.metric("Total revenue (USD)", f"${df['revenue'].sum():,.0f}")

    st.markdown("**Sales mix by category**")
    mix = (
        df.groupby("product_category")["revenue"]
        .sum()
        .sort_values(ascending=True)
    )
    fig = px.bar(
        mix,
        orientation="h",
        labels={"value": "Revenue (USD)", "product_category": ""},
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

    perishable = df[df["product_category"].isin(PERISHABLE_CATEGORIES)]
    perishable_share = perishable["revenue"].sum() / df["revenue"].sum()
    st.info(
        f"Perishable bakery items account for {perishable_share:.1%} of "
        "revenue across the three stores. Because these items have a one to "
        "two day shelf life, they are the primary target for the inventory "
        "optimisation engine."
    )


def page_demand_explorer(df: pd.DataFrame):
    st.subheader("2. Demand explorer")
    st.markdown(
        "Browse historical demand for each perishable item. Use this to "
        "spot day-of-week effects, peak hours and the spread of daily "
        "demand that the simulation engine has to work with."
    )

    store = st.sidebar.selectbox(
        "Store",
        ["All stores"] + sorted(df["store_location"].unique().tolist()),
        key="explorer_store",
    )
    perishable = df[df["product_category"].isin(PERISHABLE_CATEGORIES)]
    if store != "All stores":
        perishable = perishable[perishable["store_location"] == store]

    summary = perishable_summary(df, store)
    st.markdown("**Per-product daily demand profile**")
    st.dataframe(
        summary.style.background_gradient(
            subset=["coef_variation"], cmap="Reds"
        ),
        use_container_width=True,
    )
    st.caption(
        "A coefficient of variation above 0.5 signals that average ordering "
        "rules of thumb will leave material money on the table."
    )

    product = st.selectbox(
        "Drill into a product",
        summary.index.tolist(),
        key="explorer_product",
    )

    sub = perishable[perishable["product_detail"] == product]
    daily = (
        sub.groupby("transaction_date")["transaction_qty"]
        .sum()
        .reset_index()
    )
    daily["day_of_week"] = daily["transaction_date"].dt.day_name()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            daily,
            x="transaction_date",
            y="transaction_qty",
            title=f"Daily unit sales: {product}",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(
            daily,
            x="transaction_qty",
            nbins=20,
            title="Distribution of daily demand",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    daily["day_of_week"] = pd.Categorical(
        daily["day_of_week"], categories=dow_order, ordered=True
    )
    by_dow = daily.groupby("day_of_week", observed=True)["transaction_qty"].mean()
    fig = px.bar(
        by_dow,
        title="Average daily demand by day of week",
        labels={"value": "Mean units", "day_of_week": ""},
    )
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)


def page_simulation(df: pd.DataFrame):
    st.subheader("3. Monte Carlo simulation")
    st.markdown(
        "Pick an item, set the cost structure, and the engine sweeps every "
        "feasible order quantity, simulating ten thousand demand draws each "
        "and reporting the expected profit, the chance of running out and "
        "the average waste."
    )

    summary = perishable_summary(df, "All stores")
    col_l, col_r = st.columns([1, 1])
    with col_l:
        store = st.selectbox(
            "Store",
            ["All stores"] + sorted(df["store_location"].unique().tolist()),
            key="sim_store",
        )
        product = st.selectbox(
            "Product",
            summary.index.tolist(),
            key="sim_product",
        )
        n_samples = st.slider(
            "Simulation draws per quantity",
            1000,
            20000,
            10000,
            step=1000,
        )
    with col_r:
        suggested_price = float(summary.loc[product, "unit_price"]) * GBP_PER_USD
        sell = st.number_input(
            "Selling price per unit (GBP)",
            min_value=0.5,
            value=round(suggested_price, 2),
            step=0.1,
        )
        cost = st.number_input(
            "Wholesale cost per unit (GBP)",
            min_value=0.1,
            value=round(suggested_price * 0.4, 2),
            step=0.1,
        )
        salvage = st.number_input(
            "End-of-day salvage value (GBP)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Discount sale, staff meal, or zero if binned.",
        )
        penalty = st.number_input(
            "Stockout penalty per lost unit (GBP)",
            min_value=0.0,
            value=round((sell - cost) * 0.5, 2),
            step=0.1,
            help="Captures the goodwill cost of telling a customer 'sold out'.",
        )

    history = get_demand_series(df, product, store)
    if history.size == 0 or history.sum() == 0:
        st.warning(
            "No demand history for this product at the selected store."
        )
        return

    costs = CostInputs(
        unit_cost=cost,
        selling_price=sell,
        salvage_value=salvage,
        stockout_penalty=penalty,
    )

    sweep = sweep_order_quantities(history, costs, n_samples=n_samples)
    optimal_row = sweep[sweep["is_optimal"]].iloc[0]
    nv_q = newsvendor_quantity(history, costs)
    sq_q = status_quo_quantity(history, percentile=75)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended order Q*", int(optimal_row["order_qty"]))
    m2.metric("Expected daily profit", f"GBP {optimal_row['expected_profit']:.2f}")
    m3.metric("Stockout probability", f"{optimal_row['stockout_prob']:.1%}")
    m4.metric("Expected waste units", f"{optimal_row['expected_waste_units']:.2f}")

    st.markdown(
        f"Critical ratio: {costs.critical_ratio:.2f}. Analytical "
        f"newsvendor quantity: **{nv_q}**. Status quo benchmark "
        f"(75th percentile of history): **{sq_q}**."
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sweep["order_qty"],
            y=sweep["expected_profit"],
            mode="lines+markers",
            name="Expected profit",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sweep["order_qty"],
            y=sweep["profit_p05"],
            mode="lines",
            name="5th percentile",
            line=dict(color="#aec7e8", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sweep["order_qty"],
            y=sweep["profit_p95"],
            mode="lines",
            name="95th percentile",
            line=dict(color="#aec7e8", dash="dot"),
            fill="tonexty",
        )
    )
    fig.add_vline(
        x=int(optimal_row["order_qty"]),
        line_dash="dash",
        line_color="green",
        annotation_text=f"Q* = {int(optimal_row['order_qty'])}",
    )
    fig.update_layout(
        title=f"Profit response to order quantity (n={n_samples:,} draws each)",
        xaxis_title="Order quantity",
        yaxis_title="Profit (GBP)",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=sweep["order_qty"],
            y=sweep["fill_rate"],
            name="Fill rate",
            yaxis="y1",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=sweep["order_qty"],
            y=sweep["waste_rate"],
            name="Waste rate",
            yaxis="y2",
        )
    )
    fig2.update_layout(
        title="Service vs. waste trade-off",
        xaxis_title="Order quantity",
        yaxis=dict(title="Fill rate", range=[0, 1.05]),
        yaxis2=dict(title="Waste rate", overlaying="y", side="right", range=[0, 1.05]),
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View full simulation table"):
        st.dataframe(sweep.round(3), use_container_width=True)


def page_savings(df: pd.DataFrame):
    st.subheader("4. Status quo vs. optimised ordering")
    st.markdown(
        "Most independent cafes order to a 'safe' rule of thumb such as "
        "the 75th percentile of last week's sales. The chart below "
        "estimates the annual profit lift that a Monte Carlo recommendation "
        "would deliver, item by item, holding cost assumptions constant."
    )

    col_l, col_r = st.columns(2)
    store = col_l.selectbox(
        "Store",
        sorted(df["store_location"].unique().tolist()),
        key="sav_store",
    )
    margin_pct = col_r.slider(
        "Assumed gross margin (%)", 30, 80, 60, step=5
    )
    penalty_pct = col_r.slider(
        "Stockout penalty as % of margin", 0, 200, 50, step=10
    )

    summary = perishable_summary(df, store)
    rows = []
    for product in summary.index:
        history = get_demand_series(df, product, store)
        sell = float(summary.loc[product, "unit_price"]) * GBP_PER_USD
        cost = sell * (1 - margin_pct / 100)
        margin = sell - cost
        penalty = margin * penalty_pct / 100
        costs = CostInputs(
            unit_cost=cost,
            selling_price=sell,
            salvage_value=0.0,
            stockout_penalty=penalty,
        )
        sweep = sweep_order_quantities(history, costs, n_samples=4000)
        optimal_q = int(sweep.loc[sweep["expected_profit"].idxmax(), "order_qty"])
        baseline_q = status_quo_quantity(history, percentile=75)
        result = annualised_savings(history, costs, optimal_q, baseline_q, n_samples=4000)
        result["product"] = product
        rows.append(result)
    res = pd.DataFrame(rows)

    total_uplift = res["annual_uplift"].sum()
    avg_baseline_fill = res["baseline_fill"].mean()
    avg_optimal_fill = res["optimal_fill"].mean()
    avg_baseline_waste = res["baseline_waste_units"].mean()
    avg_optimal_waste = res["optimal_waste_units"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Annual profit uplift", f"GBP {total_uplift:,.0f}")
    c2.metric(
        "Avg fill rate change",
        f"{avg_optimal_fill:.1%}",
        f"{(avg_optimal_fill - avg_baseline_fill)*100:+.1f} pts",
    )
    c3.metric(
        "Avg waste/day change",
        f"{avg_optimal_waste:.2f} units",
        f"{(avg_optimal_waste - avg_baseline_waste):+.2f} units",
    )

    res_view = res[
        [
            "product",
            "baseline_qty",
            "optimal_qty",
            "baseline_daily_profit",
            "optimal_daily_profit",
            "daily_uplift",
            "annual_uplift",
        ]
    ].sort_values("annual_uplift", ascending=False).round(2)
    st.dataframe(res_view, use_container_width=True)

    fig = px.bar(
        res_view,
        x="annual_uplift",
        y="product",
        orientation="h",
        title="Estimated annual profit uplift by product (GBP)",
    )
    fig.update_layout(height=400, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def page_order_sheet(df: pd.DataFrame):
    st.subheader("5. Tomorrow's order sheet")
    st.markdown(
        "Generate a printable order list for one store. Costs are derived "
        "from the dataset's unit prices and the gross margin you supply."
    )

    store = st.selectbox(
        "Store",
        sorted(df["store_location"].unique().tolist()),
        key="order_store",
    )
    weekday = st.selectbox(
        "Day to order for",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        key="order_day",
    )
    margin_pct = st.slider("Gross margin (%)", 30, 80, 60, step=5, key="ord_margin")
    penalty_pct = st.slider(
        "Stockout penalty as % of margin", 0, 200, 50, step=10, key="ord_pen"
    )

    summary = perishable_summary(df, store)
    rows = []
    for product in summary.index:
        sub = df[
            (df["product_detail"] == product)
            & (df["store_location"] == store)
            & (df["day_of_week"] == weekday)
        ]
        history = (
            sub.groupby("transaction_date")["transaction_qty"].sum().values
        )
        if history.size < 5:
            continue
        sell = float(summary.loc[product, "unit_price"]) * GBP_PER_USD
        cost = sell * (1 - margin_pct / 100)
        margin = sell - cost
        penalty = margin * penalty_pct / 100
        costs = CostInputs(
            unit_cost=cost,
            selling_price=sell,
            salvage_value=0.0,
            stockout_penalty=penalty,
        )
        sweep = sweep_order_quantities(history, costs, n_samples=3000)
        opt_row = sweep.loc[sweep["expected_profit"].idxmax()]
        rows.append(
            {
                "Product": product,
                "Order qty": int(opt_row["order_qty"]),
                "Avg historical demand": round(history.mean(), 1),
                "Stockout risk": f"{opt_row['stockout_prob']:.0%}",
                "Expected profit (GBP)": round(opt_row["expected_profit"], 2),
            }
        )
    plan = pd.DataFrame(rows).sort_values("Order qty", ascending=False)
    st.dataframe(plan, use_container_width=True)
    csv = plan.to_csv(index=False)
    st.download_button(
        "Download order sheet (CSV)",
        csv,
        file_name=f"order_sheet_{store.replace(' ', '_')}_{weekday}.csv",
        mime="text/csv",
    )


PAGES = {
    "Overview": page_overview,
    "Demand explorer": page_demand_explorer,
    "Monte Carlo simulation": page_simulation,
    "Annual savings": page_savings,
    "Order sheet": page_order_sheet,
}


def main():
    header()
    df = get_data()
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built for BU7081 Programming for Business Analytics. Data: Maven "
        "Roasters transactions, Maven Analytics (2023)."
    )
    PAGES[page](df)


if __name__ == "__main__":
    main()
