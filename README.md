# CafeStock

Monte Carlo decision-support prototype for daily perishable ordering in independent cafés. Built for the BU7081 Programming for Business Analytics portfolio.

## What it does

CafeStock turns a café's own till data into a daily order recommendation that explicitly trades off the cost of waste against the cost of stockouts. The engine:

1. Bootstraps the historical demand series for each perishable product.
2. Runs a Monte Carlo simulation across every feasible order quantity (10,000 demand draws per quantity).
3. Reports the expected profit, fill rate, stockout probability and waste per quantity, and recommends the order quantity that maximises expected profit.

## Dataset

Maven Roasters transactional data: 149,116 transactions across three NYC stores between January and June 2023, sourced via the public Maven Analytics dataset. The CSV is included in `data/coffee_shop_sales.csv`.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Sign in at <https://share.streamlit.io> with the same GitHub account.
3. Click "New app", select this repo, branch `main`, file `app.py`.
4. Streamlit Cloud builds the environment from `requirements.txt` and gives you a public URL.

## Repository layout

```
app.py                  # main Streamlit entry point
src/data.py             # data loading and aggregation helpers
src/simulation.py       # Monte Carlo engine and newsvendor utilities
data/                   # Maven Roasters CSV
screenshots/            # captured screenshots used in the report
capture_screenshots.py  # Playwright script to regenerate screenshots
requirements.txt        # Python dependencies
```
