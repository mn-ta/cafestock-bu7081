"""Monte Carlo simulation engine for the single-period perishable inventory
problem (newsvendor formulation).

For each candidate order quantity Q, the engine draws N demand samples by
bootstrapping from the historical daily-demand series, then computes the
profit per draw under the cost structure:

    profit = price * sales - cost * Q + salvage * leftover - penalty * lost

where sales = min(Q, demand), leftover = max(0, Q - demand) and
lost = max(0, demand - Q). Aggregating across draws produces the expected
profit, fill rate, waste rate and a percentile risk band for each Q. The
quantity that maximises expected profit is reported as the recommended
order, with the analytical newsvendor critical ratio included for
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostInputs:
    unit_cost: float
    selling_price: float
    salvage_value: float = 0.0
    stockout_penalty: float = 0.0

    @property
    def underage_cost(self) -> float:
        return (self.selling_price - self.unit_cost) + self.stockout_penalty

    @property
    def overage_cost(self) -> float:
        return self.unit_cost - self.salvage_value

    @property
    def critical_ratio(self) -> float:
        cu, co = self.underage_cost, self.overage_cost
        return cu / (cu + co) if (cu + co) > 0 else 0.5


def bootstrap_demand(
    history: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    nonzero = history[history > 0]
    pool = nonzero if len(nonzero) >= 30 else history
    return rng.choice(pool, size=n_samples, replace=True)


def simulate_for_quantity(
    history: np.ndarray,
    order_qty: int,
    costs: CostInputs,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    demand = bootstrap_demand(history, n_samples, rng)
    sales = np.minimum(order_qty, demand)
    leftover = np.maximum(0, order_qty - demand)
    lost = np.maximum(0, demand - order_qty)
    profit = (
        costs.selling_price * sales
        - costs.unit_cost * order_qty
        + costs.salvage_value * leftover
        - costs.stockout_penalty * lost
    )
    return {
        "order_qty": int(order_qty),
        "expected_profit": float(profit.mean()),
        "profit_std": float(profit.std()),
        "profit_p05": float(np.percentile(profit, 5)),
        "profit_p95": float(np.percentile(profit, 95)),
        "fill_rate": float(sales.sum() / max(demand.sum(), 1e-9)),
        "stockout_prob": float((demand > order_qty).mean()),
        "waste_rate": float(leftover.sum() / max(order_qty * n_samples, 1e-9)),
        "expected_waste_units": float(leftover.mean()),
        "expected_lost_units": float(lost.mean()),
    }


def sweep_order_quantities(
    history: np.ndarray,
    costs: CostInputs,
    q_min: int | None = None,
    q_max: int | None = None,
    n_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if q_min is None:
        q_min = max(int(np.percentile(history[history > 0], 5)), 1)
    if q_max is None:
        q_max = int(np.percentile(history, 99)) + 5
    quantities = list(range(q_min, q_max + 1))
    rows = [
        simulate_for_quantity(history, q, costs, n_samples, rng)
        for q in quantities
    ]
    df = pd.DataFrame(rows)
    df["is_optimal"] = df["expected_profit"] == df["expected_profit"].max()
    return df


def newsvendor_quantity(history: np.ndarray, costs: CostInputs) -> int:
    cr = costs.critical_ratio
    return int(np.ceil(np.quantile(history, cr)))


def status_quo_quantity(history: np.ndarray, percentile: float = 75) -> int:
    return int(np.ceil(np.percentile(history, percentile)))


def annualised_savings(
    history: np.ndarray,
    costs: CostInputs,
    optimal_q: int,
    baseline_q: int,
    n_samples: int = 10_000,
    days_per_year: int = 312,
) -> dict:
    rng = np.random.default_rng(7)
    baseline = simulate_for_quantity(
        history, baseline_q, costs, n_samples, rng
    )
    rng = np.random.default_rng(7)
    optimal = simulate_for_quantity(
        history, optimal_q, costs, n_samples, rng
    )
    delta = optimal["expected_profit"] - baseline["expected_profit"]
    return {
        "baseline_qty": baseline_q,
        "optimal_qty": optimal_q,
        "baseline_daily_profit": baseline["expected_profit"],
        "optimal_daily_profit": optimal["expected_profit"],
        "daily_uplift": delta,
        "annual_uplift": delta * days_per_year,
        "baseline_waste_units": baseline["expected_waste_units"],
        "optimal_waste_units": optimal["expected_waste_units"],
        "baseline_fill": baseline["fill_rate"],
        "optimal_fill": optimal["fill_rate"],
    }
