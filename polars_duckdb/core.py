"""Bollinger Bands using Polars and DuckDB.

calculate_bollinger_bands: pandas rolling().mean() + rolling().std()
→ single DuckDB query with AVG and STDDEV_SAMP window functions.
"""

import duckdb
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path


def generate_synthetic_prices(
    start_date: str = "2024-04-01",
    end_date:   str = "2024-10-31",
    freq:       str = "B",
    initial_price: float = 2.5,
    volatility:    float = 0.05,
    seed:          int   = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # build business-day date range via pandas (date arithmetic only, no computation)
    import pandas as pd
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    prices = initial_price + np.cumsum(rng.normal(0, volatility, len(dates)))
    return pl.DataFrame({
        "date":     [d.date() for d in dates],
        "adjClose": prices.tolist(),
    })


def calculate_bollinger_bands(
    df: pl.DataFrame,
    window:   int   = 20,
    num_std:  float = 2.0,
    date_col: str   = "date",
    price_col: str  = "adjClose",
) -> pl.DataFrame:
    """Rolling mean ± num_std σ via a single DuckDB window query."""
    w = window - 1  # ROWS BETWEEN w PRECEDING AND CURRENT ROW = window rows
    result = duckdb.sql(f"""
        SELECT
            "{date_col}",
            "{price_col}",
            AVG("{price_col}")         OVER w AS ma,
            AVG("{price_col}")         OVER w
                - {num_std} * STDDEV_SAMP("{price_col}") OVER w AS lower_band,
            AVG("{price_col}")         OVER w
                + {num_std} * STDDEV_SAMP("{price_col}") OVER w AS upper_band
        FROM df
        WINDOW w AS (ORDER BY "{date_col}" ROWS BETWEEN {w} PRECEDING AND CURRENT ROW)
        ORDER BY "{date_col}"
    """).pl()
    return result.drop_nulls()


def plot_bollinger_bands(
    df: pl.DataFrame,
    date_col:  str = "date",
    price_col: str = "adjClose",
    window:    int = 20,
    output_path: Path = None,
    plot: bool = False,
):
    if not plot:
        return
    dates  = df[date_col].to_list()
    prices = df[price_col].to_list()
    ma     = df["ma"].to_list()
    lower  = df["lower_band"].to_list()
    upper  = df["upper_band"].to_list()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(dates, lower, upper, alpha=0.15, color="#8B6F9E", label="Bollinger Band")
    ax.plot(dates, prices, label=price_col, color="#4A90A4", linewidth=1.2)
    ax.plot(dates, ma,     label=f"{window}-Day MA", color="#D4A574",
            linewidth=1.2, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
