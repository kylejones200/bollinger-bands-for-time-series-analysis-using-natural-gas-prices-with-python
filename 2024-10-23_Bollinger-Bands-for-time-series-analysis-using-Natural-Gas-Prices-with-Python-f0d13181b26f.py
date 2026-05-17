# Description: Short example for Bollinger Bands for time series analysis using Natural Gas Prices with Python.


import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

np.random.seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def generate_synthetic_price_data(n_days=365):
    """Generate synthetic stock price data with realistic characteristics."""
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    base_price = 100
    drift = 0.0005
    volatility = 0.02

    returns = np.random.normal(drift, volatility, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame({"date": dates, "price": prices})


def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands with proper handling for trading signals.

    For visualization: Uses current values (includes today)
    For trading: Uses shifted values (only past data)
    """
    df = df.copy()

    df["sma"] = df["price"].rolling(window=window).mean()
    df["std"] = df["price"].rolling(window=window).std()
    df["upper_band"] = df["sma"] + (num_std * df["std"])
    df["lower_band"] = df["sma"] - (num_std * df["std"])

    df["sma_shifted"] = df["price"].shift(1).rolling(window=window).mean()
    df["std_shifted"] = df["price"].shift(1).rolling(window=window).std()
    df["upper_band_shifted"] = df["sma_shifted"] + (num_std * df["std_shifted"])
    df["lower_band_shifted"] = df["sma_shifted"] - (num_std * df["std_shifted"])

    return df


def generate_trading_signals(df):
    """
    Generate buy/sell signals based on Bollinger Bands.
    Uses shifted bands to avoid look-ahead bias.
    """
    df = df.copy()

    df["signal"] = 0
    df.loc[df["price"] < df["lower_band_shifted"], "signal"] = 1
    df.loc[df["price"] > df["upper_band_shifted"], "signal"] = -1

    df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
    df["returns"] = df["price"].pct_change()
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]

    return df


def prepare_dataset(n_days=365, window=20, num_std=2):
    """Build price series, bands, and trading signals."""
    df = generate_synthetic_price_data(n_days=n_days)
    df = calculate_bollinger_bands(df, window=window, num_std=num_std)
    return generate_trading_signals(df)


def compute_performance_metrics(df):
    """Return buy-and-hold vs strategy performance and signal counts."""
    total_return = (df["price"].iloc[-1] / df["price"].iloc[0] - 1) * 100
    strategy_cumulative = (1 + df["strategy_returns"]).cumprod().iloc[-1] - 1

    return {
        "total_return_pct": total_return,
        "strategy_return_pct": strategy_cumulative * 100,
        "buy_signals": int((df["signal"] == 1).sum()),
        "sell_signals": int((df["signal"] == -1).sum()),
    }


def log_performance_summary(df, metrics):
    """Log period, returns, and signal counts."""
    logger.info(
        "Period: %s to %s",
        df["date"].iloc[0].strftime("%Y-%m-%d"),
        df["date"].iloc[-1].strftime("%Y-%m-%d"),
    )
    logger.info("Buy-and-Hold Return: %.2f%%", metrics["total_return_pct"])
    logger.info(
        "Bollinger Band Strategy Return: %.2f%%", metrics["strategy_return_pct"]
    )
    logger.info(
        "Buy Signals: %d | Sell Signals: %d",
        metrics["buy_signals"],
        metrics["sell_signals"],
    )


def add_cumulative_returns(df):
    """Add cumulative buy-and-hold and strategy return columns."""
    df = df.copy()
    df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1
    df["cumulative_strategy_returns"] = (1 + df["strategy_returns"]).cumprod() - 1
    return df


def add_band_width_columns(df):
    """Add absolute and percentage Bollinger band width columns."""
    df = df.copy()
    df["band_width"] = df["upper_band"] - df["lower_band"]
    df["band_width_pct"] = (df["band_width"] / df["sma"]) * 100
    return df


def _format_date_axis(ax):
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))


def plot_price_with_bands(ax, df):
    """Panel: price, SMA, and Bollinger band envelope."""
    ax.plot(df["date"], df["price"], color="#2c3e50", linewidth=1.5, label="Price")
    ax.plot(
        df["date"], df["sma"], color="#3498db", linewidth=1.5, alpha=0.7, label="20-Day SMA"
    )
    ax.fill_between(
        df["date"],
        df["lower_band"],
        df["upper_band"],
        color="#3498db",
        alpha=0.2,
        label="Bollinger Bands (±2σ)",
    )
    ax.plot(
        df["date"],
        df["upper_band"],
        color="#3498db",
        linewidth=1,
        alpha=0.5,
        linestyle="--",
    )
    ax.plot(
        df["date"],
        df["lower_band"],
        color="#3498db",
        linewidth=1,
        alpha=0.5,
        linestyle="--",
    )
    ax.set_title("Bollinger Bands (20-Day Window, ±2σ)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    _format_date_axis(ax)


def plot_signal_markers(ax, df):
    """Panel: price with buy/sell signal markers."""
    ax.plot(df["date"], df["price"], color="#2c3e50", linewidth=1.5, label="Price")
    ax.fill_between(
        df["date"], df["lower_band"], df["upper_band"], color="#95a5a6", alpha=0.1
    )

    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]

    ax.scatter(
        buys["date"],
        buys["price"],
        color="#27ae60",
        s=80,
        marker="^",
        zorder=5,
        label="Buy Signal",
        alpha=0.8,
    )
    ax.scatter(
        sells["date"],
        sells["price"],
        color="#e74c3c",
        s=80,
        marker="v",
        zorder=5,
        label="Sell Signal",
        alpha=0.8,
    )

    ax.set_title("Trading Signals", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    _format_date_axis(ax)


def plot_returns_comparison(ax, df):
    """Panel: cumulative buy-and-hold vs strategy returns."""
    ax.plot(
        df["date"],
        df["cumulative_returns"] * 100,
        color="#95a5a6",
        linewidth=2,
        label="Buy & Hold",
        alpha=0.7,
    )
    ax.plot(
        df["date"],
        df["cumulative_strategy_returns"] * 100,
        color="#3498db",
        linewidth=2,
        label="Bollinger Strategy",
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_title("Strategy Performance Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    _format_date_axis(ax)


def plot_band_width_chart(ax, df):
    """Band width as a percentage of SMA over time."""
    ax.fill_between(df["date"], 0, df["band_width_pct"], color="#3498db", alpha=0.5)
    ax.plot(df["date"], df["band_width_pct"], color="#3498db", linewidth=1.5)
    ax.set_title(
        "Bollinger Band Width (Volatility Indicator)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Band Width (% of SMA)")
    ax.set_xlabel("Date")
    _format_date_axis(ax)


def save_analysis_figures(df, analysis_path="bollinger_bands_analysis.png", width_path="bollinger_band_width.png"):
    """Create, save, and display the main analysis and band-width charts."""
    plot_df = add_cumulative_returns(df)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    plot_price_with_bands(axes[0], plot_df)
    plot_signal_markers(axes[1], plot_df)
    plot_returns_comparison(axes[2], plot_df)
    plt.tight_layout()
    plt.savefig(analysis_path, dpi=300, bbox_inches="tight")
    plt.show()

    width_df = add_band_width_columns(df)
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_band_width_chart(ax, width_df)
    plt.tight_layout()
    plt.savefig(width_path, dpi=300, bbox_inches="tight")
    plt.show()

    return analysis_path, width_path


def main():
    df = prepare_dataset()
    metrics = compute_performance_metrics(df)
    log_performance_summary(df, metrics)

    analysis_path, width_path = save_analysis_figures(df)

    logger.info("\nSaved: %s", analysis_path)
    logger.info("Saved: %s", width_path)


if __name__ == "__main__":
    main()
