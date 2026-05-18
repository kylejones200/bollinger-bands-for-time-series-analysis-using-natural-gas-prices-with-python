#!/usr/bin/env python3
"""Bollinger Bands — Polars + DuckDB rewrite."""

import argparse
import logging
from pathlib import Path

import yaml
from core import (
    calculate_bollinger_bands,
    generate_synthetic_prices,
    plot_bollinger_bands,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Bollinger Bands — Polars + DuckDB")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)
    if args.data_path and args.data_path.exists():
        import polars as pl

        df = pl.read_csv(args.data_path, try_parse_dates=True)
    else:
        df = generate_synthetic_prices(
            config["data"]["start_date"],
            config["data"]["end_date"],
            config["data"]["frequency"],
            config["data"]["initial_price"],
            config["data"]["volatility"],
            config["data"]["seed"],
        )

    window = config["bollinger_bands"]["window"]
    num_std = config["bollinger_bands"]["num_std"]
    target = config["data"]["target_column"]
    bands = calculate_bollinger_bands(
        df, window=window, num_std=num_std, price_col=target
    )
    logging.info(f"Rows after warm-up: {bands.height}")
    logging.info(f"Latest MA:          {bands['ma'][-1]:.4f}")
    logging.info(f"Latest upper band:  {bands['upper_band'][-1]:.4f}")
    logging.info(f"Latest lower band:  {bands['lower_band'][-1]:.4f}")
    plot_bollinger_bands(
        bands,
        price_col=target,
        window=window,
        output_path=output_dir / "bollinger_bands.png",
    )
    logging.info(f"Done. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
