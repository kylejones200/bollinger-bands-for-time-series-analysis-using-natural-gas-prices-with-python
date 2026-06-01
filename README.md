# Bollinger Bands for Time Series Analysis

This project demonstrates Bollinger Bands analysis for time series data, commonly used in technical analysis of financial markets.

## Business context

Bollinger Bands are an intuitive tool for assessing a stock's price movement. We can use them to look at the underlying volatility of a stock and potential trading opportunities. These are often used in technical trading analysis and were developed by [John Bollinger](https://www.bollingerbands.com/) in the 1980s.

The premise of Bollinger Bands is straightforward and assume that stocks will revert to the mean. Bollinger Bands plot a simple moving average of a stock's price, accompanied by two standard deviation lines placed above and below the moving average. These upper and lower bands create a dynamic envelope that captures the typical range of the expected price action. This can be used to measure market volatility.

This code calculates the Bollinger Bands for a given stock, using a 20-day simple moving average and a standard deviation of 2. The resulting bands are then plotted alongside the stock's closing prices, providing a visual representation of the security's price dynamics and volatility.

## Article

Medium article: [Bollinger Bands for Time Series Analysis](https://medium.com/datadriveninvestor/bollinger-bands-for-time-series-analysis-using-natural-gas-prices-with-python-f0d13181b26f)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Bollinger Bands functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (or synthetic generation parameters)
- Bollinger Bands parameters (window size, number of standard deviations)
- Output settings

## Bollinger Bands

Bollinger Bands consist of:
- Middle Band: N-period moving average
- Upper Band: Middle band + (N-period standard deviation × multiplier)
- Lower Band: Middle band - (N-period standard deviation × multiplier)

Prices touching the upper or lower bands may indicate overbought or oversold conditions.

## Caveats

- By default, the script generates synthetic price data if no data file is provided.
- Bollinger Bands are most effective with sufficient historical data (at least 20 periods for default window).
- The bands adapt to volatility, expanding during volatile periods and contracting during stable periods.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).