import datetime

import pandas as pd
import plotsmith as ps
import timesmith as ts


def notebook_step_001() -> None:
    "Generated from Jupyter notebook: FRED Unemployment & Bollinger Bands with timesmith and plotsmith\n\nMagics and shell lines are commented out. Run with a normal Python interpreter."


def notebook_step_003() -> None:
    datetime.datetime(2010, 1, 1)
    datetime.datetime(2024, 10, 1)
    print(f"Loaded {len(df)} observations")
    print(df.head())


def timesmith_provides_a_unified_forecasting_interfa() -> None:
    forecast_horizon = 12
    ts.ForecastTask(y=df, fh=forecast_horizon, frequency="MS")
    forecaster = ts.ExponentialMovingAverageForecaster(alpha=0.3)
    forecaster.fit(df)
    forecast_result = forecaster.predict(fh=forecast_horizon)
    forecast_series = forecast_result.y_pred
    print("Forecast values:")
    print(forecast_series.head())
    print(f"\nForecast horizon: {len(forecast_series)} periods")


def plotsmith_makes_it_easy_to_visualize_forecasts() -> None:
    ps.plot_model_comparison(
        data=df,
        models={"EMA Forecast": forecast_series},
        title="Unemployment Rate Forecast with Exponential Moving Average",
        xlabel="Date",
        ylabel="Unemployment Rate (%)",
        figsize=(12, 6),
    )


def now_let_s_analyze_henry_hub_natural_gas_spot_pri() -> None:
    df1
    _raw = web.DataReader("DHHNGSP", "fred", start, end)
    col_name_gas = df1_raw.columns[0]
    df1 = df1_raw[col_name_gas].dropna()
    ps.plot_timeseries(
        df1,
        title="Henry Hub Natural Gas Spot Price",
        xlabel="Date",
        ylabel="Price (USD)",
        figsize=(12, 6),
    )


def forecast_using_timesmith_s_exponential_moving_av() -> None:
    forecast_horizon_gas = 12
    gas_forecaster = ts.ExponentialMovingAverageForecaster(alpha=0.3)
    gas_forecaster.fit(df1)
    forecast_gas_result = gas_forecaster.predict(fh=forecast_horizon_gas)
    forecast_gas_series = forecast_gas_result.y_pred
    ps.plot_model_comparison(
        data=df1,
        models={"EMA Forecast": forecast_gas_series},
        title="Henry Hub Natural Gas Spot Price Forecast",
        xlabel="Date",
        ylabel="Price (USD)",
        figsize=(12, 6),
    )


def timesmith_also_supports_naive_forecasting_method() -> None:
    naive_forecaster = ts.SimpleMovingAverageForecaster(window=1)
    naive_forecaster.fit(df1)
    naive_forecast_result = naive_forecaster.predict(fh=10)
    naive_forecast_series = naive_forecast_result.y_pred
    ps.plot_model_comparison(
        data=df1,
        models={
            "EMA Forecast": forecast_gas_series,
            "Naive Forecast": naive_forecast_series,
        },
        title="Natural Gas Price: Forecast Method Comparison",
        xlabel="Date",
        ylabel="Price (USD)",
        figsize=(14, 6),
    )


def timesmith_provides_rollingfeaturizer_for_technic() -> None:
    rolling_featurizer = ts.RollingFeaturizer(windows=[20], functions=["mean", "std"])
    rolling_featurizer.fit(df1)
    rolling_features = rolling_featurizer.transform(df1)
    df1
    _with_bb = df1.to_frame(name="price")
    df1_with_bb["MA_20"] = rolling_features.iloc[:, 0]
    df1_with_bb["std_20"] = rolling_features.iloc[:, 1]
    df1_with_bb["upper_band"] = df1_with_bb["MA_20"] + 2 * df1_with_bb["std_20"]
    df1_with_bb["lower_band"] = df1_with_bb["MA_20"] - 2 * df1_with_bb["std_20"]
    print("Bollinger Bands calculated:")
    print(df1_with_bb[["price", "MA_20", "upper_band", "lower_band"]].tail(10))


def plotsmith_makes_it_easy_to_visualize_bollinger_b() -> None:
    bands_data = pd.DataFrame(
        {"price": df1_with_bb["price"], "MA": df1_with_bb["MA_20"]}
    )
    bands = {"Bollinger Bands": (df1_with_bb["lower_band"], df1_with_bb["upper_band"])}
    ps.plot_timeseries(
        bands_data,
        bands=bands,
        title="Henry Hub Natural Gas Spot Price with Bollinger Bands",
        xlabel="Date",
        ylabel="Price (USD)",
        figsize=(14, 6),
    )


def analyze_recent_data_with_bollinger_bands() -> None:
    start_recent = datetime.datetime(2024, 4, 1)
    end_recent = datetime.datetime(2024, 10, 20)
    df2
    _raw = web.DataReader("DHHNGSP", "fred", start_recent, end_recent)
    col_name_gas2 = df2_raw.columns[0]
    df2 = df2_raw[col_name_gas2].dropna().sort_index()
    rolling_featurizer_recent = ts.RollingFeaturizer(
        windows=[20], functions=["mean", "std"]
    )
    rolling_featurizer_recent.fit(df2)
    rolling_features_recent = rolling_featurizer_recent.transform(df2)
    df2
    _with_bb = df2.to_frame(name="price")
    df2_with_bb["MA_20"] = rolling_features_recent.iloc[:, 0]
    df2_with_bb["std_20"] = rolling_features_recent.iloc[:, 1]
    df2_with_bb["upper_band"] = df2_with_bb["MA_20"] + 2 * df2_with_bb["std_20"]
    df2_with_bb["lower_band"] = df2_with_bb["MA_20"] - 2 * df2_with_bb["std_20"]
    bands_recent = {
        "Bollinger Bands": (df2_with_bb["lower_band"], df2_with_bb["upper_band"])
    }
    bands_data_recent = pd.DataFrame(
        {"price": df2_with_bb["price"], "MA": df2_with_bb["MA_20"]}
    )
    ps.plot_timeseries(
        bands_data_recent,
        bands=bands_recent,
        title="Recent Natural Gas Price with Bollinger Bands (2024)",
        xlabel="Date",
        ylabel="Price (USD)",
        figsize=(14, 6),
    )


def main() -> None:
    notebook_step_001()
    notebook_step_003()
    timesmith_provides_a_unified_forecasting_interfa()
    plotsmith_makes_it_easy_to_visualize_forecasts()
    now_let_s_analyze_henry_hub_natural_gas_spot_pri()
    forecast_using_timesmith_s_exponential_moving_av()
    timesmith_also_supports_naive_forecasting_method()
    timesmith_provides_rollingfeaturizer_for_technic()
    plotsmith_makes_it_easy_to_visualize_bollinger_b()
    analyze_recent_data_with_bollinger_bands()


if __name__ == "__main__":
    main()
