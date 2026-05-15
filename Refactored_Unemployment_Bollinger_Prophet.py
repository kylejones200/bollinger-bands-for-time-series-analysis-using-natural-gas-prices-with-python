"""Generated from Jupyter notebook: FRED Unemployment & Bollinger Bands with timesmith and plotsmith

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# %matplotlib inlineimport datetimeimport pandas as pdimport timesmith as tsimport plotsmith as psimport numpy as np# timesmith provides unified API for time series operations# plotsmith provides enhanced visualization capabilities  # Jupyter-only


# --- code cell ---

start = datetime.datetime(2010, 1, 1)end = datetime.datetime(2024, 10, 1)# Load unemployment data from FRED# Load unemployment data from FRED using timesmith# timesmith provides a simple, unified API for FRED datadf = ts.load_fred('unrate', start=start, end=end)print(f"Loaded {len(df)} observations")print(df.head())# plotsmith provides enhanced time series visualization out of the boxps.plot_timeseries(    df,    title='Unemployment Rate Time Series',    xlabel='Date',    ylabel='Unemployment Rate (%)',    figsize=(12, 6))


# --- code cell ---

# timesmith provides a unified forecasting interface
# timesmith provides: SimpleMovingAverageForecaster, ExponentialMovingAverageForecaster, 
#                     WeightedMovingAverageForecaster, MonteCarloForecaster, etc.

# Create forecast task (optional - for documentation/organization)
forecast_horizon = 12  # 12 months ahead
task = ts.ForecastTask(y=df, fh=forecast_horizon, frequency='MS')

# Use Exponential Moving Average forecaster
# This provides smooth forecasts similar to exponential smoothing
forecaster = ts.ExponentialMovingAverageForecaster(alpha=0.3)
forecaster.fit(df)

# Generate forecast
forecast_result = forecaster.predict(fh=forecast_horizon)

# Extract forecast values
forecast_series = forecast_result.y_pred
print("Forecast values:")
print(forecast_series.head())
print(f"\nForecast horizon: {len(forecast_series)} periods")

# Note: timesmith forecasters don't automatically provide confidence intervals
# For confidence intervals, you could use timesmith's bootstrap_confidence_intervals 
# or MonteCarloForecaster if available


# --- code cell ---

# plotsmith makes it easy to visualize forecasts
# plotsmith's plot_model_comparison compares actual data with model predictions
ps.plot_model_comparison(
    data=df,
    models={'EMA Forecast': forecast_series},
    title='Unemployment Rate Forecast with Exponential Moving Average',
    xlabel='Date',
    ylabel='Unemployment Rate (%)',
    figsize=(12, 6)
)


# --- code cell ---

# Now let's analyze Henry Hub Natural Gas Spot Price
df1_raw = web.DataReader('DHHNGSP', 'fred', start, end)
# FRED returns lowercase column names
col_name_gas = df1_raw.columns[0]
df1 = df1_raw[col_name_gas].dropna()

# Quick visualization with plotsmith
ps.plot_timeseries(
    df1,
    title='Henry Hub Natural Gas Spot Price',
    xlabel='Date',
    ylabel='Price (USD)',
    figsize=(12, 6)
)


# --- code cell ---

# Forecast using timesmith's Exponential Moving Average forecaster
forecast_horizon_gas = 12  # 12 months ahead
gas_forecaster = ts.ExponentialMovingAverageForecaster(alpha=0.3)
gas_forecaster.fit(df1)

forecast_gas_result = gas_forecaster.predict(fh=forecast_horizon_gas)
forecast_gas_series = forecast_gas_result.y_pred

# Visualize the forecast with plotsmith
ps.plot_model_comparison(
    data=df1,
    models={'EMA Forecast': forecast_gas_series},
    title='Henry Hub Natural Gas Spot Price Forecast',
    xlabel='Date',
    ylabel='Price (USD)',
    figsize=(12, 6)
)


# --- code cell ---

# timesmith also supports naive forecasting methods
# We can create a simple naive forecaster using SimpleMovingAverageForecaster with window=1
# This essentially uses the last value as the forecast

naive_forecaster = ts.SimpleMovingAverageForecaster(window=1)
naive_forecaster.fit(df1)
naive_forecast_result = naive_forecaster.predict(fh=10)
naive_forecast_series = naive_forecast_result.y_pred

# plotsmith can compare multiple forecasting methods using plot_model_comparison
# It compares actual data with multiple model predictions
ps.plot_model_comparison(
    data=df1,
    models={
        'EMA Forecast': forecast_gas_series,
        'Naive Forecast': naive_forecast_series
    },
    title='Natural Gas Price: Forecast Method Comparison',
    xlabel='Date',
    ylabel='Price (USD)',
    figsize=(14, 6)
)


# --- code cell ---

# timesmith provides RollingFeaturizer for technical indicators like Bollinger Bands
# RollingFeaturizer creates rolling window features (mean, std, etc.)
# We can use this to calculate Bollinger Bands components

# Create rolling features for Bollinger Bands calculation
# Window size 20, functions: mean and std
rolling_featurizer = ts.RollingFeaturizer(windows=[20], functions=['mean', 'std'])
rolling_featurizer.fit(df1)

# Transform to get rolling features
rolling_features = rolling_featurizer.transform(df1)

# Calculate Bollinger Bands manually using the rolling features
# BB = MA ± (2 * std)
df1_with_bb = df1.to_frame(name='price')
df1_with_bb['MA_20'] = rolling_features.iloc[:, 0]  # mean column
df1_with_bb['std_20'] = rolling_features.iloc[:, 1]  # std column
df1_with_bb['upper_band'] = df1_with_bb['MA_20'] + (2 * df1_with_bb['std_20'])
df1_with_bb['lower_band'] = df1_with_bb['MA_20'] - (2 * df1_with_bb['std_20'])

print("Bollinger Bands calculated:")
print(df1_with_bb[['price', 'MA_20', 'upper_band', 'lower_band']].tail(10))


# --- code cell ---

# plotsmith makes it easy to visualize Bollinger Bands
# We'll create a plot with the price, moving average, and bands

# Prepare data with bands for plotsmith
bands_data = pd.DataFrame({
    'price': df1_with_bb['price'],
    'MA': df1_with_bb['MA_20']
})

# Create bands dictionary for plotsmith (lower, upper tuples)
bands = {
    'Bollinger Bands': (
        df1_with_bb['lower_band'],
        df1_with_bb['upper_band']
    )
}

# Plot with bands using plotsmith
ps.plot_timeseries(
    bands_data,
    bands=bands,
    title='Henry Hub Natural Gas Spot Price with Bollinger Bands',
    xlabel='Date',
    ylabel='Price (USD)',
    figsize=(14, 6)
)


# --- code cell ---

# Analyze recent data with Bollinger Bands
start_recent = datetime.datetime(2024, 4, 1)
end_recent = datetime.datetime(2024, 10, 20)

df2_raw = web.DataReader('DHHNGSP', 'fred', start_recent, end_recent)
# FRED returns lowercase column names
col_name_gas2 = df2_raw.columns[0]
df2 = df2_raw[col_name_gas2].dropna().sort_index()

# Calculate Bollinger Bands for recent data
rolling_featurizer_recent = ts.RollingFeaturizer(windows=[20], functions=['mean', 'std'])
rolling_featurizer_recent.fit(df2)
rolling_features_recent = rolling_featurizer_recent.transform(df2)

df2_with_bb = df2.to_frame(name='price')
df2_with_bb['MA_20'] = rolling_features_recent.iloc[:, 0]
df2_with_bb['std_20'] = rolling_features_recent.iloc[:, 1]
df2_with_bb['upper_band'] = df2_with_bb['MA_20'] + (2 * df2_with_bb['std_20'])
df2_with_bb['lower_band'] = df2_with_bb['MA_20'] - (2 * df2_with_bb['std_20'])

# Visualize recent data with Bollinger Bands
bands_recent = {
    'Bollinger Bands': (
        df2_with_bb['lower_band'],
        df2_with_bb['upper_band']
    )
}

bands_data_recent = pd.DataFrame({
    'price': df2_with_bb['price'],
    'MA': df2_with_bb['MA_20']
})

ps.plot_timeseries(
    bands_data_recent,
    bands=bands_recent,
    title='Recent Natural Gas Price with Bollinger Bands (2024)',
    xlabel='Date',
    ylabel='Price (USD)',
    figsize=(14, 6)
)
