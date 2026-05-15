"""Generated from Jupyter notebook: Time series with fred unemployment and bollinger bands

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# ! pip install -q -r requirements.txt  # Jupyter-only


# --- code cell ---

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web

# %matplotlib inline  # Jupyter-only
from prophet import Prophet

# --- code cell ---

# !pip install -q pandas_datareader  # Jupyter-only


# --- code cell ---

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2025, 2, 20)

df = web.DataReader("unrate", "fred", start, end)
df.head()

df.plot()


df.reset_index(inplace=True)
df.columns = ["ds", "y"]
df.dropna(inplace=True)
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12)
future.tail()

future = model.make_future_dataframe(periods=12, freq="MS")
fcst = model.predict(future)
fig = model.plot(fcst)

forecast = model.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()


# --- code cell ---

import plotly.graph_objs as go


def timeseries(df, x, yhat, lower, upper, actual, save=False):

    fig = go.Figure(
        [
            go.Scatter(
                name="Measurement",
                x=df[x],
                y=df["yhat"],
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
                showlegend=False,
            ),
            go.Scatter(
                name="Upper Bound",
                x=df[x],
                y=df[upper],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Lower Bound",
                x=df[x],
                y=df[lower],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        yaxis_title="Unemployment Rate",
        title="Unemployment rate estimate using Prophet Forecast",
        hovermode="x",
    )
    fig.add_trace(
        go.Scatter(
            x=actual["ds"],
            y=actual["y"],
            mode="lines+markers",
            name="Actual values",
            showlegend=False,
        )
    )
    fig.show()

    if save:
        fig.write_html("unemployment rate.html")


timeseries(forecast, "ds", "yhat", "yhat_lower", "yhat_upper", actual=df, save=True)
df1 = web.DataReader("DHHNGSP", "fred", start, end)
df1.plot()
df1.reset_index(inplace=True)

df1.columns = ["ds", "y"]

m = Prophet()
m.fit(df1)
future = m.make_future_dataframe(periods=12, freq="MS")
fcst = m.predict(future)
fig = m.plot(fcst)
df1.set_index("ds", inplace=True)


def timeseries_trad(df, y, periods=10, save=False):
    shift = df.shift(periods=periods, freq="M")
    fig = go.Figure(
        [
            go.Scatter(
                name="Actual values",
                x=df.index,
                y=df[y],
                mode="lines+markers",
                showlegend=False,
            ),
            go.Scatter(
                name="Naive Forecast",
                x=shift.index,
                y=shift[y],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=1),
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        yaxis_title="price",
        title=f"Henry Hub Natural Gas Spot Price using Prophet Forecast with shift of {periods}",
        hovermode="x",
    )

    fig.show()

    if save:
        fig.write_html("unemployment rate.html")


timeseries_trad(df1, "y", periods=10, save=True)


# --- code cell ---


def bollinger_bands(
    df, drop: bool = True, target_col: str = "adjClose"
) -> pd.DataFrame:
    """Calculates Bollinger Bands and returns an updated DataFrame.

    :param df: DataFrame
    :param target_col: column that will be used for the calcuations
    :type target_col: str

    :return df: df with additional columns
    :rtype df: pd.DataFrame
    """
    if drop:
        df.dropna(inplace=True)

    df["20 Day MA"] = df[target_col].rolling(20).mean()
    df["20 Day MA_lower bound"] = df["20 Day MA"] - df[target_col].rolling(20).std() * 2
    df["20 Day MA_upper bound"] = df["20 Day MA"] + df[target_col].rolling(20).std() * 2

    return df


def bb_plot(df: pd.DataFrame = df, target_col: str = "adjClose"):
    """Calculates time series plot with Bollinger Bands

    :param df: DataFrame
    :param target_col: column that will be used for the calcuations
    :type target_col: str

    :return: plot
    :rtype: matplotlib.pyplot
    """

    x = df.index
    y = df[["adjClose", "20 Day MA", "20 Day MA_lower bound", "20 Day MA_upper bound"]]

    plt.fill_between(
        x, df["20 Day MA_lower bound"], df["20 Day MA_upper bound"], alpha=0.5
    )
    plt.plot(x, y)
    plt.title(f"Bollinger Bands for {target_col}")
    plt.xlabel("Date (Year/Month)")
    plt.ylabel("Price(USD)")
    plt.legend(y)
    plt.show()

    return plt


# --- code cell ---

start = datetime.datetime(2024, 4, 1)

end = datetime.datetime(2024, 10, 20)

df2 = web.DataReader("DHHNGSP", "fred", start, end)

df2["adjClose"] = df2["DHHNGSP"]
df2 = bollinger_bands(df2.sort_values(by="DATE"))
bb_plot(df2)


# --- code cell ---

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(filepath, date_column=None, parse_dates=True):
    """
    Load time series data from CSV, Parquet, or JSON using a dictionary-based dispatch.

    Parameters:
    - filepath (str): Path to the file.
    - date_column (str, optional): Column to parse as datetime. If None, tries to auto-detect.
    - parse_dates (bool): Whether to parse dates.

    Returns:
    - pd.DataFrame: Loaded DataFrame with parsed datetime column.
    """
    file_ext = os.path.splitext(filepath)[-1].lower()

    # Define file loaders
    loaders = {
        ".csv": lambda: pd.read_csv(
            filepath, parse_dates=[date_column] if date_column else parse_dates
        ),
        ".parquet": lambda: pd.read_parquet(filepath),
        ".json": lambda: pd.read_json(filepath),
    }

    # Load data based on file extension
    if file_ext in loaders:
        df = loaders[file_ext]()
    else:
        raise ValueError("Unsupported file format. Use CSV, Parquet, or JSON.")

    # Convert date column if specified
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])

    # Auto-detect date column if none is specified
    if date_column is None:
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]":
                date_column = col
                break

    # Sort by date if detected
    if date_column and date_column in df.columns:
        df = df.sort_values(by=date_column).reset_index(drop=True)

    return df


def set_plot_style(ax, df, time_column, value_columns):
    """
    Configures X and Y axes for time series plots.
    - X-axis uses 50-year intervals (e.g., 1800, 1850, 1900, ...)
    - Y-axis labels mean, 20%, and 80% dynamically.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    # X-Axis: Use 50-year spacing, ensuring the final year is included
    years = df[time_column].dt.year
    x_min, x_max = years.min(), years.max()
    x_start = (x_min // 50) * 50
    x_ticks = np.arange(x_start, x_max + 1, 50)
    if x_max not in x_ticks:
        x_ticks = np.append(x_ticks, x_max)

    ax.set_xticks(x_ticks)
    ax.set_xlim(x_start, x_max)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    # Y-Axis: Compute mean, 20%, and 80% dynamically
    all_values = np.concatenate([df[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels([f"{y_20:.2e}", f"{y_mean:.2e}", f"{y_80:.2e}"])

    plt.rcParams.update(
        {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
    )


def plot_time_series(
    df, time_column=None, value_columns=None, title=None, filename=None
):
    """
    Standardized time series plotting function with labels at the end of lines.
    """
    if time_column is None:
        time_column = next(
            (col for col in df.columns if df[col].dtype == "datetime64[ns]"), None
        )
    if time_column is None:
        raise ValueError("No datetime column found.")

    if isinstance(value_columns, str):
        value_columns = [value_columns]
    if value_columns is None:
        value_columns = df.select_dtypes(include="number").columns.tolist()
    if not value_columns:
        raise ValueError("No numeric columns found to plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set1.colors
    for i, col in enumerate(value_columns):
        ax.plot(
            df[time_column].dt.year, df[col], linewidth=2, color=colors[i % len(colors)]
        )
        last_x = (
            df[time_column].dt.year.iloc[-1]
            + (df[time_column].dt.year.max() - df[time_column].dt.year.min()) * 0.02
        )
        last_y = df[col].iloc[-1]
        ax.text(
            last_x,
            last_y,
            col,
            fontsize=12,
            color=colors[i % len(colors)],
            verticalalignment="center",
        )

    set_plot_style(ax, df, time_column, value_columns)

    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# --- code cell ---

plot_time_series(df3)


# --- code cell ---

df3 = df2.reset_index()


# --- code cell ---

df3 = df3[["DATE", "DHHNGSP"]]


# --- code cell ---

# Required Libraries
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pandas_datareader import data as web
from prophet import Prophet

# Set Global Matplotlib Style
plt.rcParams.update(
    {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
)


# Load Data Function
def load_data(filepath, date_column=None, parse_dates=True):
    file_ext = os.path.splitext(filepath)[-1].lower()

    loaders = {
        ".csv": lambda: pd.read_csv(
            filepath, parse_dates=[date_column] if date_column else parse_dates
        ),
        ".parquet": lambda: pd.read_parquet(filepath),
        ".json": lambda: pd.read_json(filepath),
    }

    if file_ext in loaders:
        df = loaders[file_ext]()
    else:
        raise ValueError("Unsupported file format. Use CSV, Parquet, or JSON.")

    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])

    if date_column is None:
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]":
                date_column = col
                break

    if date_column and date_column in df.columns:
        df = df.sort_values(by=date_column).reset_index(drop=True)

    return df


# Set Plot Style Function
def set_plot_style(ax, df, time_column, value_columns):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    # X-Axis: Use 50-year intervals
    years = pd.to_datetime(df[time_column]).dt.year
    x_min, x_max = years.min(), years.max()
    x_start = (x_min // 50) * 50
    x_ticks = np.arange(x_start, x_max + 1, 50)
    if x_max not in x_ticks:
        x_ticks = np.append(x_ticks, x_max)

    ax.set_xticks(x_ticks)
    ax.set_xlim(x_start, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Y-Axis: Compute mean, 20%, and 80% dynamically
    all_values = np.concatenate([df[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels([f"{y_20:.2e}", f"{y_mean:.2e}", f"{y_80:.2e}"])


# Plot Time Series Function
def plot_time_series(
    df, time_column=None, value_columns=None, title=None, filename=None
):
    if time_column is None:
        time_column = next(
            (col for col in df.columns if df[col].dtype == "datetime64[ns]"), None
        )
    if time_column is None:
        raise ValueError("No datetime column found.")

    if isinstance(value_columns, str):
        value_columns = [value_columns]
    if value_columns is None:
        value_columns = df.select_dtypes(include="number").columns.tolist()
    if not value_columns:
        raise ValueError("No numeric columns found to plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Greys(np.linspace(0.2, 0.8, len(value_columns)))
    for i, col in enumerate(value_columns):
        ax.plot(df[time_column].dt.year, df[col], linewidth=2, color=colors[i])
        last_x = (
            df[time_column].dt.year.iloc[-1]
            + (df[time_column].dt.year.max() - df[time_column].dt.year.min()) * 0.02
        )
        last_y = df[col].iloc[-1]
        ax.text(
            last_x,
            last_y,
            col,
            fontsize=12,
            color=colors[i],
            verticalalignment="center",
        )

    set_plot_style(ax, df, time_column, value_columns)

    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# Load Unemployment Rate Data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2025, 2, 20)
df = web.DataReader("UNRATE", "fred", start, end)
df.reset_index(inplace=True)
df.columns = ["ds", "y"]
df.dropna(inplace=True)

# Prophet Forecasting Model
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12, freq="MS")
fcst = model.predict(future)

# Plotting Prophet Forecast
fig = model.plot(fcst)
plt.title("Unemployment Rate Forecast using Prophet")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate")
plt.savefig("prophet_unemployment_forecast.png", dpi=300, bbox_inches="tight")
plt.show()


# Bollinger Bands Function
def bollinger_bands(df, drop=True, target_col="adjClose") -> pd.DataFrame:
    if drop:
        df.dropna(inplace=True)

    df["20 Day MA"] = df[target_col].rolling(20).mean()
    df["20 Day MA_lower bound"] = df["20 Day MA"] - df[target_col].rolling(20).std() * 2
    df["20 Day MA_upper bound"] = df["20 Day MA"] + df[target_col].rolling(20).std() * 2

    return df


# Bollinger Bands Plot Function
def bb_plot(df: pd.DataFrame, target_col: str = "adjClose"):
    # Fix: Explicitly set index to 'DATE'
    if df.index.name is None or df.index.name != "DATE":
        df = df.reset_index().rename(columns={"index": "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.set_index("DATE", inplace=True)

    x = df.index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(
        x,
        df["20 Day MA_lower bound"],
        df["20 Day MA_upper bound"],
        alpha=0.5,
        color="grey",
    )
    ax.plot(x, df["adjClose"], color="black", label="Adj Close")
    ax.plot(x, df["20 Day MA"], color="blue", label="20 Day MA")
    ax.set_title("Bollinger Bands")
    ax.set_xlabel("Date (Year/Month)")
    ax.set_ylabel("Price (USD)")
    set_plot_style(
        ax,
        df.reset_index(),
        "DATE",
        ["adjClose", "20 Day MA", "20 Day MA_lower bound", "20 Day MA_upper bound"],
    )
    ax.legend()

    plt.savefig("bollinger_bands.png", dpi=300, bbox_inches="tight")
    plt.show()


# Load Natural Gas Spot Price Data
start = datetime.datetime(2024, 4, 1)
end = datetime.datetime(2024, 10, 20)
df2 = web.DataReader("DHHNGSP", "fred", start, end)
df2["adjClose"] = df2["DHHNGSP"]
df2 = bollinger_bands(df2.sort_values(by="DATE"))
bb_plot(df2)


# --- code cell ---

# Required Libraries
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pandas_datareader import data as web
from prophet import Prophet

# Set Global Matplotlib Style
plt.rcParams.update(
    {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
)


# Function to Add Captions to Graphs
def add_caption(ax, topic, start_date, end_date, num_obs):
    caption = f"Graph of {topic} from {start_date} to {end_date} containing {num_obs} observations."
    fig = ax.get_figure()
    fig.text(0.5, -0.05, caption, ha="center", fontsize=10, fontstyle="italic")


# Set Plot Style Function
def set_plot_style(ax, df, time_column, value_columns):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    # X-Axis: Use 50-year intervals
    years = pd.to_datetime(df[time_column]).dt.year
    x_min, x_max = years.min(), years.max()
    x_start = (x_min // 50) * 50
    x_ticks = np.arange(x_start, x_max + 1, 50)
    if x_max not in x_ticks:
        x_ticks = np.append(x_ticks, x_max)

    ax.set_xticks(x_ticks)
    ax.set_xlim(x_start, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Y-Axis: Compute mean, 20%, and 80% dynamically
    all_values = np.concatenate([df[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels([f"{y_20:.2e}", f"{y_mean:.2e}", f"{y_80:.2e}"])


# Plot Time Series Function
def plot_time_series(
    df, time_column=None, value_columns=None, title=None, filename=None
):
    if time_column is None:
        time_column = next(
            (col for col in df.columns if df[col].dtype == "datetime64[ns]"), None
        )
    if time_column is None:
        raise ValueError("No datetime column found.")

    if isinstance(value_columns, str):
        value_columns = [value_columns]
    if value_columns is None:
        value_columns = df.select_dtypes(include="number").columns.tolist()
    if not value_columns:
        raise ValueError("No numeric columns found to plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Greys(np.linspace(0.2, 0.8, len(value_columns)))
    for i, col in enumerate(value_columns):
        ax.plot(df[time_column].dt.year, df[col], linewidth=2, color=colors[i])
        last_x = (
            df[time_column].dt.year.iloc[-1]
            + (df[time_column].dt.year.max() - df[time_column].dt.year.min()) * 0.02
        )
        last_y = df[col].iloc[-1]
        ax.text(
            last_x,
            last_y,
            col,
            fontsize=12,
            color=colors[i],
            verticalalignment="center",
        )

    set_plot_style(ax, df, time_column, value_columns)

    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)

    # Add Caption
    start_date = df[time_column].min().strftime("%Y-%m-%d")
    end_date = df[time_column].max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, title, start_date, end_date, num_obs)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# Load Unemployment Rate Data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2025, 2, 20)
df = web.DataReader("UNRATE", "fred", start, end)

# Fix: Reset index and rename correctly for FRED data
df.reset_index(inplace=True)
df.columns = ["DATE", "y"]
df["DATE"] = pd.to_datetime(df["DATE"])
df.sort_values(by="DATE", inplace=True)

# Prophet Forecasting Model
model = Prophet()
model.fit(df.rename(columns={"DATE": "ds", "y": "y"}))

future = model.make_future_dataframe(periods=12, freq="MS")
fcst = model.predict(future)

# Plotting Prophet Forecast
fig = model.plot(fcst)
plt.title("Unemployment Rate Forecast using Prophet")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate")

# Add Caption
add_caption(
    plt.gca(),
    "Unemployment Rate Forecast",
    start.strftime("%Y-%m-%d"),
    end.strftime("%Y-%m-%d"),
    len(df),
)

plt.savefig("prophet_unemployment_forecast.png", dpi=300, bbox_inches="tight")
plt.show()


# Bollinger Bands Function
def bollinger_bands(df, drop=True, target_col="adjClose") -> pd.DataFrame:
    if drop:
        df.dropna(inplace=True)

    df["20 Day MA"] = df[target_col].rolling(20).mean()
    df["20 Day MA_lower bound"] = df["20 Day MA"] - df[target_col].rolling(20).std() * 2
    df["20 Day MA_upper bound"] = df["20 Day MA"] + df[target_col].rolling(20).std() * 2

    return df


# Bollinger Bands Plot Function
def bb_plot(df: pd.DataFrame, target_col: str = "adjClose"):
    x = df.index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(
        x,
        df["20 Day MA_lower bound"],
        df["20 Day MA_upper bound"],
        alpha=0.5,
        color="grey",
    )
    ax.plot(x, df["adjClose"], color="black", label="Adj Close")
    ax.plot(x, df["20 Day MA"], color="blue", label="20 Day MA")
    ax.set_title("Bollinger Bands")
    ax.set_xlabel("Date (Year/Month)")
    ax.set_ylabel("Price (USD)")
    set_plot_style(
        ax,
        df.reset_index(),
        "DATE",
        ["adjClose", "20 Day MA", "20 Day MA_lower bound", "20 Day MA_upper bound"],
    )
    ax.legend()

    # Add Caption
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, "Bollinger Bands", start_date, end_date, num_obs)

    plt.savefig("bollinger_bands.png", dpi=300, bbox_inches="tight")
    plt.show()


# Load Natural Gas Spot Price Data
start = datetime.datetime(2024, 4, 1)
end = datetime.datetime(2024, 10, 20)
df2 = web.DataReader("DHHNGSP", "fred", start, end)

# Fix: Correctly reset index and rename for FRED data
df2.reset_index(inplace=True)
df2.rename(columns={"DATE": "DATE"}, inplace=True)
df2["adjClose"] = df2["DHHNGSP"]
df2 = bollinger_bands(df2.sort_values(by="DATE"))
df2.set_index("DATE", inplace=True)
bb_plot(df2)


# --- code cell ---

# Required Libraries
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pandas_datareader import data as web
from prophet import Prophet

# Set Global Matplotlib Style
plt.rcParams.update(
    {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
)


# Function to Add Captions to Graphs
def add_caption(ax, topic, start_date, end_date, num_obs):
    caption = f"Graph of {topic} from {start_date} to {end_date} containing {num_obs} observations."
    fig = ax.get_figure()
    fig.text(0.5, -0.05, caption, ha="center", fontsize=10, fontstyle="italic")


# Set Plot Style Function
def set_plot_style(ax, df, time_column, value_columns):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    # X-Axis: Use 50-year intervals
    years = pd.to_datetime(df[time_column]).dt.year
    x_min, x_max = years.min(), years.max()
    x_start = (x_min // 50) * 50
    x_ticks = np.arange(x_start, x_max + 1, 50)
    if x_max not in x_ticks:
        x_ticks = np.append(x_ticks, x_max)

    ax.set_xticks(x_ticks)
    ax.set_xlim(x_start, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Y-Axis: Compute mean, 20%, and 80% dynamically
    all_values = np.concatenate([df[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels([f"{y_20:.2e}", f"{y_mean:.2e}", f"{y_80:.2e}"])


# Bollinger Bands Function
def bollinger_bands(df, drop=True, target_col="adjClose") -> pd.DataFrame:
    if drop:
        df.dropna(inplace=True)

    df["20 Day MA"] = df[target_col].rolling(20).mean()
    df["20 Day MA_lower bound"] = df["20 Day MA"] - df[target_col].rolling(20).std() * 2
    df["20 Day MA_upper bound"] = df["20 Day MA"] + df[target_col].rolling(20).std() * 2

    return df


# Bollinger Bands Plot Function
def bb_plot(df: pd.DataFrame, target_col: str = "adjClose"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(
        df.index,
        df["20 Day MA_lower bound"],
        df["20 Day MA_upper bound"],
        alpha=0.5,
        color="grey",
    )
    ax.plot(df.index, df["adjClose"], color="black", label="Adj Close")
    ax.plot(df.index, df["20 Day MA"], color="blue", label="20 Day MA")
    ax.set_title("Bollinger Bands")
    ax.set_xlabel("Date (Year/Month)")
    ax.set_ylabel("Price (USD)")
    set_plot_style(
        ax,
        df.reset_index(),
        "DATE",
        ["adjClose", "20 Day MA", "20 Day MA_lower bound", "20 Day MA_upper bound"],
    )
    ax.legend()

    # Add Caption
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, "Bollinger Bands", start_date, end_date, num_obs)

    plt.savefig("bollinger_bands.png", dpi=300, bbox_inches="tight")
    plt.show()


# Load Natural Gas Spot Price Data
start = datetime.datetime(2024, 4, 1)
end = datetime.datetime(2024, 10, 20)
df2 = web.DataReader("DHHNGSP", "fred", start, end)

# Fix: Correctly reset index and rename for FRED data
df2.reset_index(inplace=True)
df2["DATE"] = pd.to_datetime(df2["DATE"])
df2.set_index("DATE", inplace=True)
df2["adjClose"] = df2["DHHNGSP"]
df2 = bollinger_bands(df2)

# Plot Bollinger Bands
bb_plot(df2)


# --- code cell ---

# Required Libraries
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pandas_datareader import data as web

# Set Global Matplotlib Style
plt.rcParams.update(
    {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
)


# Function to Add Captions to Graphs
def add_caption(ax, topic, start_date, end_date, num_obs):
    caption = f"Graph of {topic} from {start_date} to {end_date} containing {num_obs} observations."
    fig = ax.get_figure()
    fig.text(0.5, -0.1, caption, ha="center", fontsize=10, fontstyle="italic")


# Set Plot Style Function
def set_plot_style(ax, df, time_column, value_columns):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    # X-Axis: Use 50-year intervals
    years = pd.to_datetime(df[time_column]).dt.year
    x_min, x_max = years.min(), years.max()
    x_start = (x_min // 50) * 50
    x_ticks = np.arange(x_start, x_max + 1, 50)
    if x_max not in x_ticks:
        x_ticks = np.append(x_ticks, x_max)

    ax.set_xticks(x_ticks)
    ax.set_xlim(x_start, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Y-Axis: Compute mean, 20%, and 80% dynamically
    all_values = np.concatenate([df[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels([f"{y_20:.2e}", f"{y_mean:.2e}", f"{y_80:.2e}"])


# Basic Time Series Plot Function
def basic_time_series_plot(df, time_column, value_column, title=None, filename=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[time_column], df[value_column], color="black", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment Rate")
    ax.set_title(title)
    set_plot_style(ax, df, time_column, [value_column])

    # Add Caption
    start_date = df[time_column].min().strftime("%Y-%m-%d")
    end_date = df[time_column].max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, title, start_date, end_date, num_obs)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# Load Unemployment Data
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2025, 2, 20)
df_unemployment = web.DataReader("UNRATE", "fred", start, end)

# Fix: Reset index and rename for FRED data
df_unemployment.reset_index(inplace=True)
df_unemployment.columns = ["DATE", "Unemployment Rate"]
df_unemployment["DATE"] = pd.to_datetime(df_unemployment["DATE"])

# Plotting the Unemployment Rate
basic_time_series_plot(
    df_unemployment,
    "DATE",
    "Unemployment Rate",
    title="Unemployment Rate (2024)",
    filename="unemployment_rate.png",
)


# --- code cell ---

df_unemployment.head()


# --- code cell ---

# Required Libraries
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator
from pandas_datareader import data as web

# Set Global Matplotlib Style
plt.rcParams.update(
    {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
)


# Function to Add Captions to Graphs
def add_caption(ax, topic, start_date, end_date, num_obs):
    caption = f"Graph of {topic} from {start_date} to {end_date} containing {num_obs} observations."
    fig = ax.get_figure()
    fig.text(0.5, -0.1, caption, ha="center", fontsize=10, fontstyle="italic")


# Set Plot Style Function
def set_plot_style(ax, df, time_column, value_columns):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    # X-Axis: Use 5-year intervals
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.set_xlim(df[time_column].min(), df[time_column].max())
    plt.xticks(rotation=45)

    # Y-Axis: Compute mean, 20%, and 80% dynamically
    all_values = np.concatenate([df[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels([f"{y_20:.2f}", f"{y_mean:.2f}", f"{y_80:.2f}"])


# Basic Time Series Plot Function with Peak Label
def basic_time_series_plot(df, time_column, value_column, title=None, filename=None):
    # Ensure datetime format for the time column
    df[time_column] = pd.to_datetime(df[time_column])

    # Drop NaN values to prevent plotting issues
    df = df.dropna(subset=[time_column, value_column])

    # Find peak value and corresponding date
    peak_value = df[value_column].max()
    peak_date = df.loc[df[value_column] == peak_value, time_column].values[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[time_column], df[value_column], color="black", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment Rate")
    ax.set_title(title)
    set_plot_style(ax, df, time_column, [value_column])

    # Add label for peak value
    ax.annotate(
        f"Peak: {peak_value:.2f}",
        xy=(pd.to_datetime(peak_date), peak_value),
        xytext=(pd.to_datetime(peak_date), peak_value + 0.5),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
        color="black",
    )

    # Add Caption
    start_date = df[time_column].min().strftime("%Y-%m-%d")
    end_date = df[time_column].max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, title, start_date, end_date, num_obs)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# Load Unemployment Data
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2025, 1, 1)
df_unemployment = web.DataReader("UNRATE", "fred", start, end)

# Fix: Reset index and rename for FRED data
df_unemployment.reset_index(inplace=True)
df_unemployment.columns = ["DATE", "Unemployment Rate"]
df_unemployment["DATE"] = pd.to_datetime(df_unemployment["DATE"])

# Plotting the Unemployment Rate
basic_time_series_plot(
    df_unemployment,
    "DATE",
    "Unemployment Rate",
    title="Unemployment Rate (2015-2025)",
    filename="unemployment_rate.png",
)
