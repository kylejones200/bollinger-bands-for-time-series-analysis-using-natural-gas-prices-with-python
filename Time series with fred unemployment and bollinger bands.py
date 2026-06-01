"""Generated from Jupyter notebook: Time series with fred unemployment and bollinger bands

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objs as go
from matplotlib.dates import DateFormatter, YearLocator
from prophet import Prophet


def add_caption(ax, topic, start_date, end_date, num_obs):
    caption = f"Graph of {topic} from {start_date} to {end_date} containing {num_obs} observations."
    fig = ax.get_figure()
    fig.text(0.5, -0.1, caption, ha="center", fontsize=10, fontstyle="italic")


def basic_time_series_plot(df, time_column, value_column, title=None, filename=None):
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.dropna(subset=[time_column, value_column])
    peak_value = df[value_column].max()
    peak_date = df.loc[df[value_column] == peak_value, time_column].values[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[time_column], df[value_column], color="black", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment Rate")
    ax.set_title(title)
    set_plot_style(ax, df, time_column, [value_column])
    ax.annotate(
        f"Peak: {peak_value:.2f}",
        xy=(pd.to_datetime(peak_date), peak_value),
        xytext=(pd.to_datetime(peak_date), peak_value + 0.5),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
        color="black",
    )
    start_date = df[time_column].min().strftime("%Y-%m-%d")
    end_date = df[time_column].max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, title, start_date, end_date, num_obs)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


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
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, "Bollinger Bands", start_date, end_date, num_obs)
    plt.savefig("bollinger_bands.png", dpi=300, bbox_inches="tight")
    plt.show()


def bollinger_bands(df, drop=True, target_col="adjClose") -> pd.DataFrame:
    if drop:
        df.dropna(inplace=True)
    df["20 Day MA"] = df[target_col].rolling(20).mean()
    df["20 Day MA_lower bound"] = df["20 Day MA"] - df[target_col].rolling(20).std() * 2
    df["20 Day MA_upper bound"] = df["20 Day MA"] + df[target_col].rolling(20).std() * 2
    return df


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
    start_date = df[time_column].min().strftime("%Y-%m-%d")
    end_date = df[time_column].max().strftime("%Y-%m-%d")
    num_obs = len(df)
    add_caption(ax, title, start_date, end_date, num_obs)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def set_plot_style(ax, df, time_column, value_columns):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.set_xlim(df[time_column].min(), df[time_column].max())
    plt.xticks(rotation=45)
    all_values = np.concatenate([df[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels([f"{y_20:.2f}", f"{y_mean:.2f}", f"{y_80:.2f}"])


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


def notebook_step_004() -> None:
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
    model.plot(fcst)
    forecast = model.predict(future)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()


def notebook_step_005() -> None:
    timeseries(forecast, "ds", "yhat", "yhat_lower", "yhat_upper", actual=df, save=True)
    df1 = web.DataReader("DHHNGSP", "fred", start, end)
    df1.plot()
    df1.reset_index(inplace=True)
    df1.columns = ["ds", "y"]
    m = Prophet()
    m.fit(df1)
    future = m.make_future_dataframe(periods=12, freq="MS")
    fcst = m.predict(future)
    m.plot(fcst)
    df1.set_index("ds", inplace=True)
    timeseries_trad(df1, "y", periods=10, save=True)


def notebook_step_007() -> None:
    start = datetime.datetime(2024, 4, 1)
    end = datetime.datetime(2024, 10, 20)
    df2 = web.DataReader("DHHNGSP", "fred", start, end)
    df2["adjClose"] = df2["DHHNGSP"]
    df2 = bollinger_bands(df2.sort_values(by="DATE"))
    bb_plot(df2)


def notebook_step_009() -> None:
    plot_time_series(df3)


def notebook_step_010() -> None:
    df2.reset_index()


def notebook_step_011() -> None:
    df3[["DATE", "DHHNGSP"]]


def required_libraries() -> None:
    plt.rcParams.update(
        {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
    )
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2025, 2, 20)
    df = web.DataReader("UNRATE", "fred", start, end)
    df.reset_index(inplace=True)
    df.columns = ["ds", "y"]
    df.dropna(inplace=True)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=12, freq="MS")
    fcst = model.predict(future)
    model.plot(fcst)
    plt.title("Unemployment Rate Forecast using Prophet")
    plt.xlabel("Year")
    plt.ylabel("Unemployment Rate")
    plt.savefig("prophet_unemployment_forecast.png", dpi=300, bbox_inches="tight")
    plt.show()
    start = datetime.datetime(2024, 4, 1)
    end = datetime.datetime(2024, 10, 20)
    df2 = web.DataReader("DHHNGSP", "fred", start, end)
    df2["adjClose"] = df2["DHHNGSP"]
    df2 = bollinger_bands(df2.sort_values(by="DATE"))
    bb_plot(df2)


def required_libraries_2() -> None:
    plt.rcParams.update(
        {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
    )
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2025, 2, 20)
    df = web.DataReader("UNRATE", "fred", start, end)
    df.reset_index(inplace=True)
    df.columns = ["DATE", "y"]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.sort_values(by="DATE", inplace=True)
    model = Prophet()
    model.fit(df.rename(columns={"DATE": "ds", "y": "y"}))
    future = model.make_future_dataframe(periods=12, freq="MS")
    fcst = model.predict(future)
    model.plot(fcst)
    plt.title("Unemployment Rate Forecast using Prophet")
    plt.xlabel("Year")
    plt.ylabel("Unemployment Rate")
    add_caption(
        plt.gca(),
        "Unemployment Rate Forecast",
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        len(df),
    )
    plt.savefig("prophet_unemployment_forecast.png", dpi=300, bbox_inches="tight")
    plt.show()
    start = datetime.datetime(2024, 4, 1)
    end = datetime.datetime(2024, 10, 20)
    df2 = web.DataReader("DHHNGSP", "fred", start, end)
    df2.reset_index(inplace=True)
    df2.rename(columns={"DATE": "DATE"}, inplace=True)
    df2["adjClose"] = df2["DHHNGSP"]
    df2 = bollinger_bands(df2.sort_values(by="DATE"))
    df2.set_index("DATE", inplace=True)
    bb_plot(df2)


def required_libraries_3() -> None:
    plt.rcParams.update(
        {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
    )
    start = datetime.datetime(2024, 4, 1)
    end = datetime.datetime(2024, 10, 20)
    df2 = web.DataReader("DHHNGSP", "fred", start, end)
    df2.reset_index(inplace=True)
    df2["DATE"] = pd.to_datetime(df2["DATE"])
    df2.set_index("DATE", inplace=True)
    df2["adjClose"] = df2["DHHNGSP"]
    df2 = bollinger_bands(df2)
    bb_plot(df2)


def required_libraries_4() -> None:
    plt.rcParams.update(
        {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
    )
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2025, 2, 20)
    df_unemployment = web.DataReader("UNRATE", "fred", start, end)
    df_unemployment.reset_index(inplace=True)
    df_unemployment.columns = ["DATE", "Unemployment Rate"]
    df_unemployment["DATE"] = pd.to_datetime(df_unemployment["DATE"])
    basic_time_series_plot(
        df_unemployment,
        "DATE",
        "Unemployment Rate",
        title="Unemployment Rate (2024)",
        filename="unemployment_rate.png",
    )


def notebook_step_016() -> None:
    df_unemployment.head()


def required_libraries_5() -> None:
    plt.rcParams.update(
        {"font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 14}
    )
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2025, 1, 1)
    df_unemployment = web.DataReader("UNRATE", "fred", start, end)
    df_unemployment.reset_index(inplace=True)
    df_unemployment.columns = ["DATE", "Unemployment Rate"]
    df_unemployment["DATE"] = pd.to_datetime(df_unemployment["DATE"])
    basic_time_series_plot(
        df_unemployment,
        "DATE",
        "Unemployment Rate",
        title="Unemployment Rate (2015-2025)",
        filename="unemployment_rate.png",
    )


def main() -> None:
    notebook_step_004()
    notebook_step_005()
    notebook_step_007()
    notebook_step_009()
    notebook_step_010()
    notebook_step_011()
    required_libraries()
    required_libraries_2()
    required_libraries_3()
    required_libraries_4()
    notebook_step_016()
    required_libraries_5()


if __name__ == "__main__":
    main()
