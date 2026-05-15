"""Generated from Jupyter notebook: Unemployment, bollinger band, time series

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

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2024, 10, 1)

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
