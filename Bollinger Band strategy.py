"""Generated from Jupyter notebook: Example: Get financial and non-financial stock data

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import pandas as pd
import requests
from pandas.io.json import json_normalize


def getdata(stock):
    # Company Quote Group of Items
    company_quote = requests.get(
        f"https://financialmodelingprep.com/api/v3/quote/{stock}"
    )
    company_quote = company_quote.json()
    share_price = float("{0:.2f}".format(company_quote[0]["price"]))

    # Balance Sheet Group of Items
    BS = requests.get(
        f"https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/{stock}?period=quarter"
    )
    BS = BS.json()

    # Total Cash
    cash = float(
        "{0:.2f}".format(
            float(BS["financials"][0]["Cash and short-term investments"]) / 10**9
        )
    )
    # Total Debt
    debt = float("{0:.2f}".format(float(BS["financials"][0]["Total debt"]) / 10**9))

    # Income Statement Group of Items
    IS = requests.get(
        f"https://financialmodelingprep.com/api/v3/financials/income-statement/{stock}?period=quarter"
    )
    IS = IS.json()

    # Most Recent Quarterly Revenue
    qRev = float("{0:.2f}".format(float(IS["financials"][0]["Revenue"]) / 10**9))

    # Company Profile Group of Items
    company_info = requests.get(
        f"https://financialmodelingprep.com/api/v3/company/profile/{stock}"
    )
    company_info = company_info.json()

    # Chief Executive Officer
    ceo = company_info["profile"]["ceo"]

    return (share_price, cash, debt, qRev, ceo)


tickers = (
    "AAPL",
    "MSFT",
    "GOOG",
    "T",
    "CSCO",
    "INTC",
    "ORCL",
    "AMZN",
    "FB",
    "TSLA",
    "NVDA",
)

data = map(getdata, tickers)

df = pd.DataFrame(
    data,
    columns=["Share Price", "Total Cash", "Total Debt", "Q3 2019 Revenue", "CEO"],
    index=tickers,
)

print(df)

writer = pd.ExcelWriter("example.xlsx")
df.to_excel(writer, "Statistics")
writer.save()


# --- code cell ---

# getting historical data for RDS-A. This code calls the API and transforms the result into a DataFrame.

ticker = "RDS-A"
target = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
rds = pd.read_json(target)
rds = json_normalize(rds["historical"])
rds["date"] = pd.to_datetime(rds["date"])
rds.set_index("date", inplace=True)


# --- code cell ---

rds.head()


# --- code cell ---


def bollinger_bands(
    df: pd.DataFrame = df, target_col: str = "adjClose"
) -> pd.DataFrame:
    """Calculates Bollinger Bands and returns an updated DataFrame.

    :param df: DataFrame
    :param target_col: column that will be used for the calcuations
    :type target_col: str

    :return: df with additional columns
    :rtype: pd.DataFrame
    """

    df["20 Day MA"] = df[target_col].rolling(20).mean()
    df["20 Day MA_lower bound"] = df["20 Day MA"] - df[target_col].rolling(20).std() * 2
    df["20 Day MA_upper bound"] = df["20 Day MA"] + df[target_col].rolling(20).std() * 2

    return df


def bb_plot(df: pd.DataFrame = df, target_col: str = "adjClose") -> plt:
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
        x, rds["20 Day MA_lower bound"], rds["20 Day MA_upper bound"], alpha=0.5
    )
    plt.plot(x, y)
    plt.title(f"Bollinger Bands for {ticker}")
    plt.xlabel("Date (Year/Month)")
    plt.ylabel("Price(USD)")
    plt.legend(y)
    plt.show()

    return plt


# --- code cell ---

rds = bollinger_bands(rds, "adjClose")
bb_plot(rds)


# --- code cell ---

import numpy as np

df = rds


def bb_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df["Position"] = None
    # Fill our newly created position column - set to sell (-1) when the price hits the upper band, and set to buy (1) when it hits the lower band
    for row in range(len(df)):
        if (df["adjClose"].iloc[row] > df["20 Day MA_upper bound"].iloc[row]) and (
            df["adjClose"].iloc[row - 1] < df["20 Day MA_upper bound"].iloc[row - 1]
        ):
            df["Position"].iloc[row] = -1

        if (df["adjClose"].iloc[row] < df["20 Day MA_lower bound"].iloc[row]) and (
            df["adjClose"].iloc[row - 1] > df["20 Day MA_lower bound"].iloc[row - 1]
        ):
            df["Position"].iloc[row] = 1

    # Forward fill our position column to replace the "None" values with the correct long/short positions to represent the "holding" of our position
    # forward through time
    df["Position"].fillna(method="ffill", inplace=True)

    # Calculate the daily market return and multiply that by the position to determine strategy returns
    df["Market Return"] = np.log(df["adjClose"] / df["adjClose"].shift(1))
    df["Strategy Return"] = df["Market Return"] * df["Position"]

    return df


# --- code cell ---

df = bb_strategy(df)


# --- code cell ---

df["Strategy Return"].cumsum().plot()


# --- code cell ---

# Monte Carlo valuation of European call option
# in Black-Scholes-Merton model
# bsm_mcs_euro.py
#
import numpy as np

# Parameter Values
S0 = rds["adjClose"][-1]  # initial index level
K = 35.0  # strike price
T = 1  # time-to-maturity
r = 0.05  # riskless short rate
sigma = np.std(rds["changeOverTime"])  # standard deviation of the percent change
I = 100000  # number of simulations
# Valuation Algorithm
z = np.random.standard_normal(I)  # pseudorandom numbers

ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
# index values at maturity
hT = np.maximum(ST - K, 0)  # inner values at maturity
C0 = np.exp(-r * T) * np.sum(hT) / I  # Monte Carlo estimator
# Result Output
print(f"Value of the Call Option {C0:.5f}")


# --- code cell ---

rds["changeOverTime_log"] = np.log(rds["changeOverTime"])


# --- code cell ---

# version 1 using Pandas for the plot
rds[["adjClose", "20 Day MA", "20 Day MA_lower bound", "20 Day MA_upper bound"]].plot()


# --- code cell ---

len(ST)


# --- code cell ---

# set style, empty figure and axes
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

# Get index values for the X axis
x_axis = rds.index

# Plot shaded 20 Day Bollinger Band
ax.fill_between(
    x_axis, rds["20 Day MA_lower bound"], rds["20 Day MA_upper bound"], color="grey"
)

# Plot Adjust Closing Price and Moving Averages
ax.plot(x_axis, rds["adjClose"], color="blue", lw=2)
ax.plot(x_axis, rds["20 Day MA"], color="black", lw=2)

# Set Title & Show the Image
ax.set_title("20 Day Bollinger Band For RDS-A")
ax.set_xlabel("Date (Year/Month)")
ax.set_ylabel("Price(USD)")
ax.legend()
plt.show()
