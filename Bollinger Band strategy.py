"""Generated from Jupyter notebook: Example: Get financial and non-financial stock data

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pandas.io.json import json_normalize


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


def bb_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df["Position"] = None
    for row in range(len(df)):
        if (
            df["adjClose"].iloc[row] > df["20 Day MA_upper bound"].iloc[row]
            and df["adjClose"].iloc[row - 1] < df["20 Day MA_upper bound"].iloc[row - 1]
        ):
            df["Position"].iloc[row] = -1
        if (
            df["adjClose"].iloc[row] < df["20 Day MA_lower bound"].iloc[row]
            and df["adjClose"].iloc[row - 1] > df["20 Day MA_lower bound"].iloc[row - 1]
        ):
            df["Position"].iloc[row] = 1
    df["Position"].fillna(method="ffill", inplace=True)
    df["Market Return"] = np.log(df["adjClose"] / df["adjClose"].shift(1))
    df["Strategy Return"] = df["Market Return"] * df["Position"]
    return df


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


def getdata(stock):
    company_quote = requests.get(
        f"https://financialmodelingprep.com/api/v3/quote/{stock}"
    )
    company_quote = company_quote.json()
    share_price = float("{0:.2f}".format(company_quote[0]["price"]))
    BS = requests.get(
        f"https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/{stock}?period=quarter"
    )
    BS = BS.json()
    cash = float(
        "{0:.2f}".format(
            float(BS["financials"][0]["Cash and short-term investments"]) / 10**9
        )
    )
    debt = float("{0:.2f}".format(float(BS["financials"][0]["Total debt"]) / 10**9))
    IS = requests.get(
        f"https://financialmodelingprep.com/api/v3/financials/income-statement/{stock}?period=quarter"
    )
    IS = IS.json()
    qRev = float("{0:.2f}".format(float(IS["financials"][0]["Revenue"]) / 10**9))
    company_info = requests.get(
        f"https://financialmodelingprep.com/api/v3/company/profile/{stock}"
    )
    company_info = company_info.json()
    ceo = company_info["profile"]["ceo"]
    return (share_price, cash, debt, qRev, ceo)


def company_quote_group_of_items() -> None:
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


def getting_historical_data_for_rds_a_this_code_call() -> None:
    ticker = "RDS-A"
    target = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    rds = pd.read_json(target)
    rds = json_normalize(rds["historical"])
    rds["date"] = pd.to_datetime(rds["date"])
    rds.set_index("date", inplace=True)


def notebook_step_003() -> None:
    rds.head()


def notebook_step_005() -> None:
    rds = bollinger_bands(rds, "adjClose")
    bb_plot(rds)


def fill_our_newly_created_position_column_set_to_se() -> None:
    pass


def notebook_step_007() -> None:
    bb_strategy(df)


def notebook_step_008() -> None:
    df["Strategy Return"].cumsum().plot()


def monte_carlo_valuation_of_european_call_option() -> None:
    S0 = rds["adjClose"][-1]
    K = 35.0
    T = 1
    r = 0.05
    sigma = np.std(rds["changeOverTime"])
    n_sims = 100_000
    z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    hT = np.maximum(ST - K, 0)
    C0 = np.exp(-r * T) * np.sum(hT) / n_sims
    print(f"Value of the Call Option {C0:.5f}")


def notebook_step_010() -> None:
    rds["changeOverTime_log"] = np.log(rds["changeOverTime"])


def version_1_using_pandas_for_the_plot() -> None:
    rds[
        ["adjClose", "20 Day MA", "20 Day MA_lower bound", "20 Day MA_upper bound"]
    ].plot()


def notebook_step_012() -> None:
    len(ST)


def set_style_empty_figure_and_axes() -> None:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x_axis = rds.index
    ax.fill_between(
        x_axis, rds["20 Day MA_lower bound"], rds["20 Day MA_upper bound"], color="grey"
    )
    ax.plot(x_axis, rds["adjClose"], color="blue", lw=2)
    ax.plot(x_axis, rds["20 Day MA"], color="black", lw=2)
    ax.set_title("20 Day Bollinger Band For RDS-A")
    ax.set_xlabel("Date (Year/Month)")
    ax.set_ylabel("Price(USD)")
    ax.legend()
    plt.show()


def main() -> None:
    company_quote_group_of_items()
    getting_historical_data_for_rds_a_this_code_call()
    notebook_step_003()
    notebook_step_005()
    fill_our_newly_created_position_column_set_to_se()
    notebook_step_007()
    notebook_step_008()
    monte_carlo_valuation_of_european_call_option()
    notebook_step_010()
    version_1_using_pandas_for_the_plot()
    notebook_step_012()
    set_style_empty_figure_and_axes()


if __name__ == "__main__":
    main()
