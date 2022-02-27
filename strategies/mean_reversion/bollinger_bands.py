# In[ ]:
import os
from pathlib import Path

import matplotlib.pyplot as plt

from strategies.mean_reversion.cointegration import adf_test, johansen_cointegration
from strategies.utils import *


# In[ ]:

def fill_signal_long(df, entry_score, exit_score):
    state = 0
    df["L"] = 0

    for i in range(len(df)):
        if df.iloc[i]["Z"] < entry_score and state == 0:
            df.at[i, "L"] = 1
            state = 1
        elif df.iloc[i]["Z"] >= exit_score and state == 1:
            df.at[i, "L"] = -1
            state = 0

    df["L"] = df["L"].cumsum()
    return df


def fill_signal_short(df, entry_score, exit_score):
    state = 0
    df["S"] = 0

    for i in range(len(df)):
        if df.iloc[i]["Z"] > entry_score and state == 0:
            df.at[i, "S"] = 1
            state = 1
        elif df.iloc[i]["Z"] <= exit_score and state == 1:
            df.at[i, "S"] = -1
            state = 0

    df["S"] = df["S"].cumsum()
    return df


def bollinger_bands(df, long_entry, long_exit, short_entry, short_exit, window):
    """Rolling Z"""
    df["Z"] = rolling_z(df["Close"], window)
    df.dropna(how='any', inplace=True)
    df.reset_index(inplace=True, drop=True)

    """Fill Entry/Exit Signals based on Z Score"""
    df = fill_signal_long(df, long_entry, long_exit)
    df = fill_signal_short(df, short_entry, short_exit)

    """Return"""
    df["R"] = df["Close"].pct_change().shift(-1)
    df.dropna(how='any', inplace=True)

    """Strategy Return"""
    df["RT"] = (df["L"] - df["S"]) * df["R"]

    (1 + df["RT"]).plot(legend="BB Returns")
    (1 + df["R"]).plot(legend="Stock Returns")
    print(f"APR: {get_apr(df['RT'])}, Sharpe Ratio: {get_sharpe_ratio(df['RT'])}")

    plt.legend()
    plt.show()


def bollinger_bands_long_short_portfolio(df, ticker_x, ticker_y, evec, long_entry, long_exit, short_entry, short_exit,
                                         window):
    """Make stationary Series And Compute Z Score"""
    cols = [ticker_x, ticker_y]
    df["Close"] = df[cols] @ evec
    df["Z"] = rolling_z(df["Close"], window)

    df.dropna(how='any', inplace=True)
    df.reset_index(inplace=True, drop=True)

    """Fill Entry/Exit Signals"""
    df = fill_signal_long(df, long_entry, long_exit)
    df = fill_signal_short(df, short_entry, short_exit)

    """Positions (Capital Invested)"""
    positions = ["pos_" + i for i in cols]
    df[positions] = df[cols].mul(evec, axis=1)
    # df[positions] = df[positions].mul(df["L"] - df["S"], axis=0)

    """P&L and Return"""
    df["UnitPnL"] = np.sum(df[positions].values * df[cols].pct_change().shift(-1).values, axis=1)

    """Return = PnL/Capital Invested"""
    df["UnitReturn"] = df["UnitPnL"] / (df[positions].abs().sum(axis=1))
    df["BBReturn"] = (df["L"] - df["S"]) * df["UnitReturn"]

    plt.plot((1 + df["BBReturn"]).cumprod(), label="BB Returns")
    plt.plot((1 + df["UnitReturn"]).cumprod(), label="Stationary Portfolio Return")

    plt.legend()
    plt.show()
    performance_summary(df["BBReturn"], label="Bollinger Bands",
                        params={"Long Entry": long_entry, "Long Exit": long_exit, "Short Entry": short_entry,
                                "Short Exit": short_exit})


# In[ ]:

def main():
    data_dir = os.path.join(Path(os.getcwd()), "data/equity")

    ticker_x = "AUS_IDX_SNP_ASX"
    ticker_y = "CAN_IDX_SNP_TSK"
    df1 = read_df(data_dir + f"/{ticker_x}.csv", return_cols=["Close"], prefix=ticker_x)
    df2 = read_df(data_dir + f"/{ticker_y}.csv", return_cols=["Close"], prefix=ticker_y)
    df = merge_df([df1, df2])
    df.columns = [ticker_x, ticker_y]

    """Johansen Cointegration Test To Find Cointegrating Vector"""
    coint_test_res = johansen_cointegration(df, ticker_x, ticker_y)
    evec = coint_test_res.evec[0]

    """Compute Half Life"""
    df["Close"] = df[[ticker_x, ticker_y]] @ evec
    slope, p_val = adf_test(df["Close"])
    half_life = math.ceil(-math.log(2) / slope)
    print(f"half-life:{half_life:0.0f}")

    plt.plot(df["Close"], label="Stationary Portfolio")
    plt.legend()
    plt.show()

    bollinger_bands_long_short_portfolio(df, ticker_x, ticker_y, evec, -0.5, -0.1, 0.5, 0.1, window=half_life)


if __name__ == "__main__":
    main()
