# In[ ]
from matplotlib import pyplot as plt

from strategies.utils import *


# In[ ]
def cross_sectional_mean_reversion(df):
    print_dashed_line()

    tickers = df.columns
    weights = ["weight_" + i for i in tickers]

    """Mean return of all stocks"""
    mean_ret = df.pct_change().mean(axis=1)

    "Deviation from mean return"
    diff_ret = df.pct_change().subtract(mean_ret, axis=0)

    """Weight = -(r-<r>)/sum(abs(r-<r>))"""
    df[weights] = -diff_ret.div(diff_ret.abs().sum(axis=1), axis=0)
    df["mean_return"] = mean_ret

    """Return = Weight * % Change"""
    df["strategy_return"] = np.sum(df[weights].values * df[tickers].pct_change().shift(-1).values, axis=1)
    df = df.dropna(how='any')

    plt.plot((1 + df["strategy_return"]).cumprod(), label="Strategy Cumulative Return")
    plt.plot((1 + df["mean_return"]).cumprod(), label="Equal Weight Portfolio Cumulative Return")
    plt.legend()
    plt.show()

    performance_summary(df["strategy_return"], label="Cross-sectional Mean Reversion")
    performance_summary(df["mean_return"], label="Equal Weighted Portfolio")


def main():
    tickers = ["AXISBANK", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "INDUSINDBK", "SBIN"]
    df = read_df("./data/equity/NIFTY_50_STOCKS_Close.csv")[tickers]
    cross_sectional_mean_reversion(df[tickers])


# In[ ]
if __name__ == "__main__":
    main()
