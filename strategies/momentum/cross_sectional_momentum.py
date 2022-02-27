# In[ ]:
from strategies.momentum.momentum_testing import *

eps = 10 ** -10


# In[]
def momentum_rule(x, bottom, top):
    """Short if among historically worst performing stocks else go long"""
    if x < bottom:
        return -1.0
    elif x > top:
        return 1.0
    else:
        return 0.0


def cross_sectional_momentum(df: pd.DataFrame, lookback, holdday, num_stocks=5):
    return_lookback = ((df - df.shift(lookback)) / df.shift(lookback))

    return_daily = df.pct_change().shift(-1).iloc[lookback:]

    positions = return_lookback.copy()

    """Find Best & Worst Performing Return Cutoff"""
    positions["bottom"] = positions.apply(lambda x: np.sort(x)[num_stocks - 1] + eps, axis=1)
    positions["top"] = positions.apply(lambda x: np.sort(x)[-num_stocks] - eps, axis=1)

    """Place Positions Accordingly"""
    for col in positions.columns:
        positions[col] = positions.apply(lambda x: momentum_rule(x[col], x["bottom"], x["top"]), axis=1)
    positions = positions.drop(columns=["bottom", "top"])
    temp = positions.copy()
    for h in range(1, holdday):
        positions = positions + temp.shift(h).fillna(0)
    positions /= (holdday * num_stocks * 2)

    """Calculate Strategy Return From Positions"""
    strategy_return = (positions * return_daily).sum(axis=1)

    (1 + strategy_return).cumprod().plot(
        label=f"Cross-sectional Momentum Strategy ({lookback},{holdday}, {num_stocks})")
    plt.legend()
    plt.show()

    print_dashed_line()
    performance_summary(strategy_return, "Cross-sectional Momentum Strategy",
                        params={"lookback": lookback, "holding period": holdday, "stocks": num_stocks})


# In[ ]
def main():
    df = read_df("./data/equity/NIFTY_100_STOCKS_Close.csv")

    """Mask to deal with stock splits"""
    df_mask = read_df("./data/equity/NIFTY_100_STOCKS_SPLIT_MASK.csv")
    df = df * df_mask

    cross_sectional_momentum(df, lookback=25, holdday=25, num_stocks=10)


if __name__ == "__main__":
    main()
