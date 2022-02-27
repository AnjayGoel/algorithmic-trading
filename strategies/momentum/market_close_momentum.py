# In[ ]:
from matplotlib import pyplot as plt

from strategies.utils import *


# In[]
def market_close_momentum(df: pd.DataFrame, threshold=0.005):
    """315: Price at 3:15 PM. Market Closes At 3:30 PM"""
    df["longs"] = (df["315"] > (1 + threshold) * df["Close"].shift(1)).astype(int)
    df["shorts"] = (df["315"] < (1 - threshold) * df["Close"].shift(1)).astype(int)
    df["positions"] = df["shorts"] + df["longs"]
    df["return"] = df["positions"] * (df["Close"] / df["315"] - 1)
    (1 + df["return"]).cumprod().plot(label=f"Market Close Momentum Strategy")
    plt.legend()
    plt.show()
    print_dashed_line()
    performance_summary(df["return"], "Momentum Strategy", params={})


# In[ ]
def main():
    df = read_df("./data/equity/IND_IDX_NIFTY_50_DETAILED.csv")
    df = df[df.index > np.datetime64('2020-01-01')]
    market_close_momentum(df)


if __name__ == "__main__":
    main()
