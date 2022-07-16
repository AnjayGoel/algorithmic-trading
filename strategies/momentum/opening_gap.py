# In[ ]:

from .momentum_testing import *


# In[]
def opening_gap_strategy(df: pd.DataFrame, z_entry_score=0.1, rolling_window=90, show_results=True,return_pos=False):
    df["std"] = df["Close"].pct_change().rolling(window=rolling_window).std().shift()
    df = df.dropna(subset=["std"])

    df["longs"] = (df["Open"] > df["High"].shift(1) * (1 + z_entry_score * df["std"])).astype(int)
    df["shorts"] = (df["Open"] < df["Low"].shift(1) * (1 - z_entry_score * df["std"])).astype(int)
    df["position"] = df["longs"] - df["shorts"]

    df["ret"] = (df["Close"] / df["Open"] - 1) * df["position"]

    if show_results:
        (1 + df["ret"]).cumprod().plot(label=f"Buy on Gap Strategy")
        plt.legend()
        plt.show()
        print_dashed_line()
        performance_summary(df["ret"], "Momentum Strategy", params={})
    else:
        if return_pos:
            return df["ret"], df["positions"]
        else:
            return df["ret"]


# In[ ]
def main():
    df = read_df("./data/equity/IND_IDX_NIFTY_50.csv")
    opening_gap_strategy(df)


if __name__ == "__main__":
    main()
