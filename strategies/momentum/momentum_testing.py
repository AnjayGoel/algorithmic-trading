# In[ ]:

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from strategies.utils import *


# In[ ]
def momentum_test(dff, p_val_thresh=0.3):
    corr = []
    lookbacks = [1, 5, 10, 25, 60, 120, 250]
    holddays = [1, 5, 10, 25, 60, 120, 250]
    for lookback in lookbacks:
        for holdday in holddays:
            df = dff.copy(deep=True)
            df["ret_past"] = df["Close"].pct_change(lookback)
            df["ret_fut"] = df["Close"].shift(-holdday).pct_change(periods=holdday)
            choose = range(0, len(df), min(holdday, lookback))
            df = df.iloc[choose, :]
            df = df.dropna(subset=["ret_past", "ret_fut"])
            if len(df) < 3:
                continue
            """Correlation Coeff And Its Significance"""
            r, p_val = pearsonr(df["ret_past"], df["ret_fut"])
            if p_val > p_val_thresh:
                continue
            corr.append({"lookback": lookback, "holddays": holdday, "correlation": r, "significance": p_val})
    print_dashed_line()
    df_corr = pd.DataFrame(corr)
    print_df(df_corr, len(df_corr))
    return corr


def momentum_strategy(df, lookback, holdday):
    df["long"] = df["Close"] > df["Close"].shift(lookback)
    df["short"] = df["Close"] < df["Close"].shift(lookback)
    df["pos"] = 0
    for h in range(holdday):
        df.loc[df["long"].shift(h).fillna(False), "pos"] += 1
        df.loc[df["short"].shift(h).fillna(False), "pos"] -= 1
    df["ret"] = df["pos"] * df["Close"].pct_change().shift(-1) / holdday
    (1 + df["ret"]).cumprod().plot(label=f"Momentum Strategy ({lookback},{holdday})")
    plt.legend()
    plt.show()

    performance_summary(df["ret"], "Momentum Strategy", params={"lookback": lookback, "holding period": holdday})


# In[ ]
def main():
    df = read_df("./data/debt/US_TWO_YEAR_FUT.csv")
    mom_test_res = momentum_test(df)
    momentum_strategy(df, 60, 60)


if __name__ == "__main__":
    main()
