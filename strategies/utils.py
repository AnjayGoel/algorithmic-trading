import math
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tabulate import tabulate

pd.options.mode.chained_assignment = None


def read_df(path, return_cols=None, prefix=""):
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if return_cols is not None:
        df = df[return_cols]
    if prefix != "":
        df = df.add_prefix(prefix + "_")
    return df


def merge_df(dfs):
    return reduce(
        lambda left, right: pd.merge(
            left, right,
            left_index=True, right_index=True), dfs
    )


def print_df(df, n=10):
    print(tabulate(df.head(n), headers=df.columns, tablefmt="fancy_grid"))


def log_df(df):
    return df.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)


def print_dashed_line():
    print("-" * 80)


def vec_norm(x):
    x = np.array(x)
    return x / x[0]


def get_slr_coeff(x, y):
    # print(f"x,y:{len(x)},{len(y)}")
    X = sm.add_constant(x)  # adding a constant
    model = sm.OLS(y, X).fit()
    return model.params[1]


def get_sharpe_ratio(ts: pd.Series, risk_free_rate=0, annualized=True):
    sharpe_ratio = (ts.mean() - risk_free_rate) / ts.std()
    if annualized:
        sharpe_ratio *= math.sqrt(252)
    return sharpe_ratio


def get_apr(ts):
    return np.prod(1 + ts) ** (252 / len(ts)) - 1


def rolling_z(ts, window):
    return (ts - ts.rolling(window=window).mean()) / ts.rolling(window=window).std()


def get_max_drawdown(ts):
    ts = (1 + ts).cumprod()
    mx = np.maximum.accumulate(ts)
    dd = (mx - ts) / mx
    return np.max(dd)


def performance_summary(ts, label, params=None, is_cuml=False):
    if is_cuml:
        ts = ts.pct_change()
    print(f"\t{label}")
    if params is not None:
        print(tabulate(zip(params.keys(), params.values()), headers=["Param", "Value"], tablefmt="fancy_grid"))
    print(tabulate([["APR", get_apr(ts)], ["Sharpe Ratio", get_sharpe_ratio(ts)], ["Max DD", get_max_drawdown(ts)]],
                   headers=["Statistic", "Value"],
                   tablefmt="fancy_grid"))
