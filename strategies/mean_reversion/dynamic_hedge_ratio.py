# In[ ]
import os
from pathlib import Path

from matplotlib import pyplot as plt
from pykalman import KalmanFilter

from ..mean_reversion.adf_and_hurst import adf_test
from ..utils import *


# In[ ]
def calculate_beta_ols(df, ticker_x, ticker_y, lookback, series_type="Close"):
    """Calculate SLR Beta Using A Lookback Period"""
    df["x"] = 0.0
    for i in range(lookback, len(df)):
        df_temp = df.iloc[i - lookback:i]
        df["x"][i] = -get_slr_coeff(df_temp[ticker_x + "_" + series_type], df_temp[ticker_y + "_" + series_type])

    df["y"] = 1
    return df


def calculate_beta_kalman(df, ticker_one, ticker_two, series_type="Close"):
    """Use Kalman Filter To Dynamically Calculate Beta"""
    x = df[ticker_one + "_" + series_type].values
    y = df[ticker_two + "_" + series_type].values
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([x, np.ones(x.shape)]).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )

    state_means, state_covs = kf.filter(y)

    df["x"] = -state_means[:, 0]
    df["y"] = 1
    return df


def dynamic_beta_mean_reversion_strategy(df, ticker_x, ticker_y, window, series_type="Close"):
    cols = [ticker_x + "_" + series_type, ticker_y + "_" + series_type]

    """Dot product, Unit portfolio value"""
    df["Close"] = np.sum(df[cols].values * df[["x", "y"]].values, axis=1)

    """Rolling Z score (numUnits)"""
    df["Z"] = -rolling_z(df["Close"], window)
    df = df.dropna(how="any")[:-1]

    """Positions (Capital Invested)"""
    positions = ["pos_" + i for i in cols]
    df[positions] = df[cols].values * df[["x", "y"]].values
    df[positions] = df[positions].mul(df["Z"], axis=0)

    """P&L and Return"""
    df["PnL"] = np.sum(df[positions].values * df[cols].pct_change().shift(-1).values, axis=1)

    """Return = PnL/Capital Invested"""
    df["Return"] = df["PnL"] / (df[positions].abs().sum(axis=1))

    plt.style.use('seaborn-bright')
    plt.plot(np.cumprod(1 + df["Return"]) - 1, label=f"Strategy Return, Window:{window}")
    plt.legend()
    plt.show()

    performance_summary(df["Return"], label="Dynamic Hedge Ratio")

    return df


def main():
    data_dir = os.path.join(Path(os.getcwd()), "data/equity")

    # Fails on this pair
    # ticker_one = "IND_IDX_NIFTY_50"
    # ticker_two = "IND_IDX_NIFTY_BANK"

    ticker_one = "AUS_IDX_SNP_ASX"
    ticker_two = "CAN_IDX_SNP_TSK"

    df1 = read_df(data_dir + f"/{ticker_one}.csv", return_cols=["Open", "Close"], prefix=ticker_one)
    df2 = read_df(data_dir + f"/{ticker_two}.csv", return_cols=["Open", "Close"], prefix=ticker_two)
    df = merge_df([df1, df2])

    """Calculate daily beta"""
    df = calculate_beta_kalman(df, ticker_one, ticker_two)
    # df = calculate_beta_ols(df, ticker_one, ticker_two, lookback=20)

    """Calculate closing price of unit portfolio"""
    cols = [ticker_one + "_Close", ticker_two + "_Close"]
    df["Close"] = np.sum(df[cols].values * df[["x", "y"]].values, axis=1)

    plt.plot(df["Close"], label="Stationary Portfolio")
    plt.legend()
    plt.show()

    print_dashed_line()
    slope, p_val = adf_test(df["Close"])
    half_life = max(20, math.ceil(-math.log(2) / slope))

    dynamic_beta_mean_reversion_strategy(df, ticker_one, ticker_two, window=half_life)


# In[ ]
if __name__ == "__main__":
    main()
