# In[]:
import os

import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from strategies.mean_reversion.adf_and_hurst import adf_test
from strategies.utils import *


# In[ ]:

def simple_linear_regression(df, x, y):
    X = sm.add_constant(df[x])  # adding a constant
    model = sm.OLS(df[y], X).fit()
    predictions = model.predict(X)
    summary = model.summary()

    plt.style.use('seaborn-bright')
    plt.figure(figsize=(10, 10))
    plt.plot(df[x], label=x)
    plt.plot(df[y], label=y)
    plt.plot(predictions, label="Prediction")
    plt.plot(df[y] - predictions, label="Residual")
    plt.legend()
    plt.show()
    print(summary)
    return model.params[1]


def johansen_cointegration(df, ticker_one, ticker_two):
    jres = coint_johansen(df[[ticker_one, ticker_two]], det_order=0, k_ar_diff=0)

    summary = []

    for i in range(jres.trace_stat.shape[0]):
        summary.append([
            f"r<={i}", jres.trace_stat[i], jres.trace_stat_crit_vals[i][0], jres.trace_stat_crit_vals[i][1],
            jres.trace_stat_crit_vals[i][2]])

    print("Johansen Cointegration Test")
    print(tabulate(summary, headers=["NULL:", "Trace Statistics", "Crit 90%", "Crit 95%", "Crit 99%"]))

    print("\n")
    summary = []

    for i in range(jres.max_eig_stat.shape[0]):
        summary.append([
            f"r<={i}", jres.max_eig_stat[i], jres.max_eig_stat_crit_vals[i][0], jres.max_eig_stat_crit_vals[i][1],
            jres.max_eig_stat_crit_vals[i][2]])
    print(tabulate(summary, headers=["NULL:", "Eigen Statistics", "Crit 90%", "Crit 95%", "Crit 99%"]))

    print("eigen values: " + " ".join(f"{i:0.6f}" for i in jres.eig))
    print("eigen vectors: " + " ".join(f"{i}" for i in jres.evec))
    print_dashed_line()
    return jres


def generate_stationary_series(df, series_one, series_two, coint_test_res, beta):
    """Generates Stationary Series From Johansen Test Results & SLR Coeff"""
    df_t = pd.DataFrame()
    df_t["P1 (Johansen Test)"] = df[[series_one, series_two]] @ vec_norm(coint_test_res.evec[0])
    df_t["P2 (Johansen Test)"] = df[[series_one, series_two]] @ vec_norm(coint_test_res.evec[1])
    df_t["P3 (SLR Coeff)"] = df[[series_one, series_two]] @ vec_norm((beta, -1))
    return df_t


def const_beta_pair_mean_reversion_strategy(df, ticker_one, ticker_two, evec, window, plot_tickers=False):
    evec = vec_norm(evec)
    cols = [ticker_one, ticker_two]

    """Dot product, Unit portfolio value"""
    df["Close"] = df[cols] @ evec

    """Rolling Z score (numUnits)"""
    df["Z"] = -rolling_z(df["Close"], window)
    df = df.dropna(how="any")[:-1]

    """Positions (Capital Invested)"""
    positions = ["pos_" + i for i in cols]
    df[positions] = (df[cols].mul(evec, axis=1)).mul(df["Z"], axis=0)

    """P&L and Return"""
    df["PnL"] = np.sum(df[positions].values * df[cols].pct_change().shift(-1).values, axis=1)
    df["Return"] = df["PnL"] / (df[positions].abs().sum(axis=1))

    # print_df(df, n=10)
    plt.style.use('seaborn-bright')
    plt.plot(np.cumprod(1 + df["Return"]), label="Cumulative Return Of Strategy")
    # plt.plot(df["Close"], label="Cumulative PnL Of Instrument")
    if plot_tickers:
        plt.plot(df[ticker_one], label=ticker_one)
        plt.plot(df[ticker_two], label=ticker_two)
    plt.legend()
    plt.show()
    performance_summary(df['Return'], label="Constant Beta Mean Reversion")

    return df


# In[ ]:
def main():
    ticker_one = "AUS_IDX_SNP_ASX"
    ticker_two = "CAN_IDX_SNP_TSK"

    data_dir = os.path.join(os.getcwd(), "data/equity/")

    df_one = read_df(data_dir + f"{ticker_one}.csv", return_cols=["Close"], prefix=ticker_one)
    df_two = read_df(data_dir + f"{ticker_two}.csv", return_cols=["Close"], prefix=ticker_two)
    df = merge_df([df_one, df_two])
    df.columns = [ticker_one, ticker_two]

    print_dashed_line()
    """Use SLR"""
    beta = simple_linear_regression(df, ticker_one, ticker_two)

    print("\n")
    print("Stationary Test of SLR Spread")
    adf_test(df[ticker_two] - beta * df[ticker_one])
    print_dashed_line()

    "Use Johansen cointegration test"
    coint_test_res = johansen_cointegration(df, ticker_one, ticker_two)

    """Generate stationary time series from results above"""
    df_t = generate_stationary_series(df, ticker_one, ticker_two, coint_test_res, beta)
    slopes = []
    for portfolio in df_t.columns:
        print(portfolio)
        slope, p_val = adf_test(df_t[portfolio])
        slopes.append(slope)
        plt.plot(df_t[portfolio], label=portfolio)
    plt.title("Stationary Portfolios")
    plt.legend()
    plt.show()

    half_life = int(-math.log(2) / slopes[0])

    const_beta_pair_mean_reversion_strategy(df, ticker_one, ticker_two, coint_test_res.evec[0],
                                            min(half_life, 90))


if __name__ == "__main__":
    main()
