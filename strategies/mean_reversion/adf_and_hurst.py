# In[ ]:
import os
from pathlib import Path

import matplotlib.pyplot as plt
import scipy
import scipy.stats
from arch.unitroot import VarianceRatio
from statsmodels.tsa.stattools import *

from strategies.utils import *


# In[ ]:

# Simple visualization to understand random walk and ar(1) process
def plot_ar_one_process():
    plt.rcParams['figure.figsize'] = [20, 20]
    fig, ax = plt.subplots(nrows=5, ncols=5)
    lmdas = np.linspace(-1.1, 1.1, 23)
    for i in range(23):
        lmda = lmdas[i]
        lnt = 200
        x = np.random.normal(0, 1, size=(lnt,)) + random.uniform(0.01, 1)
        for j in range(1, lnt):
            x[j] += lmda * x[j - 1] + 0.4 * j / lnt
        ax[int(i / 5)][i % 5].plot(x)
        ax[int(i / 5)][i % 5].title.set_text(f"{lmda:0.3f}")

    plt.savefig("dnt.png", dpi=300)

    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


# In[ ]:

def hurst_stat(ts, lags=100):
    """Computes hurst coefficient and variance ratio test statistics"""
    rng = range(2, lags)
    ts = np.log(ts)
    tau = [np.var(np.subtract(ts[lag:], ts[:-lag])) for lag in rng]
    poly = np.polyfit(np.log(rng), np.log(tau), 1)

    # test_stat = np.var(np.subtract(ts[lags:], ts[:-lags])) / (lags * np.var(np.subtract(ts[1:], ts[:-1])))
    return poly[0] / 2  # , test_stat


def ratio_test(prices, k=5):
    """Borrowed from https://mingze-gao.com/measures/lomackinlay1988/"""

    log_prices = np.log(prices)
    rets = np.diff(log_prices)
    T = len(rets)
    mu = np.mean(rets)
    var_1 = np.var(rets, ddof=1, dtype=np.float64)
    rets_k = (log_prices - np.roll(log_prices, k))[k:]
    m = k * (T - k + 1) * (1 - k / T)
    var_k = 1 / m * np.sum(np.square(rets_k - k * mu))

    # Variance Ratio
    vr = var_k / var_1

    # Phi1
    phi1 = 2 * (2 * k - 1) * (k - 1) / (3 * k * T)

    # Phi2
    def delta(j):
        res = 0
        for t in range(j + 1, T + 1):
            t -= 1  # array index is t-1 for t-th element
            res += np.square((rets[t] - mu) * (rets[t - j] - mu))
        return res / ((T - 1) * var_1) ** 2

    phi2 = 0
    for j in range(1, k):
        phi2 += (2 * (k - j) / k) ** 2 * delta(j)

    return vr, (vr - 1) / np.sqrt(phi1), (vr - 1) / np.sqrt(phi2)


def adf_test(ts):
    """Computes ADF slope coefficient and its p-value"""
    diff = np.diff(ts)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ts[:-1], diff)
    result = adfuller(ts, regression="c", maxlag=0)
    """
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    #print(f"slope: {slope}, stat:{slope/std_err}, pval:{p_value}, r:{r_value}, stderr:{std_err}")
    """
    print(f"rho: {slope:0.3f}, adf-test pValue: {result[1]:0.3f}")
    return slope, result[1]


def stationarity_analysis(data, adf_sig_level):
    """ADF Test"""
    rho, p_rho = adf_test(data)
    half_life = int(-math.log(2) / rho)
    print(f"ADF Test: rho:{rho}, p-value:{p_rho}, half life:{half_life}")
    if p_rho < adf_sig_level:
        print("Null hypothesis rejected at 5% level. No unit root")
    else:
        print("Cannot reject null hypothesis at 5% level. Unit root present")

    """Variance Ratio Test"""
    vr, z_homo, z_hetro = ratio_test(data, k=half_life)
    p_homo = scipy.stats.norm.sf(abs(z_homo)) * 2
    p_hetro = scipy.stats.norm.sf(abs(z_hetro)) * 2
    print(
        f"Variance Ratio:{vr}, p-val under homo: {scipy.stats.norm.sf(abs(z_homo)) * 2}, p-val under hetero: {scipy.stats.norm.sf(abs(z_hetro)) * 2}")

    print(VarianceRatio(data, lags=half_life).summary().as_text())

    """Hurst coefficient"""
    hurst_coeff = hurst_stat(data)
    print(f"Hurst coefficient: {hurst_coeff}")

    return rho, p_rho, half_life, vr, p_homo, p_hetro, hurst_coeff


def linear_mean_reversion_strategy(df, window):
    """Simple strategy holding -1*(rolling z score) with window of half life"""
    df["Quantity"] = -rolling_z(df["Close"], window)
    # print(df["Quantity"])
    df["Return"] = df["Close"].pct_change().shift(-1)
    # print(tabulate(df.tail(5), headers=df.columns))
    df = df.dropna(how="any")[:-1]
    df["PnL"] = df["Return"] * df["Quantity"]
    plt.style.use('seaborn-bright')
    plt.plot(np.cumsum(df["Return"]), label="Cumulative Return Of Instrument")
    plt.plot(np.cumsum(df["PnL"]), label="Cumulative PnL Of Strategy")
    plt.legend()
    plt.show()


# In[ ]

def main():
    adf_sig_level = 0.4
    file_path = os.path.join(Path(os.getcwd()), "data/forex/MERGED.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df[df.index > datetime(year=2017, month=1, day=1)]
    df["EURJPY"] = df["USDEUR"] / df["USDJPY"]
    # df = df[["EURJPY"]]
    for ticker in df.columns[:1]:
        df_t = df[[ticker]].dropna(how="any")
        df_t.columns = ["Close"]
        print("-" * 100)
        print(f"Using {ticker}")
        stationary_result = stationarity_analysis(df_t["Close"].values, adf_sig_level)
        half_life = stationary_result[2]
        if stationary_result[1] < adf_sig_level:
            linear_mean_reversion_strategy(df_t, min(60, half_life))
        else:
            plt.plot(df[ticker], label=ticker)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    # plot_ar_one_process()
    main()
