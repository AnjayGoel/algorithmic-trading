# In[ ]
import math

from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import coint

from strategies.mean_reversion.cointegration import adf_test, const_beta_pair_mean_reversion_strategy, \
    johansen_cointegration
from strategies.utils import *


# In[ ]

def index_arbitrage(df, index_ticker):
    print_dashed_line()
    """Index unit root test"""
    print(f"{index_ticker} Unit root test")
    adf_test(df[index_ticker])

    """Find cointegrated stocks (alpha = 0.1)"""
    cointegrated_tickers = []
    for ticker in df.columns:
        if ticker == index_ticker:
            continue
        res = coint(df[ticker], df[index_ticker])
        # print(f"{ticker}, p-value:{res[1]:0.3f}")
        if res[1] < 0.1:
            cointegrated_tickers.append(ticker)

    """Ensure cointegrated stocks are non stationary"""
    print_dashed_line()
    for ticker in cointegrated_tickers:
        print(f"Cointegrated ticker: {ticker}")
        adf_test(df[ticker])
    print_dashed_line()

    df_index = df[index_ticker]
    df_coint = df[cointegrated_tickers]

    dfn = pd.DataFrame()
    dfn["Index"] = np.cumprod(1 + df_index.pct_change())

    """Daily return is mean return aka. return on equal weight portfolio"""
    dfn["Portfolio"] = np.cumprod(1 + df_coint.pct_change().mean(axis=1))
    dfn = dfn.dropna(how='any')

    """
    res = johansen_cointegration(dfn, "Index", "Portfolio")
    print(res.evec)
    dfn["Spread1"] = dfn[["Index", "Portfolio"]] @ vec_norm(res.evec[0])
    dfn["Spread2"] = dfn[["Index", "Portfolio"]] @ vec_norm(res.evec[1])
    # plt.plot(dfn["Portfolio"], label="Return on equal weight portfolio")
    """

    beta = get_slr_coeff(dfn["Index"].values, dfn["Portfolio"].values)
    dfn["Spread"] = dfn["Index"] * beta - dfn["Portfolio"]
    print("Spread ADF-Test")
    rho, pval = adf_test(dfn["Spread"])
    half_life = int(-math.log(2) / rho)
    print(f"Half life: {half_life}")
    plt.plot(dfn["Spread"], label="Spread")
    plt.legend()
    plt.show()

    const_beta_pair_mean_reversion_strategy(dfn, "Index", "Portfolio", (beta, -1), window=100)
    print_dashed_line()
    plt.plot(dfn["Index"], label="Index")
    plt.plot(dfn["Portfolio"], label="Portfolio")
    plt.legend()
    plt.show()
    performance_summary(dfn["Index"], "Index", is_cuml=True)
    print_dashed_line()
    performance_summary(dfn["Portfolio"], "Portfolio", is_cuml=True)


def main():
    df = read_df("./data/equity/NIFTY_50_STOCKS_Close.csv")

    index_arbitrage(df, "NIFTYFIFTY")
    """
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')
    plt.matshow(corr)
    plt.show()
    """


# In[ ]
if __name__ == "__main__":
    main()
