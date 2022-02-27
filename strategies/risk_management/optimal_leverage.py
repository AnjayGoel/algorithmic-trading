# In[]
from matplotlib import pyplot as plt
from scipy.stats import pearson3

from strategies.utils import *


# In[]
def fit_and_generate_pearson(prices: pd.Series, size=100000, show_plots=True):
    """Fit Pearson Type III Distribution & Generate Random Samples"""
    returns = prices.pct_change().dropna()
    param = pearson3.fit(returns)
    if show_plots:
        x = np.linspace(-0.1, 0.1, 100)
        pdf_fitted = pearson3.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        plt.title('Fitted pearson 3 distribution vs histogram')
        plt.plot(x, pdf_fitted, 'r-')  # ,x,pdf,'b--')
        plt.hist(returns, bins=100, density=True, alpha=.3)
        plt.show()
    """model.rvs draws random sample from distribution"""
    return pearson3.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=size)


def kelly_leverage(price: pd.Series, risk_free_rate=1.03 ** (1 / 365) - 1):
    returns = price.pct_change()
    return (returns.mean() - risk_free_rate) / returns.var()


def kelly_leverage_multivariate(prices: pd.DataFrame, risk_free_rate=1.03 ** (1 / 365) - 1, max_leverage=-1):
    returns = prices.pct_change() - risk_free_rate
    mean = returns.mean()
    cov = returns.cov()
    leverages = np.linalg.inv(cov).dot(mean)
    if max_leverage > 0:
        leverages = leverages * max_leverage / (np.sum(leverages))
    return leverages


def get_compounded_growth_rate(rvs, leverage):
    return np.mean(np.log(leverage * rvs + 1))


# In[]
def main():
    print_dashed_line()
    df = read_df("./data/equity/IND_IDX_NIFTY_50.csv", prefix="nifty")
    hist_kelly_leverage = kelly_leverage(df["nifty_Close"])
    print(f"Kelly leverage using historical data: {hist_kelly_leverage:0.3f}")

    rvs = fit_and_generate_pearson(df["nifty_Open"])
    leverages = np.linspace(1, 6, 100)
    comp_growth_rates = [get_compounded_growth_rate(rvs, leverage) for leverage in leverages]
    max_idx = np.argmax(comp_growth_rates)
    print(
        f"Best compounded growth rate and optimal leverage: {comp_growth_rates[max_idx] * 100:0.3f}%, {leverages[max_idx]:0.3f}")

    plt.plot(leverages, comp_growth_rates, 'r-')
    plt.title("Compounded growth (in %) vs Leverage using simulated data")
    plt.show()


if __name__ == "__main__":
    main()
