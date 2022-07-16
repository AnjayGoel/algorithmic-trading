# In[ ]:
import os
from pathlib import Path

from scipy.stats import pearson3

from ..mean_reversion.bollinger_bands import bollinger_bands
from ..utils import *


# In[ ]:
def monte_carlo(df, strategy, num_simulations=100):
    obs_returns = strategy(df)
    obs_mean = obs_returns.mean()
    print(f"Actual return: {obs_mean}")

    returns = df["Close"].pct_change().dropna()
    param = pearson3.fit(returns)  # Scipy only has pearson3

    nums_better = 0
    for i in range(num_simulations):
        sim_returns = pearson3.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=len(returns))
        sim_mean = np.mean(sim_returns)

        df = pd.DataFrame(np.cumprod(1 + sim_returns), columns=["Close"])
        strat_returns = strategy(df)
        strat_mean = strat_returns.mean()

        if strat_mean > obs_mean:
            nums_better += 1
        print(f"sim: {i}, mkt mean return : {sim_mean}, strat mean return: {strat_mean}")

    print(f"Monte Carlo p-value: {nums_better / num_simulations:0.3f}")


# In[ ]:
def main():
    data_dir = os.path.join(Path(os.getcwd()), "data/equity")
    df = read_df(data_dir + "/IND_IDX_NIFTY_50.csv")
    monte_carlo(df, lambda x: bollinger_bands(x, show_results=False))


if __name__ == "__main__":
    main()
