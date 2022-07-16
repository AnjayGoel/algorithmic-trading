# In[ ]:
import os
from pathlib import Path

from scipy.stats import pearson3

from ..mean_reversion.bollinger_bands import bollinger_bands
from ..utils import *


# In[ ]:
def randomize_positions(df, strategy, num_simulations=100):
    mkt_returns = df["Close"].pct_change().dropna().to_numpy()

    obs_returns, pos = strategy(df, return_pos=True)
    obs_mean = obs_returns.mean()
    print(f"Actual return: {obs_mean}")

    """Pad array since bollinger bands drops initial values (for moving avg calculations)"""
    pos = np.pad(pos.to_numpy(), (0, len(mkt_returns) - len(pos)), "constant")

    nums_better = 0
    for i in range(num_simulations):
        np.random.shuffle(pos)  # Inplace
        strat_mean = np.dot(mkt_returns, pos) / len(pos)

        if strat_mean > obs_mean:
            nums_better += 1
        print(f"sim: {i}, strat mean return: {strat_mean}")

    print(f"Randomized positions p-value: {nums_better / num_simulations:0.3f}")


# In[ ]:
def main():
    data_dir = os.path.join(Path(os.getcwd()), "data/equity")
    df = read_df(data_dir + "/IND_IDX_NIFTY_50.csv")
    randomize_positions(df, lambda x, return_pos: bollinger_bands(x, show_results=False, return_pos=return_pos))


if __name__ == "__main__":
    main()
