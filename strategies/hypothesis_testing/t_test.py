# In[ ]:
import os
from pathlib import Path

from scipy.stats.mstats import ttest_onesamp

from ..mean_reversion.bollinger_bands import bollinger_bands
from ..utils import *


# In[ ]:
def t_test(df, strategy):
    """First strategy in the book. Essentially a t-test"""
    returns = strategy(df)
    stat, pval = ttest_onesamp(returns, popmean=0)
    print(f"t-stat (sharpe-ratio * sqrt(n)): {stat}, p-val: {pval}")


# In[ ]:
def main():
    data_dir = os.path.join(Path(os.getcwd()), "data/equity")
    df = read_df(data_dir + "/IND_IDX_NIFTY_50.csv")
    t_test(df, lambda x: bollinger_bands(df, show_results=False))


if __name__ == "__main__":
    main()
