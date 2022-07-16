# In[ ]
import matplotlib.pyplot as plt

from .adf_and_hurst import adf_test
from .dynamic_hedge_ratio import calculate_beta_ols, dynamic_beta_mean_reversion_strategy
from ..utils import *


# In[ ]
def main():
    df = read_df("./data/forex/MERGED.csv")[["USDCAD", "USDAUD"]].dropna(
        how="any")
    cadusd = 1 / df["USDCAD"]
    audusd = 1 / df["USDAUD"]

    df = pd.DataFrame()
    df["CAD_Close"] = cadusd
    df["AUD_Close"] = audusd

    lookback = 60
    df = calculate_beta_ols(df, ticker_x="CAD", ticker_y="AUD", lookback=lookback, series_type="Close")
    df = df[lookback:]
    df["Close"] = np.sum(df[["CAD_Close", "AUD_Close"]].values * df[["x", "y"]].values, axis=1)

    print_dashed_line()
    rho, p_val = adf_test(df["Close"])
    half_life = min(max(3, int(-math.log(2) / rho)), 30)
    print(f"Half-life: {half_life}")
    df = dynamic_beta_mean_reversion_strategy(df, "CAD", "AUD", window=half_life)
    df[["CAD_Close", "AUD_Close"]].plot()
    plt.show()
    plt.plot(df["Close"], label="Stationary Series")
    plt.legend()
    plt.show()
    plt.plot(np.cumsum(df["PnL"]), label="PnL")
    plt.legend()
    plt.show()


# In[ ]
if __name__ == "__main__":
    main()
