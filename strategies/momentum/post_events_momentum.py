# In[ ]
from matplotlib import pyplot as plt

from strategies.data_download import *


# In[ ]
def fill_positions(x):
    """Fill cells that correspond to a corporate announcement"""
    if pd.isna(x["symbol"]):
        x = x.apply(lambda x: 0)
    else:
        symbols = x["symbol"].split("|")
        for symbol in x.index:
            if symbol in symbols:
                x[symbol] = 1
            else:
                x[symbol] = 0
    return x


def prepare_dataset():
    """Fetch And Prepare Dataset"""

    """
    fetch_nifty_constituents_data(index="NIFTY 100",
                                  start_date=datetime.date(2015, 1, 1),
                                  end_date=datetime.date(2020, 1, 1))
    fetch_nse_event_calender(start_year=2015, end_year=2020)
    fetch_nse_dividends_date(start_year=2015, end_year=2020)
    """

    nifty_100_stocks = list(pd.read_csv("./data/equity/NIFTY_100_STOCKS_LIST.csv")["Symbol"])
    nifty_100 = pd.read_csv("./data/equity/IND_IDX_NIFTY_100.csv", parse_dates=["Date"])

    """Corporate Events"""
    corp_events = pd.read_csv("./data/equity/IND_EQ_EVENTS.csv", parse_dates=["date"],
                              date_parser=lambda x: datetime.strptime(x, "%d-%b-%Y"))
    corp_events = corp_events.groupby("date")["symbol"].apply(list)
    corp_events = corp_events.apply(lambda x: [i for i in x if i in nifty_100_stocks])
    dct = {}
    for idx, symbols in corp_events.iteritems():
        if len(symbols) == 0:
            continue
        evt_date = idx
        nxt_date = nifty_100[nifty_100["Date"] > evt_date]["Date"].min()
        if nxt_date not in dct.keys():
            dct[nxt_date] = []
        dct[nxt_date].extend(symbols)

    lst = []
    for date, symbols in dct.items():
        lst.append({"Date": date, "Symbols": "|".join(symbols)})
    df_events = pd.DataFrame.from_records(lst)
    df_events.to_csv("./data/equity/NIFTY_100_POST_EVENT_DATE.csv", index=False)

    """Dividends"""
    ex_dividends = pd.read_csv("./data/equity/IND_EQ_DIVIDENDS.csv", parse_dates=["exDate"],
                               date_parser=lambda x: datetime.strptime(x, "%d-%b-%Y"))
    ex_dividends = ex_dividends.groupby("exDate")["symbol"].apply(list)
    ex_dividends = ex_dividends.apply(lambda x: [i for i in x if i in nifty_100_stocks])
    dct = {}
    for idx, symbols in ex_dividends.iteritems():
        if len(symbols) == 0:
            continue
        evt_date = idx
        nxt_date = nifty_100[nifty_100["Date"] >= evt_date]["Date"].min()
        if nxt_date not in dct.keys():
            dct[nxt_date] = []
        dct[nxt_date].extend(symbols)

    lst = []
    for date, symbols in dct.items():
        lst.append({"Date": date, "Symbols": "|".join(symbols)})
    df_ex_div = pd.DataFrame.from_records(lst)
    df_ex_div.to_csv("./data/equity/NIFTY_100_EX_DIV_DATE.csv", index=False)

    """Merge both df and remove dividends from corporate events"""
    df_events.columns = ["a"]
    df_ex_div.columns = ["b"]
    df = pd.merge(df_events, df_ex_div, how="outer", left_index=True, right_index=True).fillna("")
    df["symbol"] = df.apply(lambda x: "|".join(list(set(x["b"].split("|")).difference(set(x["a"].split("|"))))), axis=1)
    df.to_csv("./data/equity/NIFTY_100_NON_DIV_EVENTS.csv")

    df_events = pd.read_csv("./data/equity/NIFTY_100_NON_DIV_EVENTS.csv", index_col="Date", parse_dates=["Date"])
    df_close = pd.read_csv("./data/equity/NIFTY_100_CONSTITUENTS_DATA_Close.csv", index_col="Date",
                           parse_dates=["Date"])

    """Prepare Final Dataset"""
    df_earn_ann = df_close.copy().join(df_events)
    df_earn_ann = df_earn_ann.apply(fill_positions, axis=1).drop(columns='symbol')
    df_earn_ann.to_csv("./data/equity/NIFTY_100_POST_EARN_ANN.csv")


def post_events_momentum_strategy(df_open, df_close, df_earn_ann, lookback=90, threshold=0.5):
    ret_c2o = (df_open - df_close.shift(1)) / df_close.shift(1)
    std_c2o = ret_c2o.rolling(lookback).std()

    longs = ((ret_c2o >= threshold * std_c2o) & df_earn_ann).astype(int)
    shorts = ((ret_c2o <= -1 * threshold * std_c2o) & df_earn_ann).astype(int)

    positions = longs - shorts
    positions = positions.div((positions.abs().sum(axis=1)), axis=0)

    df_ret = positions * (df_close - df_open) / df_open

    returns = df_ret.sum(axis=1)

    (1 + returns).cumprod().plot(label=f"Earnings announcement drift")
    plt.legend()
    plt.show()

    print_dashed_line()
    performance_summary(returns, "Earnings announcement drift returns", params={})


# In[3]
def main():
    df_open = read_df("./data/equity/NIFTY_100_STOCKS_Open.csv")
    df_close = read_df("./data/equity/NIFTY_100_STOCKS_Close.csv")

    """Stock Split Mask"""
    df_mask = read_df("./data/equity/NIFTY_100_STOCKS_SPLIT_MASK.csv")
    df_open = df_mask * df_open
    df_close = df_mask * df_close

    df_earn_ann = read_df("./data/equity/NIFTY_100_POST_EARN_ANN.csv")
    post_events_momentum_strategy(df_open, df_close, df_earn_ann)


if __name__ == "__main__":
    main()
