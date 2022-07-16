# In[ ]
import os
from datetime import date

import requests
from nsepy import get_history

from strategies.utils import *


# In[ ]
def fetch_nse_url(url, start_year, end_year):
    """Fetch Data From NSE Website"""
    data = []
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0",
        "Host": "www.nseindia.com",
    }
    home = "https://www.nseindia.com"
    params = {"index": "equities",
              "from_date": "07-05-2021",
              "to_date": "07-05-2022"
              }
    sess = requests.session()
    sess.get(home, headers=headers)
    print(f"fetching {url}")
    for i in range(start_year, end_year):
        params["from_date"] = date(year=i, month=1, day=1).strftime("%d-%m-%Y")
        params["to_date"] = date(year=i + 1, month=1, day=1).strftime("%d-%m-%Y")
        resp = sess.get(url=url, params=params, headers=headers).json()
        print(f"year: {i}, resp len:{len(resp)}")
        data.extend(resp)
    df = pd.DataFrame.from_records(data)
    return df


def fetch_nse_dividends_date(start_year=2015, end_year=2020):
    """Fetch Ex Dividend Dates"""
    url = "https://www.nseindia.com/api/corporates-corporateActions"
    df = fetch_nse_url(url, start_year, end_year)
    df["subject"] = df["subject"].str.lower()
    df = df[(df["subject"].str.contains("dividend"))]
    df.to_csv("./data/equity/IND_EQ_DIVIDENDS.csv", index=False)
    return df


def fetch_nse_event_calender(start_year=2015, end_year=2020):
    """Fetch Corporate Announcements Dates"""
    url = "https://www.nseindia.com/api/event-calendar"
    df = fetch_nse_url(url, start_year, end_year)
    df.to_csv("./data/equity/IND_EQ_EVENTS.csv", index=False)
    return df


def fetch_nifty_constituents_data(index="NIFTY 100", start_date=date(2015, 1, 1), end_date=date(2020, 1, 1)):
    """Fetch Daily Data For Index Components"""
    idx_comp = list(pd.read_csv(f"./data/equity/{index.replace(' ', '_')}_STOCKS_LIST.csv")["Symbol"])

    for ticker in idx_comp:
        data = get_history(ticker, start=start_date, end=end_date)
        data.to_csv(f"./data/equity/{index.replace(' ', '_')}/{ticker}.csv")
        print(f"fetched {ticker}....")


def merge_multiple_tickers():
    """Merge Individual Files Into Two (Open & Close) Price Files"""
    idx_comp = list(pd.read_csv(f"./data/equity/NIFTY_100_STOCKS_LIST.csv")["Symbol"])
    for tp in ["Open", "Close"]:
        df = None
        for ticker in idx_comp:
            data = pd.read_csv(f"./data/equity/NIFTY_100/{ticker}.csv", parse_dates=["Date"],
                               index_col="Date")[[tp]]

            data.columns = [ticker]
            data = data.fillna(-1)
            if df is None:
                df = data
            else:
                df = df.join(data)

            print(f"fetched {ticker}....")
        df.to_csv(f"./data/equity/NIFTY_100_CONSTITUENTS_DATA_{tp}.csv")
        print("done")


def merge_csvs():
    """Merge Forex Files"""
    dfs = []
    for file in os.listdir("./data/forex/"):
        filepath = "./data/forex/" + file
        if os.path.isdir(filepath):
            continue
        df = read_df(filepath)
        df.columns = [file[:-4]]
        dfs.append(df)

    df = reduce(
        lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True), dfs
    )
    df.to_csv("./data/forex/MERGED.csv")
    print(tabulate(df.tail(5), headers=df.columns))


# In[ ]
if __name__ == "__main__":
    pass
    # merge_multiple_tickers()
    # fix_df()
    # merge_currencies_csv()
