import time, os, json
import requests
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

def _to_unix(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())

def download_close(ticker, start, end, retries=4):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")
    if os.path.exists(cache_path):
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True)["close"]
        return s
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = dict(period1=_to_unix(start), period2=_to_unix(end), interval="1d", events="div,splits")
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if r.status_code == 200:
                d = r.json()
                res = d["chart"]["result"][0]
                ts = res["timestamp"]
                adj = res["indicators"]["adjclose"][0]["adjclose"]
                idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
                s = pd.Series(adj, index=idx, name="close").dropna()
                s = s[~s.index.duplicated(keep="first")]
                s.to_csv(cache_path, header=True)
                time.sleep(0.4)
                return s
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {ticker}: {last_err}")
