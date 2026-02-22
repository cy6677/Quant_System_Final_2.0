import datetime
import os
import time
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from config import load_config


class UniverseProvider:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _normalize_ticker(self, ticker):
        if pd.isna(ticker):
            return None
        clean = str(ticker).strip().upper().replace(".", "-").replace(" ", "")
        return clean if clean else None

    def _is_cache_valid(self):
        universe_file = self.config["paths"]["universe_file"]
        if not os.path.exists(universe_file):
            return False
        last_modified = os.path.getmtime(universe_file)
        days_old = (datetime.datetime.now().timestamp() - last_modified) / 86400
        return days_old < self.config["universe"]["cache_days"]

    def _load_cache(self):
        universe_file = self.config["paths"]["universe_file"]
        if os.path.exists(universe_file):
            return pd.read_csv(universe_file)
        return None

    def _save_cache(self, df):
        os.makedirs(os.path.dirname(self.config["paths"]["universe_file"]), exist_ok=True)
        df.to_csv(self.config["paths"]["universe_file"], index=False)

    def fetch_sp500(self):
        print("📥 抓取 S&P 500 成分股...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        retry = self.config["download"]["retry"]
        sleep_s = self.config["download"]["sleep_between_retries"]
        timeout = self.config["download"]["request_timeout"]

        for i in range(retry):
            try:
                resp = requests.get(url, headers=self.headers, timeout=timeout)
                resp.raise_for_status()
                tables = pd.read_html(StringIO(resp.text))
                df = tables[0]

                # 穩健地找到 Symbol 欄位
                symbol_col = None
                for col in df.columns:
                    if 'Symbol' in col or 'Ticker' in col:
                        symbol_col = col
                        break
                if symbol_col is None:
                    symbol_col = df.columns[0]  # 通常第一個是 Symbol

                df = df.rename(columns={
                    symbol_col: "Ticker",
                    "GICS Sector": "Sector",
                    "GICS Sub-Industry": "Industry",
                })
                df["Ticker"] = df["Ticker"].apply(self._normalize_ticker)
                df["Type"] = "Stock"
                df = df.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"])
                cols = ["Ticker", "Sector", "Industry", "Type"]
                return df[cols].copy()
            except Exception as e:
                print(f"⚠️ 抓取失敗 (第 {i+1}/{retry})：{e}")
                time.sleep(sleep_s)

        cached = self._load_cache()
        if cached is not None:
            print("✅ Wikipedia 抓取失敗，改用本地 cache")
            return cached
        raise RuntimeError("❌ 無法取得 S&P500 成分股，也找不到 cache")

    def build_universe(self, include_extra_etf=True):
        if self._is_cache_valid():
            print("✅ 使用快取 Universe")
            return self._load_cache()

        df = self.fetch_sp500()
        if include_extra_etf:
            extra = pd.DataFrame(
                {
                    "Ticker": self.config["universe"]["extra_etfs"],
                    "Sector": "ETF",
                    "Industry": "ETF",
                    "Type": "ETF",
                }
            )
            df = pd.concat([df, extra], ignore_index=True)

        df["Sector"] = df["Sector"].fillna("Unknown")
        df["Industry"] = df["Industry"].fillna("Unknown")
        self._save_cache(df)
        return df


class PriceDownloader:
    def __init__(self, config=None):
        self.config = config or load_config()
        os.makedirs(self.config["paths"]["prices_dir"], exist_ok=True)

    def _normalize_yf_columns(self, df, ticker):
        if isinstance(df.columns, pd.MultiIndex):
            # 嘗試取出特定 ticker 的數據
            if ticker in df.columns.get_level_values(1):
                df = df.xs(ticker, axis=1, level=1)
            else:
                # 否則取第一層（標準欄位）
                df.columns = df.columns.get_level_values(0)
        # 標準化列名為小寫
        df.columns = [str(c).lower() for c in df.columns]
        return df

    def _download_one(self, ticker) -> Tuple[str, pd.DataFrame | None]:
        retry = self.config["download"]["retry"]
        sleep_s = self.config["download"]["sleep_between_retries"]
        start = self.config["data"]["price"]["start_date"]
        end = self.config["data"]["price"]["end_date"]

        for _ in range(retry):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                time.sleep(0.2)
                if df.empty:
                    continue
                df = self._normalize_yf_columns(df, ticker)
                required = {"close", "high", "low", "volume"}
                if not required.issubset(set(df.columns)):
                    continue
                return ticker, df
            except Exception:
                time.sleep(sleep_s)
        return ticker, None

    def download_all(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        results = {}
        failed = []
        max_workers = self.config["download"]["max_workers"]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ticker, df in tqdm(executor.map(self._download_one, tickers), total=len(tickers)):
                if df is None or df.empty:
                    failed.append(ticker)
                else:
                    results[ticker] = df
                    df.to_parquet(os.path.join(self.config["paths"]["prices_dir"], f"{ticker}.parquet"))

        if failed:
            pd.DataFrame({"Ticker": failed}).to_csv(self.config["paths"]["failed_file"], index=False)
        return results

    def load_prices(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        data = {}
        for t in tickers:
            path = os.path.join(self.config["paths"]["prices_dir"], f"{t}.parquet")
            if os.path.exists(path):
                data[t] = pd.read_parquet(path)
        return data
