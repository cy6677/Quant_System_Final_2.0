import os
from typing import Tuple, List, Dict, Optional

import pandas as pd

from config import loadconfig, ensuredirs
from layers.data_layer import UniverseProvider, PriceDownloader
from layers.historical_sp500 import HistoricalSP500


class quant_pipeline:
    """
    負責：
    - 構建 universe（美股 + 額外 ETF）
    - 如 config 設定，用歷史 SP500 成份做 point-in-time universe（減少 survivor bias）
    - 確保 price data 存在，並 load 成 {ticker: DataFrame}
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = loadconfig(config_path)
        ensuredirs(self.config)

        self.universe_provider = UniverseProvider(self.config)
        self.price_downloader = PriceDownloader(self.config)

        self.historical_universe: Optional[Dict[pd.Timestamp, List[str]]] = None
        self.historical_df: Optional[pd.DataFrame] = None
        self.tickers: List[str] = []

        # 是否使用 point-in-time SP500
        if self.config["universe"].get("usehistorical", False):
            self._load_historical_universe()

    # ----------------------------
    # Universe (point-in-time)
    # ----------------------------

    def _load_historical_universe(self) -> None:
        """
        用 HistoricalSP500 構建 date -> tickers 映射，做基本 survivor bias 緩解。
        """
        hist = HistoricalSP500(cachedir=self.config["paths"]["rawdata"])
        df = hist.download()
        # df: columns 可能包含 ["date","symbol"] / ["Date","Ticker"]，根據你現有 class 調整
        self.historical_df = df
        self.historical_universe = (
            df.groupby("date")["symbol"].apply(list).to_dict()
        )

    def get_universe_at(self, date: pd.Timestamp) -> List[str]:
        """
        根據日期取得當時 universe：
        - 如果 usehistorical=False，返回固定 tickers + extraetfs
        - 如果 usehistorical=True，用歷史成份按最近 available date 回溯
        """
        extra_etfs: List[str] = self.config["universe"].get("extraetfs", [])

        if not self.config["universe"].get("usehistorical", False):
            # 固定 universe（今日觀察 sample）+ 額外 ETF
            return list(set(self.tickers + extra_etfs))

        # point-in-time 模式
        if self.historical_universe is None:
            return extra_etfs

        dt = pd.to_datetime(date)
        if dt in self.historical_universe:
            base_list = self.historical_universe[dt]
        else:
            # 找最近早於該日期的成份
            available_dates = sorted(self.historical_universe.keys())
            past_dates = [d for d in available_dates if d <= dt]
            if not past_dates:
                base_list = []
            else:
                last_date = past_dates[-1]
                base_list = self.historical_universe[last_date]

        return list(set(base_list + extra_etfs))

    # ----------------------------
    # Universe 構建 & 清洗
    # ----------------------------

    def build_universe(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        返回：
        - universedf: 包含 Ticker 等欄位嘅 DataFrame
        - tickers: universe 內所有 ticker list
        如 config.dataquality.usecleanuniverse=True 則用已清洗檔案。
        """
        clean_path = self.config["paths"]["cleanuniversefile"]
        use_clean = self.config["dataquality"].get("usecleanuniverse", False)

        if use_clean and os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
        else:
            df = self.universe_provider.build_universe()
            # 你舊 code 可能有 missing/spike 檢查，仍可在 UniverseProvider 入面處理
            if use_clean:
                os.makedirs(os.path.dirname(clean_path), exist_ok=True)
                df.to_csv(clean_path, index=False)

        tickers = df["Ticker"].dropna().astype(str).tolist()
        self.tickers = tickers
        return df, tickers

    # ----------------------------
    # Price data
    # ----------------------------

    def ensure_prices(
        self,
        tickers: List[str],
        min_ratio: float = 0.5,
    ) -> List[str]:
        """
        確保已有足夠 price 檔案，否則觸發 downloader。
        min_ratio: 如現有檔案數 < min_ratio * len(tickers)，就觸發批量下載。
        """
        prices_dir = self.config["paths"]["pricesdir"]
        os.makedirs(prices_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(prices_dir) if f.endswith(".parquet")]

        if len(existing_files) >= len(tickers) * min_ratio:
            return tickers

        # 下載缺失 ticker 的價格
        missing = [t for t in tickers if f"{t}.parquet" not in existing_files]
        if missing:
            self.price_downloader.download_prices(missing)

        return tickers

    def load_prices(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        將 parquet 價格載入成 {ticker: DataFrame}。
        DataFrame index=Date，columns 至少包含 ["open","high","low","close","volume"]。
        """
        if tickers is None:
            tickers = self.tickers

        prices_dir = self.config["paths"]["pricesdir"]
        prices: Dict[str, pd.DataFrame] = {}

        for t in tickers:
            path = os.path.join(prices_dir, f"{t}.parquet")
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            # 確保 datetime index
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif df.index.name is None:
                df.index = pd.to_datetime(df.index)
            prices[t] = df.sort_index()

        return prices

QuantPipeline = quant_pipeline
