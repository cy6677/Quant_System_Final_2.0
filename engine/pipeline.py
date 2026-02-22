import os
from typing import Tuple, List, Dict, Optional

import pandas as pd

from config import load_config, ensuredirs
from layers.data_layer import UniverseProvider, PriceDownloader
from layers.historical_sp500 import HistoricalSP500


class quant_pipeline:
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        ensuredirs(self.config)

        self.universe_provider = UniverseProvider(self.config)
        self.price_downloader = PriceDownloader(self.config)

        self.historical_universe: Optional[Dict[pd.Timestamp, List[str]]] = None
        self.historical_df: Optional[pd.DataFrame] = None
        self.tickers: List[str] = []

        if self.config["universe"].get("use_historical", False):
            self._load_historical_universe()

    def _load_historical_universe(self) -> None:
        hist = HistoricalSP500(cachedir=self.config["paths"]["raw_data"])
        df = hist.download()
        # 確保欄位名稱一致
        if "date" in df.columns and "symbol" in df.columns:
            self.historical_df = df
            self.historical_universe = (
                df.groupby("date")["symbol"].apply(list).to_dict()
            )
        else:
            print("警告: 歷史成分股格式不符，停用 point-in-time")
            self.historical_universe = {}

    def get_universe_at(self, date: pd.Timestamp) -> List[str]:
        extra_etfs: List[str] = self.config["universe"].get("extra_etfs", [])

        if not self.config["universe"].get("use_historical", False):
            return list(set(self.tickers + extra_etfs))

        if self.historical_universe is None:
            return extra_etfs

        dt = pd.to_datetime(date)
        if dt in self.historical_universe:
            base_list = self.historical_universe[dt]
        else:
            available_dates = sorted(self.historical_universe.keys())
            past_dates = [d for d in available_dates if d <= dt]
            if not past_dates:
                base_list = []
            else:
                last_date = past_dates[-1]
                base_list = self.historical_universe[last_date]

        return list(set(base_list + extra_etfs))

    def build_universe(self) -> Tuple[pd.DataFrame, List[str]]:
        clean_path = self.config["paths"]["clean_universe_file"]
        use_clean = self.config["data_quality"].get("use_clean_universe", False)

        if use_clean and os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
        else:
            df = self.universe_provider.build_universe()
            if use_clean:
                os.makedirs(os.path.dirname(clean_path), exist_ok=True)
                df.to_csv(clean_path, index=False)

        tickers = df["Ticker"].dropna().astype(str).tolist()
        self.tickers = tickers
        return df, tickers

    def ensure_prices(
        self,
        tickers: List[str],
        min_ratio: float = 0.5,
    ) -> List[str]:
        prices_dir = self.config["paths"]["prices_dir"]
        os.makedirs(prices_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(prices_dir) if f.endswith(".parquet")]

        if len(existing_files) >= len(tickers) * min_ratio:
            return tickers

        missing = [t for t in tickers if f"{t}.parquet" not in existing_files]
        if missing:
            self.price_downloader.download_all(missing)

        return tickers

    def load_prices(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        if tickers is None:
            tickers = self.tickers

        prices_dir = self.config["paths"]["prices_dir"]
        prices: Dict[str, pd.DataFrame] = {}

        for t in tickers:
            path = os.path.join(prices_dir, f"{t}.parquet")
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif df.index.name is None:
                df.index = pd.to_datetime(df.index)
            prices[t] = df.sort_index()

        return prices


QuantPipeline = quant_pipeline
