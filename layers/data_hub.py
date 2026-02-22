"""
data_hub.py - 統一數據入口

整合 PriceManager (取代舊 PriceLoader) 以及 FundamentalsLoader, MacroLoader。
兼容原有 DataHub 介面，內部使用 data_layer.PriceDownloader 實現高效下載。
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

from layers.data_layer import PriceDownloader
from layers.fundamentals_layer import FundamentalsLoader
from layers.macro_layer import MacroLoader
from layers.technical_layer import TechnicalIndicator
from config import load_config


class PriceManager:
    """
    價格數據管理器，取代舊 PriceLoader
    提供 download, load, load_ohlcv 方法，內部使用 PriceDownloader 下載並讀取 parquet
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.downloader = PriceDownloader(config)
        self.prices_dir = Path(config["paths"]["prices_dir"])

    def download(self, symbols: Optional[List[str]] = None,
                 start: Optional[str] = None, end: Optional[str] = None,
                 force: bool = False) -> pd.DataFrame:
        """
        下載指定股票數據（若 force=True 或檔案缺失則下載）
        返回合併後嘅 DataFrame，欄位: date, Symbol, Open, High, Low, Close, Volume
        """
        if symbols is None:
            symbols = self.config["universe"].get("symbols", [])

        if force:
            self.downloader.download_all(symbols)
        else:
            # 檢查缺失
            missing = [t for t in symbols if not (self.prices_dir / f"{t}.parquet").exists()]
            if missing:
                self.downloader.download_all(missing)

        # 返回合併數據（兼容舊版 PriceLoader.download 返回值）
        return self._load_combined(symbols, start, end)

    def load(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        載入所有股票收盤價，返回 pivot table (date x ticker)
        """
        df = self._load_combined(None, start, end)
        if df.empty:
            return pd.DataFrame()
        pivot = df.pivot(index="date", columns="Symbol", values="Close")
        return pivot

    def load_ohlcv(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        載入所有股票 OHLCV 數據，返回 MultiIndex DataFrame (date, Symbol)
        """
        df = self._load_combined(None, start, end)
        if df.empty:
            return pd.DataFrame()
        df = df.set_index(["date", "Symbol"]).sort_index()
        return df

    def _load_combined(self, symbols: Optional[List[str]] = None,
                       start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        內部方法：從 parquet 讀取指定 symbols 嘅數據，合併成一個 DataFrame
        """
        if symbols is None:
            symbols = [f.stem for f in self.prices_dir.glob("*.parquet")]

        dfs = []
        for ticker in symbols:
            path = self.prices_dir / f"{ticker}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                # 確保欄位名稱小寫
                df.columns = [col.lower() for col in df.columns]
                df["symbol"] = ticker
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])

        if start:
            combined = combined[combined["date"] >= pd.to_datetime(start)]
        if end:
            combined = combined[combined["date"] <= pd.to_datetime(end)]

        return combined


class DataHub:
    """
    統一數據入口，提供價格、基本面、宏觀、技術指標
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.price = PriceManager(self.config)
        self.fundamentals = FundamentalsLoader(config=self.config)
        self.macro = MacroLoader(config=self.config)

    def load_price(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """等價於 self.price.load()"""
        return self.price.load(start, end)

    def load_ohlcv(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """等價於 self.price.load_ohlcv()"""
        return self.price.load_ohlcv(start, end)

    def build_technical(self, start: Optional[str] = None, end: Optional[str] = None) -> TechnicalIndicator:
        """
        建立技術指標計算器，需要先有價格數據
        """
        close_df = self.price.load(start, end)
        ohlcv_df = self.price.load_ohlcv(start, end)
        return TechnicalIndicator(close_df, ohlcv_df)
