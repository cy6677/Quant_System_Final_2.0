"""
歷史 S&P 500 成分股下載器
Source: https://github.com/datasets/s-and-p-500-companies
"""
import pandas as pd
import requests
import io
from pathlib import Path
from typing import Optional, List
import datetime

class HistoricalSP500:
    """處理歷史成分股 (point-in-time)"""

    URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

    def __init__(self, cache_dir: str = "data/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "sp500_constituents_history.parquet"
        self._data: Optional[pd.DataFrame] = None

    def download(self, force: bool = False) -> pd.DataFrame:
        """下載歷史成分股，返回 DataFrame 包含 date, symbol"""
        if not force and self.cache_path.exists():
            print("📂 使用快取歷史成分股")
            df = pd.read_parquet(self.cache_path)
        else:
            print("📥 下載歷史成分股 from GitHub...")
            resp = requests.get(self.URL, timeout=15)
            resp.raise_for_status()
            # 嘗試多種方式讀取 CSV
            try:
                df = pd.read_csv(io.StringIO(resp.text))
            except Exception as e:
                print(f"⚠️ 無法用 pandas 直接讀取: {e}")
                # fallback: 嘗試用更寬鬆的方式
                df = pd.read_csv(io.StringIO(resp.text), on_bad_lines='skip')
            
            # 檢查欄位名稱 (可能係大細寫問題)
            print(f"原始欄位: {list(df.columns)}")
            
            # 標準化欄位名稱 (全部轉小寫)
            df.columns = [col.lower().strip() for col in df.columns]
            
            # 確認有 date 同 symbol
            if 'date' not in df.columns:
                # 可能係其他名，嘗試搵類似名
                possible_date_cols = [c for c in df.columns if 'date' in c or 'Date' in c or 'DATE' in c]
                if possible_date_cols:
                    df = df.rename(columns={possible_date_cols[0]: 'date'})
                else:
                    raise ValueError(f"無法找到 date 欄位，現有欄位: {df.columns}")
            
            if 'symbol' not in df.columns:
                possible_sym_cols = [c for c in df.columns if 'symbol' in c or 'Symbol' in c or 'ticker' in c]
                if possible_sym_cols:
                    df = df.rename(columns={possible_sym_cols[0]: 'symbol'})
                else:
                    raise ValueError(f"無法找到 symbol 欄位，現有欄位: {df.columns}")
            
            # 確保 date 係 datetime
            df['date'] = pd.to_datetime(df['date'])
            # 只保留需要嘅欄位
            df = df[['date', 'symbol']].drop_duplicates().sort_values('date')
            df.to_parquet(self.cache_path)
            print(f"✅ 已儲存至 {self.cache_path}")
        self._data = df
        return df

    def get_universe_at(self, date: datetime.date) -> List[str]:
        """取得指定日期嘅成分股 list (point-in-time)"""
        if self._data is None:
            self.download()
        # 選出該日期或之前最後一期的成分股
        mask = self._data['date'] <= pd.to_datetime(date)
        if not mask.any():
            # 如果日期早於數據起點，返回空 list（或者你可以 fallback 到最近一期）
            return []
        last_entry = self._data[mask].groupby('symbol').last().reset_index()
        return last_entry['symbol'].tolist()