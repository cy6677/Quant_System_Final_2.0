import simfin as sf
from simfin.names import *
import pandas as pd
from pathlib import Path
from config import load_config


class FundamentalsLoader:
    def __init__(self, api_key=None, config=None):
        self.config = config or load_config()
        sf.set_api_key(api_key or self.config['data']['fundamentals'].get('api_key'))
        sf.set_data_dir(self.config['paths']['raw_data'])

    def _load_mapping(self):
        mapping_path = Path(
            self.config.get("data", {})
                .get("fundamentals", {})
                .get("mapping_file", "data/simfin_mapping.csv")
        )
        if not mapping_path.exists():
            return None
        df = pd.read_csv(mapping_path)
        df.columns = [c.strip() for c in df.columns]
        if "Ticker" not in df.columns or "SimFinId" not in df.columns:
            raise KeyError(f"mapping 檔案欄位錯誤，需要 Ticker, SimFinId: {list(df.columns)}")
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        return df

    def _get_simfin_ids(self, symbols):
        mapping = self._load_mapping()
        if mapping is not None:
            return mapping[mapping["Ticker"].isin(symbols)]["SimFinId"].dropna().unique().tolist()

        # fallback: 從 companies 取得 ticker
        companies = sf.load_companies(market='us')
        ticker_col = None
        for col in [TICKER, "Ticker"]:
            if col in companies.columns:
                ticker_col = col
                break
        id_col = None
        for col in [SIMFIN_ID, "SimFinId"]:
            if col in companies.columns:
                id_col = col
                break
        if ticker_col is None or id_col is None:
            print("⚠️ companies 冇 Ticker 或 SimFinId 欄位，無法過濾")
            return []
        companies["TickerNorm"] = companies[ticker_col].astype(str).str.strip().str.upper()
        return companies[companies["TickerNorm"].isin(symbols)][id_col].unique().tolist()

    def _get_id_col(self, df):
        for col in [SIMFIN_ID, "SimFinId"]:
            if col in df.columns:
                return col
        return None

    def download_quarterly(self, symbols):
        df = sf.load_income(variant='quarterly', market='us')
        simfin_ids = self._get_simfin_ids(symbols)
        id_col = self._get_id_col(df)
        if id_col is None:
            raise KeyError("找不到 SimFinId 欄位於 income 數據中")

        if simfin_ids and len(simfin_ids) > 0:
            df = df[df[id_col].isin(simfin_ids)]

        # 只保留常用的欄位（如果存在）
        possible_cols = [id_col, REPORT_DATE, CURRENCY, REVENUE, GROSS_PROFIT,
                         OPERATING_INCOME, NET_INCOME, EPS]
        keep_cols = [c for c in possible_cols if c in df.columns]
        df = df[keep_cols]

        out_path = Path(self.config['paths']['raw_data']) / "fundamentals_quarterly.parquet"
        df.to_parquet(out_path)
        return df

    def load_latest(self, as_of_date):
        df = pd.read_parquet(Path(self.config['paths']['raw_data']) / "fundamentals_quarterly.parquet")
        id_col = self._get_id_col(df)
        if id_col is None:
            raise KeyError("找不到 SimFinId 欄位")
        report_date_col = REPORT_DATE if REPORT_DATE in df.columns else "Report Date"
        df = df[df[report_date_col] <= pd.to_datetime(as_of_date)]
        latest = df.sort_values(report_date_col).groupby(id_col).last().reset_index()
        return latest
