#!/usr/bin/env python3
"""
產生 SimFin ID 與 Ticker 的 mapping 檔案
用法: python -m scripts.generate_simfin_mapping
"""
import pandas as pd
import simfin as sf
from simfin.names import SIMFIN_ID, TICKER
from config import load_config
from layers.data_layer import UniverseProvider


def _extract_ticker_column(df):
    if TICKER in df.columns:
        return df, TICKER
    if "Ticker" in df.columns:
        return df, "Ticker"
    if df.index.name in [TICKER, "Ticker"]:
        df = df.reset_index()
        return df, df.columns[0]
    if isinstance(df.index, pd.MultiIndex):
        if TICKER in df.index.names or "Ticker" in df.index.names:
            df = df.reset_index()
            ticker_col = TICKER if TICKER in df.columns else "Ticker"
            return df, ticker_col
    raise KeyError(f"找不到 Ticker 欄位，現有欄位: {list(df.columns)} | index: {df.index.names}")


def main():
    config = load_config()

    # 取得 S&P500 tickers
    u = UniverseProvider()
    universe_df = u.build_universe(include_extra_etf=False)
    tickers = set(universe_df["Ticker"].dropna().unique().tolist())

    # 讀 SimFin companies
    sf.set_api_key(config["data"]["fundamentals"].get("api_key"))
    sf.set_data_dir(config["paths"]["raw_data"])
    companies = sf.load_companies(market="us")

    companies, ticker_col = _extract_ticker_column(companies)

    # 找 SimFinId 欄位
    id_col = None
    for col in [SIMFIN_ID, "SimFinId"]:
        if col in companies.columns:
            id_col = col
            break
    if id_col is None:
        raise KeyError(f"companies 冇 SimFinId 欄位: {list(companies.columns)}")

    companies["TickerNorm"] = companies[ticker_col].astype(str).str.strip().str.upper()

    mapping = companies[companies["TickerNorm"].isin(tickers)][[ticker_col, id_col]].drop_duplicates()
    mapping = mapping.rename(columns={ticker_col: "Ticker", id_col: "SimFinId"})

    # 輸出 mapping
    out_path = config.get("data", {}).get("fundamentals", {}).get("mapping_file", "data/simfin_mapping.csv")
    mapping.to_csv(out_path, index=False)
    print(f"✅ 已產生 mapping：{out_path}（共 {len(mapping)} 筆）")


if __name__ == "__main__":
    main()
