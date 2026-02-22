import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from config import load_config
from layers.data_hub import DataHub
from layers.data_layer import UniverseProvider
from utils.validation import validate_missing, validate_spikes, validate_spikes_by_ticker
from utils.reporting import append_report
from utils.logging import get_logger


def run_update():
    cfg = load_config()
    logger = get_logger("update_pipeline", cfg["paths"]["log_file"])

    hub = DataHub()
    processed_dir = Path(hub.price.config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. 取得 universe
    u_provider = UniverseProvider()
    universe_df = u_provider.build_universe()
    tickers = universe_df["Ticker"].dropna().unique().tolist()
    logger.info(f"Universe Ready: {len(tickers)} tickers")

    # 2. 下載價格數據 (force 確保最新)
    hub.price.download(symbols=tickers, force=True)
    close_df = hub.price.load()                     # pivot table: date x ticker
    ohlcv_df = hub.price.load_ohlcv()               # multi-index: (date, ticker)

    # 3. 建立 TechnicalIndicator 物件
    tech = hub.build_technical()

    # 4. 計算常用技術指標
    print("📊 計算技術指標...")
    tech.add_rsi(14)
    tech.add_atr(14)
    tech.add_adx(14)
    tech.add_vwap()

    # 5. 為策略A預先計算所需指標
    print("📊 預先計算策略A指標...")
    tech.add_atr(5)
    tech.add_atr(20)

    # 計算 rolling high/low (20日)
    high_max_dict = {}
    low_min_dict = {}
    for ticker in close_df.columns:
        high_max_dict[ticker] = close_df[ticker].rolling(20).max()
        low_min_dict[ticker] = close_df[ticker].rolling(20).min()

    high_max_df = pd.DataFrame(high_max_dict, index=close_df.index)
    low_min_df = pd.DataFrame(low_min_dict, index=close_df.index)
    tech.indicators['HIGH_MAX_20'] = high_max_df
    tech.indicators['LOW_MIN_20'] = low_min_df

    # 計算 20日平均成交量
    if ohlcv_df is not None:
        volume_series = ohlcv_df['Volume']
        volume_pivot = volume_series.unstack(level=1)
        vol_ma20 = volume_pivot.rolling(20).mean()
        tech.indicators['VOL_MA20'] = vol_ma20
    else:
        logger.warning("⚠️ 無法計算成交量指標，將用 close 代替（不準確）")
        tech.indicators['VOL_MA20'] = close_df.rolling(20).mean() * 1e6

    # 6. 儲存 indicators
    tech.save_indicators(processed_dir / "indicators")
    tech.save_unified(processed_dir / "indicators_all.parquet")
    logger.info(f"✅ 指標已儲存至 {processed_dir / 'indicators_all.parquet'}")

    # 7. 嘗試下載基本面 (如果沒有 API key 會 fail，但唔影響)
    try:
        hub.fundamentals.download_quarterly(tickers)
    except Exception as e:
        logger.warning(f"⚠️ 基本面下載失敗 (可忽略): {e}")

    try:
        hub.macro.download_all()
    except Exception as e:
        logger.warning(f"⚠️ Macro 下載失敗 (可忽略): {e}")

    # 8. 數據質量檢查
    max_missing = cfg["data_quality"]["max_missing_pct"]
    max_spikes = cfg["data_quality"]["max_spike_count"]

    missing = validate_missing(close_df, max_missing_pct=max_missing)
    spike_counts = validate_spikes_by_ticker(close_df, threshold=0.5)

    bad_missing = set(missing.index)
    bad_spikes = set(spike_counts[spike_counts > max_spikes].index)

    # 9. 生成 clean universe
    clean_df = universe_df[~universe_df["Ticker"].isin(bad_missing | bad_spikes)].copy()
    clean_path = Path(cfg["paths"]["clean_universe_file"])
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)

    logger.info(f"Clean universe saved: {clean_path} (rows={len(clean_df)})")

    if bad_missing:
        logger.info(f"Removed for missing: {sorted(list(bad_missing))}")
    if bad_spikes:
        logger.info(f"Removed for spikes: {sorted(list(bad_spikes))}")

    # 10. 記錄報告
    missing_count = len(missing)
    spike_total = validate_spikes(close_df)

    report_path = processed_dir / "update_report.csv"
    append_report(
        report_path,
        {
            "rows_price": len(close_df),
            "missing_cols": missing_count,
            "spike_count": spike_total,
            "removed_missing": len(bad_missing),
            "removed_spikes": len(bad_spikes),
            "clean_universe_rows": len(clean_df),
        },
    )

    logger.info("✅ Update completed.")


if __name__ == "__main__":
    run_update()
