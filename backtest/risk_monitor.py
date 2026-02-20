import json
import os
import pandas as pd
import numpy as np


def load_portfolio_state(path):
    if not os.path.exists(path):
        return {"cash_usd": 0.0, "positions": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_portfolio_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _get_hist_slice(df, asof):
    asof = pd.to_datetime(asof)
    if asof in df.index:
        return df.loc[:asof]
    loc_idx = df.index.get_indexer([asof], method="pad")[0]
    if loc_idx == -1:
        return None
    return df.iloc[: loc_idx + 1]


def _latest_close(df, asof):
    hist = _get_hist_slice(df, asof)
    if hist is None or hist.empty:
        return None
    return float(hist["Close"].iloc[-1])


# --------------------------------------------------
# check_market_filter — 增強版（SPY MA200 + VIX + Breadth）
# --------------------------------------------------

def check_market_filter(
    price_data: dict,
    asof,
    spy_symbol: str = "SPY",
    vix_symbol: str = "^VIX",
    ma_window: int = 200,
    vix_threshold: float = 30.0,
    breadth_tickers: list = None,
    breadth_ma: int = 50,
    breadth_min_pct: float = 0.40,
) -> dict:
    """
    三重 market regime filter：
    1. SPY > MA200        — 趨勢牛市
    2. VIX < vix_threshold — 恐慌不高
    3. Market Breadth > breadth_min_pct — 市場廣度足夠

    :return: {
        "ok": bool,          # 全部通過才 True
        "spy_ok": bool,
        "vix_ok": bool,
        "breadth_ok": bool,
        "message": str
    }
    """
    messages = []
    spy_ok = True
    vix_ok = True
    breadth_ok = True

    # --- 1. SPY MA200 ---
    if spy_symbol in price_data:
        df = price_data[spy_symbol]
        hist = _get_hist_slice(df, asof)
        if hist is not None and len(hist) >= ma_window:
            close_col = "Close" if "Close" in hist.columns else "close"
            ma = hist[close_col].tail(ma_window).mean()
            latest = float(hist[close_col].iloc[-1])
            spy_ok = latest >= ma
            messages.append(
                f"{'✅' if spy_ok else '🚨'} SPY {latest:.2f} | MA{ma_window} {ma:.2f}"
            )
        else:
            messages.append(f"⚠️ SPY 歷史不足 {ma_window} 日，視為通過")
    else:
        messages.append(f"⚠️ 找不到 {spy_symbol}，SPY filter 跳過")

    # --- 2. VIX Filter ---
    if vix_symbol in price_data:
        df = price_data[vix_symbol]
        hist = _get_hist_slice(df, asof)
        if hist is not None and not hist.empty:
            close_col = "Close" if "Close" in hist.columns else "close"
            vix_val = float(hist[close_col].iloc[-1])
            vix_ok = vix_val < vix_threshold
            messages.append(
                f"{'✅' if vix_ok else '🚨'} VIX {vix_val:.1f} | threshold {vix_threshold:.0f}"
            )
    else:
        messages.append("⚠️ VIX 資料不存在，跳過 VIX filter")

    # --- 3. Market Breadth（% of stocks above MA50）---
    if breadth_tickers and len(breadth_tickers) > 0:
        above_ma = 0
        total_checked = 0
        for ticker in breadth_tickers:
            if ticker not in price_data:
                continue
            df = price_data[ticker]
            hist = _get_hist_slice(df, asof)
            if hist is None or len(hist) < breadth_ma:
                continue
            close_col = "Close" if "Close" in hist.columns else "close"
            ma_val = hist[close_col].tail(breadth_ma).mean()
            latest_px = float(hist[close_col].iloc[-1])
            total_checked += 1
            if latest_px >= ma_val:
                above_ma += 1

        if total_checked > 0:
            breadth_pct = above_ma / total_checked
            breadth_ok = breadth_pct >= breadth_min_pct
            messages.append(
                f"{'✅' if breadth_ok else '🚨'} Breadth {breadth_pct:.1%} "
                f"({above_ma}/{total_checked} > MA{breadth_ma}) | min {breadth_min_pct:.0%}"
            )
        else:
            messages.append("⚠️ Breadth tickers 無足夠資料，跳過")
    else:
        breadth_ok = True  # 無提供 tickers，視為通過

    overall_ok = spy_ok and vix_ok and breadth_ok
    return {
        "ok":         overall_ok,
        "spy_ok":     spy_ok,
        "vix_ok":     vix_ok,
        "breadth_ok": breadth_ok,
        "message":    " | ".join(messages),
    }


# --------------------------------------------------
# evaluate_positions — 修正 drawdown 基準
# --------------------------------------------------

def evaluate_positions(
    state: dict,
    price_data: dict,
    asof,
    low_window: int = 50,
    max_drawdown_from_entry: float = -0.20,
    max_drawdown_from_peak: float = -0.30,
) -> list:
    """
    逐持倉評估風險，觸發以下 alert：
    1. 價格跌破 low_window 日低位
    2. 相對 avg_cost（入場價）的回撤超過 max_drawdown_from_entry
    3. 相對入場後最高價的回撤超過 max_drawdown_from_peak

    ✅ 修正：drawdown 從 entry_price 計，唔係從全歷史最高計。
    """
    alerts = []
    positions = state.get("positions", {})

    for ticker, info in positions.items():
        if ticker not in price_data:
            alerts.append(f"⚠️ {ticker} 無價格資料，跳過風險檢查。")
            continue

        df = price_data[ticker]
        hist = _get_hist_slice(df, asof)
        if hist is None or len(hist) < low_window:
            alerts.append(f"⚠️ {ticker} 歷史不足 {low_window} 日，跳過風險檢查。")
            continue

        close_col = "Close" if "Close" in hist.columns else "close"
        latest = float(hist[close_col].iloc[-1])
        low_n = float(hist[close_col].tail(low_window).min())

        entry_price = info.get("avg_cost", None)
        entry_date  = info.get("entry_date", None)

        if entry_price is None or entry_price <= 0:
            entry_price = latest
            entry_date  = None

        # --- 1. 跌破 N 日低位 ---
        if latest < low_n:
            alerts.append(
                f"🚨 {ticker} 跌破 {low_window}日低位 ({latest:.2f} < {low_n:.2f})"
            )

        # --- 2. 相對入場價回撤（From Entry）---
        dd_from_entry = (latest - entry_price) / entry_price
        if dd_from_entry <= max_drawdown_from_entry:
            alerts.append(
                f"🚨 {ticker} 入場後回撤 {dd_from_entry*100:.1f}% "
                f"(entry {entry_price:.2f} → now {latest:.2f}, "
                f"limit {max_drawdown_from_entry*100:.0f}%)"
            )

        # --- 3. 相對入場後最高價回撤（From Peak Since Entry）---
        if entry_date is not None:
            try:
                entry_dt = pd.to_datetime(entry_date)
                hist_since_entry = hist.loc[hist.index >= entry_dt]
            except Exception:
                hist_since_entry = hist
        else:
            hist_since_entry = hist

        if not hist_since_entry.empty:
            peak_since_entry = float(hist_since_entry[close_col].max())
            dd_from_peak = (latest - peak_since_entry) / peak_since_entry if peak_since_entry > 0 else 0.0
            if dd_from_peak <= max_drawdown_from_peak:
                alerts.append(
                    f"🚨 {ticker} 入場後最高位回撤 {dd_from_peak*100:.1f}% "
                    f"(peak {peak_since_entry:.2f} → now {latest:.2f}, "
                    f"limit {max_drawdown_from_peak*100:.0f}%)"
                )

    return alerts
