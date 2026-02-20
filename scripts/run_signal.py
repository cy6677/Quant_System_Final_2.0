import pandas as pd
import os
import datetime
import requests

from engine.pipeline import QuantPipeline
from risk_monitor import load_portfolio_state, check_market_filter, evaluate_positions

# ==========================================
# 📲 Telegram 配置 (請填入你的資料)
# ==========================================
TG_TOKEN = ""
TG_CHAT_ID = ""

PORTFOLIO_PATH = "data/portfolio_state.json"


def send_telegram_message(message):
    if "YOUR_" in TG_TOKEN:
        print("⚠️ Telegram Token 未設定，只輸出到 console：")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Telegram 通知已發送")
        else:
            print(f"❌ Telegram 發送失敗: {response.text}")
    except Exception as e:
        print(f"❌ 連接 Telegram 失敗: {e}")


def main():
    print("=" * 60)
    print("📡 QUANT SIGNAL: 生成今日交易信號")
    print("=" * 60)

    pipeline = QuantPipeline()

    # Step 1: Universe
    print("\n[Step 1] 檢查 Universe...")
    universe_df, tickers = pipeline.build_universe()

    # Step 2: 更新數據
    print("[Step 2] 檢查/更新數據...")
    pipeline.ensure_prices(tickers)

    # Step 3: 載入數據
    print("[Step 3] 載入數據...")
    price_data = pipeline.load_prices(tickers)
    print(f"✅ 已載入 {len(price_data)} 隻股票")

    # Step 4: 讀取持倉
    print("[Step 4] 讀取持倉...")
    state = load_portfolio_state(PORTFOLIO_PATH)
    current_positions = state.get("positions", {})
    cash_usd = state.get("cash_usd", 0.0)

    # Step 5: 市場風險開關 (SPY < MA200 就暫停加倉)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    market_check = check_market_filter(price_data, today_str, symbol="SPY", ma_window=200)

    # Step 6: 個股風險檢查（50日低 + 最大回撤30%）
    alerts = evaluate_positions(state, price_data, today_str, low_window=50, max_drawdown=-0.30)

    # Step 7: 計算信號
    # （原邏輯保持不變，放返你原本 generate_signals 內容）


if __name__ == "__main__":
    main()