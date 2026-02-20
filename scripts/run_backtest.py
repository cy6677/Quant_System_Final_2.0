#!/usr/bin/env python3
"""
萬能回測腳本
用法：
    python scripts/run_backtest.py --strategy long      # 只 run LongTerm
    python scripts/run_backtest.py --strategy a         # 策略 A (原始版)
    python scripts/run_backtest.py --strategy a_fast    # 策略 A (快速版)
    python scripts/run_backtest.py --strategy b         # 策略 B
    python scripts/run_backtest.py --strategy c         # 策略 C
    python scripts/run_backtest.py --strategy all       # 比較 long, a, b, c
    唔加參數預設 = long
"""
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import load_config
from engine.pipeline import QuantPipeline
from backtest.universal_backtester import UniversalBacktester, TransactionCostModel, PerformanceAnalyzer

from strategies.strategy_long_term import LongTermStrategy
from strategies.strategy_a import StrategyA_VCPBreakout
from strategies.strategy_b import StrategyB_AVWAPPullback
from strategies.strategy_c import StrategyC_BollingerReversion

def ticker_classifier(ticker: str, extra_etfs: list) -> str:
    return "ETF" if ticker in extra_etfs else "default"

def run_single_strategy(
    strategy_class,
    strategy_params: dict,
    prices_dict: dict,
    start_date: str,
    end_date: str,
    backtester_params: dict,
    cost_model,
    pipeline,
    benchmark_series=None,
) -> dict:
    strategy = strategy_class(**strategy_params)
    strategy.pipeline = pipeline

    backtester = UniversalBacktester(**backtester_params, cost_model=cost_model)
    equity_df = backtester.run(
        strategy=strategy,
        prices=prices_dict,          # ← 用 prices_dict 呢個參數
        start_date=start_date,
        end_date=end_date,
    )

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(equity_df, benchmark_series=benchmark_series)
    metrics["Strategy"] = strategy.name if hasattr(strategy, "name") else "Strategy"
    return metrics, equity_df

    # 後面如果仲有 analyzer、metrics，就照你原本 code 保留



    # 如果策略有統計報告，就印出（唔影響 equity_df）
    if hasattr(strategy, 'report_stats'):
        strategy.report_stats()

    if equity_df is None or equity_df.empty:
        return {"Error": "No trades"}, equity_df

    analyzer = PerformanceAnalyzer()
    metrics, _ = analyzer.analyze(equity_df, benchmark_prices=benchmark_series)
    metrics["Trade Count"] = len(backtester.trade_log) // 2
    return metrics, equity_df

def main():
    parser = argparse.ArgumentParser(description="萬能回測腳本")
    parser.add_argument("--strategy", type=str, default="long",
                        choices=["long", "a", "a_fast", "b", "c", "all"],
                        help="策略: long / a / a_fast / b / c / all")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 70)
    print(f"🚀 回測策略: {args.strategy.upper()}")
    print("=" * 70)

    pipeline = QuantPipeline()
    print("\n📦 準備 Universe...")
    cfg = load_config()

    # 簡化測試：只用 SPY，用 yfinance 直接抓數
    import yfinance as yf
    print("⚙️ TEST_MODE: 只用 SPY 價格數據做測試")
    tickers = ["SPY"]
    data = yf.download(
        "SPY",
        start=cfg["data"]["price"]["start_date"],
        end=cfg["data"]["price"]["end_date"],
        progress=False,
    )
    data = data.rename(columns=str.lower)
    if "adj close" in data.columns and "close" not in data.columns:
        data = data.rename(columns={"adj close": "close"})
    data.index.name = "Date"
    prices_dict = {"SPY": data}

    print(f"✅ Universe: {len(tickers)} 隻股票")
    print("📥 載入價格數據...")
    print(f"✅ 成功載入 {len(prices_dict)} 隻股票")


    start_date = cfg["data"]["price"]["start_date"] or "2015-01-01"
    end_date = cfg["data"]["price"]["end_date"] or datetime.now().strftime("%Y-%m-%d")

    cost_cfg = cfg.get("cost_model", {})
    extra_etfs = cfg["universe"].get("extra_etfs", [])
    cost_model = TransactionCostModel(
        commission_rate=cfg["backtest"]["commission_rate"],
        slippage=cfg["backtest"]["slippage"],
        min_commission=cfg["backtest"]["min_commission"],
        per_ticker_costs=cost_cfg,
        ticker_classifier=lambda t: ticker_classifier(t, extra_etfs)
    )

    backtester_params = {
        "initial_capital": cfg["backtest"]["initial_capital"],
        "calendar_ticker": cfg["backtest"]["calendar_ticker"],
        "allow_fractional": cfg["backtest"]["allow_fractional"],
        "min_trade_value": 5.0,
    }

    benchmark = price_data.get("SPY")
    bench_series = benchmark["Close"] if benchmark is not None else None

    strategies = {
        "long": {
            "name": "Long Term Strategy",
            "class": LongTermStrategy,
            "params": {
                "top_n": 15,
                "max_sector_count": 4,
                "rebalance_freq": "Q",
                "fundamentals_df": universe_df,
            },
        },
        "a": {
            "name": "Strategy A - VCP Breakout (Original)",
            "class": StrategyA_VCPBreakout,
            "params": {
                "risk_per_trade": 0.005,
                "min_price": 5.0,
                "min_avg_dollar_vol": 20e6,
                "max_hold_days": 30,
                "atr_short": 5,
                "atr_long": 20,
                "contraction_ratio": 0.7,
                "lookback_high": 20,
                "breakout_buffer": 0.1,
                "rvol_threshold": 1.5,
                "stop_loss_atr": 1.5,
                "target_r": 2.0,
                "trail_atr": 3.0,
            },
        },

        "b": {
            "name": "Strategy B - AVWAP Pullback",
            "class": StrategyB_AVWAPPullback,
            "params": {
                "risk_per_trade": 0.005,
                "min_price": 5.0,
                "min_avg_dollar_vol": 20e6,
                "max_hold_days": 20,
                "anchor_type": "maxvolume",
                "avwap_touch_pct": 0.25,
                "rsi_period": 2,
                "rsi_oversold": 10,
                "confirm_reversal": True,
                "trend_ma_short": 50,
                "trend_ma_long": 200,
                "stop_loss_atr": 2.0,
                "target_r": 2.0,
            },
        },
        "c": {
            "name": "Strategy C - Bollinger Reversion",
            "class": StrategyC_BollingerReversion,
            "params": {
                "risk_per_trade": 0.005,
                "min_price": 5.0,
                "min_avg_dollar_vol": 20e6,
                "max_hold_days": 5,
                "bb_period": 20,
                "bb_std": 2,
                "adx_period": 14,
                "adx_threshold": 25,
                "adx_disable": 30,
                "stop_loss_atr": 1.5,
            },
        },
    }

    if args.strategy == "all":
        results = []
        compare = ["long", "a", "b", "c"]
        for key in compare:
            s = strategies[key]
            print(f"\n▶️ 執行: {s['name']}")
            try:
                metrics, _ = run_single_strategy(
                    strategy_class=s["class"],
                    strategy_params=s["params"],
                    prices_dict=price_data,
                    start_date=start_date,
                    end_date=end_date,
                    backtester_params=backtester_params,
                    cost_model=cost_model,
                    pipeline=pipeline,
                    benchmark_series=bench_series,
                )
                metrics["Strategy"] = s["name"]
                results.append(metrics)
                print(f"   ✅ CAGR: {metrics.get('CAGR', 0):.2%} | Sharpe: {metrics.get('Sharpe', 0):.2f} | MaxDD: {metrics.get('Max Drawdown', 0):.2%}")
            except Exception as e:
                print(f"   ❌ 失敗: {e}")
                results.append({"Strategy": s["name"], "Error": str(e)})

        df_results = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("📊 所有策略表現比較")
        print("=" * 70)
        print(df_results.to_string(index=False))
        df_results.to_csv("strategy_comparison.csv", index=False)
        print("\n💾 比較結果已儲存至: strategy_comparison.csv")

    else:
        s = strategies[args.strategy]
        print(f"\n▶️ 執行: {s['name']}")
        metrics, equity_df = run_single_strategy(
            strategy_class=s["class"],
            strategy_params=s["params"],
            prices_dict=price_data,
            start_date=start_date,
            end_date=end_date,
            backtester_params=backtester_params,
            cost_model=cost_model,
            pipeline=pipeline,
            benchmark_series=bench_series,
        )

        if "Error" in metrics:
            print(f"❌ {metrics['Error']}")
            return

        print("\n" + "=" * 40)
        print(f"📊 {s['name']} 表現")
        print("=" * 40)
        print(f"總回報: {metrics.get('Total Return', 0):.2%}")
        print(f"年化 (CAGR): {metrics.get('CAGR', 0):.2%}")
        print(f"Sharpe: {metrics.get('Sharpe', 0):.2f}")
        print(f"最大回撤: {metrics.get('Max Drawdown', 0):.2%}")
        if "Alpha" in metrics:
            print(f"Alpha: {metrics['Alpha']:.2%}")
            print(f"Beta: {metrics['Beta']:.2f}")

        out_name = f"backtest_{args.strategy}.csv"
        equity_df.to_csv(out_name, index=False)
        print(f"\n💾 權益曲線已儲存至: {out_name}")

if __name__ == "__main__":
    main()