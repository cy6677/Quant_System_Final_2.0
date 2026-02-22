#!/usr/bin/env python3
"""
萬能回測腳本
用法：
    python scripts/run_backtest.py --strategy long
    python scripts/run_backtest.py --strategy a
    python scripts/run_backtest.py --strategy b
    python scripts/run_backtest.py --strategy c
    python scripts/run_backtest.py --strategy all
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
    if hasattr(strategy, "pipeline"):
        strategy.pipeline = pipeline

    backtester = UniversalBacktester(**backtester_params, cost_model=cost_model)
    equity_df = backtester.run(
        strategy=strategy,
        prices=prices_dict,
        start_date=start_date,
        end_date=end_date,
    )

    if equity_df is None or equity_df.empty:
        return {"Error": "No trades"}, equity_df

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(equity_df, benchmark_series=benchmark_series)
    metrics["Strategy"] = strategy.name if hasattr(strategy, "name") else "Strategy"

    if hasattr(backtester, "trade_log") and not backtester.trade_log.empty:
        metrics["Trade Count"] = len(backtester.trade_log) // 2

    return metrics, equity_df


def main():
    parser = argparse.ArgumentParser(description="��能回測腳本")
    parser.add_argument("--strategy", type=str, default="long",
                        choices=["long", "a", "b", "c", "all"],
                        help="策略: long / a / b / c / all")
    parser.add_argument("--use-pipeline", action="store_true", default=False,
                        help="使用完整 pipeline 載入數據（預設用 SPY 測試）")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 70)
    print(f"🚀 回測策略: {args.strategy.upper()}")
    print("=" * 70)

    pipeline = QuantPipeline()

    if args.use_pipeline:
        print("\n📦 使用 Pipeline 準備 Universe...")
        universe_df, tickers = pipeline.build_universe()
        pipeline.ensure_prices(tickers)
        prices_dict = pipeline.load_prices(tickers)
    else:
        import yfinance as yf
        print("⚙️  TEST_MODE: 只用 SPY 價格數據做測試")
        tickers = ["SPY"]
        data = yf.download(
            "SPY",
            start=cfg["data"]["price"]["start_date"],
            end=cfg["data"]["price"]["end_date"],
            progress=False,
        )
        # ✅ 修正：確保 column 名用 Title Case
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        col_map = {}
        for c in data.columns:
            cl = c.lower().strip()
            if cl == "open": col_map[c] = "Open"
            elif cl == "high": col_map[c] = "High"
            elif cl == "low": col_map[c] = "Low"
            elif cl in ("close", "adj close"): col_map[c] = "Close"
            elif cl == "volume": col_map[c] = "Volume"
            else: col_map[c] = c
        data = data.rename(columns=col_map)
        data.index.name = "Date"
        prices_dict = {"SPY": data}
        universe_df = pd.DataFrame({"Ticker": tickers})

    print(f"✅ Universe: {len(tickers)} 隻股票")
    print(f"✅ 成功載入 {len(prices_dict)} 隻股票")

    start_date = cfg["data"]["price"]["start_date"] or "2015-01-01"
    end_date = cfg["data"]["price"]["end_date"] or datetime.now().strftime("%Y-%m-%d")

    extra_etfs = cfg["universe"].get("extra_etfs", [])
    cost_model = TransactionCostModel(
        commission_rate=cfg["backtest"]["commission_rate"],
        slippage=cfg["backtest"]["slippage"],
        min_commission=cfg["backtest"]["min_commission"],
        per_ticker_costs=cfg.get("cost_model", {}),
        ticker_classifier=lambda t: ticker_classifier(t, extra_etfs)
    )

    backtester_params = {
        "initial_capital": cfg["backtest"]["initial_capital"],
        "calendar_ticker": cfg["backtest"]["calendar_ticker"],
        "allow_fractional": cfg["backtest"]["allow_fractional"],
    }

    # ✅ 修正：用 prices_dict 取 benchmark
    benchmark = prices_dict.get("SPY")
    bench_series = None
    if benchmark is not None:
        close_col = "Close" if "Close" in benchmark.columns else "close"
        bench_series = benchmark[close_col]

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
            "name": "Strategy A - VCP Breakout",
            "class": StrategyA_VCPBreakout,
            "params": {
                "riskpertrade": 0.005,
                "minprice": 5.0,
                "minavgdollarvol": 20e6,
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
        for key in ["long", "a", "b", "c"]:
            s = strategies[key]
            print(f"\n▶️ 執行: {s['name']}")
            try:
                metrics, _ = run_single_strategy(
                    strategy_class=s["class"],
                    strategy_params=s["params"],
                    prices_dict=prices_dict,
                    start_date=start_date,
                    end_date=end_date,
                    backtester_params=backtester_params,
                    cost_model=cost_model,
                    pipeline=pipeline,
                    benchmark_series=bench_series,
                )
                metrics["Strategy"] = s["name"]
                results.append(metrics)
                print(f"   ✅ CAGR: {metrics.get('CAGR', 0):.2%} | "
                      f"Sharpe: {metrics.get('Sharpe', 0):.2f} | "
                      f"MaxDD: {metrics.get('Max Drawdown', 0):.2%}")
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
            prices_dict=prices_dict,
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
        equity_df.to_csv(out_name)
        print(f"\n💾 權益曲線已儲存至: {out_name}")


if __name__ == "__main__":
    main()
