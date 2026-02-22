import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import load_config
from engine.pipeline import QuantPipeline
from backtest.optimizer import StrategyOptimizer
from backtest.universal_backtester import TransactionCostModel
from strategies.strategy_a import StrategyA_VCPBreakout


def ticker_classifier(ticker: str, extra_etfs: list) -> str:
    if ticker in extra_etfs:
        return "ETF"
    return "default"


def main():
    cfg = load_config()

    print("=" * 60)
    print("🚀 啟動參數優化")
    print("=" * 60)

    pipeline = QuantPipeline()
    universe_df, tickers = pipeline.build_universe()
    print(f"✅ Universe: {len(tickers)} 隻股票")

    pipeline.ensure_prices(tickers)
    price_data = pipeline.load_prices(tickers)
    print(f"✅ 載入 {len(price_data)} 隻股票價格")

    param_space = {
        "atr_short": [3, 5, 7, 10],
        "atr_long": [15, 20, 25, 30, 40],
        "contraction_ratio": (0.5, 0.9),
        "lookback_high": [10, 15, 20, 25],
        "breakout_buffer": (0.0, 0.5),
        "rvol_threshold": (1.2, 2.0),
        "stop_loss_atr": (1.0, 3.0),
        "target_r": (1.5, 3.0),
        "trail_atr": (2.0, 4.0),
    }

    fixed_params = {
        "max_hold_days": 30,
        "riskpertrade": 0.005,
        "minprice": 5.0,
        "minavgdollarvol": 20e6,
    }

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

    optimizer = StrategyOptimizer(
        strategy_class=StrategyA_VCPBreakout,
        param_space=param_space,
        fixed_params=fixed_params,
        prices_dict=price_data,
        start_date=start_date,
        end_date=end_date,
        metric="sharpe",
        n_trials=30,
        initial_capital=cfg["backtest"]["initial_capital"],
        cost_model=cost_model,
        allow_fractional=cfg["backtest"]["allow_fractional"],
        calendar_ticker=cfg["backtest"]["calendar_ticker"],
    )

    # ✅ 修正：用 optimize_single_period 而唔係 optimize()
    result = optimizer.optimize_single_period(
        save_best_to="best_params.json"
    )

    print(f"\n📊 最佳參數: {result['best_params']}")
    print(f"📊 最佳 Sharpe (in-sample): {result['best_metric_in_sample']:.4f}")
    print("🎉 完成！")


if __name__ == "__main__":
    main()
