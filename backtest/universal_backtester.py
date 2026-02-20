from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable, Any
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd

# =========================
# 基本結構：Order / Position / BaseStrategy
# =========================

@dataclass
class order:
    ticker: str
    order_type: str  # "market", "limit", "stop", "target_weight"
    quantity: float = 0.0
    target_weight: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class position:
    qty: float
    avg_cost: float

class base_strategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.Series],
        current_portfolio_value: float,
        positions: Dict[str, position],
        cash: float,
    ) -> List[order]:
        raise NotImplementedError

# =========================
# TransactionCostModel
# =========================

class transaction_cost_model:
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage: float = 0.001,
        min_commission: float = 1.0,
        per_ticker_costs: Optional[Dict[str, Dict[str, float]]] = None,
        ticker_classifier: Optional[Callable[[str], str]] = None,
    ):
        self.default_commission_rate = commission_rate
        self.default_slippage = slippage
        self.default_min_commission = min_commission
        self.per_ticker_costs = per_ticker_costs or {}
        self.ticker_classifier = ticker_classifier or (lambda t: "default")
        if "default" not in self.per_ticker_costs:
            self.per_ticker_costs["default"] = {
                "commission_rate": commission_rate,
                "slippage": slippage,
                "min_commission": min_commission,
            }

    def _get_ticker_config(self, ticker: str) -> Dict[str, float]:
        if ticker in self.per_ticker_costs:
            return self.per_ticker_costs[ticker]
        category = self.ticker_classifier(ticker)
        if category in self.per_ticker_costs:
            return self.per_ticker_costs[category]
        return self.per_ticker_costs["default"]

    def calc_commission(self, ticker: str, trade_value: float) -> float:
        cfg = self._get_ticker_config(ticker)
        comm = trade_value * cfg["commission_rate"]
        return max(comm, cfg["min_commission"])

    def apply_slippage(self, ticker: str, price: float, qty: float) -> float:
        cfg = self._get_ticker_config(ticker)
        if qty > 0:
            return price * (1 + cfg["slippage"])
        elif qty < 0:
            return price * (1 - cfg["slippage"])
        return price

# =========================
# Execution Model：T+1 fill, ADV limit
# =========================

class execution_model:
    def __init__(
        self,
        prices: Dict[str, pd.DataFrame],
        max_participation: float = 0.05,
        allow_fractional: bool = True,
    ):
        self.prices = prices
        self.max_participation = max_participation
        self.allow_fractional = allow_fractional

    def _get_next_bar(self, ticker: str, date: pd.Timestamp) -> Optional[pd.Series]:
        df = self.prices.get(ticker)
        if df is None:
            return None
        try:
            loc = df.index.get_loc(date)
        except KeyError:
            return None
        next_idx = loc + 1
        if next_idx >= len(df.index):
            return None
        return df.iloc[next_idx]

    def execute_orders(
        self,
        date: pd.Timestamp,
        orders: List[order],
        positions: Dict[str, position],
        cash: float,
        portfolio_value: float,
        cost_model: transaction_cost_model,
    ):
        fills: List[Dict[str, Any]] = []
        new_positions = dict(positions)
        new_cash = cash

        expanded_orders: List[order] = []
        for o in orders:
            if o.order_type == "target_weight":
                if o.target_weight is None:
                    continue
                px_bar = self._get_next_bar(o.ticker, date)
                if px_bar is None:
                    continue
                price = float(px_bar["open"])
                current_pos = new_positions.get(o.ticker, position(qty=0.0, avg_cost=0.0))
                target_value = portfolio_value * o.target_weight
                current_value = current_pos.qty * price
                diff_value = target_value - current_value
                qty = diff_value / price if price > 0 else 0.0
                if not self.allow_fractional:
                    qty = math.floor(qty)
                if abs(qty) < 1e-8:
                    continue
                expanded_orders.append(order(
                    ticker=o.ticker,
                    order_type="market",
                    quantity=qty,
                    metadata=o.metadata,
                ))
            else:
                expanded_orders.append(o)

        desired_qty_by_ticker: Dict[str, float] = {}
        for o in expanded_orders:
            desired_qty_by_ticker[o.ticker] = desired_qty_by_ticker.get(o.ticker, 0.0) + o.quantity

        for ticker, desired_qty in desired_qty_by_ticker.items():
            if abs(desired_qty) < 1e-8:
                continue
            bar = self._get_next_bar(ticker, date)
            if bar is None:
                continue
            day_open = float(bar["open"])
            day_volume = float(bar.get("volume", 0.0))

            max_qty_by_volume = float("inf")
            if day_volume > 0 and self.max_participation is not None:
                max_qty_by_volume = max(day_volume * self.max_participation, 0.0)

            qty = desired_qty
            if not self.allow_fractional:
                qty = math.floor(qty)
            if abs(qty) > max_qty_by_volume:
                scale = max_qty_by_volume / abs(qty) if abs(qty) > 0 else 0.0
                qty = qty * scale
            if abs(qty) < 1e-8:
                continue

            raw_price = day_open
            price = cost_model.apply_slippage(ticker, raw_price, qty)
            trade_value = price * qty
            commission = cost_model.calc_commission(ticker, abs(trade_value))

            if qty > 0:
                total_cost = trade_value + commission
                if total_cost > new_cash:
                    if new_cash <= 0:
                        continue
                    scale = new_cash / total_cost
                    qty = qty * scale
                    if not self.allow_fractional:
                        qty = math.floor(qty)
                    if abs(qty) < 1e-8:
                        continue
                    trade_value = price * qty
                    commission = cost_model.calc_commission(ticker, abs(trade_value))

            pos = new_positions.get(ticker, position(qty=0.0, avg_cost=0.0))
            new_qty = pos.qty + qty
            if abs(new_qty) < 1e-12:
                new_positions.pop(ticker, None)
            else:
                if pos.qty == 0:
                    new_avg_cost = price
                else:
                    if (pos.qty > 0 and qty > 0) or (pos.qty < 0 and qty < 0):
                        new_avg_cost = (pos.qty * pos.avg_cost + qty * price) / new_qty
                    else:
                        new_avg_cost = price if abs(qty) > abs(pos.qty) else pos.avg_cost
                new_positions[ticker] = position(qty=new_qty, avg_cost=new_avg_cost)

            new_cash -= (trade_value + commission)
            fills.append({
                "date": bar.name,
                "signal_date": date,
                "ticker": ticker,
                "qty": qty,
                "price": price,
                "value": trade_value,
                "commission": commission,
            })

        return new_positions, new_cash, fills

# =========================
# UniversalBacktester
# =========================

class universal_backtester:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        calendar_ticker: str = "SPY",
        allow_fractional: bool = True,
        max_total_risk: float = 0.1,
        risk_field: str = "risk_per_trade",
        cost_model: Optional[transaction_cost_model] = None,
        execution_model_factory: Optional[Callable[[Dict[str, pd.DataFrame]], execution_model]] = None,
        min_trade_value: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.calendar_ticker = calendar_ticker
        self.allow_fractional = allow_fractional
        self.max_total_risk = max_total_risk
        self.risk_field = risk_field
        self.cost_model = cost_model or transaction_cost_model()
        self.execution_model_factory = execution_model_factory
        self.min_trade_value = min_trade_value

    def _compute_portfolio_value(self, date, positions, cash, prices):
        total = cash
        for ticker, pos in positions.items():
            df = prices.get(ticker)
            if df is None or date not in df.index:
                continue
            px = float(df.loc[date, "close"])
            total += pos.qty * px
        return total

    def _compute_open_risk(self, positions, stop_dict):
        total_risk = 0.0
        for ticker, pos in positions.items():
            stop = stop_dict.get(ticker)
            if stop is None:
                continue
            if pos.qty > 0:
                per_share_risk = max(pos.avg_cost - stop, 0.0)
            else:
                per_share_risk = max(stop - pos.avg_cost, 0.0)
            total_risk += abs(per_share_risk * pos.qty)
        return total_risk

    def run(
        self,
        strategy: base_strategy,
        prices: Optional[Dict[str, pd.DataFrame]] = None,
        prices_dict: Optional[Dict[str, pd.DataFrame]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        stop_levels_provider=None,
    ) -> pd.DataFrame:
        if prices is None and prices_dict is not None:
            prices = prices_dict
        if prices is None:
            raise ValueError("prices or prices_dict must be provided")

        calendar_df = prices[self.calendar_ticker]
        idx = calendar_df.index
        if start_date is not None:
            idx = idx[idx >= pd.to_datetime(start_date)]
        if end_date is not None:
            idx = idx[idx <= pd.to_datetime(end_date)]
        if len(idx) < 2:
            raise ValueError("Not enough dates for backtest")

        positions: Dict[str, position] = {}
        cash = self.initial_capital
        exec_model = (
            self.execution_model_factory(prices)
            if self.execution_model_factory is not None
            else execution_model(prices=prices, allow_fractional=self.allow_fractional)
        )

        records: List[Dict[str, Any]] = []
        fills_all: List[Dict[str, Any]] = []

        for i in range(len(idx) - 1):
            date = idx[i]
            universe_prices = {}
            for ticker, df in prices.items():
                if date in df.index:
                    universe_prices[ticker] = df.loc[date]

            portfolio_value = self._compute_portfolio_value(date, positions, cash, prices)
            orders = strategy.on_bar(
                date=date,
                universe_prices=universe_prices,
                current_portfolio_value=portfolio_value,
                positions=positions,
                cash=cash,
            )

            stop_dict_existing: Dict[str, float] = {}
            if stop_levels_provider is not None:
                stop_dict_existing = stop_levels_provider(date, positions)

            open_risk_nominal = self._compute_open_risk(positions, stop_dict_existing)
            current_risk_ratio = open_risk_nominal / portfolio_value if portfolio_value > 0 else 0.0

            if current_risk_ratio >= self.max_total_risk:
                filtered_orders = []
                for o in orders:
                    pos = positions.get(o.ticker)
                    if pos is None:
                        continue
                    if (pos.qty > 0 and o.quantity < 0) or (pos.qty < 0 and o.quantity > 0):
                        filtered_orders.append(o)
                orders_to_execute = filtered_orders
            else:
                orders_to_execute = orders

            new_positions, new_cash, fills = exec_model.execute_orders(
                date=date,
                orders=orders_to_execute,
                positions=positions,
                cash=cash,
                portfolio_value=portfolio_value,
                cost_model=self.cost_model,
            )
            positions = new_positions
            cash = new_cash
            fills_all.extend(fills)

            end_of_day_value = self._compute_portfolio_value(date, positions, cash, prices)
            records.append({
                "date": date,
                "equity": end_of_day_value,
                "cash": cash,
                "positions_count": len(positions),
            })

        equity_df = pd.DataFrame.from_records(records).set_index("date")
        self._last_equity = equity_df
        self._last_fills = pd.DataFrame(fills_all) if fills_all else pd.DataFrame()
        return equity_df

    @property
    def trade_log(self) -> pd.DataFrame:
        return getattr(self, "_last_fills", pd.DataFrame())

# =========================
# PerformanceAnalyzer — FULLY EXPANDED
# =========================

class performance_analyzer:
    """
    完整版 PerformanceAnalyzer。
    新增指標：
    - sortino       : 年化 Sortino Ratio（只懲罰下行波動）
    - calmar        : Ann Return / |Max Drawdown|
    - win_rate      : 正回報日佔比
    - profit_factor : Σ正回報 / |Σ負回報|
    - omega_ratio   : Ω ratio（threshold=0）
    - recovery_factor: Total Return / |Max Drawdown|
    - max_consec_losses: 最多連續負回報日數
    - alpha / beta  : 相對 benchmark（需傳入 benchmark_series）
    - total_return  : 累計回報
    - avg_daily_ret : 平均日回報
    - trade_stats   : 如傳入 trade_log，計算交易層面指標
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate

    def analyze(
        self,
        equity_df: pd.DataFrame,
        benchmark_series: Optional[pd.Series] = None,
        trade_log: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if equity_df.empty:
            return result

        equity = equity_df["equity"].astype(float)
        returns = equity.pct_change().dropna()
        if returns.empty:
            return result

        # --- 基礎指標 ---
        n_days = len(returns)
        ann_factor = 252
        daily_rf = (1 + self.risk_free_rate) ** (1 / ann_factor) - 1

        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
        ann_ret = float((1 + returns.mean()) ** ann_factor - 1)
        ann_vol = float(returns.std() * np.sqrt(ann_factor))
        sharpe = float((ann_ret - self.risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0

        # Max Drawdown
        cum_max = equity.cummax()
        dd_series = equity / cum_max - 1
        max_dd = float(dd_series.min())

        # --- Sortino Ratio ---
        downside_returns = returns[returns < daily_rf] - daily_rf
        downside_std = float(np.sqrt((downside_returns ** 2).mean()) * np.sqrt(ann_factor)) if len(downside_returns) > 0 else 0.0
        sortino = float((ann_ret - self.risk_free_rate) / downside_std) if downside_std > 0 else 0.0

        # --- Calmar Ratio ---
        calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0

        # --- Win Rate (日) ---
        win_rate = float((returns > 0).mean())

        # --- Profit Factor ---
        pos_sum = float(returns[returns > 0].sum())
        neg_sum = float(abs(returns[returns < 0].sum()))
        profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else float("inf")

        # --- Omega Ratio (threshold = risk_free_rate daily) ---
        gains = (returns - daily_rf).clip(lower=0).sum()
        losses = (daily_rf - returns).clip(lower=0).sum()
        omega = float(gains / losses) if losses > 0 else float("inf")

        # --- Recovery Factor ---
        recovery_factor = float(total_ret / abs(max_dd)) if max_dd < 0 else float("inf")

        # --- Max Consecutive Losses ---
        loss_streak = 0
        max_loss_streak = 0
        for r in returns:
            if r < 0:
                loss_streak += 1
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                loss_streak = 0

        # --- 填入 result ---
        result["total_return"]       = total_ret
        result["ann_return"]         = ann_ret
        result["ann_vol"]            = ann_vol
        result["sharpe"]             = sharpe
        result["sortino"]            = sortino
        result["calmar"]             = calmar
        result["max_drawdown"]       = max_dd
        result["win_rate"]           = win_rate
        result["profit_factor"]      = profit_factor
        result["omega_ratio"]        = omega
        result["recovery_factor"]    = recovery_factor
        result["max_consec_losses"]  = float(max_loss_streak)
        result["avg_daily_ret"]      = float(returns.mean())

        # --- Alpha / Beta vs Benchmark ---
        if benchmark_series is not None:
            bench_ret = benchmark_series.pct_change().dropna()
            aligned = pd.concat([returns, bench_ret], axis=1, join="inner").dropna()
            if len(aligned) > 10:
                strat_r = aligned.iloc[:, 0]
                bench_r = aligned.iloc[:, 1]
                cov_matrix = np.cov(strat_r, bench_r)
                beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 0.0
                bench_ann = float((1 + bench_r.mean()) ** ann_factor - 1)
                alpha = float(ann_ret - (self.risk_free_rate + beta * (bench_ann - self.risk_free_rate)))
                result["alpha"]          = alpha
                result["beta"]           = beta
                result["benchmark_ann"]  = bench_ann
                result["benchmark_corr"] = float(strat_r.corr(bench_r))

        # --- Trade-level 指標（如傳入 trade_log）---
        if trade_log is not None and not trade_log.empty and "value" in trade_log.columns:
            pnl_col = trade_log["value"]
            wins = pnl_col[pnl_col > 0]
            losses = pnl_col[pnl_col < 0]
            result["trade_count"]       = float(len(trade_log))
            result["trade_win_rate"]    = float(len(wins) / len(trade_log)) if len(trade_log) > 0 else 0.0
            result["avg_win"]           = float(wins.mean()) if len(wins) > 0 else 0.0
            result["avg_loss"]          = float(losses.mean()) if len(losses) > 0 else 0.0
            trade_pf_num = float(wins.sum())
            trade_pf_den = float(abs(losses.sum()))
            result["trade_profit_factor"] = float(trade_pf_num / trade_pf_den) if trade_pf_den > 0 else float("inf")

        return result

    def summary_table(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """輸出格式化指標表格"""
        labels = {
            "total_return":       ("Total Return",        "{:.2%}"),
            "ann_return":         ("Ann. Return",         "{:.2%}"),
            "ann_vol":            ("Ann. Volatility",     "{:.2%}"),
            "sharpe":             ("Sharpe Ratio",        "{:.3f}"),
            "sortino":            ("Sortino Ratio",       "{:.3f}"),
            "calmar":             ("Calmar Ratio",        "{:.3f}"),
            "max_drawdown":       ("Max Drawdown",        "{:.2%}"),
            "win_rate":           ("Win Rate (daily)",    "{:.2%}"),
            "profit_factor":      ("Profit Factor",       "{:.3f}"),
            "omega_ratio":        ("Omega Ratio",         "{:.3f}"),
            "recovery_factor":    ("Recovery Factor",     "{:.3f}"),
            "max_consec_losses":  ("Max Consec. Losses",  "{:.0f}"),
            "alpha":              ("Alpha (ann.)",        "{:.2%}"),
            "beta":               ("Beta",                "{:.3f}"),
            "trade_win_rate":     ("Trade Win Rate",      "{:.2%}"),
            "trade_profit_factor":("Trade Profit Factor", "{:.3f}"),
            "trade_count":        ("# Trades",            "{:.0f}"),
        }
        rows = []
        for key, (label, fmt) in labels.items():
            if key in metrics:
                val = metrics[key]
                try:
                    display = fmt.format(val)
                except Exception:
                    display = str(val)
                rows.append({"Metric": label, "Value": display})
        return pd.DataFrame(rows).set_index("Metric")


# Aliases（向後兼容）
UniversalBacktester  = universal_backtester
PerformanceAnalyzer  = performance_analyzer
TransactionCostModel = transaction_cost_model
BaseStrategy         = base_strategy
Order                = order
Position             = position
