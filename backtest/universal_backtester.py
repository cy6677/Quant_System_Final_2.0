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
        universe_prices: Dict[str, pd.Series],   # 當日各 ticker 的 OHLCV Series
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

        # 先將 target_weight 轉換為 market orders
        expanded_orders: List[order] = []
        for o in orders:
            if o.order_type == "target_weight":
                if o.target_weight is None:
                    continue
                px_bar = self._get_next_bar(o.ticker, date)
                if px_bar is None:
                    continue
                price = float(px_bar["open"]) if "open" in px_bar else float(px_bar["Open"])
                current_pos = new_positions.get(o.ticker, position(qty=0.0, avg_cost=0.0))
                target_value = portfolio_value * o.target_weight
                current_value = current_pos.qty * price
                diff_value = target_value - current_value
                qty = diff_value / price if price > 0 else 0.0
                if not self.allow_fractional:
                    qty = math.floor(qty) if qty > 0 else math.ceil(qty)
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

        # 按 ticker 聚合 desired qty
        desired_qty_by_ticker: Dict[str, float] = {}
        for o in expanded_orders:
            desired_qty_by_ticker[o.ticker] = desired_qty_by_ticker.get(o.ticker, 0.0) + o.quantity

        for ticker, desired_qty in desired_qty_by_ticker.items():
            if abs(desired_qty) < 1e-8:
                continue
            bar = self._get_next_bar(ticker, date)
            if bar is None:
                continue
            day_open = float(bar["open"]) if "open" in bar else float(bar["Open"])
            day_volume = float(bar.get("volume", bar.get("Volume", 0.0)))

            # 成交量限制
            max_qty_by_volume = float('inf')
            if self.max_participation is not None and self.max_participation > 0 and day_volume > 0:
                max_qty_by_volume = day_volume * self.max_participation

            qty = desired_qty
            if not self.allow_fractional:
                qty = math.floor(qty) if qty > 0 else math.ceil(qty)
            if abs(qty) > max_qty_by_volume:
                scale = max_qty_by_volume / abs(qty) if abs(qty) > 0 else 0.0
                qty = qty * scale
            if abs(qty) < 1e-8:
                continue

            raw_price = day_open
            price = cost_model.apply_slippage(ticker, raw_price, qty)
            trade_value = price * qty
            commission = cost_model.calc_commission(ticker, abs(trade_value))

            # 現金檢查（買入）
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
                # 更新平均成本（簡化處理）
                if pos.qty == 0:
                    new_avg_cost = price
                else:
                    if (pos.qty > 0 and qty > 0) or (pos.qty < 0 and qty < 0):
                        new_avg_cost = (pos.qty * pos.avg_cost + qty * price) / new_qty
                    else:
                        # 減倉時不改變 avg_cost
                        new_avg_cost = pos.avg_cost
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
        cost_model: Optional[transaction_cost_model] = None,
        execution_model_factory: Optional[Callable[[Dict[str, pd.DataFrame]], execution_model]] = None,
        min_trade_value: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.calendar_ticker = calendar_ticker
        self.allow_fractional = allow_fractional
        self.max_total_risk = max_total_risk
        self.cost_model = cost_model or transaction_cost_model()
        self.execution_model_factory = execution_model_factory
        self.min_trade_value = min_trade_value

    def _compute_portfolio_value(self, date, positions, cash, prices):
        total = cash
        for ticker, pos in positions.items():
            df = prices.get(ticker)
            if df is None or date not in df.index:
                continue
            px = float(df.loc[date, "close"]) if "close" in df.columns else float(df.loc[date, "Close"])
            total += pos.qty * px
        return total

    def run(
        self,
        strategy: base_strategy,
        prices: Optional[Dict[str, pd.DataFrame]] = None,
        prices_dict: Optional[Dict[str, pd.DataFrame]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
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
            # 構建當日的 universe_prices (每個 ticker 的 Series)
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

            new_positions, new_cash, fills = exec_model.execute_orders(
                date=date,
                orders=orders,
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
# PerformanceAnalyzer
# =========================

class performance_analyzer:
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

        # Sortino Ratio
        excess_returns = returns - daily_rf
        downside = excess_returns[excess_returns < 0]
        downside_std = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 0 else 0.0
        sortino = float((ann_ret - self.risk_free_rate) / downside_std) if downside_std > 0 else 0.0

        # Calmar
        calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0

        # Win Rate
        win_rate = float((returns > 0).mean())

        # Profit Factor
        pos_sum = float(returns[returns > 0].sum())
        neg_sum = float(abs(returns[returns < 0].sum()))
        profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else float("inf")

        # Omega
        gains = (returns - daily_rf).clip(lower=0).sum()
        losses = (daily_rf - returns).clip(lower=0).sum()
        omega = float(gains / losses) if losses > 0 else float("inf")

        # Recovery Factor
        recovery_factor = float(total_ret / abs(max_dd)) if max_dd < 0 else float("inf")

        # Max Consecutive Losses
        loss_streak = 0
        max_loss_streak = 0
        for r in returns:
            if r < 0:
                loss_streak += 1
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                loss_streak = 0

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

        # Alpha / Beta vs Benchmark
        if benchmark_series is not None:
            bench_ret = benchmark_series.pct_change().dropna()
            aligned = pd.concat([returns, bench_ret], axis=1, join="inner").dropna()
            if len(aligned) > 10:
                strat_r = aligned.iloc[:, 0]
                bench_r = aligned.iloc[:, 1]
                strat_excess = strat_r - daily_rf
                bench_excess = bench_r - daily_rf
                cov_matrix = np.cov(strat_excess, bench_excess)
                beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 0.0
                alpha = float((strat_excess.mean() - beta * bench_excess.mean()) * ann_factor)
                result["alpha"]          = alpha
                result["beta"]           = beta
                result["benchmark_ann"]  = float((1 + bench_r.mean()) ** ann_factor - 1)
                result["benchmark_corr"] = float(strat_r.corr(bench_r))

        # Trade-level 指標
        if trade_log is not None and not trade_log.empty and "value" in trade_log.columns:
            pnl_col = trade_log["value"]   # 正值代表買入支出？不適合直接作為盈虧
            # 更合理的 trade-level 指標需要逐筆交易配對，這裡簡化
            result["trade_count"] = float(len(trade_log))
            # 粗略估算：用 value 的正負？不準確，可省略
            # 建議另寫函數計算

        return result

    def summary_table(self, metrics: Dict[str, float]) -> pd.DataFrame:
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


# Aliases
UniversalBacktester  = universal_backtester
PerformanceAnalyzer  = performance_analyzer
TransactionCostModel = transaction_cost_model
BaseStrategy         = base_strategy
Order                = order
Position             = position
