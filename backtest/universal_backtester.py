from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
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
        self.pipeline = None
        # ✅ 策略可提供：回測時只需要最近多少日歷史（加速）
        # 沒提供就由 backtester fallback
        # self.history_window = 400

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.DataFrame],  # 傳入到 date 為止「最近 N 日」hist DF
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

    def calc_commission(self, ticker: str, trade_value_abs: float) -> float:
        cfg = self._get_ticker_config(ticker)
        comm = trade_value_abs * cfg["commission_rate"]
        return max(comm, cfg["min_commission"])

    def apply_slippage(self, ticker: str, price: float, qty: float) -> float:
        cfg = self._get_ticker_config(ticker)
        if qty > 0:
            return price * (1 + cfg["slippage"])
        elif qty < 0:
            return price * (1 - cfg["slippage"])
        return price

# =========================
# Execution Model：T+1 open fill, ADV limit
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
        if df is None or df.empty:
            return None
        idx_pos = df.index.get_indexer([date])[0]
        if idx_pos < 0:
            return None
        nxt = idx_pos + 1
        if nxt >= len(df.index):
            return None
        return df.iloc[nxt]

    @staticmethod
    def _get_col(bar: pd.Series, names: List[str], default: Optional[float] = None) -> Optional[float]:
        for n in names:
            if n in bar.index:
                try:
                    return float(bar[n])
                except Exception:
                    return default
        return default

    def execute_orders(
        self,
        date: pd.Timestamp,  # signal_date
        orders: List[order],
        positions: Dict[str, position],
        cash: float,
        portfolio_value: float,
        cost_model: transaction_cost_model,
    ):
        fills: List[Dict[str, Any]] = []
        new_positions = dict(positions)
        new_cash = float(cash)

        # target_weight -> market (用 T+1 open 估)
        expanded: List[order] = []
        for o in orders:
            if o.order_type == "target_weight":
                if o.target_weight is None:
                    continue
                nb = self._get_next_bar(o.ticker, date)
                if nb is None:
                    continue
                opx = self._get_col(nb, ["Open", "open"])
                cpx = self._get_col(nb, ["Close", "close"])
                px = opx if opx is not None else cpx
                if px is None or px <= 0:
                    continue

                cur = new_positions.get(o.ticker, position(qty=0.0, avg_cost=0.0))
                target_value = portfolio_value * o.target_weight
                cur_value = cur.qty * px
                diff_value = target_value - cur_value
                qty = diff_value / px

                if not self.allow_fractional:
                    qty = math.floor(qty) if qty > 0 else math.ceil(qty)
                if abs(qty) < 1e-8:
                    continue

                expanded.append(order(ticker=o.ticker, order_type="market", quantity=float(qty), metadata=o.metadata))
            else:
                expanded.append(o)

        desired_by_t: Dict[str, float] = {}
        for o in expanded:
            desired_by_t[o.ticker] = desired_by_t.get(o.ticker, 0.0) + float(o.quantity)

        for ticker, desired_qty in desired_by_t.items():
            if abs(desired_qty) < 1e-8:
                continue

            bar = self._get_next_bar(ticker, date)  # fill at T+1
            if bar is None:
                continue

            day_open = self._get_col(bar, ["Open", "open"])
            day_close = self._get_col(bar, ["Close", "close"])
            raw_price = day_open if day_open is not None else day_close
            if raw_price is None or raw_price <= 0:
                continue

            vol = self._get_col(bar, ["Volume", "volume"], default=0.0) or 0.0

            max_qty = float("inf")
            if self.max_participation and self.max_participation > 0 and vol > 0:
                max_qty = float(vol) * float(self.max_participation)

            qty = float(desired_qty)
            if not self.allow_fractional:
                qty = math.floor(qty) if qty > 0 else math.ceil(qty)

            if abs(qty) > max_qty:
                qty = qty * (max_qty / abs(qty))

            if abs(qty) < 1e-8:
                continue

            price = cost_model.apply_slippage(ticker, float(raw_price), qty)
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
                    new_avg = price
                else:
                    if (pos.qty > 0 and qty > 0) or (pos.qty < 0 and qty < 0):
                        new_avg = (pos.qty * pos.avg_cost + qty * price) / new_qty
                    else:
                        new_avg = pos.avg_cost
                new_positions[ticker] = position(qty=float(new_qty), avg_cost=float(new_avg))

            new_cash -= (trade_value + commission)
            fills.append({
                "fill_date": bar.name,
                "signal_date": date,
                "ticker": ticker,
                "qty": float(qty),
                "price": float(price),
                "value": float(trade_value),
                "commission": float(commission),
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
        self.initial_capital = float(initial_capital)
        self.calendar_ticker = calendar_ticker
        self.allow_fractional = allow_fractional
        self.max_total_risk = max_total_risk
        self.cost_model = cost_model or transaction_cost_model()
        self.execution_model_factory = execution_model_factory
        self.min_trade_value = min_trade_value

    @staticmethod
    def _get_close(df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        if df is None or df.empty or date not in df.index:
            return None
        if "Close" in df.columns:
            return float(df.loc[date, "Close"])
        if "close" in df.columns:
            return float(df.loc[date, "close"])
        return None

    def _compute_portfolio_value(self, date: pd.Timestamp, positions: Dict[str, position], cash: float, prices: Dict[str, pd.DataFrame]) -> float:
        total = float(cash)
        for ticker, pos in positions.items():
            df = prices.get(ticker)
            px = self._get_close(df, date)
            if px is None:
                continue
            total += float(pos.qty) * float(px)
        return float(total)

    def _get_calendar_index(self, prices: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        if self.calendar_ticker in prices and not prices[self.calendar_ticker].empty:
            return pd.DatetimeIndex(prices[self.calendar_ticker].index)
        for _, df in prices.items():
            if df is not None and not df.empty:
                return pd.DatetimeIndex(df.index)
        raise ValueError("No valid price data for calendar")

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

        idx = self._get_calendar_index(prices)
        if start_date is not None:
            idx = idx[idx >= pd.to_datetime(start_date)]
        if end_date is not None:
            idx = idx[idx <= pd.to_datetime(end_date)]
        if len(idx) < 2:
            raise ValueError("Not enough dates for backtest")

        positions: Dict[str, position] = {}
        cash = float(self.initial_capital)

        exec_model = (
            self.execution_model_factory(prices)
            if self.execution_model_factory is not None
            else execution_model(prices=prices, allow_fractional=self.allow_fractional)
        )

        # ✅ 加速：策略可提供 history_window（預設 400）
        history_window = int(getattr(strategy, "history_window", 400))
        history_window = max(history_window, 50)

        records: List[Dict[str, Any]] = []
        fills_all: List[Dict[str, Any]] = []

        for i in range(len(idx) - 1):
            signal_date = idx[i]
            fill_date = idx[i + 1]

            universe_prices: Dict[str, pd.DataFrame] = {}
            for ticker, df in prices.items():
                if df is None or df.empty:
                    continue
                if signal_date in df.index:
                    # ✅ 只取最近 N 日，避免每次傳整段歷史（大幅加速）
                    hist = df.loc[:signal_date].tail(history_window)
                    universe_prices[ticker] = hist

            pv = self._compute_portfolio_value(signal_date, positions, cash, prices)
            orders = strategy.on_bar(
                date=signal_date,
                universe_prices=universe_prices,
                current_portfolio_value=pv,
                positions=positions,
                cash=cash,
            )

            positions, cash, fills = exec_model.execute_orders(
                date=signal_date,
                orders=orders,
                positions=positions,
                cash=cash,
                portfolio_value=pv,
                cost_model=self.cost_model,
            )
            fills_all.extend(fills)

            eod_value = self._compute_portfolio_value(fill_date, positions, cash, prices)
            records.append({
                "date": fill_date,
                "equity": float(eod_value),
                "cash": float(cash),
                "positions_count": int(len(positions)),
            })

        equity_df = pd.DataFrame.from_records(records).set_index("date")
        self._last_equity = equity_df
        self._last_fills = pd.DataFrame(fills_all) if fills_all else pd.DataFrame()
        return equity_df

    @property
    def trade_log(self) -> pd.DataFrame:
        return getattr(self, "_last_fills", pd.DataFrame())

# =========================
# PerformanceAnalyzer (保持你原來即可)
# =========================

class performance_analyzer:
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = float(risk_free_rate)

    def analyze(
        self,
        equity_df: pd.DataFrame,
        benchmark_series: Optional[pd.Series] = None,
        trade_log: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if equity_df is None or equity_df.empty:
            return result

        equity = equity_df["equity"].astype(float)
        returns = equity.pct_change().dropna()
        if returns.empty:
            return result

        ann_factor = 252
        daily_rf = (1 + self.risk_free_rate) ** (1 / ann_factor) - 1

        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
        ann_ret = float((1 + returns.mean()) ** ann_factor - 1)
        ann_vol = float(returns.std() * np.sqrt(ann_factor))
        sharpe = float((ann_ret - self.risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0

        cum_max = equity.cummax()
        dd_series = equity / cum_max - 1
        max_dd = float(dd_series.min())

        result["total_return"] = total_ret
        result["ann_return"] = ann_ret
        result["ann_vol"] = ann_vol
        result["sharpe"] = sharpe
        result["max_drawdown"] = max_dd

        if benchmark_series is not None and isinstance(benchmark_series, pd.Series) and len(benchmark_series) > 10:
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
                result["alpha"] = alpha
                result["beta"] = beta

        if trade_log is not None and not trade_log.empty:
            result["trade_count"] = float(len(trade_log))

        return result

    def summary_table(self, metrics: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([metrics]).T.rename(columns={0: "Value"})

# Aliases
UniversalBacktester  = universal_backtester
PerformanceAnalyzer  = performance_analyzer
TransactionCostModel = transaction_cost_model
BaseStrategy         = base_strategy
Order                = order
Position             = position
