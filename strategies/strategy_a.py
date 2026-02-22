import pandas as pd
import numpy as np

from backtest.universal_backtester import BaseStrategy, Order
from utils.utils import compute_atr


class StrategyA_VCPBreakout(BaseStrategy):
    def __init__(
        self,
        riskpertrade: float = 0.005,
        minprice: float = 5.0,
        minavgdollarvol: float = 20e6,
        atr_short: int = 5,
        atr_long: int = 20,
        contraction_ratio: float = 0.7,
        lookback_high: int = 20,
        breakout_buffer: float = 0.1,
        rvol_threshold: float = 1.5,
        stop_loss_atr: float = 1.5,
        target_r: float = 2.0,
        trail_atr: float = 3.0,
        max_hold_days: int = 30,
        # ✅ 新增：VCP 多重收縮參數
        min_contractions: int = 2,
        contraction_window: int = 60,
        rs_lookback: int = 252,
        rs_min_pct: float = 0.70,
        **kwargs,
    ):
        super().__init__(name="StrategyA_VCP")
        self.riskpertrade = riskpertrade
        self.minprice = minprice
        self.minavgdollarvol = minavgdollarvol

        self.atr_short = atr_short
        self.atr_long = atr_long
        self.contraction_ratio = contraction_ratio
        self.lookback_high = lookback_high
        self.breakout_buffer = breakout_buffer
        self.rvol_threshold = rvol_threshold
        self.stop_loss_atr = stop_loss_atr
        self.target_r = target_r
        self.trail_atr = trail_atr
        self.max_hold_days = max_hold_days

        # ✅ 新增
        self.min_contractions = min_contractions
        self.contraction_window = contraction_window
        self.rs_lookback = rs_lookback
        self.rs_min_pct = rs_min_pct

        self.state: dict = {}

    # ------------------------------------------------------------------
    # Helper：position size
    # ------------------------------------------------------------------
    def _compute_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
    ) -> int:
        if entry_price <= 0 or stop_loss <= 0:
            return 0
        per_share_risk = entry_price - stop_loss
        if per_share_risk <= 0:
            return 0
        capital_risk = portfolio_value * self.riskpertrade
        shares = capital_risk / per_share_risk
        shares = int(np.floor(shares))
        return max(shares, 0)

    def _passes_universe_filters(self, ticker: str, hist: pd.DataFrame) -> bool:
        if len(hist) < 20:
            return False
        close = hist["Close"].iloc[-1]
        if close < self.minprice:
            return False
        vol = hist["Volume"].tail(20).mean()
        dollar_vol = close * vol
        if dollar_vol < self.minavgdollarvol:
            return False
        return True

    # ------------------------------------------------------------------
    # ✅ 新增：多重收縮檢測
    # ------------------------------------------------------------------
    def _count_contractions(self, hist: pd.DataFrame) -> int:
        """
        計算最近 contraction_window 日內有幾次 ATR 收縮。
        定義：ATR 連續下降 >= 5 天算一次收縮。
        """
        atr = compute_atr(hist, self.atr_short).tail(self.contraction_window)
        if len(atr) < 10:
            return 0

        contractions = 0
        streak = 0
        for i in range(1, len(atr)):
            if pd.isna(atr.iloc[i]) or pd.isna(atr.iloc[i - 1]):
                streak = 0
                continue
            if atr.iloc[i] < atr.iloc[i - 1]:
                streak += 1
                if streak >= 5:
                    contractions += 1
                    streak = 0
            else:
                streak = 0
        return contractions

    # ------------------------------------------------------------------
    # ✅ 新增：Relative Strength vs SPY
    # ------------------------------------------------------------------
    def _calc_rs_rank(
        self, ticker: str, hist: pd.DataFrame, universe_prices: dict
    ) -> float:
        """
        計算相對 SPY 嘅 RS 排名百分比。
        返回 0~1，越高越強。
        """
        if "SPY" not in universe_prices:
            return 1.0  # 如果冇 SPY 數據，預設通過

        spy_df = universe_prices["SPY"]
        if spy_df is None or spy_df.empty:
            return 1.0

        lookback = self.rs_lookback
        if len(hist) < lookback or len(spy_df) < lookback:
            return 1.0

        ticker_ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[-lookback]) - 1
        spy_close = spy_df["Close"] if "Close" in spy_df.columns else spy_df["close"]
        spy_ret = (spy_close.iloc[-1] / spy_close.iloc[-lookback]) - 1

        # RS = ticker return / SPY return（唔用 percentile rank 因為需要全 universe）
        if spy_ret == 0:
            return 1.0
        rs_ratio = ticker_ret / abs(spy_ret)
        # 簡化：如果 ticker 回報比 SPY 高就通過
        return 1.0 if rs_ratio >= self.rs_min_pct else rs_ratio

    # ------------------------------------------------------------------
    # on_bar：入場 + 出場
    # ------------------------------------------------------------------
    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: dict,
        current_portfolio_value: float,
        positions: dict,
        cash: float,
    ):
        orders = []

        # 1) 新入場
        for ticker, df in universe_prices.items():
            if ticker in positions:
                continue
            if date not in df.index:
                continue

            hist = df.loc[:date]
            if len(hist) < max(self.atr_long, 120, self.lookback_high, self.rs_lookback) + 5:
                continue

            if not self._passes_universe_filters(ticker, hist):
                continue

            atr_short = compute_atr(hist, self.atr_short).iloc[-1]
            atr_long = compute_atr(hist, self.atr_long).iloc[-1]
            if pd.isna(atr_short) or pd.isna(atr_long) or atr_long == 0:
                continue

            # ✅ 改進：基本收縮條件
            contraction_cond = (atr_short / atr_long) <= self.contraction_ratio

            atr14 = compute_atr(hist, 14).iloc[-120:]
            if len(atr14) >= 120:
                pct_rank = (atr14 <= atr_short).sum() / 120.0
                contraction_cond = contraction_cond or (pct_rank <= 0.2)

            if not contraction_cond:
                continue

            # ✅ 新增：多重收縮檢測
            n_contractions = self._count_contractions(hist)
            if n_contractions < self.min_contractions:
                continue

            # ✅ 新增：RS Rating 過濾
            rs = self._calc_rs_rank(ticker, hist, universe_prices)
            if rs < self.rs_min_pct:
                continue

            highest_last = hist["High"].rolling(self.lookback_high).max().iloc[-2]
            close_today = hist["Close"].iloc[-1]
            if pd.isna(highest_last):
                continue
            if close_today <= highest_last + self.breakout_buffer * atr_short:
                continue

            vol_avg = hist["Volume"].tail(20).mean()
            if vol_avg == 0 or (hist["Volume"].iloc[-1] / vol_avg) < self.rvol_threshold:
                continue

            entry_price = close_today
            recent_low = hist["Low"].tail(self.lookback_high).min()
            stop_loss = min(
                recent_low,
                entry_price - self.stop_loss_atr * atr_short,
            )

            shares = self._compute_position_size(entry_price, stop_loss, current_portfolio_value)
            if shares <= 0:
                continue

            self.state[ticker] = {
                "entry_date": date,
                "entry_price": float(entry_price),
                "stop_loss": float(stop_loss),
                "highest": float(entry_price),
                "bars": 0,
                "partial": False,
            }

            orders.append(
                Order(
                    ticker=ticker,
                    order_type="market",
                    quantity=shares,
                )
            )

        # 2) 現有持倉：更新 bars / highest，檢查出場
        for ticker, pos in positions.items():
            if ticker not in universe_prices:
                continue
            df = universe_prices[ticker]
            if date not in df.index:
                continue

            current_price = float(df.loc[date, "Close"])
            s = self.state.get(ticker)

            if s is None:
                s = {
                    "entry_date": date,
                    "entry_price": float(pos.avg_cost),
                    "stop_loss": float(pos.avg_cost * (1 - self.stop_loss_atr * 0.01)),
                    "highest": float(pos.avg_cost),
                    "bars": 0,
                    "partial": False,
                }
                self.state[ticker] = s

            s["bars"] += 1
            s["highest"] = max(s["highest"], current_price)

            exit_orders = self._check_specific_exits(
                ticker=ticker,
                pos=pos,
                state=s,
                current_price=current_price,
                date=date,
                universe_prices=universe_prices,
            )
            orders.extend(exit_orders)

            if any(o.quantity == -pos.qty for o in exit_orders if o.ticker == ticker):
                self.state.pop(ticker, None)

        return orders

    # ------------------------------------------------------------------
    # 出場邏輯：partial TP + trailing stop + max_hold_days
    # ------------------------------------------------------------------
    def _check_specific_exits(
        self,
        ticker: str,
        pos,
        state: dict,
        current_price: float,
        date: pd.Timestamp,
        universe_prices: dict,
    ):
        orders = []

        entry = state["entry_price"]
        stop = state["stop_loss"]
        if (entry - stop) <= 0:
            return orders

        r_multiple = (current_price - entry) / (entry - stop)

        # 1) partial take-profit
        if r_multiple >= self.target_r and not state.get("partial", False):
            half_shares = pos.qty / 2
            if half_shares > 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        order_type="market",
                        quantity=-half_shares,
                    )
                )
                state["partial"] = True

        # 2) trailing stop
        hist = universe_prices[ticker].loc[:date]
        if len(hist) >= 20:
            atr = compute_atr(hist, 14).iloc[-1]
        else:
            atr = np.nan

        if not pd.isna(atr):
            trail_stop = state["highest"] - self.trail_atr * atr
            trail_stop = max(trail_stop, stop)

            if current_price < trail_stop and pos.qty > 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        order_type="market",
                        quantity=-pos.qty,
                    )
                )

        # 3) max holding days
        if state["bars"] >= self.max_hold_days and pos.qty > 0:
            orders.append(
                Order(
                    ticker=ticker,
                    order_type="market",
                    quantity=-pos.qty,
                )
            )

        return orders
