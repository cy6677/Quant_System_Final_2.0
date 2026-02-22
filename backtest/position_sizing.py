from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


class position_sizer:
    def __init__(
        self,
        portfolio_value: float,
        risk_per_trade: float = 0.01,
        atr_multiplier: float = 2.0,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.10,
        corr_threshold: float = 0.70,
        corr_penalty: float = 0.50,
    ):
        self.portfolio_value  = portfolio_value
        self.risk_per_trade   = risk_per_trade
        self.atr_multiplier   = atr_multiplier
        self.kelly_fraction   = kelly_fraction
        self.max_position_pct = max_position_pct
        self.corr_threshold   = corr_threshold
        self.corr_penalty     = corr_penalty

    @staticmethod
    def calc_atr(price_df: pd.DataFrame, window: int = 14) -> float:
        high  = price_df["high"] if "high" in price_df.columns else price_df["High"]
        low   = price_df["low"]  if "low"  in price_df.columns else price_df["Low"]
        close = price_df["close"] if "close" in price_df.columns else price_df["Close"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(window).mean().iloc[-1])

    def atr_based(
        self,
        ticker: str,
        entry_price: float,
        price_df: pd.DataFrame,
        atr_window: int = 14,
    ) -> Dict[str, float]:
        atr = self.calc_atr(price_df, atr_window)
        if atr <= 0:
            return {"shares": 0, "stop_price": entry_price, "atr": 0}

        stop_distance = self.atr_multiplier * atr
        stop_price    = entry_price - stop_distance
        risk_dollars  = self.portfolio_value * self.risk_per_trade
        shares        = risk_dollars / stop_distance

        max_shares = (self.portfolio_value * self.max_position_pct) / entry_price
        shares = min(shares, max_shares)

        return {
            "shares":      float(shares),
            "stop_price":  float(stop_price),
            "atr":         float(atr),
            "stop_dist":   float(stop_distance),
            "risk_dollars": float(risk_dollars),
            "position_pct": float(shares * entry_price / self.portfolio_value),
        }

    def kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float,
    ) -> Dict[str, float]:
        if avg_win <= 0 or avg_loss <= 0:
            return {"shares": 0, "kelly_f": 0.0, "position_pct": 0.0}

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        kelly_f = (p * b - q) / b

        kelly_f = max(0.0, kelly_f) * self.kelly_fraction
        kelly_f = min(kelly_f, self.max_position_pct)

        position_value = self.portfolio_value * kelly_f
        shares = position_value / entry_price if entry_price > 0 else 0

        return {
            "shares":       float(shares),
            "kelly_f":      float(kelly_f),
            "position_pct": float(kelly_f),
            "position_value": float(position_value),
        }

    def corr_adjusted(
        self,
        target_weights: Dict[str, float],
        returns_dict: Dict[str, pd.Series],
        lookback: int = 60,
    ) -> Dict[str, float]:
        tickers = [t for t in target_weights if t in returns_dict]
        if len(tickers) <= 1:
            return target_weights

        returns_df = pd.DataFrame({t: returns_dict[t].tail(lookback) for t in tickers}).dropna()
        if returns_df.empty or len(returns_df) < 20:
            return target_weights

        corr_matrix = returns_df.corr()
        adjusted_weights = dict(target_weights)

        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i >= j:
                    continue
                corr_val = abs(corr_matrix.loc[t1, t2])
                if corr_val >= self.corr_threshold:
                    adjusted_weights[t1] = adjusted_weights.get(t1, 0) * self.corr_penalty
                    adjusted_weights[t2] = adjusted_weights.get(t2, 0) * self.corr_penalty

        return adjusted_weights

    def size_position(
        self,
        ticker: str,
        entry_price: float,
        price_df: pd.DataFrame,
        method: str = "atr",
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
    ) -> Dict[str, float]:
        if method == "atr":
            return self.atr_based(ticker, entry_price, price_df)
        elif method == "kelly":
            return self.kelly(win_rate, avg_win, avg_loss, entry_price)
        elif method == "fixed":
            shares = (self.portfolio_value * self.risk_per_trade) / entry_price
            return {"shares": float(shares), "position_pct": self.risk_per_trade}
        else:
            raise ValueError(f"Unknown method: {method}. Use 'atr', 'kelly', or 'fixed'.")


PositionSizer = position_sizer
