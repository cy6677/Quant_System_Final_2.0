from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


class position_sizer:
    """
    三種 position sizing 方法：
    1. atr_based     — 每筆固定風險金額 / ATR-based stop distance
    2. kelly         — 半 Kelly（根據歷史勝率+賠率計算最優倉位）
    3. corr_adjusted — 高相關倉位自動縮小（降低集中風險）
    """

    def __init__(
        self,
        portfolio_value: float,
        risk_per_trade: float = 0.01,   # 每筆交易願意虧損的最大比例
        atr_multiplier: float = 2.0,    # stop = entry - atr_multiplier * ATR
        kelly_fraction: float = 0.5,    # 半 Kelly（保守）
        max_position_pct: float = 0.10, # 單倉上限
        corr_threshold: float = 0.70,   # 相關系數超過此值視為高度相關
        corr_penalty: float = 0.50,     # 高相關倉位縮小至原本的比例
    ):
        self.portfolio_value  = portfolio_value
        self.risk_per_trade   = risk_per_trade
        self.atr_multiplier   = atr_multiplier
        self.kelly_fraction   = kelly_fraction
        self.max_position_pct = max_position_pct
        self.corr_threshold   = corr_threshold
        self.corr_penalty     = corr_penalty

    # --------------------------------------------------
    # 1. ATR-Based Sizing
    # --------------------------------------------------

    @staticmethod
    def calc_atr(price_df: pd.DataFrame, window: int = 14) -> float:
        """計算 ATR（Average True Range）"""
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
        """
        Position sizing based on ATR stop.
        Risk $ = portfolio_value * risk_per_trade
        Stop distance = atr_multiplier * ATR
        Shares = Risk $ / Stop distance
        """
        atr = self.calc_atr(price_df, atr_window)
        if atr <= 0:
            return {"shares": 0, "stop_price": entry_price, "atr": 0}

        stop_distance = self.atr_multiplier * atr
        stop_price    = entry_price - stop_distance
        risk_dollars  = self.portfolio_value * self.risk_per_trade
        shares        = risk_dollars / stop_distance

        # 上限：不超過 max_position_pct
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

    # --------------------------------------------------
    # 2. Kelly Criterion
    # --------------------------------------------------

    def kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float,
    ) -> Dict[str, float]:
        """
        半 Kelly position sizing。
        f* = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
        使用 half-Kelly 控制風險。
        """
        if avg_win <= 0 or avg_loss <= 0:
            return {"shares": 0, "kelly_f": 0.0, "position_pct": 0.0}

        # Kelly formula
        b = avg_win / avg_loss  # 賠率
        p = win_rate
        q = 1 - win_rate
        kelly_f = (p * b - q) / b

        # 半 Kelly
        kelly_f = max(0.0, kelly_f) * self.kelly_fraction

        # 上限
        kelly_f = min(kelly_f, self.max_position_pct)

        position_value = self.portfolio_value * kelly_f
        shares = position_value / entry_price if entry_price > 0 else 0

        return {
            "shares":       float(shares),
            "kelly_f":      float(kelly_f),
            "position_pct": float(kelly_f),
            "position_value": float(position_value),
        }

    # --------------------------------------------------
    # 3. Correlation-Adjusted Sizing
    # --------------------------------------------------

    def corr_adjusted(
        self,
        target_weights: Dict[str, float],
        returns_dict: Dict[str, pd.Series],
        lookback: int = 60,
    ) -> Dict[str, float]:
        """
        調整 target weights：高度相關的持倉組合自動縮小。
        返回調整後嘅 weights。
        """
        tickers = [t for t in target_weights if t in returns_dict]
        if len(tickers) <= 1:
            return target_weights

        # 計算相關矩陣
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
                    # 兩個都縮小
                    adjusted_weights[t1] = adjusted_weights.get(t1, 0) * self.corr_penalty
                    adjusted_weights[t2] = adjusted_weights.get(t2, 0) * self.corr_penalty

        return adjusted_weights

    # --------------------------------------------------
    # 組合方法：推薦用法
    # --------------------------------------------------

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
        """統一入口，選擇 sizing 方法"""
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
