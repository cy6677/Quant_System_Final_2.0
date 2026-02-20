from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from backtest.universal_backtester import base_strategy, order, position


class strategy_cross_sectional_momentum(base_strategy):
    """
    Cross-Sectional Momentum Strategy（學術 12-1 Month Momentum）。

    Alpha Source：Jegadeesh & Titman (1993)，最強 factor 之一。
    邏輯：
    - 每月月底 rebalance
    - 計算每隻股票過去 lookback_months 回報（排除最近 skip_months）
    - 買入 top_pct 強勢股，可選沽出 bottom_pct 弱勢股（long/short）
    - 用 equal weight 或 score-weighted 分配倉位
    """

    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        top_pct: float = 0.20,
        bottom_pct: float = 0.0,          # >0 時做 long-short
        max_positions: int = 20,
        target_weight_per_stock: float = 0.05,
        min_price: float = 5.0,
        rebalance_freq: str = "monthly",  # "monthly" or "weekly"
        use_score_weight: bool = False,   # True: 按 momentum score 分配 weight
    ):
        super().__init__("CrossSectionalMomentum")
        self.lookback_months         = lookback_months
        self.skip_months             = skip_months
        self.top_pct                 = top_pct
        self.bottom_pct              = bottom_pct
        self.max_positions           = max_positions
        self.target_weight_per_stock = target_weight_per_stock
        self.min_price               = min_price
        self.rebalance_freq          = rebalance_freq
        self.use_score_weight        = use_score_weight
        self._last_rebalance_date: Optional[pd.Timestamp] = None
        self._price_history: Dict[str, List[float]] = {}

    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        if self._last_rebalance_date is None:
            return True
        if self.rebalance_freq == "monthly":
            return date.month != self._last_rebalance_date.month
        elif self.rebalance_freq == "weekly":
            return (date - self._last_rebalance_date).days >= 7
        return False

    def _update_price_history(
        self, date: pd.Timestamp, universe_prices: Dict[str, pd.Series]
    ):
        for ticker, bar in universe_prices.items():
            close_val = bar.get("close", bar.get("Close", np.nan))
            if pd.isna(close_val):
                continue
            if ticker not in self._price_history:
                self._price_history[ticker] = []
            self._price_history[ticker].append((date, float(close_val)))
            # 只保留需要的最大歷史
            max_days = (self.lookback_months + 2) * 23
            self._price_history[ticker] = self._price_history[ticker][-max_days:]

    def _calc_momentum_scores(
        self, date: pd.Timestamp, universe_prices: Dict[str, pd.Series]
    ) -> pd.Series:
        """
        計算每隻股票的 12-1 month momentum score。
        Score = Return over [t-lookback, t-skip]
        """
        lookback_days = self.lookback_months * 21
        skip_days     = self.skip_months * 21
        scores = {}

        for ticker, history in self._price_history.items():
            if len(history) < lookback_days + skip_days:
                continue
            prices_arr = np.array([h[1] for h in history])
            try:
                price_now      = prices_arr[-(skip_days + 1)]    # t-skip
                price_lookback = prices_arr[-(lookback_days + skip_days + 1)]  # t-lookback
                if price_lookback > 0:
                    scores[ticker] = (price_now - price_lookback) / price_lookback
            except IndexError:
                continue

        # 過濾低價股
        filtered = {}
        for ticker, score in scores.items():
            if ticker in universe_prices:
                close_val = universe_prices[ticker].get("close", universe_prices[ticker].get("Close", 0))
                if float(close_val) >= self.min_price:
                    filtered[ticker] = score

        return pd.Series(filtered).dropna()

    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.Series],
        current_portfolio_value: float,
        positions: Dict[str, position],
        cash: float,
    ) -> List[order]:
        self._update_price_history(date, universe_prices)

        if not self._should_rebalance(date):
            return []

        self._last_rebalance_date = date
        scores = self._calc_momentum_scores(date, universe_prices)

        if scores.empty:
            return []

        n_stocks = len(scores)
        n_long   = max(1, min(self.max_positions, int(n_stocks * self.top_pct)))
        n_short  = max(0, int(n_stocks * self.bottom_pct)) if self.bottom_pct > 0 else 0

        sorted_scores = scores.sort_values(ascending=False)
        long_tickers  = sorted_scores.head(n_long).index.tolist()
        short_tickers = sorted_scores.tail(n_short).index.tolist() if n_short > 0 else []

        # 計算 target weights
        if self.use_score_weight:
            long_scores  = sorted_scores[long_tickers]
            long_scores  = (long_scores - long_scores.min()).clip(lower=0)
            total_score  = long_scores.sum()
            if total_score > 0:
                long_weights = {t: (long_scores[t] / total_score) * (1 - n_short * self.target_weight_per_stock)
                                for t in long_tickers}
            else:
                long_weights = {t: self.target_weight_per_stock for t in long_tickers}
        else:
            long_weights  = {t: self.target_weight_per_stock for t in long_tickers}

        short_weights = {t: -self.target_weight_per_stock for t in short_tickers}
        target_weights = {**long_weights, **short_weights}

        # 生成 orders（target_weight 型）
        orders = []
        all_relevant = set(long_tickers + short_tickers + list(positions.keys()))

        for ticker in all_relevant:
            tw = target_weights.get(ticker, 0.0)
            orders.append(order(
                ticker=ticker,
                order_type="target_weight",
                target_weight=tw,
                metadata={"strategy": "momentum", "score": float(scores.get(ticker, 0))},
            ))

        return orders


class strategy_short_term_reversal(base_strategy):
    """
    Short-Term Mean Reversion Strategy（5日反轉）。

    Alpha Source：Lo & MacKinlay (1990)。
    邏輯：
    - 每週 rebalance
    - 計算每隻股票過去 reversal_days 回報
    - 買入 bottom_pct 弱勢股（預期反轉），沽出 top_pct 強勢股
    - 適合高流動性大型股，換手率較高
    """

    def __init__(
        self,
        reversal_days: int = 5,
        top_pct: float = 0.20,
        bottom_pct: float = 0.20,
        max_positions: int = 20,
        target_weight_per_stock: float = 0.04,
        min_price: float = 10.0,
        min_adv_usd: float = 1e6,     # 最低平均成交額（過濾非流動股）
    ):
        super().__init__("ShortTermReversal")
        self.reversal_days           = reversal_days
        self.top_pct                 = top_pct
        self.bottom_pct              = bottom_pct
        self.max_positions           = max_positions
        self.target_weight_per_stock = target_weight_per_stock
        self.min_price               = min_price
        self.min_adv_usd             = min_adv_usd
        self._price_history: Dict[str, List] = {}
        self._last_rebalance: Optional[pd.Timestamp] = None

    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        if self._last_rebalance is None:
            return True
        return (date - self._last_rebalance).days >= 7

    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.Series],
        current_portfolio_value: float,
        positions: Dict[str, position],
        cash: float,
    ) -> List[order]:
        # 更新歷史
        for ticker, bar in universe_prices.items():
            close_val  = bar.get("close",  bar.get("Close",  np.nan))
            volume_val = bar.get("volume", bar.get("Volume", 0))
            if pd.isna(close_val):
                continue
            if ticker not in self._price_history:
                self._price_history[ticker] = []
            self._price_history[ticker].append((float(close_val), float(volume_val)))
            self._price_history[ticker] = self._price_history[ticker][-(self.reversal_days + 5):]

        if not self._should_rebalance(date):
            return []
        self._last_rebalance = date

        # 計算 reversal scores
        scores = {}
        for ticker, history in self._price_history.items():
            if len(history) < self.reversal_days + 1:
                continue
            prices_arr  = np.array([h[0] for h in history])
            volumes_arr = np.array([h[1] for h in history])

            current_price = prices_arr[-1]
            past_price    = prices_arr[-(self.reversal_days + 1)]
            if past_price <= 0 or current_price < self.min_price:
                continue

            avg_volume = volumes_arr[-self.reversal_days:].mean()
            avg_dollar_vol = avg_volume * current_price
            if avg_dollar_vol < self.min_adv_usd:
                continue

            ret = (current_price - past_price) / past_price
            scores[ticker] = ret

        if not scores:
            return []

        scores_s = pd.Series(scores).dropna()
        n = len(scores_s)
        n_long  = max(1, min(self.max_positions, int(n * self.bottom_pct)))
        n_short = max(1, min(self.max_positions, int(n * self.top_pct)))

        sorted_s      = scores_s.sort_values()
        long_tickers  = sorted_s.head(n_long).index.tolist()   # 最弱 → 反彈
        short_tickers = sorted_s.tail(n_short).index.tolist()  # 最強 → 回調

        target_weights = {}
        for t in long_tickers:
            target_weights[t] = self.target_weight_per_stock
        for t in short_tickers:
            target_weights[t] = -self.target_weight_per_stock

        orders = []
        all_relevant = set(list(target_weights.keys()) + list(positions.keys()))
        for ticker in all_relevant:
            tw = target_weights.get(ticker, 0.0)
            orders.append(order(
                ticker=ticker,
                order_type="target_weight",
                target_weight=tw,
                metadata={"strategy": "reversal", "score": float(scores.get(ticker, 0))},
            ))
        return orders
