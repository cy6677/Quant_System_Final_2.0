from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class portfolio_optimizer:
    """
    Portfolio-level weight optimization。
    支援：
    1. risk_parity      — 每個資產貢獻相等風險
    2. mean_variance    — 最大化 Sharpe（Markowitz）
    3. equal_weight     — 1/N 等權（baseline）
    4. min_variance     — 最小化波幅
    """

    def __init__(
        self,
        lookback: int = 252,
        min_weight: float = 0.0,
        max_weight: float = 0.30,
        risk_free_rate: float = 0.04,
        annualize_factor: int = 252,
    ):
        self.lookback          = lookback
        self.min_weight        = min_weight
        self.max_weight        = max_weight
        self.risk_free_rate    = risk_free_rate
        self.annualize_factor  = annualize_factor

    def _get_returns(
        self, prices_dict: Dict[str, pd.DataFrame], tickers: List[str], asof: str
    ) -> pd.DataFrame:
        price_list = []
        for t in tickers:
            if t not in prices_dict:
                continue
            df = prices_dict[t]
            col = "close" if "close" in df.columns else "Close"
            s = df.loc[df.index <= asof, col].tail(self.lookback + 1)
            if len(s) < 30:
                continue
            price_list.append(s.rename(t))

        if not price_list:
            return pd.DataFrame()

        prices_df = pd.concat(price_list, axis=1).dropna()
        return prices_df.pct_change().dropna()

    # --------------------------------------------------
    # 1. Risk Parity（等風險貢獻）
    # --------------------------------------------------

    def risk_parity(
        self,
        returns_df: pd.DataFrame,
        n_iter: int = 1000,
        tol: float = 1e-8,
    ) -> Dict[str, float]:
        """
        Risk Parity: 每個資產貢獻相等的組合風險。
        用 Cyclical Coordinate Descent 求解。
        """
        if returns_df.empty:
            return {}

        cov = returns_df.cov().values * self.annualize_factor
        n = cov.shape[0]
        tickers = list(returns_df.columns)
        w = np.ones(n) / n

        for _ in range(n_iter):
            w_old = w.copy()
            for i in range(n):
                # Marginal Risk Contribution
                sigma_w = cov @ w
                mrc = sigma_w[i]
                # 解析解更新（近似）
                a = cov[i, i]
                b = (cov[i, :] @ w) - cov[i, i] * w[i]
                # target: w[i] * mrc = budget / n
                target_rc = (w @ (cov @ w)) / n
                # Newton step
                w[i] = max(1e-6, (-b + np.sqrt(b**2 + 4 * a * target_rc)) / (2 * a))
            w = w / w.sum()
            if np.max(np.abs(w - w_old)) < tol:
                break

        # 應用 weight 上下限
        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / w.sum()

        return {tickers[i]: float(w[i]) for i in range(n)}

    # --------------------------------------------------
    # 2. Mean-Variance（最大化 Sharpe）
    # --------------------------------------------------

    def mean_variance(
        self,
        returns_df: pd.DataFrame,
        n_portfolios: int = 5000,
    ) -> Dict[str, float]:
        """
        Monte Carlo 模擬找最大 Sharpe 組合。
        """
        if returns_df.empty:
            return {}

        tickers = list(returns_df.columns)
        n = len(tickers)
        mu  = returns_df.mean().values * self.annualize_factor
        cov = returns_df.cov().values  * self.annualize_factor
        rf  = self.risk_free_rate

        best_sharpe = -np.inf
        best_w      = np.ones(n) / n

        rng = np.random.default_rng(42)
        for _ in range(n_portfolios):
            w = rng.dirichlet(np.ones(n))
            w = np.clip(w, self.min_weight, self.max_weight)
            w = w / w.sum()
            p_ret = float(w @ mu)
            p_vol = float(np.sqrt(w @ cov @ w))
            if p_vol > 0:
                sharpe = (p_ret - rf) / p_vol
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w = w

        return {tickers[i]: float(best_w[i]) for i in range(n)}

    # --------------------------------------------------
    # 3. Equal Weight
    # --------------------------------------------------

    def equal_weight(self, tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: float(min(w, self.max_weight)) for t in tickers}

    # --------------------------------------------------
    # 4. Minimum Variance
    # --------------------------------------------------

    def min_variance(
        self,
        returns_df: pd.DataFrame,
        n_portfolios: int = 5000,
    ) -> Dict[str, float]:
        if returns_df.empty:
            return {}

        tickers = list(returns_df.columns)
        n = len(tickers)
        cov = returns_df.cov().values * self.annualize_factor

        best_vol = np.inf
        best_w   = np.ones(n) / n

        rng = np.random.default_rng(42)
        for _ in range(n_portfolios):
            w = rng.dirichlet(np.ones(n))
            w = np.clip(w, self.min_weight, self.max_weight)
            w = w / w.sum()
            p_vol = float(np.sqrt(w @ cov @ w))
            if p_vol < best_vol:
                best_vol = p_vol
                best_w   = w

        return {tickers[i]: float(best_w[i]) for i in range(n)}

    # --------------------------------------------------
    # 統一入口
    # --------------------------------------------------

    def optimize(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        asof: str,
        method: str = "risk_parity",
    ) -> Dict[str, float]:
        returns_df = self._get_returns(prices_dict, tickers, asof)
        available  = list(returns_df.columns) if not returns_df.empty else tickers

        if method == "risk_parity":
            return self.risk_parity(returns_df)
        elif method == "mean_variance":
            return self.mean_variance(returns_df)
        elif method == "equal_weight":
            return self.equal_weight(available)
        elif method == "min_variance":
            return self.min_variance(returns_df)
        else:
            raise ValueError(f"Unknown method: {method}. Use risk_parity / mean_variance / equal_weight / min_variance")


PortfolioOptimizer = portfolio_optimizer
