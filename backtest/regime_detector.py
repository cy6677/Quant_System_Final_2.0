from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from hmmlearn import hmm
    _HMM_AVAILABLE = True
except ImportError:
    warnings.warn("hmmlearn not installed. pip install hmmlearn")
    _HMM_AVAILABLE = False


class regime_detector:
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_fit: int = 504,
        refit_freq: int = 63,
        regime_weights: Optional[Dict[int, float]] = None,
        random_state: int = 42,
    ):
        self.n_regimes     = n_regimes
        self.lookback_fit  = lookback_fit
        self.refit_freq    = refit_freq
        self.random_state  = random_state
        self._model        = None
        self._last_fit_date: Optional[pd.Timestamp] = None
        self._fit_counter  = 0

        if regime_weights is None:
            self.regime_weights = {i: max(0.5, 1.0 - i * 0.25) for i in range(n_regimes)}
        else:
            self.regime_weights = regime_weights

    def _build_features(self, returns: pd.Series) -> np.ndarray:
        vol5 = returns.rolling(5).std().fillna(returns.std())
        features = np.column_stack([returns.values, vol5.values])
        return features

    def fit(self, spy_prices: pd.Series, asof: Optional[pd.Timestamp] = None) -> bool:
        if not _HMM_AVAILABLE:
            return False

        series = spy_prices.loc[:asof] if asof is not None else spy_prices
        returns = series.pct_change().dropna().tail(self.lookback_fit)
        if len(returns) < 100:
            return False

        features = self._build_features(returns)

        try:
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=self.random_state,
                tol=1e-4,
            )
            model.fit(features)
            self._model = model
            self._last_fit_date = asof
            return True
        except Exception as e:
            warnings.warn(f"HMM fit failed: {e}")
            return False

    def predict_regime(
        self,
        spy_prices: pd.Series,
        asof: Optional[pd.Timestamp] = None,
    ) -> int:
        if self._model is None:
            return self._vol_regime_fallback(spy_prices, asof)

        series = spy_prices.loc[:asof] if asof is not None else spy_prices
        returns = series.pct_change().dropna().tail(max(20, self.lookback_fit))
        if len(returns) < 20:
            return 0

        features = self._build_features(returns)

        try:
            hidden_states = self._model.predict(features)
            current_state = int(hidden_states[-1])

            means = self._model.means_
            state_vols = {s: abs(means[s][1]) for s in range(self.n_regimes)}
            sorted_states = sorted(state_vols, key=lambda s: state_vols[s])
            regime_rank = {sorted_states[i]: i for i in range(self.n_regimes)}
            return regime_rank.get(current_state, 0)

        except Exception:
            return self._vol_regime_fallback(spy_prices, asof)

    def _vol_regime_fallback(
        self, spy_prices: pd.Series, asof: Optional[pd.Timestamp] = None
    ) -> int:
        series = spy_prices.loc[:asof] if asof is not None else spy_prices
        returns = series.pct_change().dropna()
        if len(returns) < 20:
            return 0

        current_vol = float(returns.tail(20).std() * np.sqrt(252))
        long_vol    = float(returns.tail(252).std() * np.sqrt(252))
        vol_ratio   = current_vol / long_vol if long_vol > 0 else 1.0

        if vol_ratio < 0.8:
            return 0
        elif vol_ratio < 1.3:
            return 1
        else:
            return self.n_regimes - 1

    def get_position_scale(
        self,
        spy_prices: pd.Series,
        asof: Optional[pd.Timestamp] = None,
        auto_refit: bool = True,
    ) -> Tuple[int, float]:
        if auto_refit and _HMM_AVAILABLE:
            self._fit_counter += 1
            if self._fit_counter >= self.refit_freq or self._model is None:
                self.fit(spy_prices, asof)
                self._fit_counter = 0

        regime = self.predict_regime(spy_prices, asof)
        scale  = self.regime_weights.get(regime, 1.0)
        return regime, scale

    def get_regime_history(
        self,
        spy_prices: pd.Series,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        series = spy_prices.loc[start_date:end_date]
        if len(series) < self.lookback_fit:
            self.fit(spy_prices, pd.to_datetime(start_date) + pd.DateOffset(years=2))
        else:
            self.fit(spy_prices, pd.to_datetime(end_date))

        results = []
        for date in series.index:
            regime, scale = self.get_position_scale(spy_prices, asof=date, auto_refit=False)
            results.append({"date": date, "regime": regime, "scale": scale})

        return pd.DataFrame(results).set_index("date")


RegimeDetector = regime_detector
