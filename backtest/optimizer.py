from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Callable, List, Union, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.universal_backtester import (
    UniversalBacktester,
    PerformanceAnalyzer,
    TransactionCostModel,
)
from config import loadconfig

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    warnings.warn("optuna not installed. Falling back to random search. pip install optuna")
    _OPTUNA_AVAILABLE = False
    import random


@dataclass
class trial_result:
    params: Dict[str, Any]
    metric: float
    equity_df: Optional[pd.DataFrame] = None


class strategy_optimizer:
    """
    單策略參數優化器。
    - 預設使用 Optuna (TPE Bayesian Search)
    - 如 optuna 未安裝，自動退化成 random search
    - Walk-forward equity curve 使用鏈式正規化
    """

    def __init__(
        self,
        strategy_class: type,
        param_space: Dict[str, Union[List[Any], Tuple[float, float], Any]],
        prices_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        fixed_params: Optional[Dict[str, Any]] = None,
        metric: str = "sharpe",
        n_trials: int = 50,
        initial_capital: float = 100000.0,
        cost_model: Optional[TransactionCostModel] = None,
        allow_fractional: bool = True,
        calendar_ticker: str = "SPY",
    ):
        self.strategy_class    = strategy_class
        self.param_space       = param_space
        self.prices_dict       = prices_dict
        self.start_date        = start_date
        self.end_date          = end_date
        self.fixed_params      = fixed_params or {}
        self.metric            = metric.lower()
        self.n_trials          = n_trials
        self.initial_capital   = initial_capital
        self.cost_model        = cost_model or self._default_cost_model()
        self.allow_fractional  = allow_fractional
        self.calendar_ticker   = calendar_ticker
        self.trials: List[trial_result] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric: float = -np.inf

    # --------------------------------------------------
    # ✅ 修正：config key 統一用 underscore
    # --------------------------------------------------

    def _default_cost_model(self) -> TransactionCostModel:
        cfg = loadconfig()
        bt = cfg.get("backtest", {})
        return TransactionCostModel(
            commission_rate=bt.get("commission_rate", 0.001),
            slippage=bt.get("slippage", 0.001),
            min_commission=bt.get("min_commission", 1.0),
        )

    # --------------------------------------------------
    # 參數 sampling
    # --------------------------------------------------

    def _suggest_params_optuna(self, trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, spec in self.param_space.items():
            if isinstance(spec, list):
                params[key] = trial.suggest_categorical(key, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = trial.suggest_int(key, low, high)
                else:
                    params[key] = trial.suggest_float(key, float(low), float(high))
            else:
                params[key] = spec
        params.update(self.fixed_params)
        return params

    def _sample_params_random(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, spec in self.param_space.items():
            if isinstance(spec, list):
                params[key] = random.choice(spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = random.randint(low, high)
                else:
                    params[key] = random.uniform(float(low), float(high))
            else:
                params[key] = spec
        params.update(self.fixed_params)
        return params

    # --------------------------------------------------
    # 單次 backtest
    # --------------------------------------------------

    def _run_single_backtest(
        self, params: Dict[str, Any], start_date: str, end_date: str
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        try:
            strategy = self.strategy_class(**params)
            backtester = UniversalBacktester(
                initial_capital=self.initial_capital,
                calendar_ticker=self.calendar_ticker,
                allow_fractional=self.allow_fractional,
                cost_model=self.cost_model,
            )
            equity_df = backtester.run(
                strategy=strategy,
                prices=self.prices_dict,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            return {}, None

        if equity_df is None or equity_df.empty:
            return {}, None

        analyzer = PerformanceAnalyzer()
        trade_log = backtester.trade_log
        metrics = analyzer.analyze(
            equity_df,
            trade_log=trade_log if not trade_log.empty else None,
        )
        metrics["trade_count"] = float(len(trade_log)) if not trade_log.empty else 0.0
        return metrics, equity_df

    def _extract_metric(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return -np.inf
        val = metrics.get(self.metric, -np.inf)
        if self.metric == "max_drawdown":
            return -abs(float(val))
        return float(val)

    # --------------------------------------------------
    # Optuna 優化核心
    # --------------------------------------------------

    def _run_optuna_study(
        self, train_start: str, train_end: str, n_trials: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        def objective(trial):
            params = self._suggest_params_optuna(trial)
            metrics, _ = self._run_single_backtest(params, train_start, train_end)
            return self._extract_metric(metrics)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_params.update(self.fixed_params)
        for key, spec in self.param_space.items():
            if not isinstance(spec, (list, tuple)):
                best_params[key] = spec
        return best_params, float(best_trial.value)

    def _run_random_search(
        self, train_start: str, train_end: str, n_trials: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        best_metric = -np.inf
        best_params = None
        for _ in range(n_trials):
            params = self._sample_params_random()
            metrics, _ = self._run_single_backtest(params, train_start, train_end)
            value = self._extract_metric(metrics)
            if value > best_metric:
                best_metric = value
                best_params = params
        return best_params, best_metric

    def _optimize(
        self, train_start: str, train_end: str, n_trials: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        if _OPTUNA_AVAILABLE:
            return self._run_optuna_study(train_start, train_end, n_trials)
        else:
            return self._run_random_search(train_start, train_end, n_trials)

    # --------------------------------------------------
    # 單段優化
    # --------------------------------------------------

    def optimize_single_period(
        self,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        save_best_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        train_start = train_start or self.start_date
        train_end   = train_end   or self.end_date

        best_params, best_metric_val = self._optimize(train_start, train_end, self.n_trials)
        self.best_params = best_params
        self.best_metric = best_metric_val

        result: Dict[str, Any] = {
            "mode":                 "single_period",
            "metric":               self.metric,
            "optimizer":            "optuna" if _OPTUNA_AVAILABLE else "random",
            "train_start":          train_start,
            "train_end":            train_end,
            "best_metric_in_sample": best_metric_val,
            "best_params":          best_params,
        }

        if best_params is not None and test_start and test_end:
            oos_metrics, oos_equity = self._run_single_backtest(best_params, test_start, test_end)
            result["test_start"]   = test_start
            result["test_end"]     = test_end
            result["oos_metrics"]  = oos_metrics

        if save_best_to is not None and best_params is not None:
            path = Path(save_best_to)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    # --------------------------------------------------
    # Walk-Forward 優化
    # --------------------------------------------------

    def optimize_walk_forward(
        self,
        window_train_years: int = 3,
        window_test_years: int = 1,
        step_years: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if step_years is None:
            step_years = window_test_years

        calendar = self.prices_dict[self.calendar_ticker]
        all_dates = calendar.index
        start = pd.to_datetime(self.start_date)
        end   = pd.to_datetime(self.end_date)
        mask = (all_dates >= start) & (all_dates <= end)
        dates_in_range = all_dates[mask]

        if len(dates_in_range) == 0:
            raise ValueError("No dates in given start/end range for walk-forward")

        segments: List[Dict[str, Any]] = []
        all_test_equity: List[pd.DataFrame] = []
        cur_train_start = dates_in_range[0]

        while True:
            cur_train_end = cur_train_start + pd.DateOffset(years=window_train_years)
            cur_test_end  = cur_train_end   + pd.DateOffset(years=window_test_years)

            if cur_train_end >= end:
                break

            train_start_str = cur_train_start.strftime("%Y-%m-%d")
            train_end_str   = min(cur_train_end, end).strftime("%Y-%m-%d")
            test_start_str  = (cur_train_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            test_end_str    = min(cur_test_end, end).strftime("%Y-%m-%d")

            if pd.to_datetime(test_start_str) > pd.to_datetime(test_end_str):
                break

            best_params_seg, best_metric_seg = self._optimize(
                train_start_str, train_end_str, self.n_trials
            )

            if best_params_seg is None:
                break

            test_metrics, test_equity = self._run_single_backtest(
                best_params_seg, test_start_str, test_end_str
            )

            segments.append({
                "train_start":          train_start_str,
                "train_end":            train_end_str,
                "test_start":           test_start_str,
                "test_end":             test_end_str,
                "best_metric_in_sample": float(best_metric_seg),
                "best_params":          best_params_seg,
                "test_metrics":         test_metrics,
            })

            if test_equity is not None and not test_equity.empty:
                all_test_equity.append(test_equity)

            cur_train_start = cur_train_start + pd.DateOffset(years=step_years)
            if cur_train_start >= end:
                break

        result: Dict[str, Any] = {
            "mode":     "walk_forward",
            "metric":   self.metric,
            "optimizer": "optuna" if _OPTUNA_AVAILABLE else "random",
            "segments": segments,
        }

        if all_test_equity:
            chained: List[pd.DataFrame] = []
            running_capital = self.initial_capital

            for seg_eq in all_test_equity:
                if seg_eq.empty:
                    continue
                seg_start_val = float(seg_eq["equity"].iloc[0])
                if seg_start_val <= 0:
                    continue
                scale = running_capital / seg_start_val
                seg_scaled = seg_eq.copy()
                seg_scaled["equity"] = seg_eq["equity"] * scale
                if "cash" in seg_eq.columns:
                    seg_scaled["cash"] = seg_eq["cash"] * scale
                running_capital = float(seg_scaled["equity"].iloc[-1])
                chained.append(seg_scaled)

            if chained:
                combined = pd.concat(chained).sort_index()
                analyzer = PerformanceAnalyzer()
                overall_metrics = analyzer.analyze(combined)
                result["overall_oos_metrics"] = overall_metrics
                result["final_capital"]       = running_capital
                result["wf_return"]           = (running_capital - self.initial_capital) / self.initial_capital

        if save_dir is not None:
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            with (path / "walk_forward_result.json").open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result


# Aliases
StrategyOptimizer = strategy_optimizer
TrialResult       = trial_result
