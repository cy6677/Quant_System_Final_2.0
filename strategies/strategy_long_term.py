import pandas as pd
import numpy as np
from scipy.stats import zscore

from backtest.universal_backtester import BaseStrategy, Order


class LongTermStrategy(BaseStrategy):
    def __init__(self, top_n=15, max_sector_count=4, rebalance_freq="Q",
                 fundamentals_df=None):
        super().__init__("LongTerm_Mom_Vol")
        self.top_n = top_n
        self.max_sector_count = max_sector_count
        self.min_price = 5.0
        self.rebalance_freq = rebalance_freq
        self.fundamentals_df = fundamentals_df
        self._last_rebalance = None
        self.pipeline = None

    def _is_rebalance_day(self, date):
        period = date.to_period(self.rebalance_freq)
        if self._last_rebalance != period:
            self._last_rebalance = period
            return True
        return False

    # ✅ 修正：on_bar 簽名統一為 5 個參數
    def on_bar(self, date, universe_prices, current_portfolio_value,
               positions=None, cash=None):
        if not self._is_rebalance_day(date):
            return []

        target_weights = self.generate_signals(date, universe_prices, self.fundamentals_df)

        orders = []
        for ticker, weight in target_weights.items():
            orders.append(Order(ticker, "TARGET_WEIGHT", target_weight=weight))
        return orders

    def generate_signals(self, current_date, universe_prices, fundamentals_df=None):
        # 獲取當日允許的 tickers
        if self.pipeline and self.pipeline.config["universe"].get("use_historical", False):
            allowed_tickers = set(self.pipeline.get_universe_at(current_date))
        else:
            allowed_tickers = set(universe_prices.keys())

        candidates = []
        current_dt = pd.to_datetime(current_date)

        # ✅ 新增：從 fundamentals 建立 sector_map + quality_map
        sector_map = {}
        quality_map = {}  # ticker -> quality score (ROE, margin stability)
        if fundamentals_df is not None and "Ticker" in fundamentals_df.columns:
            sector_map = dict(
                zip(
                    fundamentals_df["Ticker"],
                    fundamentals_df.get("Sector", pd.Series(["Unknown"] * len(fundamentals_df))),
                )
            )
            # 嘗試讀取 Quality 指標
            if "ROE" in fundamentals_df.columns:
                quality_map = dict(
                    zip(fundamentals_df["Ticker"], fundamentals_df["ROE"].fillna(0))
                )
            elif "Net Income" in fundamentals_df.columns and "Total Equity" in fundamentals_df.columns:
                roe = fundamentals_df["Net Income"] / fundamentals_df["Total Equity"].replace(0, np.nan)
                quality_map = dict(zip(fundamentals_df["Ticker"], roe.fillna(0)))

        for ticker in allowed_tickers:
            if ticker not in universe_prices:
                continue
            df = universe_prices[ticker]

            if current_dt not in df.index:
                loc_idx = df.index.get_indexer([current_dt], method="pad")[0]
                if loc_idx == -1:
                    continue
                hist_data = df.iloc[: loc_idx + 1]
            else:
                hist_data = df.loc[:current_dt]

            if len(hist_data) < 252:
                continue

            close_col = "Close" if "Close" in hist_data.columns else "close"
            latest = hist_data.iloc[-1]
            if latest[close_col] < self.min_price or latest["Volume"] == 0:
                continue

            try:
                close = hist_data[close_col]
                p_lag = close.iloc[-21]
                p_base = close.iloc[-252]
                mom_score = (p_lag / p_base) - 1 if p_base > 0 else np.nan

                daily_ret = close.pct_change().tail(60)
                vol_score = daily_ret.std() * np.sqrt(252)

                if pd.isna(mom_score) or pd.isna(vol_score) or vol_score == 0:
                    continue

                # ✅ 新增：Quality score
                q_score = quality_map.get(ticker, 0.0)

                # ✅ 新增：Momentum stability（近 3 個月 vs 近 6 個月回報一致性）
                if len(close) >= 126:
                    mom_3m = (close.iloc[-1] / close.iloc[-63]) - 1
                    mom_6m = (close.iloc[-1] / close.iloc[-126]) - 1
                    mom_consistency = 1.0 if (mom_3m > 0 and mom_6m > 0) else 0.5
                else:
                    mom_consistency = 0.5

                candidates.append({
                    "Ticker": ticker,
                    "Momentum": mom_score,
                    "Volatility": vol_score,
                    "Quality": float(q_score),
                    "MomConsistency": mom_consistency,
                    "Sector": sector_map.get(ticker, "Unknown"),
                })
            except Exception:
                continue

        df = pd.DataFrame(candidates)
        if df.empty:
            return {}

        # ✅ 改進：Multi-Factor Composite Score
        df["Momentum_Z"] = zscore(df["Momentum"])
        df["Quality_Z"] = zscore(df["Quality"]) if df["Quality"].std() > 0 else 0.0
        df["Composite_Score"] = (
            0.50 * df["Momentum_Z"]
            + 0.20 * df["Quality_Z"]
            + 0.30 * df["MomConsistency"]
        )

        df = df.sort_values(by="Composite_Score", ascending=False)

        selected = []
        sector_count = {}
        for _, row in df.iterrows():
            sector = row["Sector"]
            if sector_count.get(sector, 0) < self.max_sector_count:
                selected.append(row)
                sector_count[sector] = sector_count.get(sector, 0) + 1
            if len(selected) >= self.top_n:
                break

        selected_df = pd.DataFrame(selected)
        if selected_df.empty:
            return {}

        selected_df["Inv_Vol"] = 1 / selected_df["Volatility"]
        selected_df["Raw_Weight"] = selected_df["Inv_Vol"]
        selected_df["Final_Weight"] = selected_df["Raw_Weight"] / selected_df["Raw_Weight"].sum()

        return dict(zip(selected_df["Ticker"], selected_df["Final_Weight"]))
