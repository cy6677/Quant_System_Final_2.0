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
        self.pipeline = None   # 將由 QuantPipeline 注入

    def _is_rebalance_day(self, date):
        period = date.to_period(self.rebalance_freq)
        if self._last_rebalance != period:
            self._last_rebalance = period
            return True
        return False

    def on_bar(self, date, universe_prices, current_portfolio_value):
        if not self._is_rebalance_day(date):
            return []

        target_weights = self.generate_signals(date, universe_prices, self.fundamentals_df)

        orders = []
        for ticker, weight in target_weights.items():
            orders.append(Order(ticker, "TARGET_WEIGHT", target_weight=weight))
        return orders

    def generate_signals(self, current_date, universe_prices, fundamentals_df=None):
        # 獲取當日允許的 tickers（歷史成分股 + ETF）
        if self.pipeline and self.pipeline.config["universe"].get("use_historical", False):
            allowed_tickers = set(self.pipeline.get_universe_at(current_date))
        else:
            allowed_tickers = set(universe_prices.keys())

        candidates = []
        current_dt = pd.to_datetime(current_date)

        sector_map = {}
        if fundamentals_df is not None and "Ticker" in fundamentals_df.columns:
            sector_map = dict(
                zip(
                    fundamentals_df["Ticker"],
                    fundamentals_df.get("Sector", pd.Series(["Unknown"] * len(fundamentals_df))),
                )
            )

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

            latest = hist_data.iloc[-1]
            if latest["Close"] < self.min_price or latest["Volume"] == 0:
                continue

            try:
                p_lag = hist_data["Close"].iloc[-21]
                p_base = hist_data["Close"].iloc[-252]
                mom_score = (p_lag / p_base) - 1 if p_base > 0 else np.nan

                daily_ret = hist_data["Close"].pct_change().tail(60)
                vol_score = daily_ret.std() * np.sqrt(252)

                if pd.isna(mom_score) or pd.isna(vol_score) or vol_score == 0:
                    continue

                candidates.append({
                    "Ticker": ticker,
                    "Momentum": mom_score,
                    "Volatility": vol_score,
                    "Sector": sector_map.get(ticker, "Unknown"),
                })
            except Exception:
                continue

        df = pd.DataFrame(candidates)
        if df.empty:
            return {}

        df["Momentum_Z"] = zscore(df["Momentum"])
        df["Composite_Score"] = df["Momentum_Z"]

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