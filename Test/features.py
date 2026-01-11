"""
Feature Engineering: Calculate technical indicators for crypto data
Simplified and cleaned version, fully compatible with SignalPredictor
"""

import numpy as np
from loguru import logger
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

import config


class FeatureEngineer:
    """Calculate technical indicators for crypto price data"""

    def __init__(self):
        self.indicators_config = config.TECHNICAL_INDICATORS

    def add_all_features(self, df):
        """Add all technical indicators to dataframe"""

        try:
            df = df.copy()
            df = self._add_basic_features(df)
            df = self._add_moving_averages(df)
            df = self._add_rsi(df)
            df = self._add_macd(df)
            df = self._add_bollinger_bands(df)
            df = self._add_atr(df)
            df = self._add_volume_features(df)
            df = self._add_momentum_indicators(df)
            df = self._add_volatility_indicators(df)
            df = self._add_trend_strength(df)
            df = self._add_support_resistance(df)
            df = self._add_market_structure(df)
            df = self._calculate_composite_score(df)

            # Handle NaNs and infinities
            df = df.ffill().bfill().fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            df = df.dropna(how="all")

            return df

        except Exception as e:
            logger.error(f"Error adding features: {e}")
            raise

    # ===== ORIGINAL METHODS =====

    def _add_basic_features(self, df):
        df["returns"] = df["close"].pct_change(fill_method=None)
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["price_change"] = df["close"] - df["open"]
        df["price_change_pct"] = (df["close"] - df["open"]) / df["open"] * 100
        df["hl_range"] = df["high"] - df["low"]
        df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
        return df

    def _add_moving_averages(self, df):
        for period in self.indicators_config["moving_averages"]:
            sma = SMAIndicator(df["close"], window=period)
            df[f"ma_{period}"] = sma.sma_indicator()
            df[f"price_to_ma_{period}"] = (
                (df["close"] - df[f"ma_{period}"]) / df[f"ma_{period}"] * 100
            )

        for period in self.indicators_config["ema_periods"]:
            ema = EMAIndicator(df["close"], window=period)
            df[f"ema_{period}"] = ema.ema_indicator()

        return df

    def _add_rsi(self, df):
        period = self.indicators_config["rsi_period"]
        rsi = RSIIndicator(df["close"], window=period)
        df["rsi"] = rsi.rsi()
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        return df

    def _add_macd(self, df):
        macd_cfg = self.indicators_config["macd"]
        macd = MACD(
            df["close"],
            window_fast=macd_cfg["fast"],
            window_slow=macd_cfg["slow"],
            window_sign=macd_cfg["signal"],
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        df["macd_bullish"] = (
            (df["macd"] > df["macd_signal"])
            & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)
        df["macd_bearish"] = (
            (df["macd"] < df["macd_signal"])
            & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        ).astype(int)
        return df

    def _add_bollinger_bands(self, df):
        bb_cfg = self.indicators_config["bollinger"]
        bb = BollingerBands(
            df["close"], window=bb_cfg["period"], window_dev=bb_cfg["std"]
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )
        return df

    def _add_atr(self, df):
        period = self.indicators_config["atr_period"]
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = df["atr"] / df["close"] * 100
        return df

    def _add_volume_features(self, df):
        df["volume_change"] = df["volume"].pct_change(fill_method=None)
        df["volume_ma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
        df["pv_trend"] = df["returns"] * df["volume"]
        return df

    def _add_momentum_indicators(self, df):
        df["roc_3h"] = df["close"].pct_change(3, fill_method=None) * 100
        df["roc_6h"] = df["close"].pct_change(6, fill_method=None) * 100
        df["roc_12h"] = df["close"].pct_change(12, fill_method=None) * 100
        df["price_acceleration"] = df["close"].diff().diff()
        df["volume_momentum"] = (
            (df["volume"] * df["close"].pct_change(fill_method=None)).rolling(6).sum()
        )
        df["rel_volume"] = df["volume"] / df["volume"].rolling(24).mean()
        return df

    def _add_volatility_indicators(self, df):
        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = abs(df["high"] - df["close"].shift(1))
        df["low_close"] = abs(df["low"] - df["close"].shift(1))
        df["volatility_3h"] = (
            df["close"].pct_change(fill_method=None).rolling(3).std() * 100
        )
        df["volatility_6h"] = (
            df["close"].pct_change(fill_method=None).rolling(6).std() * 100
        )
        df["volatility_24h"] = (
            df["close"].pct_change(fill_method=None).rolling(24).std() * 100
        )
        df["volatility_ratio"] = df["volatility_6h"] / df["volatility_24h"]
        return df

    def _add_trend_strength(self, df):
        df["plus_dm"] = df["high"].diff().clip(lower=0)
        df["minus_dm"] = (-df["low"].diff()).clip(lower=0)
        tr = df[["high_low", "high_close", "low_close"]].max(axis=1)
        df["plus_di"] = df["plus_dm"].rolling(14).sum() / tr.rolling(14).sum() * 100
        df["minus_di"] = df["minus_dm"].rolling(14).sum() / tr.rolling(14).sum() * 100
        df["trend_strength"] = (
            abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"]) * 100
        )
        return df

    def _add_support_resistance(self, df, window=24):
        df["resistance_24h"] = df["high"].rolling(window).max()
        df["support_24h"] = df["low"].rolling(window).min()
        df["resistance_48h"] = df["high"].rolling(window * 2).max()
        df["support_48h"] = df["low"].rolling(window * 2).min()
        df["dist_from_resistance"] = (
            (df["resistance_24h"] - df["close"]) / df["close"] * 100
        )
        df["dist_from_support"] = (df["close"] - df["support_24h"]) / df["close"] * 100
        pivot = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["pivot_point"] = pivot
        df["resistance_1"] = 2 * pivot - df["low"].shift(1)
        df["support_1"] = 2 * pivot - df["high"].shift(1)
        df["resistance_2"] = pivot + (df["high"].shift(1) - df["low"].shift(1))
        df["support_2"] = pivot - (df["high"].shift(1) - df["low"].shift(1))
        return df

    def _add_market_structure(self, df):
        df["is_higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["is_lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
        df["swing_high"] = (
            (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
        ).astype(int)
        df["swing_low"] = (
            (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))
        ).astype(int)
        ema_fast = df["close"].ewm(span=8).mean()
        ema_slow = df["close"].ewm(span=21).mean()
        df["trend_direction"] = (ema_fast > ema_slow).astype(int)
        df["consecutive_up"] = (df["close"] > df["open"]).astype(int)
        df["consecutive_down"] = (df["close"] < df["open"]).astype(int)
        return df

    def _calculate_composite_score(self, df):
        scores = []
        for idx in range(len(df)):
            score = 0
            try:
                rsi = df["rsi"].iloc[idx]
                if rsi < 30:
                    score += 20
                elif rsi < 40:
                    score += 10
                elif rsi > 70:
                    score -= 20
                elif rsi > 60:
                    score -= 10
                else:
                    score += 5
                if df["macd"].iloc[idx] > df["macd_signal"].iloc[idx]:
                    score += 15
                else:
                    score -= 15
                if df["trend_strength"].iloc[idx] > 25:
                    score += 10
                if df["rel_volume"].iloc[idx] > 1.5:
                    score += 10
                elif df["rel_volume"].iloc[idx] < 0.7:
                    score -= 5
                if df["roc_6h"].iloc[idx] > 2:
                    score += 10
                elif df["roc_6h"].iloc[idx] < -2:
                    score -= 10
                bb_pos = df["bb_position"].iloc[idx]
                if bb_pos < 0.2:
                    score += 10
                elif bb_pos > 0.8:
                    score -= 10
            except:
                score = 0
            scores.append(score)
        df["composite_score"] = scores
        return df

    # ===== UTILITY =====

    def prepare_for_chronos(self, df):
        return df["close"].values

    def get_latest_indicators(self, df):
        latest = df.iloc[-1]
        return {
            "rsi": float(latest.get("rsi", 50)),
            "macd": float(latest.get("macd", 0)),
            "macd_signal": float(latest.get("macd_signal", 0)),
            "bb_position": float(latest.get("bb_position", 0.5)),
            "atr_pct": float(latest.get("atr_pct", 0)),
            "volume_ratio": float(latest.get("volume_ratio", 1)),
            "roc_6h": float(latest.get("roc_6h", 0)),
            "trend_strength": float(latest.get("trend_strength", 0)),
            "rel_volume": float(latest.get("rel_volume", 1)),
            "volatility_6h": float(latest.get("volatility_6h", 0)),
            "composite_score": float(latest.get("composite_score", 0)),
            "dist_from_resistance": float(latest.get("dist_from_resistance", 0)),
            "dist_from_support": float(latest.get("dist_from_support", 0)),
            "trend_direction": int(latest.get("trend_direction", 1)),
        }
