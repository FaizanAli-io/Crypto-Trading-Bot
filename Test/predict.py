"""
Live Trading Signal Predictor (Simplified)
Generates BUY / SELL / HOLD signals
"""

import json
import numpy as np
from loguru import logger
from datetime import datetime, timedelta

from loader import ModelLoader
from collector import DataCollector
from features import FeatureEngineer


class SignalPredictor:
    def __init__(self, binance_client=None):
        self.data_collector = DataCollector(binance_client)
        self.feature_engineer = FeatureEngineer()
        self.model_loader = ModelLoader()

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.interval = None
        self.horizon_minutes = None
        self.shift_candles = None

        self.confidence_thresholds = {
            1: 0.70,
            3: 0.60,
            6: 0.60,
            12: 0.55,
            24: 0.55,
        }

    def load_model(self, symbol, interval, horizon_minutes):
        result = self.model_loader.load_model(
            symbol=symbol,
            interval=interval,
            horizon_minutes=horizon_minutes,
            silent=True,
        )

        if not result["success"]:
            return False

        self.model = result["model"]
        self.scaler = result["scaler"]
        self.feature_names = result["feature_names"]
        self.interval = interval
        self.horizon_minutes = horizon_minutes
        self.shift_candles = ModelLoader.calculate_horizon_shift(
            horizon_minutes, interval
        )
        return True

    def predict_signal(
        self,
        symbol,
        interval,
        horizon_minutes,
        days=10,
        estimate_price=False,
    ):
        if not self.load_model(symbol, interval, horizon_minutes):
            raise RuntimeError("Model not found")

        min_conf = self.confidence_thresholds.get(horizon_minutes, 0.60)

        df = self.data_collector.get_realtime_data(
            symbol=symbol,
            days=days,
            interval="1m",
            include_ongoing=False,
        )

        if df is None or len(df) < 200:
            raise RuntimeError("Insufficient data")

        df = self.feature_engineer.add_all_features(df)

        current_price = float(df.iloc[-1]["close"])
        current_time = df.index[-1]

        X = df.iloc[-1:][self.feature_names]
        X = X.ffill().bfill().fillna(0).values
        X = np.nan_to_num(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        prob_down, prob_up = self.model.predict_proba(X)[0]
        confidence = max(prob_up, prob_down)

        if confidence < min_conf:
            signal = "HOLD"
        elif prob_up > prob_down:
            signal = "BUY"
        else:
            signal = "SELL"

        price_estimate = None
        if estimate_price and signal != "HOLD":
            price_estimate = self._estimate_price(df, current_price)

        return {
            "symbol": symbol,
            "interval": interval,
            "timestamp": current_time.isoformat(),
            "prediction_time": datetime.now().isoformat(),
            "horizon_minutes": horizon_minutes,
            "current_price": current_price,
            "signal": signal,
            "confidence": confidence,
            "prob_up": float(prob_up),
            "prob_down": float(prob_down),
            "price_estimate": price_estimate,
            "valid_until": (
                datetime.now() + timedelta(minutes=horizon_minutes)
            ).isoformat(),
        }

    def _estimate_price(self, df, current_price):
        df = df.tail(1440)

        changes = []
        for i in range(len(df) - self.shift_candles):
            c = df.iloc[i]["close"]
            f = df.iloc[i + self.shift_candles]["close"]
            changes.append((f - c) / c)

        changes = np.array(changes)
        mean = changes.mean()
        std = changes.std()

        target = current_price * (1 + mean)
        return {
            "target_price": round(target, 2),
            "expected_change_pct": round(mean * 100, 2),
            "range": {
                "low": round(target - current_price * std, 2),
                "high": round(target + current_price * std, 2),
            },
        }


# ========================
# CLI
# ========================
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from binance.client import Client

    load_dotenv()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    client = Client(api_key, api_secret) if api_key else Client()
    predictor = SignalPredictor(client)

    result = predictor.predict_signal(
        symbol="BTCUSDT",
        interval="1h",
        horizon_minutes=60,
        estimate_price=True,
    )

    logger.info(json.dumps(result, indent=2))
