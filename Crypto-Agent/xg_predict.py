"""
Simple XGBoost Predictor - NO CACHING
Direct Binance fetching for immediate results
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from model_loader import ModelLoader
from feature_engineering import FeatureEngineer
from data_collector import DataCollector


class SimpleSignalPredictor:
    """Simplified predictor without caching"""

    def __init__(self, model_root="models_2026-01-01"):
        self.model_loader = ModelLoader(model_root=model_root)
        self.feature_engineer = FeatureEngineer()
        self.data_collector = DataCollector()

        # Model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.interval = None
        self.horizon_minutes = None
        self.shift_candles = None

    def load_model(self, symbol, interval, horizon_minutes):
        """Load model for specific symbol/interval"""
        result = self.model_loader.load_model(
            symbol=symbol,
            interval=interval,
            horizon_minutes=horizon_minutes,
            use_smc=False,
            silent=True,
        )

        if result["success"]:
            self.model = result["model"]
            self.scaler = result["scaler"]
            self.feature_names = result["feature_names"]
            self.interval = interval
            self.horizon_minutes = horizon_minutes
            self.shift_candles = self.calculate_horizon_shift(horizon_minutes, interval)
            return True
        return False

    @staticmethod
    def calculate_horizon_shift(horizon_minutes: int, interval: str) -> int:
        """Calculate number of candles to shift"""
        return ModelLoader.calculate_horizon_shift(horizon_minutes, interval)

    def predict_signal(self, symbol, min_confidence=0.60):
        """Generate prediction for symbol"""
        import math

        try:
            # Fetch fresh data directly from Binance (no cache)
            logger.info(f"[{symbol}-{self.interval}] Fetching data from Binance...")
            df = self.data_collector.get_realtime_data(
                symbol=symbol, days=14, interval=self.interval, include_ongoing=False
            )

            if df is None or len(df) == 0:
                raise ValueError(f"No data fetched for {symbol}")

            logger.info(f"[{symbol}-{self.interval}] Fetched {len(df)} rows")

            # Feature engineering
            logger.info(f"[{symbol}-{self.interval}] Adding features...")
            df = self.feature_engineer.add_all_features(df)

            if df is None or len(df) == 0:
                raise ValueError(f"No data after feature engineering for {symbol}")

            logger.info(f"[{symbol}-{self.interval}] After features: {len(df)} rows")

            # Get current price
            current_price = float(df.iloc[-1]["close"])
            if math.isnan(current_price) or math.isinf(current_price):
                raise ValueError(f"Invalid price: {current_price}")

            current_time = df.index[-1]

            # Extract and prepare features
            feature_data = df.iloc[-1:][self.feature_names]
            feature_data = feature_data.ffill().bfill().fillna(0)
            X = feature_data.values
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

            # Apply scaler
            if self.scaler is not None:
                X = self.scaler.transform(X)
                X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

            # Make prediction
            pred_proba = self.model.predict_proba(X)[0]
            pred_proba = np.nan_to_num(pred_proba, nan=0.5, posinf=0.5, neginf=0.5)

            # Sanitize probabilities
            prob_down = float(pred_proba[0])
            prob_up = float(pred_proba[1])

            if math.isnan(prob_down) or math.isinf(prob_down):
                prob_down = 0.5
            if math.isnan(prob_up) or math.isinf(prob_up):
                prob_up = 0.5

            confidence = max(prob_down, prob_up)
            if math.isnan(confidence) or math.isinf(confidence):
                confidence = 0.5

            predicted_direction = 1 if prob_up > 0.5 else 0

            # Determine signal
            if confidence < min_confidence:
                signal = "HOLD"
                reason = f"Low confidence ({confidence:.1%})"
            elif predicted_direction == 1:
                signal = "BUY"
                reason = f"Upward prediction ({confidence:.1%})"
            else:
                signal = "SELL"
                reason = f"Downward prediction ({confidence:.1%})"

            # Safe float conversion helper
            def safe_float(val):
                if val is None:
                    return None
                try:
                    f = float(val)
                    return None if (math.isnan(f) or math.isinf(f)) else f
                except:
                    return None

            # Build clean result
            result = {
                "symbol": symbol,
                "interval": self.interval,
                "timestamp": current_time.isoformat(),
                "prediction_time": datetime.now().isoformat(),
                "horizon_minutes": self.horizon_minutes,
                "current_price": safe_float(current_price),
                "signal": signal,
                "predicted_direction": "UP" if predicted_direction == 1 else "DOWN",
                "confidence": safe_float(confidence),
                "prob_up": safe_float(prob_up),
                "prob_down": safe_float(prob_down),
                "reason": reason,
                "valid_until": (
                    datetime.now() + timedelta(minutes=self.horizon_minutes)
                ).isoformat(),
            }

            logger.info(
                f"✅ [{symbol}-{self.interval}] {signal} | Confidence: {confidence:.1%}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ [{symbol}-{self.interval}] Error: {e}")
            raise


def predict_all_models_simple():
    """Run predictions for all models without caching"""
    logger.info("=" * 60)
    logger.info("SIMPLE PREDICTOR - NO CACHE")
    logger.info("=" * 60)

    predictor = SimpleSignalPredictor()

    # Get all available models
    models = predictor.model_loader.get_all_available_models()
    logger.info(f"Found {len(models)} models")

    predictions = []
    failures = []

    for model_info in models:
        symbol = model_info["symbol"]
        interval = model_info["interval"]
        horizon = model_info["horizon_minutes"]

        try:
            # Load model
            if not predictor.load_model(symbol, interval, horizon):
                raise ValueError("Failed to load model")

            # Generate prediction
            result = predictor.predict_signal(symbol)
            predictions.append(result)

        except Exception as e:
            logger.error(f"Failed {symbol}-{interval}: {e}")
            failures.append({"symbol": symbol, "interval": interval, "error": str(e)})

    logger.info("=" * 60)
    logger.info(f"✅ Successful: {len(predictions)} | ❌ Failed: {len(failures)}")
    logger.info("=" * 60)

    return {
        "batch_timestamp": datetime.now().isoformat(),
        "total_models": len(models),
        "successful_predictions": len(predictions),
        "failed_predictions": len(failures),
        "predictions": predictions,
        "failures": failures,
    }


if __name__ == "__main__":
    results = predict_all_models_simple()
    print(f"\n✅ Generated {results['successful_predictions']} predictions")
    print(f"❌ Failed {results['failed_predictions']} predictions")
