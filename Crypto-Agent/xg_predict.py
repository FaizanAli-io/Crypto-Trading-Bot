"""
Live Trading Signal Predictor
Generates BUY/SELL/HOLD signals with optional price estimation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import joblib

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from smc_features import SMCFeatureEngineer, integrate_smc_into_feature_engineer
from model_loader import ModelLoader

# Initialize
smc = SMCFeatureEngineer(
    swing_lookback=5,  # Lookback for swing points
    fvg_threshold=0.001,  # Minimum gap size (0.1%)
)


class SignalPredictor:
    """Generate live trading signals from trained XGBoost models"""

    def __init__(self, binance_client=None):
        self.data_collector = DataCollector(binance_client)
        self.feature_engineer = FeatureEngineer()
        self.model_loader = ModelLoader()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.signals_dir = Path("signals")
        self.signals_dir.mkdir(exist_ok=True)

        # Data cache: symbol -> (df, timestamp)
        self.data_cache = {}
        self.cache_ttl_seconds = 60

        # Confidence thresholds by horizon
        self.confidence_thresholds = {
            1: 0.70,  # 1h needs 70% confidence
            3: 0.60,  # 3h needs 60% confidence
            6: 0.60,  # 6h needs 60% confidence
            12: 0.55,  # 12h needs 55% confidence
            24: 0.55,  # 24h needs 55% confidence
        }

    def _get_cached_data(self, symbol, days):
        """Get data from cache if fresh, otherwise fetch and cache."""
        now = datetime.now()
        if symbol in self.data_cache:
            df, timestamp = self.data_cache[symbol]
            age_seconds = (now - timestamp).total_seconds()
            if age_seconds < self.cache_ttl_seconds:
                return df

        # Fetch fresh data
        df = self.data_collector.get_realtime_data(
            symbol=symbol, days=days, interval="1m", include_ongoing=False
        )
        self.data_cache[symbol] = (df, now)
        return df

    @staticmethod
    def interval_to_minutes(interval: str) -> int:
        """Convert interval string to minutes"""
        return ModelLoader.interval_to_minutes(interval)

    @staticmethod
    def calculate_horizon_shift(horizon_minutes: int, interval: str) -> int:
        """Calculate number of candles to shift based on prediction horizon"""
        return ModelLoader.calculate_horizon_shift(horizon_minutes, interval)

    def load_model(
        self,
        symbol="BTCUSDT",
        interval="1h",
        horizon_minutes=60,
        use_smc=None,
        silent=False,
    ):
        """
        Load trained XGBoost model for specific symbol, interval, and horizon
        (Delegates to centralized ModelLoader)
        """
        result = self.model_loader.load_model(
            symbol=symbol,
            interval=interval,
            horizon_minutes=horizon_minutes,
            use_smc=use_smc,
            silent=silent,
        )

        if result["success"]:
            self.model = result["model"]
            self.scaler = result["scaler"]
            self.feature_names = result["feature_names"]
            self.use_smc = result["is_smc"]
            self.interval = interval
            self.horizon_minutes = horizon_minutes
            self.shift_candles = self.calculate_horizon_shift(horizon_minutes, interval)
            return True
        else:
            return False

    def load_latest_model_for_symbol(self, symbol="BTCUSDT"):
        """
        Load the most recent model for a symbol (regardless of interval/horizon)
        (Delegates to centralized ModelLoader)
        """
        result = self.model_loader.load_latest_model_for_symbol(symbol=symbol)

        if result["success"]:
            self.model = result["model"]
            self.scaler = result["scaler"]
            self.feature_names = result["feature_names"]
            self.use_smc = result["is_smc"]
            self.interval = result.get("interval")
            self.horizon_minutes = result.get("horizon_minutes")
            self.shift_candles = self.calculate_horizon_shift(
                self.horizon_minutes, self.interval
            )
            return True
        else:
            return False

    # ================================================================
    # FOR PredictSignal CLASS
    # ================================================================

    def predict_signal(
        self,
        symbol,
        interval,
        horizon_minutes,
        custom_confidence=None,
        estimate_price=False,
        days=10,
        silent=False,
    ):
        """
        Generate trading signal for a symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Minutes ahead to predict (e.g., 60, 360)
            custom_confidence: Override default confidence threshold
            estimate_price: Whether to estimate target price
            days: Days of historical data
            silent: Suppress logging output

        Returns:
            dict: Signal with prediction details
        """
        if not silent:
            logger.info(
                f"Generating signal for {symbol} - {interval} interval, {horizon_minutes}min horizon"
            )

        # Load model
        if not self.load_model(symbol, interval, horizon_minutes, silent=True):
            raise ValueError(
                f"Could not load model for {symbol} with interval={interval}, horizon={horizon_minutes}min"
            )

        # Get confidence threshold (use horizon_minutes for lookup or default)
        min_confidence = (
            custom_confidence
            if custom_confidence
            else self.confidence_thresholds.get(horizon_minutes, 0.60)
        )

        # Fetch recent data with correct interval (use cache)
        df = self._get_cached_data(symbol, days)

        if df is None or len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")

        # Add SMC features if model was trained with them
        if self.use_smc:
            integrate_smc_into_feature_engineer(self.feature_engineer)

        # Add technical features (and SMC if enabled)
        df = self.feature_engineer.add_all_features(df)

        # Get current price and latest features
        current_price = float(df.iloc[-1]["close"])
        current_time = df.index[-1]

        # Extract features
        feature_data = df.iloc[-1:][self.feature_names]
        feature_data = feature_data.ffill().bfill().fillna(0)
        X = feature_data.values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Make prediction
        pred_proba = self.model.predict_proba(X)[0]
        prob_down = float(pred_proba[0])
        prob_up = float(pred_proba[1])

        predicted_direction = 1 if prob_up > 0.5 else 0
        confidence = float(max(pred_proba))

        # Determine signal
        if confidence < min_confidence:
            signal = "HOLD"
            reason = f"Low confidence ({confidence:.1%} < {min_confidence:.0%})"
        elif predicted_direction == 1:
            signal = "BUY"
            reason = f"Upward prediction with {confidence:.1%} confidence"
        else:
            signal = "SELL"
            reason = f"Downward prediction with {confidence:.1%} confidence"

        # Price estimation
        price_estimate = None
        if estimate_price and signal != "HOLD":
            price_estimate = self._estimate_target_price(
                df, current_price, predicted_direction, confidence, self.shift_candles
            )

        # Build result
        result = {
            "symbol": symbol,
            "interval": interval,
            "timestamp": current_time.isoformat(),
            "prediction_time": datetime.now().isoformat(),
            "horizon_minutes": horizon_minutes,
            "horizon_candles": self.shift_candles,
            "current_price": current_price,
            "signal": signal,
            "predicted_direction": "UP" if predicted_direction == 1 else "DOWN",
            "confidence": confidence,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "min_confidence_threshold": min_confidence,
            "reason": reason,
            "price_estimate": price_estimate,
            "valid_until": (
                datetime.now() + timedelta(minutes=horizon_minutes)
            ).isoformat(),
        }

        # Save signal
        # self._save_signal(result)
        if not silent:
            self._print_signal(result)

        return result

    def _predict_with_loaded_model(
        self,
        symbol,
        custom_confidence=None,
        estimate_price=False,
        days=5,
        silent=True,
    ):
        """
        Make prediction using already loaded model (for live monitoring)
        Skips model loading step for efficiency
        """
        # Get confidence threshold
        min_confidence = (
            custom_confidence
            if custom_confidence
            else self.confidence_thresholds.get(self.horizon_minutes, 0.60)
        )

        # Fetch recent data
        df = self.data_collector.get_realtime_data(
            symbol=symbol,
            days=days,
            interval=self.interval,
            include_ongoing=False,
        )

        if df is None or len(df) < 200:
            return None

        # Add SMC features if needed
        if self.use_smc:
            integrate_smc_into_feature_engineer(self.feature_engineer)

        # Add features
        df = self.feature_engineer.add_all_features(df)

        # Get current data
        current_price = float(df.iloc[-1]["close"])
        current_time = df.index[-1]

        # Prepare features
        feature_data = df.iloc[-1:][self.feature_names]
        feature_data = feature_data.ffill().bfill().fillna(0)
        X = feature_data.values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # Scale if available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict
        pred_proba = self.model.predict_proba(X)[0]
        prob_down = float(pred_proba[0])
        prob_up = float(pred_proba[1])

        predicted_direction = 1 if prob_up > 0.5 else 0
        confidence = float(max(pred_proba))

        # Determine signal
        if confidence < min_confidence:
            signal = "HOLD"
        elif predicted_direction == 1:
            signal = "LONG"
        else:
            signal = "SHORT"

        return {
            "symbol": symbol,
            "interval": self.interval,
            "horizon_minutes": self.horizon_minutes,
            "current_price": current_price,
            "signal": signal,
            "predicted_direction": "UP" if predicted_direction == 1 else "DOWN",
            "confidence": confidence,
            "prob_up": prob_up,
            "prob_down": prob_down,
        }

    def _estimate_target_price(
        self, df, current_price, direction, confidence, shift_candles
    ):
        """
        Estimate target price using historical movement patterns

        Args:
            shift_candles: Number of candles ahead (from self.shift_candles)

        WARNING: This is a rough estimate and should NOT be treated as precise.
        The model only predicts DIRECTION, not exact price.
        """
        # Calculate historical movements for this shift
        df_recent = df.tail(1440)  # Last 1440 candles for analysis

        # Calculate actual price changes over shift_candles periods
        price_changes = []
        for i in range(len(df_recent) - shift_candles):
            current = df_recent.iloc[i]["close"]
            future = df_recent.iloc[i + shift_candles]["close"]
            pct_change = (future - current) / current
            price_changes.append(pct_change)

        price_changes = np.array(price_changes)

        # Remove outliers (beyond 2 standard deviations)
        mean_change = np.mean(price_changes)
        std_change = np.std(price_changes)

        filtered_changes = price_changes[
            np.abs(price_changes - mean_change) < 2 * std_change
        ]

        if direction == 1:  # UP prediction
            # Use positive changes only
            up_moves = filtered_changes[filtered_changes > 0]
            if len(up_moves) > 0:
                # Weight by confidence: higher confidence = use higher percentile
                percentile = (
                    50 + (confidence - 0.5) * 60
                )  # Range: 50th to 80th percentile
                expected_change = np.percentile(up_moves, percentile)
            else:
                expected_change = mean_change
        else:  # DOWN prediction
            # Use negative changes only
            down_moves = filtered_changes[filtered_changes < 0]
            if len(down_moves) > 0:
                percentile = (
                    50 - (confidence - 0.5) * 60
                )  # Range: 50th to 20th percentile
                expected_change = np.percentile(down_moves, percentile)
            else:
                expected_change = mean_change

        # Calculate target price
        target_price = current_price * (1 + expected_change)

        # Calculate price range (confidence interval)
        price_std = current_price * std_change
        lower_bound = target_price - price_std
        upper_bound = target_price + price_std

        return {
            "method": "historical_average",
            "target_price": round(target_price, 2),
            "expected_change_pct": round(expected_change * 100, 2),
            "price_range": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2),
            },
            "warning": "This is a statistical estimate based on past patterns. NOT a guarantee.",
        }

    def _save_signal(self, signal):
        """Save signal to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"signal_{signal['symbol']}_{signal['interval']}_{signal['horizon_minutes']}m_{timestamp}.json"
        filepath = self.signals_dir / filename

        with open(filepath, "w") as f:
            json.dump(signal, f, indent=2)

        logger.info(f"Signal saved to {filepath}")

        # Also update "latest" file for easy access
        latest_file = (
            self.signals_dir
            / f"latest_{signal['symbol']}_{signal['interval']}_{signal['horizon_minutes']}m.json"
        )
        with open(latest_file, "w") as f:
            json.dump(signal, f, indent=2)

    def _print_signal(self, signal):
        """Pretty print signal"""
        logger.info("\n" + "=" * 70)
        logger.info(f"TRADING SIGNAL: {signal['symbol']}")
        logger.info("=" * 70)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Interval: {signal['interval']}")
        logger.info(
            f"Horizon: {signal['horizon_minutes']} minutes ({signal['horizon_candles']} candles)"
        )
        logger.info(f"Current Price: ${signal['current_price']:,.2f}")
        logger.info(f"\nPREDICTION:")
        logger.info(f"  Direction: {signal['predicted_direction']}")
        logger.info(f"  Confidence: {signal['confidence']:.1%}")
        logger.info(f"  Prob UP: {signal['prob_up']:.1%}")
        logger.info(f"  Prob DOWN: {signal['prob_down']:.1%}")
        logger.info(f"\nSIGNAL: {signal['signal']}")
        logger.info(f"  Reason: {signal['reason']}")

        if signal.get("price_estimate"):
            est = signal["price_estimate"]
            logger.info(f"\nPRICE ESTIMATE (Rough):")
            logger.info(
                f"  Target: ${est['target_price']:,.2f} ({est['expected_change_pct']:+.2f}%)"
            )
            logger.info(
                f"  Range: ${est['price_range']['lower']:,.2f} - ${est['price_range']['upper']:,.2f}"
            )
            logger.info(f"  ⚠️  {est['warning']}")

        logger.info(f"\nValid until: {signal['valid_until']}")
        logger.info("=" * 70 + "\n")

    def get_latest_signal(self, symbol, interval, horizon_minutes):
        """Retrieve latest saved signal"""
        latest_file = (
            self.signals_dir / f"latest_{symbol}_{interval}_{horizon_minutes}m.json"
        )

        if not latest_file.exists():
            return None

        with open(latest_file, "r") as f:
            return json.load(f)

    def get_signal_history(self, symbol, interval, horizon_minutes, days=7):
        """Get all signals from past N days"""
        cutoff_date = datetime.now() - timedelta(days=days)

        pattern = f"signal_{symbol}_{interval}_{horizon_minutes}m_*.json"
        signal_files = list(self.signals_dir.glob(pattern))

        signals = []
        for file in signal_files:
            try:
                timestamp_str = file.stem.split("_")[-2] + file.stem.split("_")[-1]
                file_date = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")

                if file_date >= cutoff_date:
                    with open(file, "r") as f:
                        signals.append(json.load(f))
            except:
                continue

        # Sort by timestamp
        signals.sort(key=lambda x: x["prediction_time"], reverse=True)

        return signals

    def get_all_available_models(self):
        """
        Parse all model files and extract symbol, interval, and horizon_minutes
        (Delegates to centralized ModelLoader)

        Returns:
            list: List of dicts with model info
        """
        return self.model_loader.get_all_available_models()

    def predict_all_models(self, estimate_price=True, days=10):
        """
        Run predictions on all available models and save to a single JSON file

        Args:
            estimate_price: Whether to estimate target price
            days: Number of days of historical data to use

        Returns:
            dict: All predictions with metadata
        """
        logger.info(
            f"Running predictions for {len(self.get_all_available_models())} models..."
        )

        # Get all models
        models = self.get_all_available_models()

        if not models:
            logger.error("No models found!")
            return None

        # Run predictions
        all_predictions = []
        failed_predictions = []

        for i, model_info in enumerate(models, 1):
            symbol = model_info["symbol"]
            interval = model_info["interval"]
            horizon_minutes = model_info["horizon_minutes"]

            try:
                signal = self.predict_signal(
                    symbol=symbol,
                    interval=interval,
                    horizon_minutes=horizon_minutes,
                    estimate_price=estimate_price,
                    days=days,
                    silent=True,
                )
                all_predictions.append(signal)

            except Exception as e:
                logger.error(f"{symbol}-{interval}: {e}")
                failed_predictions.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "horizon_minutes": horizon_minutes,
                        "error": str(e),
                    }
                )

        # Build result
        result = {
            "batch_timestamp": datetime.now().isoformat(),
            "total_models": len(models),
            "successful_predictions": len(all_predictions),
            "failed_predictions": len(failed_predictions),
            "predictions": all_predictions,
            "failures": failed_predictions,
        }

        logger.info(
            f"Predictions complete: {len(all_predictions)} success, {len(failed_predictions)} failed"
        )

        # Save to file
        # self._save_batch_predictions(result)

        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PREDICTION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total Models: {result['total_models']}")
        logger.info(f"Successful: {result['successful_predictions']}")
        logger.info(f"Failed: {result['failed_predictions']}")
        logger.info(f"{'='*70}\n")

        return result

    def _save_batch_predictions(self, batch_result):
        """Save batch predictions to a single JSON file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_predictions_{timestamp}.json"
        filepath = self.signals_dir / filename

        with open(filepath, "w") as f:
            json.dump(batch_result, f, indent=2)

        logger.info(f"Batch predictions saved to {filepath}")

        # Also save as "latest_batch"
        latest_file = self.signals_dir / "latest_batch_predictions.json"
        with open(latest_file, "w") as f:
            json.dump(batch_result, f, indent=2)

    def _print_batch_summary(self, results):
        """Print batch predictions in a formatted table"""
        logger.info("\n" + "=" * 100)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 100)
        logger.info(
            f"{'Symbol':<12} {'Interval':<10} {'Signal':<8} {'Direction':<10} {'Confidence':<12} {'Current Price':<15} {'Target Price':<15}"
        )
        logger.info("-" * 100)

        predictions = results.get("predictions", [])
        for pred in predictions:
            symbol = pred.get("symbol", "N/A")
            interval = pred.get("interval", "N/A")
            signal = pred.get("signal", "N/A")
            direction = pred.get("predicted_direction", "N/A")
            confidence = pred.get("confidence", 0)
            current_price = pred.get("current_price", 0)

            target_price = "N/A"
            if pred.get("price_estimate") and isinstance(pred["price_estimate"], dict):
                target_price = f"${pred['price_estimate'].get('target_price', 'N/A')}"

            logger.info(
                f"{symbol:<12} {interval:<10} {signal:<8} {direction:<10} "
                f"{confidence:>10.1%}  ${current_price:>13,.2f}  {target_price:>14}"
            )

        logger.info("-" * 100)
        logger.info(
            f"Total Predictions: {results['successful_predictions']} | Failed: {results['failed_predictions']}"
        )
        logger.info("=" * 100 + "\n")


# ========================
# CLI Interface
# ========================
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from binance.client import Client

    load_dotenv()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    if api_key:
        binance_client = Client(api_key, api_secret)
    else:
        binance_client = Client()

    predictor = SignalPredictor(binance_client)

    # Run predictions for ALL available models
    logger.info("\n" + "=" * 70)
    logger.info("LIVE PREDICTIONS - ALL CRYPTOCURRENCIES")
    logger.info("=" * 70)

    results = predictor.predict_all_models(estimate_price=True, days=10)

    # Print summary table
    if results and results["predictions"]:
        predictor._print_batch_summary(results)
    else:
        logger.warning("No predictions available")
