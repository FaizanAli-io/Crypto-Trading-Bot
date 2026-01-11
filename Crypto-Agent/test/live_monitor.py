#!/usr/bin/env python3
"""
Live Trading Monitor
Makes predictions with all available models every minute
Displays only high-confidence signals in a clean format
"""

import sys
import time
from datetime import datetime

from xg_predict import SignalPredictor

from loguru import logger

MIN_CONFIDENCE = 0.80
NEXT_SCAN_INTERVAL = 30


class LiveMonitor:
    def __init__(self, min_confidence=MIN_CONFIDENCE):
        # Quiet down noisy modules: only show errors
        logger.remove()
        logger.add(sys.stderr, level="ERROR", format="{message}")

        self.predictor = SignalPredictor()
        self.min_confidence = min_confidence
        self.loaded_models = {}  # Cache for loaded models

        # Load all models at startup
        print("Loading models...")
        self._load_all_models()
        print(f"✓ Loaded {len(self.loaded_models)} models\n")

    def _load_all_models(self):
        """Load all available models once at startup"""
        models = self.predictor.get_all_available_models()

        for model_info in models:
            try:
                key = (
                    model_info["symbol"],
                    model_info["interval"],
                    model_info["horizon_minutes"],
                )

                # Load model
                success = self.predictor.load_model(
                    symbol=model_info["symbol"],
                    interval=model_info["interval"],
                    horizon_minutes=model_info["horizon_minutes"],
                    silent=True,
                )

                if success:
                    # Store loaded model components
                    self.loaded_models[key] = {
                        "model": self.predictor.model,
                        "scaler": self.predictor.scaler,
                        "feature_names": self.predictor.feature_names,
                        "use_smc": self.predictor.use_smc,
                        "interval": self.predictor.interval,
                        "horizon_minutes": self.predictor.horizon_minutes,
                        "shift_candles": self.predictor.shift_candles,
                    }
            except:
                pass

    def scan_all_models(self):
        """Scan all models and return high-confidence signals"""
        signals = []

        for key, model_cache in self.loaded_models.items():
            symbol, interval, horizon_minutes = key

            try:
                # Restore cached model to predictor
                self.predictor.model = model_cache["model"]
                self.predictor.scaler = model_cache["scaler"]
                self.predictor.feature_names = model_cache["feature_names"]
                self.predictor.use_smc = model_cache["use_smc"]
                self.predictor.interval = model_cache["interval"]
                self.predictor.horizon_minutes = model_cache["horizon_minutes"]
                self.predictor.shift_candles = model_cache["shift_candles"]

                # Make prediction (skip model loading)
                signal = self.predictor._predict_with_loaded_model(
                    symbol=symbol,
                    custom_confidence=None,
                    estimate_price=False,
                    days=5,
                    silent=True,
                )

                # Filter by confidence
                if signal and signal["confidence"] >= self.min_confidence:
                    signals.append(signal)

            except:
                continue

        return signals

    def print_signals(self, signals):
        """Display high-confidence signals in a clean format"""
        if not signals:
            print("\n⚪ No high-confidence signals at this time")
            return

        # Sort by confidence (highest first)
        signals.sort(key=lambda x: x["confidence"], reverse=True)

        print(f"\n{'=' * 60}")
        print(f"🎯 HIGH-CONFIDENCE SIGNALS (>= {self.min_confidence*100:.0f}%)")
        print(f"{'=' * 60}")
        print(
            f"{'Symbol':<10} {'Interval':<10} {'Horizon':<10} {'Signal':<8} {'Confidence':<12} {'Price':<15}"
        )
        print(f"{'-' * 60}")

        for sig in signals:
            signal_icon = "🟢" if sig["signal"] == "LONG" else "🔴"

            print(
                f"{sig['symbol']:<10} {sig['interval']:<10} {sig['horizon_minutes']:>3}min    "
                f"{signal_icon} {sig['signal']:<6} {sig['confidence']*100:>5.1f}%      "
                f"{sig['current_price']:>12,.2f}"
            )

        print(f"{'=' * 60}\n")

    def run(self):
        """Main monitoring loop"""
        print(
            f"🚀 Starting Live Monitor (min confidence: {self.min_confidence*100:.0f}%)"
        )
        print("📊 Scanning all models every minute...")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"\n{'=' * 60}")
                print(f"📡 SCAN #{iteration} - {now}")
                print(f"{'=' * 60}")

                # Scan all models
                signals = self.scan_all_models()

                # Display results
                self.print_signals(signals)

                # Wait NEXT_SCAN_INTERVAL seconds
                print(f"⏳ Next scan in {NEXT_SCAN_INTERVAL} seconds...")
                time.sleep(NEXT_SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\n👋 Monitor stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise


def main():
    monitor = LiveMonitor(min_confidence=MIN_CONFIDENCE)
    monitor.run()


if __name__ == "__main__":
    main()
