#!/usr/bin/env python3
"""
Live Trading Monitor (Streaming)
Prints each prediction as soon as it is made (table format)
"""

import sys
import time
from loguru import logger
from datetime import datetime

from loader import ModelLoader
from predict import SignalPredictor


MIN_CONFIDENCE = 0.50
SCAN_INTERVAL = 30


class LiveMonitor:
    def __init__(self, min_confidence=MIN_CONFIDENCE):
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

        self.min_confidence = min_confidence
        self.predictor = SignalPredictor()
        self.model_loader = ModelLoader()

        print("🔄 Discovering models...")
        self.models = self.model_loader.list_available_models()
        print(f"✓ Found {len(self.models)} models\n")

    def run(self):
        print(
            f"🚀 Live Monitor started (min confidence: {self.min_confidence*100:.0f}%)"
        )
        print("📡 Streaming predictions as they are made")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                print("=" * 90)
                print(
                    f"🕒 SCAN #{iteration} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                print("=" * 90)

                print(
                    f"{'SYMBOL':<10} {'INTERVAL':<8} {'HORIZON':<8} "
                    f"{'SIGNAL':<10} {'CONF %':<8} {'PRICE':>12}"
                )
                print("-" * 90)

                for model in self.models:
                    symbol = model["symbol"]
                    interval = model["interval"]
                    horizon = f"{model['horizon_minutes']}m"

                    try:
                        result = self.predictor.predict_signal(
                            symbol=symbol,
                            interval=interval,
                            horizon_minutes=model["horizon_minutes"],
                            days=5,
                            estimate_price=False,
                        )

                        if result["confidence"] < self.min_confidence:
                            print(
                                f"{symbol:<10} {interval:<8} {horizon:<8} "
                                f"{'HOLD':<10} {result['confidence']*100:<7.1f} {'-':>12}",
                                flush=True,
                            )
                            continue

                        icon = "🟢 BUY" if result["signal"] == "BUY" else "🔴 SELL"

                        print(
                            f"{symbol:<10} {interval:<8} {horizon:<8} "
                            f"{icon:<10} {result['confidence']*100:<7.1f} "
                            f"{result['current_price']:>12,.2f}",
                            flush=True,
                        )

                    except Exception as e:
                        print(
                            f"{symbol:<10} {interval:<8} {horizon:<8} "
                            f"{'ERROR':<10} {'-':<8} {str(e)[:12]:>12}",
                            flush=True,
                        )

                print(f"\n⏳ Next scan in {SCAN_INTERVAL} seconds...\n")
                time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\n👋 Monitor stopped")


def main():
    monitor = LiveMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
