"""
Historical Model Validator
Tests model accuracy on historical data from 1/1/2026 to 1/10/2026
Evaluates directional accuracy (UP vs DOWN predictions)
"""

import os
import json
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from binance.client import Client
from datetime import datetime, timedelta

from loader import ModelLoader
from collector import DataCollector
from features import FeatureEngineer


class ModelValidator:
    def __init__(self, binance_client=None):
        self.data_collector = DataCollector(binance_client)
        self.feature_engineer = FeatureEngineer()
        self.model_loader = ModelLoader()

        self.start_date = datetime(2026, 1, 1)
        self.end_date = datetime(2026, 1, 10)
        self.days_range = (self.end_date - self.start_date).days

        logger.info(
            f"Validator initialized for {self.start_date.date()} to {self.end_date.date()}"
        )

    def validate_model(self, symbol, interval, horizon_minutes):
        """
        Validate a single model on historical data
        Returns accuracy metrics for directional predictions
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating {symbol} - {interval} - {horizon_minutes}min horizon")
        logger.info(f"{'='*60}")

        # Load model
        result = self.model_loader.load_model(
            symbol=symbol,
            interval=interval,
            horizon_minutes=horizon_minutes,
            silent=True,
        )

        if not result["success"]:
            logger.error(f"❌ Model not found")
            return None

        model = result["model"]
        scaler = result["scaler"]
        feature_names = result["feature_names"]

        # Calculate shift for predictions
        shift_candles = ModelLoader.calculate_horizon_shift(horizon_minutes, interval)

        # Fetch historical data (need extra days for feature calculation)
        logger.info(f"Fetching historical data...")
        df = self.data_collector.get_historical_data_by_date_range(
            symbol=symbol,
            start_date=self.start_date
            - timedelta(days=30),  # Extra 30 days for indicators
            days=self.days_range + 30,
            interval=interval,
        )

        if df is None or len(df) < 200:
            logger.error("❌ Insufficient historical data")
            return None

        logger.info(f"✅ Fetched {len(df)} candles")

        # Add technical features
        logger.info(f"Engineering features...")
        df = self.feature_engineer.add_all_features(df)
        logger.info(f"✅ Features added: {len(df)} rows remaining")

        # Filter to validation period only
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        if len(df) < shift_candles + 10:
            logger.error(f"❌ Not enough data in validation period: {len(df)} rows")
            return None

        logger.info(
            f"Validation period: {len(df)} candles from {df.index[0]} to {df.index[-1]}"
        )

        # Make predictions and track results
        predictions = []
        correct_direction = 0
        total_predictions = 0

        logger.info(f"Making predictions with {shift_candles} candle shift...")

        for i in range(len(df) - shift_candles):
            try:
                current_row = df.iloc[i]
                future_row = df.iloc[i + shift_candles]

                current_price = float(current_row["close"])
                future_price = float(future_row["close"])
                actual_direction = "UP" if future_price > current_price else "DOWN"

                # Extract features for prediction
                X = df.iloc[i : i + 1][feature_names]
                X = X.ffill().bfill().fillna(0).values
                X = np.nan_to_num(X)

                if scaler is not None:
                    X = scaler.transform(X)

                # Make prediction
                prob_down, prob_up = model.predict_proba(X)[0]
                predicted_direction = "UP" if prob_up > prob_down else "DOWN"
                confidence = max(prob_up, prob_down)

                # Check if correct
                is_correct = predicted_direction == actual_direction
                if is_correct:
                    correct_direction += 1
                total_predictions += 1

                predictions.append(
                    {
                        "timestamp": current_row.name.isoformat(),
                        "current_price": round(current_price, 2),
                        "future_price": round(future_price, 2),
                        "price_change_pct": round(
                            ((future_price - current_price) / current_price) * 100, 3
                        ),
                        "actual_direction": actual_direction,
                        "predicted_direction": predicted_direction,
                        "prob_up": round(float(prob_up), 4),
                        "prob_down": round(float(prob_down), 4),
                        "confidence": round(float(confidence), 4),
                        "correct": is_correct,
                    }
                )

            except Exception as e:
                logger.error(f"Error at index {i}: {e}")
                continue

        # Calculate metrics
        accuracy = (
            (correct_direction / total_predictions * 100)
            if total_predictions > 0
            else 0
        )

        # Calculate accuracy by confidence buckets
        df_predictions = pd.DataFrame(predictions)

        high_conf = df_predictions[df_predictions["confidence"] >= 0.70]
        medium_conf = df_predictions[
            (df_predictions["confidence"] >= 0.60)
            & (df_predictions["confidence"] < 0.70)
        ]
        low_conf = df_predictions[df_predictions["confidence"] < 0.60]

        high_conf_accuracy = (
            (high_conf["correct"].sum() / len(high_conf) * 100)
            if len(high_conf) > 0
            else 0
        )
        medium_conf_accuracy = (
            (medium_conf["correct"].sum() / len(medium_conf) * 100)
            if len(medium_conf) > 0
            else 0
        )
        low_conf_accuracy = (
            (low_conf["correct"].sum() / len(low_conf) * 100)
            if len(low_conf) > 0
            else 0
        )

        # Calculate accuracy on UP vs DOWN predictions
        up_predictions = df_predictions[df_predictions["predicted_direction"] == "UP"]
        down_predictions = df_predictions[
            df_predictions["predicted_direction"] == "DOWN"
        ]

        up_accuracy = (
            (up_predictions["correct"].sum() / len(up_predictions) * 100)
            if len(up_predictions) > 0
            else 0
        )
        down_accuracy = (
            (down_predictions["correct"].sum() / len(down_predictions) * 100)
            if len(down_predictions) > 0
            else 0
        )

        results = {
            "symbol": symbol,
            "interval": interval,
            "horizon_minutes": horizon_minutes,
            "validation_period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "days": self.days_range,
            },
            "total_predictions": total_predictions,
            "correct_predictions": correct_direction,
            "overall_accuracy": round(accuracy, 2),
            "accuracy_by_confidence": {
                "high_confidence_70plus": {
                    "count": len(high_conf),
                    "accuracy": round(high_conf_accuracy, 2),
                },
                "medium_confidence_60_70": {
                    "count": len(medium_conf),
                    "accuracy": round(medium_conf_accuracy, 2),
                },
                "low_confidence_under_60": {
                    "count": len(low_conf),
                    "accuracy": round(low_conf_accuracy, 2),
                },
            },
            "accuracy_by_direction": {
                "up_predictions": {
                    "count": len(up_predictions),
                    "accuracy": round(up_accuracy, 2),
                },
                "down_predictions": {
                    "count": len(down_predictions),
                    "accuracy": round(down_accuracy, 2),
                },
            },
            "sample_predictions": predictions[:10],  # First 10 for reference
        }

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 VALIDATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Predictions: {total_predictions}")
        logger.info(f"Correct: {correct_direction} ({accuracy:.2f}%)")
        logger.info(f"\n📈 By Confidence Level:")
        logger.info(
            f"  High (≥70%): {len(high_conf)} predictions, {high_conf_accuracy:.2f}% accuracy"
        )
        logger.info(
            f"  Medium (60-70%): {len(medium_conf)} predictions, {medium_conf_accuracy:.2f}% accuracy"
        )
        logger.info(
            f"  Low (<60%): {len(low_conf)} predictions, {low_conf_accuracy:.2f}% accuracy"
        )
        logger.info(f"\n📉 By Direction:")
        logger.info(
            f"  UP predictions: {len(up_predictions)} total, {up_accuracy:.2f}% accuracy"
        )
        logger.info(
            f"  DOWN predictions: {len(down_predictions)} total, {down_accuracy:.2f}% accuracy"
        )
        logger.info(f"{'='*60}\n")

        return results

    def validate_all_models(self):
        """
        Validate all available models
        """
        logger.info("🔍 Discovering available models...")
        available_models = self.model_loader.list_available_models()

        if not available_models:
            logger.error("❌ No models found")
            return []

        logger.info(f"✅ Found {len(available_models)} models\n")

        all_results = []
        successful = 0
        failed = 0

        for idx, model_info in enumerate(available_models, 1):
            logger.info(f"\n{'#'*60}")
            logger.info(f"Model {idx}/{len(available_models)}")
            logger.info(f"{'#'*60}")

            result = self.validate_model(
                symbol=model_info["symbol"],
                interval=model_info["interval"],
                horizon_minutes=model_info["horizon_minutes"],
            )

            if result:
                all_results.append(result)
                successful += 1
            else:
                failed += 1

        # Save results to file
        output_file = (
            f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\n{'#'*60}")
        logger.info(f"🏁 VALIDATION COMPLETE")
        logger.info(f"{'#'*60}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📁 Results saved to: {output_file}")

        # Print summary statistics
        if all_results:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 OVERALL SUMMARY")
            logger.info(f"{'='*60}")

            df_summary = pd.DataFrame(all_results)
            avg_accuracy = df_summary["overall_accuracy"].mean()
            best_model = df_summary.loc[df_summary["overall_accuracy"].idxmax()]
            worst_model = df_summary.loc[df_summary["overall_accuracy"].idxmin()]

            logger.info(f"Average Accuracy: {avg_accuracy:.2f}%")
            logger.info(f"\n🏆 Best Model:")
            logger.info(
                f"  {best_model['symbol']} {best_model['interval']} {best_model['horizon_minutes']}min"
            )
            logger.info(f"  Accuracy: {best_model['overall_accuracy']:.2f}%")
            logger.info(f"\n⚠️  Worst Model:")
            logger.info(
                f"  {worst_model['symbol']} {worst_model['interval']} {worst_model['horizon_minutes']}min"
            )
            logger.info(f"  Accuracy: {worst_model['overall_accuracy']:.2f}%")

            # Accuracy by horizon
            logger.info(f"\n📈 Accuracy by Horizon:")
            for horizon in sorted(df_summary["horizon_minutes"].unique()):
                horizon_df = df_summary[df_summary["horizon_minutes"] == horizon]
                avg_acc = horizon_df["overall_accuracy"].mean()
                logger.info(f"  {horizon}min: {avg_acc:.2f}% (n={len(horizon_df)})")

            # Accuracy by symbol
            logger.info(f"\n💰 Accuracy by Symbol:")
            for symbol in sorted(df_summary["symbol"].unique()):
                symbol_df = df_summary[df_summary["symbol"] == symbol]
                avg_acc = symbol_df["overall_accuracy"].mean()
                logger.info(f"  {symbol}: {avg_acc:.2f}% (n={len(symbol_df)})")

            logger.info(f"{'='*60}\n")

        return all_results


def validate_specific_models(symbol=None, interval=None, horizon_minutes=None):
    """
    Validate specific model(s) based on filters
    """
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    client = Client(api_key, api_secret) if api_key else Client()

    validator = ModelValidator(client)

    if symbol and interval and horizon_minutes:
        # Single model
        result = validator.validate_model(symbol, interval, horizon_minutes)
        if result:
            output_file = f"validation_{symbol}_{interval}_{horizon_minutes}min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"📁 Results saved to: {output_file}")
        return [result] if result else []
    else:
        # All models
        return validator.validate_all_models()


if __name__ == "__main__":
    import sys

    logger.info("🚀 Starting Model Validator")
    logger.info("=" * 60)

    if len(sys.argv) > 1:
        # Specific model validation
        if len(sys.argv) == 4:
            symbol = sys.argv[1]
            interval = sys.argv[2]
            horizon_minutes = int(sys.argv[3])

            logger.info(
                f"Validating single model: {symbol} {interval} {horizon_minutes}min"
            )
            validate_specific_models(symbol, interval, horizon_minutes)
        else:
            logger.error("Usage: python validator.py [SYMBOL INTERVAL HORIZON_MINUTES]")
            logger.error("Example: python validator.py BTCUSDT 1h 60")
            logger.error("Or run without arguments to validate all models")
    else:
        # Validate all models
        logger.info("Validating ALL models...")
        validate_specific_models()
