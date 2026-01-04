"""
Model Validator: Test XGBoost prediction accuracy for directional forecasting
Modified to use trained XGBoost models with feature engineering
"""

import json
import numpy as np

from pathlib import Path
from datetime import datetime

from config import SUPPORTED_CRYPTOS
from model_loader import ModelLoader
from data_collector import DataCollector
from feature_engineering import FeatureEngineer

from loguru import logger


class ModelValidator:
    """Validate and track XGBoost model directional prediction accuracy"""

    def __init__(self, binance_client=None):
        self.data_collector = DataCollector(binance_client)
        self.feature_engineer = FeatureEngineer()
        self.model_loader = ModelLoader()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)

    @staticmethod
    def interval_to_minutes(interval: str) -> int:
        """Convert interval string to minutes"""
        return ModelLoader.interval_to_minutes(interval)

    @staticmethod
    def calculate_horizon_shift(horizon_minutes: int, interval: str) -> int:
        """Calculate number of candles to shift based on prediction horizon"""
        return ModelLoader.calculate_horizon_shift(horizon_minutes, interval)

    def load_model(
        self, symbol="BTCUSDT", interval="1h", horizon_minutes=60, use_smc=False
    ):
        """
        Load trained XGBoost model for specific symbol, interval, and horizon
        (Delegates to centralized ModelLoader)

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Prediction horizon in minutes (e.g., 60, 360)
            use_smc: If True, try to load SMC model first, fallback to simple model

        Returns:
            bool: Success status
        """
        result = self.model_loader.load_model(
            symbol=symbol,
            interval=interval,
            horizon_minutes=horizon_minutes,
            use_smc=use_smc,
        )

        if result["success"]:
            self.model = result["model"]
            self.scaler = result["scaler"]
            self.feature_names = result["feature_names"]
            self.interval = interval
            self.horizon_minutes = horizon_minutes
            self.shift_candles = self.calculate_horizon_shift(horizon_minutes, interval)
            self.model_type = "smc" if result["is_smc"] else "simple"
            return True
        else:
            return False

    def load_latest_model_for_symbol(self, symbol="BTCUSDT"):
        """
        Load the most recent model for a symbol (regardless of interval/horizon)
        (Delegates to centralized ModelLoader)

        Args:
            symbol: Trading pair

        Returns:
            bool: Success status
        """
        result = self.model_loader.load_latest_model_for_symbol(symbol=symbol)

        if result["success"]:
            self.model = result["model"]
            self.scaler = result["scaler"]
            self.feature_names = result["feature_names"]
            self.interval = result.get("interval")
            self.horizon_minutes = result.get("horizon_minutes")
            self.shift_candles = self.calculate_horizon_shift(
                self.horizon_minutes, self.interval
            )
            self.model_type = "smc" if result["is_smc"] else "simple"
            return True
        else:
            return False

    # ================================================================
    # FOR ModelValidator CLASS
    # ================================================================

    def backtest(
        self,
        symbol="BTCUSDT",
        interval="1h",
        horizon_minutes=60,
        days=10,
        min_confidence=0.7,
        use_smc=False,
    ):
        """
        Backtest XGBoost model on historical data (directional accuracy only)

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            interval (str): Candle interval (e.g., "15m", "1h")
            horizon_minutes (int): Minutes ahead to predict (must match trained model)
            days (int): Days of historical data to test
            min_confidence (float): Minimum prediction probability to count (0.5-1.0)
            use_smc (bool): If True, try to load SMC model first, fallback to simple model

        Returns:
            dict: Backtesting results with metrics
        """
        logger.info(
            f"Backtesting {symbol} - {interval} interval, {horizon_minutes}min predictions over {days} days..."
        )

        # Load model for this symbol, interval, and horizon
        if not self.load_model(symbol, interval, horizon_minutes, use_smc=use_smc):
            raise ValueError(
                f"Could not load model for {symbol} with interval={interval}, horizon={horizon_minutes}min"
            )

        # Get shift_candles from loaded model
        shift_candles = self.shift_candles
        logger.info(
            f"Using shift_candles={shift_candles} for {horizon_minutes}min prediction"
        )

        # Fetch historical data (need extra for indicators)
        total_days = days  # Extra for technical indicators
        df = self.data_collector.get_realtime_data(
            symbol=symbol, days=total_days, interval=interval  # Use correct interval
        )

        if df is None or len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")

        logger.info(f"Fetched {len(df)} candles")

        # Add all technical features
        logger.info("Calculating technical indicators...")
        df = self.feature_engineer.add_all_features(df)

        # Store predictions vs actual
        predictions = []
        actuals = []
        confidences = []
        timestamps = []
        current_prices = []
        future_prices = []

        # Walk forward through history
        start_idx = 200  # Wait for indicators to stabilize
        end_idx = len(df) - shift_candles  # Use shift_candles instead of horizon

        # Test every shift_candles
        step = shift_candles

        logger.info(f"Testing from index {start_idx} to {end_idx}, step={step}")

        for i in range(start_idx, end_idx, step):
            try:
                # Get current price
                current_price = float(df.iloc[i]["close"])

                # Get actual future price (shift_candles later)
                future_price = float(df.iloc[i + shift_candles]["close"])

                # Actual direction: 1 if price went up, 0 if down
                actual_direction = 1 if future_price > current_price else 0

                # Extract features for this point in time
                feature_data = df.iloc[i : i + 1][self.feature_names]

                # Handle missing values
                feature_data = feature_data.ffill().bfill().fillna(0)

                # Convert to numpy array
                X = feature_data.values

                # Handle infinite values
                X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

                # Apply scaler if available
                if self.scaler is not None:
                    X = self.scaler.transform(X)

                # Make prediction
                pred_proba = self.model.predict_proba(X)[0]
                predicted_direction = int(pred_proba[1] > 0.5)
                confidence = float(max(pred_proba))

                # Only count predictions above minimum confidence
                if confidence < min_confidence:
                    continue

                # Store results
                predictions.append(predicted_direction)
                actuals.append(actual_direction)
                confidences.append(confidence)
                timestamps.append(df.index[i])
                current_prices.append(current_price)
                future_prices.append(future_price)

                if len(predictions) % 50 == 0:
                    logger.info(f"Progress: {len(predictions)} predictions made")

            except Exception as e:
                logger.warning(f"Error at index {i}: {e}")
                continue

        if len(predictions) == 0:
            raise ValueError("No successful predictions made during backtest")

        # Calculate metrics
        metrics = self._calculate_directional_metrics(
            predictions, actuals, confidences, current_prices, future_prices
        )

        # Save results
        results = {
            "symbol": symbol,
            "interval": interval,
            "prediction_horizon_minutes": horizon_minutes,
            "prediction_horizon_candles": shift_candles,
            "test_period_days": days,
            "min_confidence": min_confidence,
            "num_predictions": len(predictions),
            "metrics": metrics,
            "predictions": predictions,
            "actuals": actuals,
            "confidences": confidences,
            "current_prices": current_prices,
            "future_prices": future_prices,
            "timestamps": [str(t) for t in timestamps],
        }

        self._print_results(symbol, interval, horizon_minutes, results, metrics)

        return results

    def backtest_for_simulation(
        self,
        symbol="BTCUSDT",
        interval="1h",
        horizon_minutes=60,
        days=30,
        min_confidence=0.7,
        use_smc=False,
    ):
        """
        Backtest XGBoost model on historical data with full dataframe

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            interval (str): Candle interval (e.g., "15m", "1h")
            horizon_minutes (int): Minutes ahead to predict
            days (int): Days of historical data to test
            min_confidence (float): Minimum prediction probability to count (0.5-1.0)
            use_smc (bool): If True, try to load SMC model first, fallback to simple model

        Returns:
            dict: Backtesting results with metrics and full dataframe
        """
        logger.info(
            f"Backtesting {symbol} - {interval} interval, {horizon_minutes}min predictions over {days} days..."
        )

        # Load model for this symbol, interval, and horizon
        if not self.load_model(symbol, interval, horizon_minutes, use_smc=use_smc):
            raise ValueError(
                f"Could not load model for {symbol} with interval={interval}, horizon={horizon_minutes}min"
            )

        # Get shift_candles from loaded model
        shift_candles = self.shift_candles
        logger.info(
            f"Using shift_candles={shift_candles} for {horizon_minutes}min prediction"
        )

        # Fetch historical data (need extra for indicators)
        total_days = days + 9  # Extra for technical indicators
        df = self.data_collector.get_realtime_data(
            symbol=symbol, days=total_days, interval=interval  # Use correct interval
        )

        if df is None or len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")

        logger.info(f"Fetched {len(df)} candles")

        # Add all technical features
        logger.info("Calculating technical indicators...")
        df = self.feature_engineer.add_all_features(df)

        # Store the full dataframe for price estimation (IMPORTANT!)
        df_full = df.copy()

        # Store predictions vs actual
        predictions = []
        actuals = []
        confidences = []
        timestamps = []
        current_prices = []
        future_prices = []

        # Walk forward through history
        start_idx = 200
        end_idx = len(df) - shift_candles  # Use shift_candles instead of horizon

        # Test every shift_candles
        step = shift_candles

        logger.info(f"Testing from index {start_idx} to {end_idx}, step={step}")

        for i in range(start_idx, end_idx, step):
            try:
                # Get current price
                current_price = float(df.iloc[i]["close"])

                # Get actual future price (shift_candles later)
                future_price = float(df.iloc[i + shift_candles]["close"])

                # Actual direction: 1 if price went up, 0 if down
                actual_direction = 1 if future_price > current_price else 0

                # Extract features for this point in time
                feature_data = df.iloc[i : i + 1][self.feature_names]

                # Handle missing values
                feature_data = feature_data.ffill().bfill().fillna(0)

                # Convert to numpy array
                X = feature_data.values

                # Handle infinite values
                X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

                # Apply scaler if available
                if self.scaler is not None:
                    X = self.scaler.transform(X)

                # Make prediction
                pred_proba = self.model.predict_proba(X)[0]
                predicted_direction = int(pred_proba[1] > 0.5)
                confidence = float(max(pred_proba))

                # Only count predictions above minimum confidence
                if confidence < min_confidence:
                    continue

                # Store results
                predictions.append(predicted_direction)
                actuals.append(actual_direction)
                confidences.append(confidence)
                timestamps.append(df.index[i])
                current_prices.append(current_price)
                future_prices.append(future_price)

                if len(predictions) % 50 == 0:
                    logger.info(f"Progress: {len(predictions)} predictions made")

            except Exception as e:
                logger.warning(f"Error at index {i}: {e}")
                continue

        if len(predictions) == 0:
            raise ValueError("No successful predictions made during backtest")

        # Calculate metrics
        metrics = self._calculate_directional_metrics(
            predictions, actuals, confidences, current_prices, future_prices
        )

        # Build results dictionary
        results = {
            "symbol": symbol,
            "interval": interval,
            "prediction_horizon_minutes": horizon_minutes,
            "prediction_horizon_candles": shift_candles,
            "test_period_days": days,
            "min_confidence": min_confidence,
            "num_predictions": len(predictions),
            "metrics": metrics,
            "predictions": predictions,
            "actuals": actuals,
            "confidences": confidences,
            "current_prices": current_prices,
            "future_prices": future_prices,
            "timestamps": [str(t) for t in timestamps],
            "dataframe": df_full,  # Include full dataframe for price estimation
        }

        # Save results (without dataframe)
        self._save_results(symbol, results, horizon_minutes)
        self._print_results(symbol, interval, horizon_minutes, results, metrics)

        return results

    def _calculate_directional_metrics(
        self, predictions, actuals, confidences, current_prices, future_prices
    ):
        """
        Calculate directional prediction metrics with robust validation

        Accuracy is calculated as: (correct predictions / total predictions) * 100
        Where correct = predicted direction matches actual price movement direction
        """
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)
        current_prices = np.array(current_prices)
        future_prices = np.array(future_prices)

        # Validation: ensure all arrays have same length
        assert (
            len(predictions) == len(actuals) == len(confidences)
        ), "Array length mismatch in metrics calculation"

        # Validation: ensure predictions and actuals are binary (0 or 1)
        assert np.all(
            (predictions == 0) | (predictions == 1)
        ), "Predictions must be binary (0 or 1)"
        assert np.all(
            (actuals == 0) | (actuals == 1)
        ), "Actuals must be binary (0 or 1)"

        # Overall directional accuracy: exact match between prediction and actual
        correct = predictions == actuals
        num_correct = correct.sum()
        total_predictions = len(predictions)
        directional_accuracy = (num_correct / total_predictions) * 100

        # Separate accuracy for UP and DOWN predictions
        up_mask = predictions == 1
        down_mask = predictions == 0

        up_accuracy = correct[up_mask].mean() * 100 if up_mask.sum() > 0 else 0
        down_accuracy = correct[down_mask].mean() * 100 if down_mask.sum() > 0 else 0

        # High confidence predictions (top 25%)
        high_conf_threshold = np.percentile(confidences, 75)
        high_conf_mask = confidences >= high_conf_threshold

        high_conf_accuracy = (
            correct[high_conf_mask].mean() * 100 if high_conf_mask.sum() > 0 else 0
        )

        # Calculate actual returns if we followed predictions
        actual_returns = (future_prices - current_prices) / current_prices

        # If predicted UP and it went UP -> positive return
        # If predicted DOWN and it went DOWN -> positive return (in reality you'd short)
        # If wrong direction -> negative return
        strategy_returns = []
        for i in range(len(predictions)):
            if predictions[i] == 1:  # Predicted UP
                strategy_returns.append(actual_returns[i])
            else:  # Predicted DOWN
                strategy_returns.append(
                    -actual_returns[i]
                )  # Inverse return (as if shorting)

        strategy_returns = np.array(strategy_returns)
        avg_return_per_trade = strategy_returns.mean() * 100

        # Win rate (profitable trades)
        profitable_trades = (strategy_returns > 0).sum()
        win_rate = (profitable_trades / len(strategy_returns)) * 100

        return {
            "directional_accuracy": float(directional_accuracy),
            "up_predictions_accuracy": float(up_accuracy),
            "down_predictions_accuracy": float(down_accuracy),
            "high_confidence_accuracy": float(high_conf_accuracy),
            "high_confidence_threshold": float(high_conf_threshold),
            "avg_confidence": float(confidences.mean()),
            "num_up_predictions": int(up_mask.sum()),
            "num_down_predictions": int(down_mask.sum()),
            "num_high_confidence": int(high_conf_mask.sum()),
            "avg_return_per_trade_pct": float(avg_return_per_trade),
            "win_rate": float(win_rate),
            "total_trades": len(predictions),
        }

    def _print_results(self, symbol, interval, horizon_minutes, results, metrics):
        """Pretty print backtest results with verification of calculation accuracy"""
        logger.info(f"\n{'='*70}")
        logger.info(
            f"BACKTEST RESULTS: {symbol} ({interval} interval, {horizon_minutes}min predictions)"
        )
        logger.info(f"{'='*70}")
        logger.info(f"Total Predictions: {metrics['total_trades']}")
        logger.info(f"  UP predictions:   {metrics['num_up_predictions']}")
        logger.info(f"  DOWN predictions: {metrics['num_down_predictions']}")
        logger.info(f"\nDIRECTIONAL ACCURACY:")
        logger.info(f"  Overall:          {metrics['directional_accuracy']:.2f}%")
        logger.info(f"  UP predictions:   {metrics['up_predictions_accuracy']:.2f}%")
        logger.info(f"  DOWN predictions: {metrics['down_predictions_accuracy']:.2f}%")

        # Manual verification of overall accuracy
        predictions = np.array(results["predictions"])
        actuals = np.array(results["actuals"])
        correct_count = (predictions == actuals).sum()
        manual_accuracy = (correct_count / len(predictions)) * 100
        logger.info(
            f"  Verified:         {manual_accuracy:.2f}% ({correct_count}/{len(predictions)} correct)"
        )

        logger.info(f"\nCONFIDENCE ANALYSIS:")
        logger.info(f"  Average confidence: {metrics['avg_confidence']:.2%}")
        logger.info(
            f"  High confidence (top 25%) accuracy: {metrics['high_confidence_accuracy']:.2f}%"
        )
        logger.info(
            f"  High confidence threshold: {metrics['high_confidence_threshold']:.2%}"
        )
        logger.info(f"\nTRADING PERFORMANCE (hypothetical):")
        logger.info(f"  Win rate:                {metrics['win_rate']:.2f}%")
        logger.info(
            f"  Avg return per trade:    {metrics['avg_return_per_trade_pct']:.3f}%"
        )
        logger.info(f"{'='*70}\n")

    def _save_results(self, symbol, results, horizon):
        """Save validation results to file"""
        filename = f"{symbol}_backtest_{horizon}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

    def backtest_multiple_symbols(
        self,
        symbols=None,
        interval="1h",
        horizon_minutes=60,
        days=30,
        min_confidence=0.7,
        use_smc=False,
        output_file=None,
    ):
        """
        Test multiple symbols with same configuration and save results to file

        Args:
            symbols: List of trading pairs
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Prediction horizon in minutes
            days: Days to test
            min_confidence: Minimum confidence threshold
            use_smc: Whether to use SMC models
            output_file: Path to save results (auto-generated if None)

        Returns:
            dict: Results for each symbol
        """
        # Default to config-defined symbols when none provided
        symbols = symbols or list(SUPPORTED_CRYPTOS.values())

        logger.info(
            f"Testing {len(symbols)} symbols with {interval} interval, {horizon_minutes}min horizon"
        )

        all_results = {}
        summary_stats = []

        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {symbol}")
            logger.info(f"{'='*60}")

            try:
                results = self.backtest(
                    symbol=symbol,
                    interval=interval,
                    horizon_minutes=horizon_minutes,
                    days=days,
                    min_confidence=min_confidence,
                    use_smc=use_smc,
                )
                all_results[symbol] = results

                # Collect summary for comparison
                summary_stats.append(
                    {
                        "symbol": symbol,
                        "directional_accuracy": results["metrics"][
                            "directional_accuracy"
                        ],
                        "win_rate": results["metrics"]["win_rate"],
                        "avg_return_per_trade_pct": results["metrics"][
                            "avg_return_per_trade_pct"
                        ],
                        "total_trades": results["metrics"]["total_trades"],
                        "avg_confidence": results["metrics"]["avg_confidence"],
                    }
                )

            except Exception as e:
                logger.error(f"Error testing {symbol}: {e}")
                import traceback

                logger.debug(traceback.format_exc())
                continue

        # Compare results
        self._compare_results(summary_stats, interval, horizon_minutes)

        # Save to file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                self.results_dir
                / f"multi_backtest_{interval}_{horizon_minutes}min_{timestamp}.json"
            )
        else:
            output_file = Path(output_file)

        # Prepare data for JSON (remove dataframe objects)
        results_to_save = {}
        for symbol, result in all_results.items():
            result_copy = result.copy()
            if "dataframe" in result_copy:
                del result_copy["dataframe"]
            results_to_save[symbol] = result_copy

        # Add summary metadata
        save_data = {
            "test_configuration": {
                "interval": interval,
                "horizon_minutes": horizon_minutes,
                "days": days,
                "min_confidence": min_confidence,
                "use_smc": use_smc,
                "tested_symbols": symbols,
                "timestamp": datetime.now().isoformat(),
            },
            "summary_statistics": summary_stats,
            "detailed_results": results_to_save,
        }

        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"\n✅ Results saved to {output_file}")

        return all_results

    def _compare_results(self, summary_stats, interval, horizon_minutes):
        """Compare results across symbols with detailed statistics"""
        if not summary_stats:
            logger.warning("No results to compare")
            return

        logger.info(f"\n{'='*80}")
        logger.info(
            f"COMPARISON: {interval} interval, {horizon_minutes}min predictions"
        )
        logger.info(f"{'='*80}")
        logger.info(
            f"{'Symbol':<10} {'Trades':<8} {'Dir.Acc':<10} {'WinRate':<10} {'AvgRet%':<10} {'AvgConf':<10}"
        )
        logger.info("-" * 80)

        for stats in summary_stats:
            logger.info(
                f"{stats['symbol']:<10} "
                f"{stats['total_trades']:<8} "
                f"{stats['directional_accuracy']:>8.2f}%  "
                f"{stats['win_rate']:>8.2f}%  "
                f"{stats['avg_return_per_trade_pct']:>8.3f}%  "
                f"{stats['avg_confidence']:>8.2%}"
            )

        # Calculate aggregates
        avg_dir_acc = sum(s["directional_accuracy"] for s in summary_stats) / len(
            summary_stats
        )
        avg_win_rate = sum(s["win_rate"] for s in summary_stats) / len(summary_stats)
        total_trades = sum(s["total_trades"] for s in summary_stats)

        logger.info("-" * 80)
        logger.info(f"AVERAGES:")
        logger.info(f"  Directional Accuracy: {avg_dir_acc:.2f}%")
        logger.info(f"  Win Rate: {avg_win_rate:.2f}%")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from binance.client import Client

    load_dotenv()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    if api_key:
        binance_client = Client(api_key, api_secret)
        logger.info("✅ Using authenticated Binance client")
    else:
        binance_client = Client()
        logger.info("⚠️ Using public Binance client (rate limits apply)")

    # Use config-defined symbols for consistency across the codebase
    SYMBOLS = list(SUPPORTED_CRYPTOS.values())

    # Test configuration - adjust these parameters as needed
    INTERVAL = "15m"  # Must match your trained models
    HORIZON_MINUTES = 15  # Must match your trained models
    DAYS = 5  # Historical period to test
    MIN_CONFIDENCE = 0.75  # Only count predictions above this threshold
    USE_SMC = True  # Try SMC models first, fallback to simple models

    logger.info("=" * 80)
    logger.info("MULTI-SYMBOL BACKTEST VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Testing {len(SYMBOLS)} cryptocurrencies")
    logger.info(f"Interval: {INTERVAL}")
    logger.info(f"Horizon: {HORIZON_MINUTES} minutes")
    logger.info(f"Period: {DAYS} days")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE:.0%}")
    logger.info(f"Model Type: {'SMC (with fallback)' if USE_SMC else 'Simple'}")
    logger.info("=" * 80)

    validator = ModelValidator(binance_client)

    results = validator.backtest_multiple_symbols(
        symbols=SYMBOLS,
        interval=INTERVAL,
        horizon_minutes=HORIZON_MINUTES,
        days=DAYS,
        min_confidence=MIN_CONFIDENCE,
        use_smc=USE_SMC,
    )

    logger.info("\n✅ Backtest completed for all symbols!")
    logger.info(f"Results saved to validation_results/ directory")
