"""
Model Validator: Test XGBoost prediction accuracy for directional forecasting
Modified to use trained XGBoost models with feature engineering
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path
import joblib

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer

class ModelValidator:
    """Validate and track XGBoost model directional prediction accuracy"""
    
    def __init__(self, binance_client=None):
        self.data_collector = DataCollector(binance_client)
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)
        self.model_dir = Path("models")
     
    @staticmethod
    def interval_to_minutes(interval: str) -> int:
        """Convert interval string to minutes"""
        interval_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
        }
        return interval_map.get(interval, 60)
    
    @staticmethod
    def calculate_horizon_shift(horizon_minutes: int, interval: str) -> int:
        """
        Calculate number of candles to shift based on prediction horizon
        
        Args:
            horizon_minutes: How many minutes ahead to predict
            interval: Candle interval (e.g., "15m", "1h")
        
        Returns:
            Number of candles to shift
            
        Example:
            interval="15m", horizon_minutes=60 -> shift=4 candles
            interval="1h", horizon_minutes=60 -> shift=1 candle
            interval="15m", horizon_minutes=30 -> shift=2 candles
        """
        interval_minutes = ModelValidator.interval_to_minutes(interval)
        shift = horizon_minutes / interval_minutes
        
        if shift != int(shift):
            logger.warning(
                f"Horizon {horizon_minutes}min is not evenly divisible by interval {interval} "
                f"({interval_minutes}min). Using shift={int(shift)} candles."
            )
        
        return int(shift)
    def load_model(self, symbol="BTCUSDT", interval="1h", horizon_minutes=60, use_smc=False):
        """
        Load trained XGBoost model for specific symbol, interval, and horizon
        Automatically selects SMC model if use_smc=True and available, otherwise falls back to simple model
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Prediction horizon in minutes (e.g., 60, 360)
            use_smc: If True, try to load SMC model first, fallback to simple model
        
        Returns:
            bool: Success status
            
        Example:
            # Load SMC model if available, otherwise simple model
            validator.load_model(symbol="ETHUSDT", interval="15m", horizon_minutes=60, use_smc=True)
        """
        try:
            # Determine which model to try first
            if use_smc:
                # Try SMC model first
                smc_pattern = f"xgb_model_{symbol}_{interval}_{horizon_minutes}min_smc_*.pkl"
                smc_files = list(self.model_dir.glob(smc_pattern))
                
                if smc_files:
                    logger.info(f"🎯 SMC model found for {symbol} ({interval}, {horizon_minutes}min)")
                    return self._load_model_files(symbol, interval, horizon_minutes, model_type="smc")
                else:
                    logger.warning(f"⚠️ SMC model not found for {symbol} ({interval}, {horizon_minutes}min)")
                    logger.info(f"📦 Falling back to simple model...")
            
            # Try simple model (default or fallback)
            simple_pattern = f"xgb_model_{symbol}_{interval}_{horizon_minutes}min_*.pkl"
            simple_files = list(self.model_dir.glob(simple_pattern))
            
            # Exclude SMC models from simple search
            simple_files = [f for f in simple_files if "_smc_" not in f.name]
            
            if simple_files:
                logger.info(f"📦 Simple model found for {symbol} ({interval}, {horizon_minutes}min)")
                return self._load_model_files(symbol, interval, horizon_minutes, model_type="simple")
            
            # No models found
            logger.error(f"❌ No model found for {symbol} with interval={interval}, horizon={horizon_minutes}min")
            logger.info(f"Looking for patterns:")
            logger.info(f"  - SMC: xgb_model_{symbol}_{interval}_{horizon_minutes}min_smc_*.pkl")
            logger.info(f"  - Simple: xgb_model_{symbol}_{interval}_{horizon_minutes}min_*.pkl")
            logger.info(f"\nAvailable models in {self.model_dir}:")
            
            # Show available models to help user
            all_models = list(self.model_dir.glob("xgb_model_*.pkl"))
            if all_models:
                for model in sorted(all_models)[-10:]:  # Show last 10
                    logger.info(f"  - {model.name}")
            else:
                logger.info("  (no models found)")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _load_model_files(self, symbol, interval, horizon_minutes, model_type="simple"):
        """
        Internal method to load model files (model, features, scaler)
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            horizon_minutes: Prediction horizon
            model_type: "smc" or "simple"
        
        Returns:
            bool: Success status
        """
        try:
            # Build pattern based on model type
            if model_type == "smc":
                pattern = f"xgb_model_{symbol}_{interval}_{horizon_minutes}min_smc_*.pkl"
            else:
                pattern = f"xgb_model_{symbol}_{interval}_{horizon_minutes}min_*.pkl"
            
            model_files = list(self.model_dir.glob(pattern))
            
            # Exclude SMC models if looking for simple
            if model_type == "simple":
                model_files = [f for f in model_files if "_smc_" not in f.name]
            
            if not model_files:
                return False
            
            # Get most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"📂 Loading model: {latest_model.name}")
            
            self.model = joblib.load(latest_model)
            
            # Extract timestamp from filename
            # Format: xgb_model_ETHUSDT_15m_60min_smc_20250108_143022.pkl
            # or:     xgb_model_ETHUSDT_15m_60min_20250108_143022.pkl
            parts = latest_model.stem.split('_')
            
            if model_type == "smc":
                # parts = ['xgb', 'model', 'ETHUSDT', '15m', '60min', 'smc', '20250108', '143022']
                timestamp = f"{parts[-2]}_{parts[-1]}"
                model_suffix = "smc"
            else:
                # parts = ['xgb', 'model', 'ETHUSDT', '15m', '60min', '20250108', '143022']
                timestamp = f"{parts[-2]}_{parts[-1]}"
                model_suffix = ""
            
            # Load corresponding feature names
            if model_type == "smc":
                features_file = self.model_dir / f"features_{symbol}_{interval}_{horizon_minutes}min_smc_{timestamp}.txt"
            else:
                features_file = self.model_dir / f"features_{symbol}_{interval}_{horizon_minutes}min_{timestamp}.txt"
            
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                logger.info(f"✅ Loaded {len(self.feature_names)} feature names")
            else:
                logger.error(f"❌ Feature names file not found: {features_file.name}")
                return False
            
            # Try to load scaler
            if model_type == "smc":
                scaler_file = self.model_dir / f"scaler_{symbol}_{interval}_{horizon_minutes}min_smc_{timestamp}.pkl"
            else:
                scaler_file = self.model_dir / f"scaler_{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl"
            
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info("✅ Loaded scaler")
            else:
                logger.warning("⚠️ No scaler found - predictions may be inaccurate!")
                self.scaler = None
            
            # Store loaded configuration
            self.interval = interval
            self.horizon_minutes = horizon_minutes
            self.shift_candles = self.calculate_horizon_shift(horizon_minutes, interval)
            self.model_type = model_type
            
            logger.info("="*60)
            logger.info(f"✅ {model_type.upper()} MODEL LOADED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"   Symbol:   {symbol}")
            logger.info(f"   Type:     {model_type.upper()}")
            logger.info(f"   Interval: {interval}")
            logger.info(f"   Horizon:  {horizon_minutes} minutes ({self.shift_candles} candles)")
            logger.info(f"   Features: {len(self.feature_names)}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading {model_type} model files: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_latest_model_for_symbol(self, symbol="BTCUSDT"):
        """
        Load the most recent model for a symbol (regardless of interval/horizon)
        
        Args:
            symbol: Trading pair
            
        Returns:
            bool: Success status
        """
        try:
            pattern = f"xgb_model_{symbol}_*.pkl"
            model_files = list(self.model_dir.glob(pattern))
            
            if not model_files:
                logger.error(f"No models found for {symbol}")
                return False
            
            # Get most recent
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            
            # Parse filename to extract interval and horizon
            # Format: xgb_model_ETHUSDT_15m_60min_20250108_143022.pkl
            parts = latest_model.stem.split('_')
            
            if len(parts) < 7:
                logger.error(f"Unexpected filename format: {latest_model.name}")
                return False
            
            interval = parts[3]  # "15m"
            horizon_str = parts[4]  # "60min"
            horizon_minutes = int(horizon_str.replace('min', ''))
            
            logger.info(f"Auto-detected: interval={interval}, horizon={horizon_minutes}min")
            
            return self.load_model(symbol=symbol, interval=interval, horizon_minutes=horizon_minutes)
            
        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            return False
    
    # ================================================================
    # FOR ModelValidator CLASS
    # ================================================================

    def backtest(self, symbol="BTCUSDT", interval="1h", horizon_minutes=60, 
                days=10, min_confidence=0.7, use_smc=False):
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
        logger.info(f"Backtesting {symbol} - {interval} interval, {horizon_minutes}min predictions over {days} days...")
        
        # Load model for this symbol, interval, and horizon
        if not self.load_model(symbol, interval, horizon_minutes, use_smc=use_smc):
            raise ValueError(f"Could not load model for {symbol} with interval={interval}, horizon={horizon_minutes}min")
        
        # Get shift_candles from loaded model
        shift_candles = self.shift_candles
        logger.info(f"Using shift_candles={shift_candles} for {horizon_minutes}min prediction")
        
        # Fetch historical data (need extra for indicators)
        total_days = days  # Extra for technical indicators
        df = self.data_collector.get_realtime_data(
            symbol=symbol, 
            days=total_days,
            interval=interval  # Use correct interval
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
                current_price = float(df.iloc[i]['close'])
                
                # Get actual future price (shift_candles later)
                future_price = float(df.iloc[i + shift_candles]['close'])
                
                # Actual direction: 1 if price went up, 0 if down
                actual_direction = 1 if future_price > current_price else 0
                
                # Extract features for this point in time
                feature_data = df.iloc[i:i+1][self.feature_names]
                
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
            predictions, actuals, confidences, 
            current_prices, future_prices
        )
        
        # Save results
        results = {
            'symbol': symbol,
            'interval': interval,
            'prediction_horizon_minutes': horizon_minutes,
            'prediction_horizon_candles': shift_candles,
            'test_period_days': days,
            'min_confidence': min_confidence,
            'num_predictions': len(predictions),
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'confidences': confidences,
            'current_prices': current_prices,
            'future_prices': future_prices,
            'timestamps': [str(t) for t in timestamps]
        }
        
        # self._save_results(symbol, results, interval, horizon_minutes)
        self._print_results(symbol, interval, horizon_minutes, results, metrics)
        
        return results


    def backtest_for_simulation(self, symbol="BTCUSDT", interval="1h", 
                            horizon_minutes=60, days=30, min_confidence=0.7, use_smc=False):
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
        logger.info(f"Backtesting {symbol} - {interval} interval, {horizon_minutes}min predictions over {days} days...")
        
        # Load model for this symbol, interval, and horizon
        if not self.load_model(symbol, interval, horizon_minutes, use_smc=use_smc):
            raise ValueError(f"Could not load model for {symbol} with interval={interval}, horizon={horizon_minutes}min")
        
        # Get shift_candles from loaded model
        shift_candles = self.shift_candles
        logger.info(f"Using shift_candles={shift_candles} for {horizon_minutes}min prediction")
        
        # Fetch historical data (need extra for indicators)
        total_days = days + 9  # Extra for technical indicators
        df = self.data_collector.get_realtime_data(
            symbol=symbol, 
            days=total_days,
            interval=interval  # Use correct interval
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
                current_price = float(df.iloc[i]['close'])
                
                # Get actual future price (shift_candles later)
                future_price = float(df.iloc[i + shift_candles]['close'])
                
                # Actual direction: 1 if price went up, 0 if down
                actual_direction = 1 if future_price > current_price else 0
                
                # Extract features for this point in time
                feature_data = df.iloc[i:i+1][self.feature_names]
                
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
            predictions, actuals, confidences, 
            current_prices, future_prices
        )
        
        # Build results dictionary
        results = {
            'symbol': symbol,
            'interval': interval,
            'prediction_horizon_minutes': horizon_minutes,
            'prediction_horizon_candles': shift_candles,
            'test_period_days': days,
            'min_confidence': min_confidence,
            'num_predictions': len(predictions),
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'confidences': confidences,
            'current_prices': current_prices,
            'future_prices': future_prices,
            'timestamps': [str(t) for t in timestamps],
            'dataframe': df_full  # Include full dataframe for price estimation
        }
        
        # Save results (without dataframe)
        # self._save_results(symbol, results, interval, horizon_minutes)
        self._print_results(symbol, interval, horizon_minutes, results, metrics)
        
        return results



    def _calculate_directional_metrics(self, predictions, actuals, confidences, 
                                      current_prices, future_prices):
        """Calculate directional prediction metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)
        current_prices = np.array(current_prices)
        future_prices = np.array(future_prices)
        
        # Overall directional accuracy
        correct = predictions == actuals
        directional_accuracy = correct.mean() * 100
        
        # Separate accuracy for UP and DOWN predictions
        up_mask = predictions == 1
        down_mask = predictions == 0
        
        up_accuracy = correct[up_mask].mean() * 100 if up_mask.sum() > 0 else 0
        down_accuracy = correct[down_mask].mean() * 100 if down_mask.sum() > 0 else 0
        
        # High confidence predictions (top 25%)
        high_conf_threshold = np.percentile(confidences, 75)
        high_conf_mask = confidences >= high_conf_threshold
        
        high_conf_accuracy = correct[high_conf_mask].mean() * 100 if high_conf_mask.sum() > 0 else 0
        
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
                strategy_returns.append(-actual_returns[i])  # Inverse return (as if shorting)
        
        strategy_returns = np.array(strategy_returns)
        avg_return_per_trade = strategy_returns.mean() * 100
        
        # Win rate (profitable trades)
        profitable_trades = (strategy_returns > 0).sum()
        win_rate = (profitable_trades / len(strategy_returns)) * 100
        
        return {
            'directional_accuracy': float(directional_accuracy),
            'up_predictions_accuracy': float(up_accuracy),
            'down_predictions_accuracy': float(down_accuracy),
            'high_confidence_accuracy': float(high_conf_accuracy),
            'high_confidence_threshold': float(high_conf_threshold),
            'avg_confidence': float(confidences.mean()),
            'num_up_predictions': int(up_mask.sum()),
            'num_down_predictions': int(down_mask.sum()),
            'num_high_confidence': int(high_conf_mask.sum()),
            'avg_return_per_trade_pct': float(avg_return_per_trade),
            'win_rate': float(win_rate),
            'total_trades': len(predictions)
        }
    
    def _print_results(self,symbol, interval, horizon, results, metrics):
        """Pretty print backtest results"""
        logger.info(f"\n{'='*70}")
        logger.info(f"BACKTEST RESULTS: {symbol} ({horizon}h predictions)")
        logger.info(f"{'='*70}")
        logger.info(f"Total Predictions: {metrics['total_trades']}")
        logger.info(f"  UP predictions:   {metrics['num_up_predictions']}")
        logger.info(f"  DOWN predictions: {metrics['num_down_predictions']}")
        logger.info(f"\nDIRECTIONAL ACCURACY:")
        logger.info(f"  Overall:          {metrics['directional_accuracy']:.2f}%")
        logger.info(f"  UP predictions:   {metrics['up_predictions_accuracy']:.2f}%")
        logger.info(f"  DOWN predictions: {metrics['down_predictions_accuracy']:.2f}%")
        logger.info(f"\nCONFIDENCE ANALYSIS:")
        logger.info(f"  Average confidence: {metrics['avg_confidence']:.2%}")
        logger.info(f"  High confidence (top 25%) accuracy: {metrics['high_confidence_accuracy']:.2f}%")
        logger.info(f"  High confidence threshold: {metrics['high_confidence_threshold']:.2%}")
        logger.info(f"\nTRADING PERFORMANCE (hypothetical):")
        logger.info(f"  Win rate:                {metrics['win_rate']:.2f}%")
        logger.info(f"  Avg return per trade:    {metrics['avg_return_per_trade_pct']:.3f}%")
        logger.info(f"{'='*70}\n")
    
    def _save_results(self, symbol, results,interval ,horizon):
        """Save validation results to file"""
        filename = f"{symbol}_backtest_{horizon}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def backtest_multiple_symbols(self, symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"], 
                                  horizon=1, days=30):
        """
        Test multiple symbols with same horizon
        
        Args:
            symbols: List of trading pairs
            horizon: Prediction horizon in hours
            days: Days to test
            
        Returns:
            dict: Results for each symbol
        """
        logger.info(f"Testing multiple symbols with {horizon}h horizon: {symbols}")
        
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {symbol}")
            logger.info(f"{'='*50}")
            
            try:
                results = self.backtest(
                    symbol=symbol,
                    horizon=horizon,
                    days=days
                )
                all_results[symbol] = results
            except Exception as e:
                logger.error(f"Error testing {symbol}: {e}")
                continue
        
        # Compare results
        self._compare_results(all_results, horizon)
        
        return all_results
    
    def _compare_results(self, all_results, horizon):
        """Compare results across symbols"""
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPARISON: {horizon}h Predictions")
        logger.info(f"{'='*70}")
        logger.info(f"{'Symbol':<12} {'Dir. Acc.':<12} {'Win Rate':<12} {'Avg Return':<12}")
        logger.info("-" * 70)
        
        for symbol, results in all_results.items():
            metrics = results['metrics']
            logger.info(f"{symbol:<12} {metrics['directional_accuracy']:>10.2f}%  "
                       f"{metrics['win_rate']:>10.2f}%  "
                       f"{metrics['avg_return_per_trade_pct']:>10.3f}%")
        
        logger.info(f"{'='*70}\n")


# ========================
# CLI for validation
# ========================
if __name__ == "__main__":
    import sys
    from binance.client import Client
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    
    if api_key:
        binance_client = Client(api_key, api_secret)
    else:
        binance_client = Client()
    
    validator = ModelValidator(binance_client)
    symbol = "XRPUSDT"
    validator.backtest(symbol=symbol, interval="15m", horizon_minutes=15, 
                days=10, min_confidence=0.85)
    # if len(sys.argv) > 1:
    #     command = sys.argv[1]
        
    #     if command == "backtest":
    #         # python model_validator.py backtest BTCUSDT 1 30 0.5
    #         symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
    #         horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    #         days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    #         min_conf = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
            
    #         validator.backtest(symbol, horizon, days, min_conf)
        
    #     elif command == "multi":
    #         # python model_validator.py multi BTCUSDT,ETHUSDT,BNBUSDT 1 30
    #         symbols_str = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT,ETHUSDT"
    #         symbols = symbols_str.split(',')
    #         horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    #         days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
            
    #         validator.backtest_multiple_symbols(symbols, horizon, days)
        
    #     else:
    #         print(f"Unknown command: {command}")
    
    # else:
    #     print("\nXGBoost Model Validator - Usage:")
    #     print("  Single symbol:")
    #     print("    python model_validator.py backtest BTCUSDT 1 30 0.5")
    #     print("      Args: symbol horizon days min_confidence")
    #     print("\n  Multiple symbols:")
    #     print("    python model_validator.py multi BTCUSDT,ETHUSDT 1 30")
    #     print("      Args: symbols(comma-separated) horizon days")
    #     print("\nNote: Horizon must match your trained model (e.g., 1h, 6h)")

