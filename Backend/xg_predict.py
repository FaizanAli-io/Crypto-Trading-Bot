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

# Initialize
smc = SMCFeatureEngineer(
    swing_lookback=5,      # Lookback for swing points
    fvg_threshold=0.001    # Minimum gap size (0.1%)
)


class SignalPredictor:
    """Generate live trading signals from trained XGBoost models"""
    
    def __init__(self, binance_client=None):
        self.data_collector = DataCollector(binance_client)
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_dir = Path("models")
        self.signals_dir = Path("signals")
        self.signals_dir.mkdir(exist_ok=True)
        
        # Confidence thresholds by horizon
        self.confidence_thresholds = {
            1: 0.70,   # 1h needs 70% confidence
            3: 0.60,   # 3h needs 60% confidence
            6: 0.60,   # 6h needs 60% confidence
            12: 0.55,  # 12h needs 55% confidence
            24: 0.55   # 24h needs 55% confidence
        }
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
        interval_minutes = SignalPredictor.interval_to_minutes(interval)
        shift = horizon_minutes / interval_minutes
        
        if shift != int(shift):
            logger.warning(
                f"Horizon {horizon_minutes}min is not evenly divisible by interval {interval} "
                f"({interval_minutes}min). Using shift={int(shift)} candles."
            )
        
        return int(shift)
    def load_model(self, symbol="BTCUSDT", interval="1h", horizon_minutes=60, use_smc=None):
        """
        Load trained XGBoost model for specific symbol, interval, and horizon
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Prediction horizon in minutes (e.g., 60, 360)
            use_smc: Whether to load SMC model (None=auto-detect, True=SMC only, False=non-SMC only)
        
        Returns:
            bool: Success status
            
        Example:
            # Load model trained on 15-min candles, predicting 60 minutes ahead
            trainer.load_model(symbol="ETHUSDT", interval="15m", horizon_minutes=60)
            
            # Load SMC model specifically
            trainer.load_model(symbol="ETHUSDT", interval="15m", horizon_minutes=60, use_smc=True)
        """
        try:
            # Find latest model file for this symbol, interval, and horizon
            # Pattern: xgb_model_{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl
            # OR: xgb_model_smc_{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl
            
            if use_smc is True:
                # Only look for SMC models
                pattern = f"xgb_model_smc_{symbol}_{interval}_{horizon_minutes}min_*.pkl"
            elif use_smc is False:
                # Only look for non-SMC models
                pattern = f"xgb_model_{symbol}_{interval}_{horizon_minutes}min_*.pkl"
            else:
                # Auto-detect: prefer SMC if available, otherwise use non-SMC
                smc_pattern = f"xgb_model_smc_{symbol}_{interval}_{horizon_minutes}min_*.pkl"
                smc_files = list(self.model_dir.glob(smc_pattern))
                if smc_files:
                    pattern = smc_pattern
                    logger.info("SMC model detected - will use SMC features")
                else:
                    pattern = f"xgb_model_{symbol}_{interval}_{horizon_minutes}min_*.pkl"
                    logger.info("Non-SMC model detected - will use standard features")
            
            model_files = list(self.model_dir.glob(pattern))
            
            if not model_files:
                logger.error(f"No model found for {symbol} with interval={interval}, horizon={horizon_minutes}min")
                logger.info(f"Looking for pattern: {pattern}")
                logger.info(f"Available models in {self.model_dir}:")
                
                # Show available models to help user
                all_models = list(self.model_dir.glob("xgb_model_*.pkl"))
                if all_models:
                    for model in sorted(all_models)[-5:]:  # Show last 5
                        logger.info(f"  - {model.name}")
                else:
                    logger.info("  (no models found)")
                
                return False
            
            # Get most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading model: {latest_model.name}")
            
            # Detect if this is an SMC model from filename
            is_smc_model = "_smc_" in latest_model.name or latest_model.name.startswith("xgb_model_smc_")
            self.use_smc = is_smc_model
            
            self.model = joblib.load(latest_model)
            
            # Extract timestamp from filename
            # Format: xgb_model_ETHUSDT_15m_60min_20250108_143022.pkl
            # OR: xgb_model_smc_ETHUSDT_15m_60min_20250108_143022.pkl
            parts = latest_model.stem.split('_')
            # parts = ['xgb', 'model', 'ETHUSDT', '15m', '60min', '20250108', '143022']
            # OR: ['xgb', 'model', 'smc', 'ETHUSDT', '15m', '60min', '20250108', '143022']
            timestamp = f"{parts[-2]}_{parts[-1]}"  # "20250108_143022"
            
            # Build feature file name with SMC prefix if needed
            smc_prefix = "smc_" if is_smc_model else ""
            features_file = self.model_dir / f"features_{smc_prefix}{symbol}_{interval}_{horizon_minutes}min_{timestamp}.txt"
            
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                logger.warning(f"Feature names file not found: {features_file}")
                return False
            
            # Try to load scaler
            scaler_file = self.model_dir / f"scaler_{smc_prefix}{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info("Loaded scaler")
            else:
                logger.warning("No scaler found - predictions may be inaccurate!")
                self.scaler = None
            
            # Store loaded configuration
            self.interval = interval
            self.horizon_minutes = horizon_minutes
            self.shift_candles = self.calculate_horizon_shift(horizon_minutes, interval)
            
            logger.info("✅ Model loaded successfully!")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Interval: {interval}")
            logger.info(f"   Horizon: {horizon_minutes} minutes ({self.shift_candles} candles)")
            logger.info(f"   Features: {len(self.feature_names)}")
            logger.info(f"   SMC Model: {'YES' if is_smc_model else 'NO'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
    # FOR PredictSignal CLASS
    # ================================================================

    def predict_signal(self, symbol, interval, horizon_minutes, custom_confidence=None, 
                    estimate_price=False, days=10):
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Minutes ahead to predict (e.g., 60, 360)
            custom_confidence: Override default confidence threshold
            estimate_price: Whether to estimate target price
            lookback_limit: Number of candles for history (default: 200)
        
        Returns:
            dict: Signal with prediction details
        """
        logger.info(f"Generating signal for {symbol} - {interval} interval, {horizon_minutes}min horizon")
        
        # Load model
        if not self.load_model(symbol, interval, horizon_minutes):
            raise ValueError(f"Could not load model for {symbol} with interval={interval}, horizon={horizon_minutes}min")
        
        # Get confidence threshold (use horizon_minutes for lookup or default)
        min_confidence = custom_confidence if custom_confidence else self.confidence_thresholds.get(horizon_minutes, 0.60)
        logger.info(f"Using confidence threshold: {min_confidence:.0%}")
        
        # Fetch recent data with correct interval
        df = self.data_collector.get_realtime_data(
            symbol=symbol,
            days=days,
            interval=interval,  # Use the model's interval
            include_ongoing=False
        )
        
        if df is None or len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Add SMC features if model was trained with them
        if self.use_smc:
            logger.info("Adding SMC features for prediction...")
            integrate_smc_into_feature_engineer(self.feature_engineer)

        # Add technical features (and SMC if enabled)
        df = self.feature_engineer.add_all_features(df)
        
        
        # Get current price and latest features
        current_price = float(df.iloc[-1]['close'])
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
            'symbol': symbol,
            'interval': interval,
            'timestamp': current_time.isoformat(),
            'prediction_time': datetime.now().isoformat(),
            'horizon_minutes': horizon_minutes,
            'horizon_candles': self.shift_candles,
            'current_price': current_price,
            'signal': signal,
            'predicted_direction': 'UP' if predicted_direction == 1 else 'DOWN',
            'confidence': confidence,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'min_confidence_threshold': min_confidence,
            'reason': reason,
            'price_estimate': price_estimate,
            'valid_until': (datetime.now() + timedelta(minutes=horizon_minutes)).isoformat()
        }
        
        # Save signal
        # self._save_signal(result)
        self._print_signal(result)
        
        return result


    def _estimate_target_price(self, df, current_price, direction, confidence, shift_candles):
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
            current = df_recent.iloc[i]['close']
            future = df_recent.iloc[i + shift_candles]['close']
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
                percentile = 50 + (confidence - 0.5) * 60  # Range: 50th to 80th percentile
                expected_change = np.percentile(up_moves, percentile)
            else:
                expected_change = mean_change
        else:  # DOWN prediction
            # Use negative changes only
            down_moves = filtered_changes[filtered_changes < 0]
            if len(down_moves) > 0:
                percentile = 50 - (confidence - 0.5) * 60  # Range: 50th to 20th percentile
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
            'method': 'historical_average',
            'target_price': round(target_price, 2),
            'expected_change_pct': round(expected_change * 100, 2),
            'price_range': {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2)
            },
            'warning': 'This is a statistical estimate based on past patterns. NOT a guarantee.'
        }

    def _save_signal(self, signal):
        """Save signal to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"signal_{signal['symbol']}_{signal['interval']}_{signal['horizon_minutes']}m_{timestamp}.json"
        filepath = self.signals_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(signal, f, indent=2)
        
        logger.info(f"Signal saved to {filepath}")
        
        # Also update "latest" file for easy access
        latest_file = self.signals_dir / f"latest_{signal['symbol']}_{signal['interval']}_{signal['horizon_minutes']}m.json"
        with open(latest_file, 'w') as f:
            json.dump(signal, f, indent=2)
    
    def _print_signal(self, signal):
        """Pretty print signal"""
        logger.info("\n" + "="*70)
        logger.info(f"TRADING SIGNAL: {signal['symbol']}")
        logger.info("="*70)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Interval: {signal['interval']}")
        logger.info(f"Horizon: {signal['horizon_minutes']} minutes ({signal['horizon_candles']} candles)")
        logger.info(f"Current Price: ${signal['current_price']:,.2f}")
        logger.info(f"\nPREDICTION:")
        logger.info(f"  Direction: {signal['predicted_direction']}")
        logger.info(f"  Confidence: {signal['confidence']:.1%}")
        logger.info(f"  Prob UP: {signal['prob_up']:.1%}")
        logger.info(f"  Prob DOWN: {signal['prob_down']:.1%}")
        logger.info(f"\nSIGNAL: {signal['signal']}")
        logger.info(f"  Reason: {signal['reason']}")
        
        if signal.get('price_estimate'):
            est = signal['price_estimate']
            logger.info(f"\nPRICE ESTIMATE (Rough):")
            logger.info(f"  Target: ${est['target_price']:,.2f} ({est['expected_change_pct']:+.2f}%)")
            logger.info(f"  Range: ${est['price_range']['lower']:,.2f} - ${est['price_range']['upper']:,.2f}")
            logger.info(f"  ⚠️  {est['warning']}")
        
        logger.info(f"\nValid until: {signal['valid_until']}")
        logger.info("="*70 + "\n")
    
    def get_latest_signal(self, symbol, interval, horizon_minutes):
        """Retrieve latest saved signal"""
        latest_file = self.signals_dir / f"latest_{symbol}_{interval}_{horizon_minutes}m.json"
        
        if not latest_file.exists():
            return None
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def get_signal_history(self, symbol, interval, horizon_minutes, days=7):
        """Get all signals from past N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        pattern = f"signal_{symbol}_{interval}_{horizon_minutes}m_*.json"
        signal_files = list(self.signals_dir.glob(pattern))
        
        signals = []
        for file in signal_files:
            try:
                timestamp_str = file.stem.split('_')[-2] + file.stem.split('_')[-1]
                file_date = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                
                if file_date >= cutoff_date:
                    with open(file, 'r') as f:
                        signals.append(json.load(f))
            except:
                continue
        
        # Sort by timestamp
        signals.sort(key=lambda x: x['prediction_time'], reverse=True)
        
        return signals
    
    def get_all_available_models(self):
        """
        Parse all model files and extract symbol, interval, and horizon_minutes
        
        Returns:
            list: List of dicts with model info [{'symbol': 'BTCUSDT', 'interval': '15m', 'horizon_minutes': 15}, ...]
        """
        model_files = list(self.model_dir.glob("xgb_model_*.pkl"))
        
        models_info = []
        for model_file in model_files:
            try:
                # Parse filename: xgb_model_{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl
                parts = model_file.stem.split('_')
                
                if len(parts) < 7:
                    logger.warning(f"Skipping unexpected filename format: {model_file.name}")
                    continue
                
                symbol = parts[2]  # "BTCUSDT"
                interval = parts[3]  # "15m"
                horizon_str = parts[4]  # "15min"
                horizon_minutes = int(horizon_str.replace('min', ''))
                
                models_info.append({
                    'symbol': symbol,
                    'interval': interval,
                    'horizon_minutes': horizon_minutes,
                    'model_file': model_file.name
                })
                
            except Exception as e:
                logger.warning(f"Error parsing model file {model_file.name}: {e}")
                continue
        
        logger.info(f"Found {len(models_info)} models")
        return models_info
    
    def predict_all_models(self, estimate_price=True, days=10):
        """
        Run predictions on all available models and save to a single JSON file
        
        Args:
            estimate_price: Whether to estimate target price
            days: Number of days of historical data to use
        
        Returns:
            dict: All predictions with metadata
        """
        logger.info("Starting batch prediction for all available models...")
        
        # Get all models
        models = self.get_all_available_models()
        
        if not models:
            logger.error("No models found!")
            return None
        
        # Run predictions
        all_predictions = []
        failed_predictions = []
        
        for i, model_info in enumerate(models, 1):
            symbol = model_info['symbol']
            interval = model_info['interval']
            horizon_minutes = model_info['horizon_minutes']
            
            logger.info(f"\n[{i}/{len(models)}] Processing {symbol} - {interval} - {horizon_minutes}min")
            
            try:
                signal = self.predict_signal(
                    symbol=symbol,
                    interval=interval,
                    horizon_minutes=horizon_minutes,
                    estimate_price=estimate_price,
                    days=days
                )
                all_predictions.append(signal)
                
            except Exception as e:
                logger.error(f"Failed to predict {symbol} - {interval} - {horizon_minutes}min: {e}")
                failed_predictions.append({
                    'symbol': symbol,
                    'interval': interval,
                    'horizon_minutes': horizon_minutes,
                    'error': str(e)
                })
        
        # Build result
        result = {
            'batch_timestamp': datetime.now().isoformat(),
            'total_models': len(models),
            'successful_predictions': len(all_predictions),
            'failed_predictions': len(failed_predictions),
            'predictions': all_predictions,
            'failures': failed_predictions
        }
        
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
        
        with open(filepath, 'w') as f:
            json.dump(batch_result, f, indent=2)
        
        logger.info(f"Batch predictions saved to {filepath}")
        
        # Also save as "latest_batch"
        latest_file = self.signals_dir / "latest_batch_predictions.json"
        with open(latest_file, 'w') as f:
            json.dump(batch_result, f, indent=2)


# ========================
# CLI Interface
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
    
    predictor = SignalPredictor(binance_client)
    
    # predictor.predict_signal(symbol="HBARUSDT", interval="1h", horizon_minutes=60, 
    #                 estimate_price=True, days=10)

    predictor.predict_all_models()

    # symbols = ["BTCUSDT",
    # "ETHUSDT",
    # "BNBUSDT",
    # "XRPUSDT",
    #  "ADAUSDT",
    # "DOGEUSDT",
    # "SOLUSDT",
    # "DOTUSDT",
    # "LINKUSDT",
    # "LTCUSDT" ]

    # # for symbol in symbols:
        
    # #     singal = predictor.predict_signal(symbol=symbol, interval="15m", 
    # #     horizon_minutes=15, estimate_price=True, days=10)

    # symbol = "ETHUSDT"    
    # singal = predictor.predict_signal(symbol=symbol, interval="15m", 
    #     horizon_minutes=15, estimate_price=True, days=10)