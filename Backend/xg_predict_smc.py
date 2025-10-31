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
                    estimate_price=False, days=10, use_smc=True):
        """
        Generate trading signal for a symbol with Smart Money Concepts integration
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Minutes ahead to predict (e.g., 60, 360)
            custom_confidence: Override default confidence threshold
            estimate_price: Whether to estimate target price
            days: Number of days for history (default: 10)
            use_smc: Whether to apply SMC filters (default: True)
        
        Returns:
            dict: Signal with prediction details + SMC analysis
        """
        logger.info(f"Generating signal for {symbol} - {interval} interval, {horizon_minutes}min horizon")
        
        # Load model
        if not self.load_model(symbol, interval, horizon_minutes):
            raise ValueError(f"Could not load model for {symbol} with interval={interval}, horizon={horizon_minutes}min")
        
        # Get confidence threshold
        min_confidence = custom_confidence if custom_confidence else self.confidence_thresholds.get(horizon_minutes, 0.60)
        logger.info(f"Using confidence threshold: {min_confidence:.0%}")
        
        # Fetch recent data
        df = self.data_collector.get_realtime_data(
            symbol=symbol,
            days=days,
            interval=interval,
            include_ongoing=False
        )
        
        if df is None or len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Determine if we should use SMC features
        # Priority: 1) Model was trained with SMC, 2) use_smc parameter
        should_use_smc = self.use_smc if hasattr(self, 'use_smc') else use_smc
        
        # Add SMC features if model requires them
        if should_use_smc:
            logger.info("Adding SMC features for prediction...")
            from smc_features import SMCFeatureEngineer, integrate_smc_into_feature_engineer
            integrate_smc_into_feature_engineer(self.feature_engineer)
        
        # Add technical features (and SMC if enabled)
        df = self.feature_engineer.add_all_features(df)
        
        # Get SMC indicators if enabled
        smc_indicators = None
        smc_context = None
        if should_use_smc and hasattr(self.feature_engineer, 'smc'):
            smc_indicators = self.feature_engineer.smc.get_latest_smc_indicators(df)
            smc_context = self._analyze_smc_context(smc_indicators)
            logger.info(f"SMC Analysis: {smc_context['summary']}")
        
        # Get current price and latest features
        current_price = float(df.iloc[-1]['close'])
        current_time = df.index[-1]
        
        # Extract features for model prediction
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
        
        # ===== SMC-ENHANCED SIGNAL LOGIC =====
        signal = "HOLD"
        reason = ""
        smc_adjustment = ""
        
        if use_smc and smc_context:
            # Apply SMC filters and adjustments
            signal, reason, smc_adjustment = self._apply_smc_filters(
                predicted_direction, 
                confidence, 
                min_confidence,
                smc_indicators,
                smc_context
            )
        else:
            # Original logic without SMC
            if confidence < min_confidence:
                signal = "HOLD"
                reason = f"Low confidence ({confidence:.1%} < {min_confidence:.0%})"
            elif predicted_direction == 1:
                signal = "BUY"
                reason = f"Upward prediction with {confidence:.1%} confidence"
            else:
                signal = "SELL"
                reason = f"Downward prediction with {confidence:.1%} confidence"
        
        # Price estimation with SMC levels
        price_estimate = None
        smc_targets = None
        if estimate_price and signal != "HOLD":
            price_estimate = self._estimate_target_price(
                df, current_price, predicted_direction, confidence, self.shift_candles
            )
            
            # Add SMC-based targets
            if use_smc and smc_indicators:
                smc_targets = self._get_smc_targets(current_price, smc_indicators, predicted_direction)
        
        # Build result with SMC integration
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
            'valid_until': (datetime.now() + timedelta(minutes=horizon_minutes)).isoformat(),
            
            # SMC-specific fields
            'smc_enabled': use_smc,
            'smc_adjustment': smc_adjustment if use_smc else None,
            'smc_context': smc_context if use_smc else None,
            'smc_indicators': smc_indicators if use_smc else None,
            'smc_targets': smc_targets if use_smc else None,
        }
        
        # Save and print signal
        # self._save_signal(result)
        self._print_signal_with_smc(result) if use_smc else self._print_signal(result)
        
        return result
    
    
    def _analyze_smc_context(self, smc_indicators):
        """
        Analyze SMC indicators to understand market context
        
        Returns:
            dict: SMC context analysis
        """
        context = {
            'bias': 'NEUTRAL',
            'strength': 0,
            'key_levels': [],
            'warnings': [],
            'summary': ''
        }
        
        # Determine bias from composite signal
        smc_signal = smc_indicators['smc_signal']
        
        if smc_signal > 30:
            context['bias'] = 'BULLISH'
            context['strength'] = min(100, int(smc_signal))
        elif smc_signal < -30:
            context['bias'] = 'BEARISH'
            context['strength'] = min(100, int(abs(smc_signal)))
        else:
            context['bias'] = 'NEUTRAL'
            context['strength'] = 0
        
        # Identify key levels and conditions
        if smc_indicators['in_discount']:
            context['key_levels'].append('In Discount Zone (Buy Zone)')
        elif smc_indicators['in_premium']:
            context['key_levels'].append('In Premium Zone (Sell Zone)')
        elif smc_indicators['at_equilibrium']:
            context['key_levels'].append('At Equilibrium (50%)')
        
        if smc_indicators['bullish_ob']:
            context['key_levels'].append('Bullish Order Block Present')
        if smc_indicators['bearish_ob']:
            context['key_levels'].append('Bearish Order Block Present')
        
        if smc_indicators['price_in_fvg']:
            context['key_levels'].append('Price in Fair Value Gap')
        
        # Warnings
        if smc_indicators['liquidity_sweep_high']:
            context['warnings'].append('⚠️ Liquidity Sweep High (Bearish Reversal)')
        if smc_indicators['liquidity_sweep_low']:
            context['warnings'].append('⚠️ Liquidity Sweep Low (Bullish Reversal)')
        
        if smc_indicators['inducement_long']:
            context['warnings'].append('🚨 Long Trap Detected')
        if smc_indicators['inducement_short']:
            context['warnings'].append('🚨 Short Trap Detected')
        
        if smc_indicators['choch_bullish']:
            context['warnings'].append('📊 Change of Character - Bullish')
        if smc_indicators['choch_bearish']:
            context['warnings'].append('📊 Change of Character - Bearish')
        
        # Build summary
        summary_parts = [f"{context['bias']} bias"]
        if context['strength'] > 0:
            summary_parts.append(f"(strength: {context['strength']})")
        if context['key_levels']:
            summary_parts.append(f"| {context['key_levels'][0]}")
        
        context['summary'] = ' '.join(summary_parts)
        
        return context
    
    
    def _apply_smc_filters(self, predicted_direction, confidence, min_confidence, 
                           smc_indicators, smc_context):
        """
        Apply SMC filters to enhance or override ML prediction
        
        Returns:
            tuple: (signal, reason, smc_adjustment)
        """
        signal = "HOLD"
        reason = ""
        smc_adjustment = ""
        
        # Base confidence check
        if confidence < min_confidence:
            return "HOLD", f"Low confidence ({confidence:.1%} < {min_confidence:.0%})", "Confidence too low"
        
        # Start with ML prediction
        ml_signal = "BUY" if predicted_direction == 1 else "SELL"
        
        # ===== STRONG SMC OVERRIDES =====
        
        # 1. Inducement (trap) - STRONG OVERRIDE
        if smc_indicators['inducement_long'] and ml_signal == "BUY":
            return "HOLD", "ML suggests BUY but SMC detects LONG TRAP", "🚨 Long trap - signal blocked"
        
        if smc_indicators['inducement_short'] and ml_signal == "SELL":
            return "HOLD", "ML suggests SELL but SMC detects SHORT TRAP", "🚨 Short trap - signal blocked"
        
        # 2. Premium/Discount Zone Conflicts - MODERATE OVERRIDE
        if ml_signal == "BUY" and smc_indicators['in_premium']:
            # Buying in premium is risky - require higher confidence
            if confidence < 0.75:
                return "HOLD", f"BUY signal but in PREMIUM zone (confidence {confidence:.1%} < 75%)", "⚠️ Premium zone - need higher confidence"
        
        if ml_signal == "SELL" and smc_indicators['in_discount']:
            # Selling in discount is risky - require higher confidence
            if confidence < 0.75:
                return "HOLD", f"SELL signal but in DISCOUNT zone (confidence {confidence:.1%} < 75%)", "⚠️ Discount zone - need higher confidence"
        
        # 3. Liquidity Sweeps - REVERSAL CONFIRMATION
        if smc_indicators['liquidity_sweep_low'] and ml_signal == "BUY":
            # Both agree on bullish reversal - STRONG SIGNAL
            smc_adjustment = "✅ Liquidity sweep low confirms BUY"
            signal = "BUY"
            reason = f"ML + SMC confluence: Bullish reversal ({confidence:.1%})"
            return signal, reason, smc_adjustment
        
        if smc_indicators['liquidity_sweep_high'] and ml_signal == "SELL":
            # Both agree on bearish reversal - STRONG SIGNAL
            smc_adjustment = "✅ Liquidity sweep high confirms SELL"
            signal = "SELL"
            reason = f"ML + SMC confluence: Bearish reversal ({confidence:.1%})"
            return signal, reason, smc_adjustment
        
        # 4. Market Structure Breaks
        if smc_indicators['bos_bullish'] and ml_signal == "BUY":
            smc_adjustment = "✅ Break of Structure confirms bullish continuation"
        elif smc_indicators['bos_bearish'] and ml_signal == "SELL":
            smc_adjustment = "✅ Break of Structure confirms bearish continuation"
        
        # 5. Change of Character (reversal warning)
        if smc_indicators['choch_bullish'] and ml_signal == "SELL":
            return "HOLD", "SELL signal but ChoCh suggests trend reversal to bullish", "⚠️ ChoCh bullish - holding"
        
        if smc_indicators['choch_bearish'] and ml_signal == "BUY":
            return "HOLD", "BUY signal but ChoCh suggests trend reversal to bearish", "⚠️ ChoCh bearish - holding"
        
        # ===== SMC ENHANCEMENT (signal passes filters) =====
        
        # Check for confluence with SMC bias
        if smc_context['bias'] == 'BULLISH' and ml_signal == "BUY":
            smc_adjustment = f"✅ SMC confirms BULLISH bias (strength: {smc_context['strength']})"
            signal = "BUY"
            reason = f"ML + SMC confluence: Strong bullish setup ({confidence:.1%})"
        
        elif smc_context['bias'] == 'BEARISH' and ml_signal == "SELL":
            smc_adjustment = f"✅ SMC confirms BEARISH bias (strength: {smc_context['strength']})"
            signal = "SELL"
            reason = f"ML + SMC confluence: Strong bearish setup ({confidence:.1%})"
        
        elif smc_context['bias'] != 'NEUTRAL' and smc_context['bias'] != ml_signal.replace('BUY', 'BULLISH').replace('SELL', 'BEARISH'):
            # SMC and ML disagree
            if smc_context['strength'] > 50:
                return "HOLD", f"ML suggests {ml_signal} but SMC shows strong {smc_context['bias']} bias", f"⚠️ SMC/ML divergence"
            else:
                # Weak SMC signal - trust ML
                signal = ml_signal
                reason = f"{ml_signal} with {confidence:.1%} confidence (weak SMC conflict)"
                smc_adjustment = f"⚠️ Weak SMC {smc_context['bias']} bias - following ML"
        
        else:
            # No strong SMC opinion - use ML
            signal = ml_signal
            reason = f"{ml_signal} prediction with {confidence:.1%} confidence"
            smc_adjustment = "Neutral SMC - following ML prediction"
        
        return signal, reason, smc_adjustment
    
    
    def _get_smc_targets(self, current_price, smc_indicators, predicted_direction):
        """
        Calculate SMC-based target and stop loss levels
        
        Returns:
            dict: Target prices and stop levels
        """
        targets = {
            'entry': current_price,
            'targets': [],
            'stop_loss': None,
            'risk_reward': None
        }
        
        if predicted_direction == 1:  # Bullish
            # Targets: nearby resistance levels
            if smc_indicators['dist_to_bear_ob'] > 0:
                targets['targets'].append({
                    'level': current_price * (1 + smc_indicators['dist_to_bear_ob'] / 100),
                    'type': 'Bearish Order Block'
                })
            
            if smc_indicators['dist_to_liquidity_high'] > 0:
                targets['targets'].append({
                    'level': current_price * (1 + smc_indicators['dist_to_liquidity_high'] / 100),
                    'type': 'Liquidity High'
                })
            
            # Stop loss: below support
            if smc_indicators['dist_to_bull_ob'] < 0:
                targets['stop_loss'] = current_price * (1 + smc_indicators['dist_to_bull_ob'] / 100)
            elif smc_indicators['dist_to_support'] > 0:
                targets['stop_loss'] = current_price * (1 - smc_indicators['dist_to_support'] / 100)
        
        else:  # Bearish
            # Targets: nearby support levels
            if smc_indicators['dist_to_bull_ob'] < 0:
                targets['targets'].append({
                    'level': current_price * (1 + smc_indicators['dist_to_bull_ob'] / 100),
                    'type': 'Bullish Order Block'
                })
            
            if smc_indicators['dist_to_liquidity_low'] > 0:
                targets['targets'].append({
                    'level': current_price * (1 - smc_indicators['dist_to_liquidity_low'] / 100),
                    'type': 'Liquidity Low'
                })
            
            # Stop loss: above resistance
            if smc_indicators['dist_to_bear_ob'] > 0:
                targets['stop_loss'] = current_price * (1 + smc_indicators['dist_to_bear_ob'] / 100)
            elif smc_indicators['dist_to_resistance'] > 0:
                targets['stop_loss'] = current_price * (1 + smc_indicators['dist_to_resistance'] / 100)
        
        # Calculate risk/reward
        if targets['stop_loss'] and targets['targets']:
            risk = abs(current_price - targets['stop_loss'])
            reward = abs(targets['targets'][0]['level'] - current_price)
            targets['risk_reward'] = round(reward / risk, 2) if risk > 0 else None
        
        return targets
    
    
    def _print_signal_with_smc(self, result):
        """Enhanced signal printing with SMC details"""
        print("\n" + "="*70)
        print(f"🎯 TRADING SIGNAL - {result['symbol']}")
        print("="*70)
        print(f"Signal: {result['signal']} | Direction: {result['predicted_direction']}")
        print(f"Confidence: {result['confidence']:.1%} (threshold: {result['min_confidence_threshold']:.0%})")
        print(f"Current Price: ${result['current_price']:,.2f}")
        print(f"Reason: {result['reason']}")
        
        # SMC Analysis
        if result['smc_enabled']:
            print("\n" + "-"*70)
            print("📊 SMART MONEY ANALYSIS")
            print("-"*70)
            
            if result['smc_adjustment']:
                print(f"SMC Filter: {result['smc_adjustment']}")
            
            if result['smc_context']:
                ctx = result['smc_context']
                print(f"Market Bias: {ctx['bias']} (Strength: {ctx['strength']})")
                
                if ctx['key_levels']:
                    print("\nKey Levels:")
                    for level in ctx['key_levels']:
                        print(f"  • {level}")
                
                if ctx['warnings']:
                    print("\nWarnings:")
                    for warning in ctx['warnings']:
                        print(f"  {warning}")
            
            # SMC Targets
            if result['smc_targets'] and result['signal'] != "HOLD":
                targets = result['smc_targets']
                print("\nSMC-Based Targets:")
                
                for i, target in enumerate(targets['targets'][:3], 1):
                    print(f"  Target {i}: ${target['level']:,.2f} ({target['type']})")
                
                if targets['stop_loss']:
                    print(f"  Stop Loss: ${targets['stop_loss']:,.2f}")
                
                if targets['risk_reward']:
                    print(f"  Risk/Reward: 1:{targets['risk_reward']}")
        
        print("\n" + "="*70)
        print(f"Valid until: {result['valid_until']}")
        print("="*70 + "\n")


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