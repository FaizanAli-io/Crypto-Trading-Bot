"""
ML Predictor: Main prediction module using Chronos model
Handles all price predictions for crypto currencies
"""
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from chronos import ChronosPipeline
from loguru import logger
import json
from pathlib import Path

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer

class CryptoPredictor:
    """Main predictor class using Chronos T5 model"""
    
    def __init__(self, binance_client=None):
        """
        Initialize predictor with Chronos model
        
        Args:
            binance_client: Binance client instance (optional)
        """
        self.model = None
        self.data_collector = DataCollector(binance_client)
        self.feature_engineer = FeatureEngineer()
        self.cache = {}
        self.cache_file = config.CACHE_DIR / "predictions.json"
        
        # Load cache if exists
        self._load_cache()
        
        logger.info("Initializing Crypto Predictor...")
    def _load_or_download_model(self):
        """Ensure model is loaded"""
        if self.model is None:
            logger.info("Loading Chronos model...")
            from chronos import ChronosPipeline
            self.model = ChronosPipeline.from_pretrained(
                config.MODEL_CONFIG['model_name'],
                device_map=config.MODEL_CONFIG['device'],
                torch_dtype=torch.bfloat16,
            )
            logger.info("✅ Model loaded successfully")
    def load_model(self):
        """Load Chronos model (lazy loading)"""
        if self.model is None:
            try:
                logger.info(f"Loading {config.MODEL_CONFIG['model_name']}...")
                
                self.model = ChronosPipeline.from_pretrained(
                    config.MODEL_CONFIG['model_name'],
                    device_map=config.MODEL_CONFIG['device'],
                    torch_dtype=getattr(torch, config.MODEL_CONFIG['torch_dtype']),
                )
                
                logger.info("✅ Chronos model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        
        return self.model
    

    def predict(self, symbol, crypto_name=None):
        """
        Main prediction function - predicts next 24 hours
        """
        try:
            if crypto_name is None:
                crypto_name = symbol.replace("USDT", "")
            
            # Check cache first
            if self._is_cached(symbol):
                logger.info(f"Using cached prediction for {symbol}")
                return self.cache[symbol]['prediction']
            
            logger.info(f"🔮 Generating prediction for {symbol}...")
            
            # Load model
            model = self.load_model()
            
            # Fetch historical data (7 days)
            context_length = config.PREDICTION_CONFIG['context_length']
            df = self.data_collector.get_realtime_data(symbol, limit=context_length)
            
            if df is None or len(df) < 24:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Add technical indicators
            df_with_features = self.feature_engineer.add_all_features(df)
            
            # Get current price and indicators
            current_price = float(df['close'].iloc[-1])
            indicators = self.feature_engineer.get_latest_indicators(df_with_features)
            
            # Prepare price series for Chronos
            price_history = torch.tensor(df['close'].values, dtype=torch.float32)
            
            # Predict next 24 hours
            prediction_length = config.PREDICTION_CONFIG['prediction_horizons']['medium']
            num_samples = config.PREDICTION_CONFIG['num_samples']
            
            logger.info(f"Predicting next {prediction_length} hours...")
            
            forecast = model.predict(
                context=price_history,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
            
            # ✅ Ensure forecast is numpy array
            if isinstance(forecast, torch.Tensor):
                forecast_np = forecast.cpu().numpy()
            else:
                forecast_np = np.array(forecast)
            
            # Calculate statistics across samples - ensure we have 2D array
            if forecast_np.ndim == 1:
                forecast_np = forecast_np.reshape(1, -1)
            
            forecast_median = np.median(forecast_np, axis=(0,1))  # shape -> (24,)
            forecast_low = np.quantile(forecast_np, 0.1, axis=(0,1))  # shape -> (24,)
            forecast_high = np.quantile(forecast_np, 0.9, axis=(0,1))  # shape -> (24,)
            
            forecast_median = np.asarray(forecast_median, dtype=float).flatten()
            forecast_low = np.asarray(forecast_low, dtype=float).flatten()
            forecast_high = np.asarray(forecast_high, dtype=float).flatten()

            
            
            # # Convert to float arrays to avoid ambiguity errors
            # forecast_median = np.asarray(forecast_median, dtype=float)
            # forecast_low = np.asarray(forecast_low, dtype=float)
            # forecast_high = np.asarray(forecast_high, dtype=float)
            
            # Analyze predictions
            analysis = self._analyze_predictions(
                current_price=current_price,
                predictions=forecast_median,
                low_predictions=forecast_low,
                high_predictions=forecast_high,
                indicators=indicators
            )
            print("forecast_np shape:", forecast_np.shape)
            print("forecast_median shape:", forecast_median.shape)
            print("forecast_low shape:", forecast_low.shape)
            print("forecast_high shape:", forecast_high.shape)
            # Build result (force conversion to float using .item() for numpy scalars)
            result = {
                "symbol": crypto_name,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat(),
                "predictions": {
                    "1h": float(np.ravel(forecast_median)[0]),
                    "6h": float(np.ravel(forecast_median)[5]) if len(forecast_median) > 5 else None,
                    "12h": float(np.ravel(forecast_median)[11]) if len(forecast_median) > 11 else None,
                    "24h": float(np.ravel(forecast_median)[-1]),
                },
                "price_range": {
                    "24h_low": float(np.asarray(forecast_low[-1]).item()),
                    "24h_high": float(np.asarray(forecast_high[-1]).item()),
                },
                "analysis": analysis,
                "technical_indicators": indicators,
                "confidence": analysis['confidence'],
                "signal": analysis['signal'],
            }
            
            # Cache result
            self._cache_prediction(symbol, result)
            
            logger.info(f"✅ Prediction complete for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    
    def _analyze_predictions(self, current_price, predictions, low_predictions, 
                            high_predictions, indicators):
        """
        Analyze predictions and generate trading signal
        
        Returns:
            dict: Analysis including trend, signal, confidence
        """
        # Ensure all inputs are numpy arrays and flatten them
        predictions = np.asarray(predictions).flatten()
        low_predictions = np.asarray(low_predictions).flatten()
        high_predictions = np.asarray(high_predictions).flatten()
        
        # Calculate expected price change
        predicted_24h = float(predictions[-1])
        price_change_pct = ((predicted_24h - current_price) / current_price) * 100
        
        # Determine trend
        if price_change_pct > 2:
            trend = "bullish"
            trend_emoji = "🟢"
        elif price_change_pct < -2:
            trend = "bearish"
            trend_emoji = "🔴"
        else:
            trend = "neutral"
            trend_emoji = "⚪"
        
        # Calculate volatility from prediction range
        volatility_24h = ((float(high_predictions[-1]) - float(low_predictions[-1])) / current_price) * 100
        
        # Generate trading signal based on multiple factors
        signal, confidence = self._generate_signal(
            price_change_pct=price_change_pct,
            volatility=volatility_24h,
            indicators=indicators
        )
        
        # Support/resistance levels - use scalar min/max
        support_level = float(predictions.min())
        resistance_level = float(predictions.max())
        
        return {
            "trend": trend,
            "trend_emoji": trend_emoji,
            "price_change_24h": float(price_change_pct),
            "volatility_24h": float(volatility_24h),
            "signal": signal,
            "confidence": float(confidence),
            "support_level": support_level,
            "resistance_level": resistance_level,
            "risk_level": "high" if volatility_24h > 5 else "medium" if volatility_24h > 2 else "low",
        }
    
    """
Add these methods to your CryptoPredictor class in ml_predictor.py
Place them after the predict() method
"""

    def predict_day_trade(self, symbol, crypto_name=None):
        """
        Specialized prediction for DAY TRADING
        Uses 6-hour forecast with shorter context for faster reactions
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            crypto_name: Display name (e.g., "BTC")
        
        Returns:
            dict: Prediction with day trading levels
        """
        try:
            if crypto_name is None:
                crypto_name = symbol.replace("USDT", "")
            
            # Check cache
            cache_key = f"{symbol}_daytrade"
            if self._is_cached(cache_key):
                logger.info(f"Using cached day trade prediction for {symbol}")
                return self.cache[cache_key]['prediction']
            
            logger.info(f"🎯 Day trade prediction for {symbol}...")
            
            # Load model
            model = self.load_model()
            
            # Use shorter context (3 days = 72 hours)
            context_length = 72
            df = self.data_collector.get_realtime_data(symbol, limit=context_length)
            
            if df is None or len(df) < 24:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Add features
            df_with_features = self.feature_engineer.add_all_features(df)
            current_price = float(df['close'].iloc[-1])
            indicators = self.feature_engineer.get_latest_indicators(df_with_features)
            
            # Prepare for prediction
            price_history = torch.tensor(df['close'].values, dtype=torch.float32)
            
            # Predict next 6 hours (day trading window)
            prediction_length = 6
            num_samples = 50  # More samples for better confidence
            
            logger.info(f"Predicting next {prediction_length} hours (day trading)...")
            
            forecast = model.predict(
                context=price_history,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
            
            # Process forecast
            if isinstance(forecast, torch.Tensor):
                forecast_np = forecast.cpu().numpy()
            else:
                forecast_np = np.array(forecast)
            
            if forecast_np.ndim == 1:
                forecast_np = forecast_np.reshape(1, -1)
            
            forecast_median = np.median(forecast_np, axis=(0,1)).flatten()
            forecast_low = np.quantile(forecast_np, 0.1, axis=(0,1)).flatten()
            forecast_high = np.quantile(forecast_np, 0.9, axis=(0,1)).flatten()
            
            # Analyze predictions
            analysis = self._analyze_predictions(
                current_price=current_price,
                predictions=forecast_median,
                low_predictions=forecast_low,
                high_predictions=forecast_high,
                indicators=indicators
            )
            
            # Calculate day trading levels
            entry_price = current_price * (1 - 0.003)  # 0.3% below current
            stop_loss = current_price * (1 - 0.015)    # 1.5% stop loss
            take_profit_1 = current_price * (1 + 0.01) # 1% profit
            take_profit_2 = current_price * (1 + 0.02) # 2% profit
            take_profit_3 = current_price * (1 + 0.03) # 3% profit (ambitious)
            
            # Build result
            result = {
                "symbol": crypto_name,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat(),
                "mode": "day_trading",
                "timeframe": "6h",
                "predictions": {
                    "1h": float(forecast_median[0]),
                    "3h": float(forecast_median[2]) if len(forecast_median) > 2 else None,
                    "6h": float(forecast_median[-1]),
                },
                "price_range": {
                    "6h_low": float(forecast_low[-1]),
                    "6h_high": float(forecast_high[-1]),
                },
                "day_trading_levels": {
                    "entry": round(entry_price, 8 if current_price < 1 else 2),
                    "stop_loss": round(stop_loss, 8 if current_price < 1 else 2),
                    "take_profit_1": round(take_profit_1, 8 if current_price < 1 else 2),
                    "take_profit_2": round(take_profit_2, 8 if current_price < 1 else 2),
                    "take_profit_3": round(take_profit_3, 8 if current_price < 1 else 2),
                    "risk_reward_ratio": 2.0,  # 1.5% risk for 3% reward
                },
                "analysis": analysis,
                "technical_indicators": indicators,
                "confidence": analysis['confidence'],
                "signal": analysis['signal'],
            }
            
            # Cache result
            self._cache_prediction(cache_key, result)
            
            logger.info(f"✅ Day trade prediction complete for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Day trade prediction error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


    def ensemble_predict(self, symbol, crypto_name=None):
        """
        ENSEMBLE PREDICTION: Combines multiple strategies for better accuracy
        
        Strategy combines:
        1. Chronos ML prediction
        2. Trend following (EMA crossover)
        3. Mean reversion (Bollinger Bands)
        4. Momentum confirmation (RSI + Volume)
        
        Args:
            symbol: Trading pair
            crypto_name: Display name
        
        Returns:
            dict: Enhanced prediction with ensemble confidence
        """
        try:
            if crypto_name is None:
                crypto_name = symbol.replace("USDT", "")
            
            logger.info(f"🎯 Ensemble prediction for {symbol}...")
            
            # Get base Chronos prediction
            chronos_pred = self.predict(symbol, crypto_name)
            
            # Fetch data for additional strategies
            df = self.data_collector.get_realtime_data(symbol, limit=100)
            df_features = self.feature_engineer.add_all_features(df)
            
            current_price = float(df['close'].iloc[-1])
            latest = df_features.iloc[-1]
            
            # === STRATEGY 1: Trend Following (EMA) ===
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            
            trend_score = 0
            if ema_12 > ema_26 > ema_50:
                trend_signal = "BUY"
                trend_score = 30
            elif ema_12 < ema_26 < ema_50:
                trend_signal = "SELL"
                trend_score = -30
            else:
                trend_signal = "NEUTRAL"
                trend_score = 0
            
            # === STRATEGY 2: Mean Reversion (Bollinger Bands) ===
            bb_position = float(latest['bb_position'])
            mean_reversion_score = 0
            
            if bb_position < 0.2:
                mean_reversion_signal = "BUY"  # Oversold
                mean_reversion_score = 25
            elif bb_position > 0.8:
                mean_reversion_signal = "SELL"  # Overbought
                mean_reversion_score = -25
            else:
                mean_reversion_signal = "NEUTRAL"
                mean_reversion_score = 0
            
            # === STRATEGY 3: Momentum Confirmation ===
            rsi = float(latest['rsi'])
            volume_ratio = float(latest['rel_volume'])
            roc_6h = float(latest['roc_6h'])
            
            momentum_score = 0
            if rsi < 40 and volume_ratio > 1.2 and roc_6h > 0:
                momentum_signal = "BUY"
                momentum_score = 20
            elif rsi > 60 and volume_ratio > 1.2 and roc_6h < 0:
                momentum_signal = "SELL"
                momentum_score = -20
            else:
                momentum_signal = "NEUTRAL"
                momentum_score = 0
            
            # === STRATEGY 4: Convert Chronos signal to score ===
            chronos_signal = chronos_pred['signal']
            chronos_confidence = chronos_pred['confidence']
            
            if chronos_signal == "STRONG BUY":
                chronos_score = 40 * chronos_confidence
            elif chronos_signal == "BUY":
                chronos_score = 25 * chronos_confidence
            elif chronos_signal == "HOLD":
                chronos_score = 0
            elif chronos_signal == "SELL":
                chronos_score = -25 * chronos_confidence
            else:  # STRONG SELL
                chronos_score = -40 * chronos_confidence
            
            # === ENSEMBLE CALCULATION ===
            # Weighted average of all strategies
            weights = {
                'chronos': 0.40,     # 40% weight (ML prediction)
                'trend': 0.25,       # 25% weight (trend following)
                'mean_rev': 0.20,    # 20% weight (mean reversion)
                'momentum': 0.15,    # 15% weight (momentum)
            }
            
            ensemble_score = (
                chronos_score * weights['chronos'] +
                trend_score * weights['trend'] +
                mean_reversion_score * weights['mean_rev'] +
                momentum_score * weights['momentum']
            )
            
            # Convert ensemble score to signal
            if ensemble_score >= 30:
                ensemble_signal = "STRONG BUY"
                ensemble_confidence = min(ensemble_score / 50, 0.95)
            elif ensemble_score >= 15:
                ensemble_signal = "BUY"
                ensemble_confidence = min(ensemble_score / 40, 0.85)
            elif ensemble_score >= -10:
                ensemble_signal = "HOLD"
                ensemble_confidence = 0.60
            elif ensemble_score >= -25:
                ensemble_signal = "SELL"
                ensemble_confidence = min(abs(ensemble_score) / 40, 0.80)
            else:
                ensemble_signal = "STRONG SELL"
                ensemble_confidence = min(abs(ensemble_score) / 50, 0.90)
            
            # Check agreement level
            all_signals = [chronos_signal, trend_signal, mean_reversion_signal, momentum_signal]
            buy_count = sum(1 for s in all_signals if "BUY" in s)
            sell_count = sum(1 for s in all_signals if "SELL" in s)
            agreement = max(buy_count, sell_count) / 4
            
            # Boost confidence if signals agree
            if agreement >= 0.75:  # 3 or more strategies agree
                ensemble_confidence = min(ensemble_confidence * 1.2, 0.95)
            
            # Update the prediction with ensemble results
            chronos_pred['ensemble'] = {
                'signal': ensemble_signal,
                'confidence': float(ensemble_confidence),
                'score': float(ensemble_score),
                'agreement': float(agreement),
                'strategies': {
                    'chronos': {'signal': chronos_signal, 'score': float(chronos_score)},
                    'trend_following': {'signal': trend_signal, 'score': float(trend_score)},
                    'mean_reversion': {'signal': mean_reversion_signal, 'score': float(mean_reversion_score)},
                    'momentum': {'signal': momentum_signal, 'score': float(momentum_score)},
                }
            }
            
            # Override main signal with ensemble
            chronos_pred['signal'] = ensemble_signal
            chronos_pred['confidence'] = ensemble_confidence
            chronos_pred['analysis']['signal'] = ensemble_signal
            chronos_pred['analysis']['confidence'] = ensemble_confidence
            
            logger.info(f"✅ Ensemble: {ensemble_signal} (confidence: {ensemble_confidence:.2%}, agreement: {agreement:.1%})")
            
            return chronos_pred
            
        except Exception as e:
            logger.error(f"Ensemble prediction error for {symbol}: {e}")
            # Fallback to regular prediction
            return self.predict(symbol, crypto_name)


    # Also add this improved signal generation (replaces your existing _generate_signal)
    def _generate_signal(self, price_change_pct, volatility, indicators):
        """
        IMPROVED: Generate trading signal for DAY TRADING
        Adjusted thresholds for shorter timeframes
        """
        score = 0
        
        # Factor 1: Predicted price movement (50% weight) - MORE SENSITIVE
        if price_change_pct > 3:
            score += 50
        elif price_change_pct > 1.5:
            score += 35
        elif price_change_pct > 0.5:
            score += 20
        elif price_change_pct > -0.5:
            score += 5
        elif price_change_pct > -1.5:
            score += -20
        elif price_change_pct > -3:
            score += -35
        else:
            score += -50
        
        # Factor 2: RSI (25% weight) - MORE AGGRESSIVE
        rsi = float(indicators.get('rsi', 50))
        if rsi < 35:
            score += 20
        elif rsi < 45:
            score += 10
        elif rsi > 65:
            score += -20
        elif rsi > 55:
            score += -10
        else:
            score += 5
        
        # Factor 3: MACD (15% weight)
        macd = float(indicators.get('macd', 0))
        macd_signal = float(indicators.get('macd_signal', 0))
        if macd > macd_signal:
            score += 15
        else:
            score += -15
        
        # Factor 4: Bollinger Band (10% weight)
        bb_position = float(indicators.get('bb_position', 0.5))
        if bb_position < 0.25:
            score += 10
        elif bb_position > 0.75:
            score += -10
        
        # Factor 5: Volume confirmation
        volume_ratio = float(indicators.get('rel_volume', 1.0))
        if volume_ratio > 1.5:
            score += 10
        elif volume_ratio < 0.7:
            score += -5
        
        # Factor 6: Momentum (ROC)
        roc_6h = float(indicators.get('roc_6h', 0))
        if roc_6h > 2:
            score += 10
        elif roc_6h < -2:
            score += -10
        
        # Volatility adjustment - day traders need volatility
        if volatility < 1.5:
            score += -10  # Too quiet
        elif 2 < volatility < 8:
            score += 5   # Good range
        elif volatility > 10:
            score += -5  # Too risky
        
        # Convert score to signal - RECALIBRATED
        if score >= 45:
            return "STRONG BUY", min(score / 100 + 0.25, 0.95)
        elif score >= 20:
            return "BUY", min(score / 100 + 0.20, 0.85)
        elif score >= -15:
            return "HOLD", 0.60
        elif score >= -35:
            return "SELL", min(abs(score) / 100 + 0.20, 0.80)
        else:
            return "STRONG SELL", min(abs(score) / 100 + 0.25, 0.90)
    
    def _is_cached(self, symbol):
        """Check if prediction is cached and still valid"""
        if not config.CACHE_CONFIG['enabled']:
            return False
        
        if symbol not in self.cache:
            return False
        
        cache_time = datetime.fromisoformat(self.cache[symbol]['cached_at'])
        cache_duration = timedelta(minutes=config.PREDICTION_CONFIG['cache_duration_minutes'])
        
        return datetime.now() - cache_time < cache_duration
    
    def _cache_prediction(self, symbol, prediction):
        """Cache prediction result"""
        self.cache[symbol] = {
            'prediction': prediction,
            'cached_at': datetime.now().isoformat()
        }
        self._save_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def format_prediction_for_whatsapp(self, prediction):
        """
        Format prediction results for WhatsApp message
        """
        symbol = prediction['symbol']
        current = prediction['current_price']
        pred_24h = prediction['predictions']['24h']
        analysis = prediction['analysis']
        indicators = prediction['technical_indicators']
        
        # Format prices
        if current >= 1:
            curr_str = f"${current:,.2f}"
            pred_str = f"${pred_24h:,.2f}"
            low_str = f"${prediction['price_range']['24h_low']:,.2f}"
            high_str = f"${prediction['price_range']['24h_high']:,.2f}"
        else:
            curr_str = f"${current:.6f}"
            pred_str = f"${pred_24h:.6f}"
            low_str = f"${prediction['price_range']['24h_low']:.6f}"
            high_str = f"${prediction['price_range']['24h_high']:.6f}"
        
        # Build message
        message = f"🔮 *{symbol} PRICE PREDICTION*\n\n"
        message += f"💰 *Current Price:* {curr_str}\n"
        message += f"📅 *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        
        # Predictions
        message += "📊 *24-Hour Forecast*\n"
        message += "```\n"
        message += f"Predicted: {pred_str}\n"
        message += f"Change:    {analysis['trend_emoji']} {analysis['price_change_24h']:+.2f}%\n"
        message += f"Range:     {low_str} - {high_str}\n"
        message += "```\n\n"
        
        # Hourly breakdown
        preds = prediction['predictions']
        message += "⏰ *Hourly Breakdown*\n"
        message += "```\n"
        if preds.get('1h') is not None:
            change_1h = ((preds['1h'] - current) / current) * 100
            message += f"1h:  ${preds['1h']:,.2f} ({change_1h:+.1f}%)\n"
        if preds.get('6h') is not None:
            change_6h = ((preds['6h'] - current) / current) * 100
            message += f"6h:  ${preds['6h']:,.2f} ({change_6h:+.1f}%)\n"
        if preds.get('12h') is not None:
            change_12h = ((preds['12h'] - current) / current) * 100
            message += f"12h: ${preds['12h']:,.2f} ({change_12h:+.1f}%)\n"
        message += "```\n\n"
        
        # Trading signal
        signal_emoji = {
            "STRONG BUY": "🟢🟢",
            "BUY": "🟢",
            "HOLD": "⚪",
            "SELL": "🔴",
            "STRONG SELL": "🔴🔴"
        }
        message += f"📈 *Trading Signal:* {signal_emoji.get(analysis['signal'], '⚪')} *{analysis['signal']}*\n"
        message += f"🎯 *Confidence:* {analysis['confidence']*100:.1f}%\n"
        message += f"⚠️ *Risk Level:* {analysis['risk_level'].upper()}\n\n"
        
        # Technical indicators
        if config.WHATSAPP_CONFIG['show_technical_indicators']:
            message += "📊 *Technical Indicators*\n"
            message += "```\n"
            message += f"RSI:        {indicators['rsi']:.1f}"
            if indicators['rsi'] < 30:
                message += " (Oversold)\n"
            elif indicators['rsi'] > 70:
                message += " (Overbought)\n"
            else:
                message += " (Neutral)\n"
            
            message += f"MACD:       {indicators['macd']:.2f}\n"
            message += f"Volatility: {analysis['volatility_24h']:.2f}%\n"
            message += "```\n\n"
        
        # Support/Resistance
        message += f"📍 *Support:* ${analysis['support_level']:,.2f}\n"
        message += f"📍 *Resistance:* ${analysis['resistance_level']:,.2f}\n\n"
        
        # Disclaimer
        if config.WHATSAPP_CONFIG['include_disclaimer']:
            message += "━━━━━━━━━━━━━━━━━━━━━\n"
            message += "⚠️ *Disclaimer:* This is AI-generated prediction, not financial advice. Crypto trading involves significant risk.\n\n"
        
        message += "🤖 Powered by Amazon Chronos AI"
        
        return message

    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_name": config.MODEL_CONFIG['model_name'],
            "device": config.MODEL_CONFIG['device'],
            "loaded": self.model is not None,
            "context_length": config.PREDICTION_CONFIG['context_length'],
            "prediction_horizons": config.PREDICTION_CONFIG['prediction_horizons'],
            "cache_enabled": config.CACHE_CONFIG['enabled'],
            "cached_predictions": len(self.cache)
        }