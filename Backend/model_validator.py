# """
# Model Validator: Test prediction accuracy and track performance
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from loguru import logger
# import json
# from pathlib import Path

# import config
# from data_collector import DataCollector
# from ml_predictor import CryptoPredictor

# class ModelValidator:
#     """Validate and track model prediction accuracy"""
    
#     def __init__(self, binance_client=None):
#         self.data_collector = DataCollector(binance_client)
#         self.predictor = CryptoPredictor(binance_client)
#         self.results_dir = Path("validation_results")
#         self.results_dir.mkdir(exist_ok=True)

#     def backtest(self, symbol, days=30, prediction_horizon=24):
#         """
#         Backtest model on historical data
        
#         Args:
#             symbol (str): Trading pair (e.g., "BTCUSDT")
#             days (int): Days of historical data to test
#             prediction_horizon (int): Hours ahead to predict
            
#         Returns:
#             dict: Backtesting results with metrics
#         """
#         logger.info(f"Backtesting {symbol} over {days} days...")
        
#         # Fetch historical data (need extra for context)
#         total_days = days + 10  # Extra for context window
#         df = self.data_collector.fetch_historical_data(symbol, days=total_days)
        
#         if df is None or len(df) < 200:
#             raise ValueError(f"Insufficient data for {symbol}")
        
#         # Store predictions vs actual
#         predictions = []
#         actuals = []
#         timestamps = []
        
#         # Context length needed for predictions
#         context_length = config.PREDICTION_CONFIG['context_length']
        
#         # Walk forward through history
#         # Start after context period, end before prediction horizon
#         start_idx = context_length
#         end_idx = len(df) - prediction_horizon
        
#         # Test every 24 hours to reduce compute
#         step = 24
        
#         for i in range(start_idx, end_idx, step):
#             try:
#                 # Get context timestamp
#                 context_end_time = df.index[i]
                
#                 # Use predictor's method instead of direct model access
#                 # This ensures proper model initialization and preprocessing
#                 crypto_name = symbol.replace("USDT", "")
                
#                 # Create a temporary dataframe for this backtest point
#                 # by slicing historical data up to this point
#                 historical_slice = df.iloc[:i].copy()
                
#                 # Make prediction using the predictor's predict method
#                 # Note: You'll need to modify this to work with historical data
#                 # For now, we'll use the model directly but ensure it's loaded
                
#                 # Ensure model is loaded
#                 if self.predictor.model is None:
#                     self.predictor._load_or_download_model()
                
#                 # Get context prices
#                 context_data = df.iloc[i-context_length:i]
#                 context_prices = torch.tensor(context_data['close'].values, dtype=torch.float32)
                
#                 # Make prediction
#                 forecast = self.predictor.model.predict(
#                     context=context_prices,
#                     prediction_length=prediction_horizon,
#                     num_samples=20
#                 )
                
#                 # Get median prediction for the final hour (24h ahead)
#                 y_pred = forecast.median(dim=0).values.mean(axis=0) 
#                 logger.debug(f"Forecast median shape: {np.shape(y_pred)} at index {i}")
               
#                 predicted_price = float(y_pred[-1])
                

#                 # Get actual price (24h later)
#                 actual_price = float(df.iloc[i + prediction_horizon - 1]['close'])
                
#                 predictions.append(predicted_price)
#                 actuals.append(actual_price)
#                 timestamps.append(df.index[i])
                
#                 if len(predictions) % 10 == 0:  # Log every 10 predictions
#                     logger.info(f"Progress: {len(predictions)} predictions made")
#                     logger.info(f"Latest - Date: {df.index[i].strftime('%Y-%m-%d')}, "
#                             f"Predicted: ${predicted_price:,.2f}, "
#                             f"Actual: ${actual_price:,.2f}, "
#                             f"Error: {abs(predicted_price - actual_price) / actual_price * 100:.2f}%")
                
#             except Exception as e:
#                 logger.warning(f"Error at index {i}: {e}")
#                 continue
        
#         if len(predictions) == 0:
#             raise ValueError("No successful predictions made during backtest")
        
#         # Calculate metrics
#         metrics = self._calculate_metrics(predictions, actuals)
        
#         # Save results
#         results = {
#             'symbol': symbol,
#             'test_period_days': days,
#             'prediction_horizon_hours': prediction_horizon,
#             'num_predictions': len(predictions),
#             'metrics': metrics,
#             'predictions': predictions[:50],  # Save first 50 only to reduce file size
#             'actuals': actuals[:50],
#             'timestamps': [str(t) for t in timestamps[:50]]
#         }
        
#         self._save_results(symbol, results)
        
#         logger.info(f"\n{'='*50}")
#         logger.info(f"Backtest Results for {symbol}:")
#         logger.info(f"{'='*50}")
#         logger.info(f"  Total Predictions: {len(predictions)}")
#         logger.info(f"  MAPE: {metrics['mape']:.2f}%")
#         logger.info(f"  MAE: ${metrics['mae']:.2f}")
#         logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
#         logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
#         logger.info(f"  Within 5% of Actual: {metrics['within_5_pct']:.2f}%")
#         logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
#         logger.info(f"{'='*50}\n")
        
#         return results


#     # def backtest(self, symbol, days=30, prediction_horizon=24):
#     #     """
#     #     Backtest model on historical data
        
#     #     Args:
#     #         symbol (str): Trading pair (e.g., "BTCUSDT")
#     #         days (int): Days of historical data to test
#     #         prediction_horizon (int): Hours ahead to predict
            
#     #     Returns:
#     #         dict: Backtesting results with metrics
#     #     """
#     #     logger.info(f"Backtesting {symbol} over {days} days...")
        
#     #     # Fetch historical data (need extra for context)
#     #     total_days = days + 10  # Extra for context window
#     #     df = self.data_collector.fetch_historical_data(symbol, days=total_days)
        
#     #     if df is None or len(df) < 200:
#     #         raise ValueError(f"Insufficient data for {symbol}")
        
#     #     # Store predictions vs actual
#     #     predictions = []
#     #     actuals = []
#     #     timestamps = []
        
#     #     # Context length needed for predictions
#     #     context_length = config.PREDICTION_CONFIG['context_length']
        
#     #     # Walk forward through history
#     #     # Start after context period, end before prediction horizon
#     #     start_idx = context_length
#     #     end_idx = len(df) - prediction_horizon
        
#     #     # Test every 24 hours to reduce compute
#     #     step = 24
#     #     # Ensure model is loaded before backtesting
#     #     if self.predictor.model is None:
#     #         self.predictor._load_or_download_model()
#     #     for i in range(start_idx, end_idx, step):
#     #         try:
#     #             # Get context timestamp
#     #             context_end_time = df.index[i]
                
#     #             # Use predictor's method instead of direct model access
#     #             # This ensures proper model initialization and preprocessing
#     #             crypto_name = symbol.replace("USDT", "")
                
#     #             # Create a temporary dataframe for this backtest point
#     #             # by slicing historical data up to this point
#     #             historical_slice = df.iloc[:i].copy()
                
#     #             # Make prediction using the predictor's predict method
#     #             # Note: You'll need to modify this to work with historical data
#     #             # For now, we'll use the model directly but ensure it's loaded
                
#     #             # Ensure model is loaded
#     #             if self.predictor.model is None:
#     #                 self.predictor._load_or_download_model()
                
#     #             # Get context prices
#     #             context_data = df.iloc[i-context_length:i]
#     #             context_prices = torch.tensor(context_data['close'].values, dtype=torch.float32)
                
#     #             # Make prediction
#     #             forecast = self.predictor.model.predict(
#     #                 context=context_prices,
#     #                 prediction_length=prediction_horizon,
#     #                 num_samples=20
#     #             )
                
#     #             # --- FIX START ---
#     #             # If prediction is a tensor/array, flatten and take first element
#     #             if isinstance(y_pred, (list, tuple)):
#     #                 y_pred = y_pred[0]
#     #             if hasattr(y_pred, "detach"):  # torch tensor
#     #                 y_pred = y_pred.detach().cpu().numpy()

#     #             y_pred = np.array(y_pred).squeeze()

#     #             # If multiple outputs, keep the first
#     #             if y_pred.ndim > 0:
#     #                 y_pred = y_pred.item()  # safely converts single element to Python float
#     #             # --- FIX END ---

#     #             # Get median prediction for the final hour (24h ahead)
#     #             predicted_price = float(forecast.median(dim=0).values[-1].detach().cpu().numpy())

                
#     #             # Get actual price (24h later)
#     #             actual_price = float(df.iloc[i + prediction_horizon - 1]['close'])
                
#     #             predictions.append(predicted_price)
#     #             actuals.append(actual_price)
#     #             timestamps.append(df.index[i])
                
#     #             if len(predictions) % 10 == 0:  # Log every 10 predictions
#     #                 logger.info(f"Progress: {len(predictions)} predictions made")
#     #                 logger.info(f"Latest - Date: {df.index[i].strftime('%Y-%m-%d')}, "
#     #                         f"Predicted: ${predicted_price:,.2f}, "
#     #                         f"Actual: ${actual_price:,.2f}, "
#     #                         f"Error: {abs(predicted_price - actual_price) / actual_price * 100:.2f}%")
                
#     #         except Exception as e:
#     #             logger.warning(f"Error at index {i}: {e}")
#     #             continue
        
#     #     if len(predictions) == 0:
#     #         raise ValueError("No successful predictions made during backtest")
        
#     #     # Calculate metrics
#     #     metrics = self._calculate_metrics(predictions, actuals)
        
#     #     # Save results
#     #     results = {
#     #         'symbol': symbol,
#     #         'test_period_days': days,
#     #         'prediction_horizon_hours': prediction_horizon,
#     #         'num_predictions': len(predictions),
#     #         'metrics': metrics,
#     #         'predictions': predictions[:50],  # Save first 50 only to reduce file size
#     #         'actuals': actuals[:50],
#     #         'timestamps': [str(t) for t in timestamps[:50]]
#     #     }
        
#     #     self._save_results(symbol, results)
        
#     #     logger.info(f"\n{'='*50}")
#     #     logger.info(f"Backtest Results for {symbol}:")
#     #     logger.info(f"{'='*50}")
#     #     logger.info(f"  Total Predictions: {len(predictions)}")
#     #     logger.info(f"  MAPE: {metrics['mape']:.2f}%")
#     #     logger.info(f"  MAE: ${metrics['mae']:.2f}")
#     #     logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
#     #     logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
#     #     logger.info(f"  Within 5% of Actual: {metrics['within_5_pct']:.2f}%")
#     #     logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
#     #     logger.info(f"{'='*50}\n")
        
#     #     return results
    
#     def _calculate_metrics(self, predictions, actuals):
#         """Calculate prediction accuracy metrics"""
#         predictions = np.array(predictions)
#         actuals = np.array(actuals)
        
#         # Mean Absolute Percentage Error
#         mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
#         # Root Mean Squared Error
#         rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
#         # Mean Absolute Error
#         mae = np.mean(np.abs(predictions - actuals))
        
#         # Directional Accuracy (did we predict up/down correctly?)
#         actual_directions = np.diff(actuals) > 0
#         predicted_directions = np.diff(predictions) > 0
#         directional_accuracy = np.mean(actual_directions == predicted_directions) * 100
        
#         # R² Score (coefficient of determination)
#         ss_res = np.sum((actuals - predictions) ** 2)
#         ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
#         r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
#         # Percentage of predictions within 5% of actual
#         within_5_pct = np.mean(np.abs((actuals - predictions) / actuals) < 0.05) * 100
        
#         return {
#             'mape': float(mape),
#             'rmse': float(rmse),
#             'mae': float(mae),
#             'directional_accuracy': float(directional_accuracy),
#             'r2_score': float(r2_score),
#             'within_5_pct': float(within_5_pct)
#         }
    
#     def _save_results(self, symbol, results):
#         """Save validation results to file"""
#         filename = f"{symbol}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         filepath = self.results_dir / filename
        
#         with open(filepath, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         logger.info(f"Results saved to {filepath}")
    
#     def live_validation(self, symbol, wait_hours=24):
#         """
#         Make a prediction and wait to validate against actual price
        
#         Args:
#             symbol (str): Trading pair
#             wait_hours (int): Hours to wait before checking actual
            
#         Returns:
#             dict: Prediction and actual comparison
#         """
#         crypto_name = symbol.replace("USDT", "")
        
#         # Make prediction
#         logger.info(f"Making prediction for {crypto_name}...")
#         prediction_result = self.predictor.predict(symbol, crypto_name)
        
#         predicted_24h = prediction_result['predictions']['24h']
#         current_price = prediction_result['current_price']
#         prediction_time = datetime.now()
        
#         logger.info(f"Prediction made at {prediction_time}")
#         logger.info(f"Current: ${current_price:,.2f}")
#         logger.info(f"Predicted 24h: ${predicted_24h:,.2f}")
#         logger.info(f"Expected change: {((predicted_24h - current_price) / current_price * 100):.2f}%")
        
#         # Save prediction for later validation
#         validation_record = {
#             'symbol': symbol,
#             'prediction_time': prediction_time.isoformat(),
#             'current_price': current_price,
#             'predicted_24h': predicted_24h,
#             'validation_time': (prediction_time + timedelta(hours=wait_hours)).isoformat(),
#             'status': 'pending'
#         }
        
#         self._save_pending_validation(validation_record)
        
#         logger.info(f"\nCheck back in {wait_hours} hours to validate!")
        
#         return validation_record
    
#     def check_pending_validations(self):
#         """Check all pending predictions and validate if time has passed"""
#         pending_file = self.results_dir / "pending_validations.json"
        
#         if not pending_file.exists():
#             logger.info("No pending validations")
#             return []
        
#         with open(pending_file, 'r') as f:
#             pending = json.load(f)
        
#         validated = []
#         still_pending = []
        
#         for record in pending:
#             validation_time = datetime.fromisoformat(record['validation_time'])
            
#             if datetime.now() >= validation_time:
#                 # Time to validate!
#                 try:
#                     # Get current price
#                     actual_price = self.data_collector.get_latest_price(record['symbol'])
                    
#                     # Calculate error
#                     predicted = record['predicted_24h']
#                     error_pct = abs((actual_price - predicted) / actual_price) * 100
                    
#                     # Was direction correct?
#                     predicted_direction = "up" if predicted > record['current_price'] else "down"
#                     actual_direction = "up" if actual_price > record['current_price'] else "down"
#                     direction_correct = predicted_direction == actual_direction
                    
#                     record['actual_price'] = actual_price
#                     record['error_pct'] = error_pct
#                     record['direction_correct'] = direction_correct
#                     record['status'] = 'validated'
#                     record['validated_at'] = datetime.now().isoformat()
                    
#                     validated.append(record)
                    
#                     logger.info(f"\n✅ Validated: {record['symbol']}")
#                     logger.info(f"   Predicted: ${predicted:,.2f}")
#                     logger.info(f"   Actual: ${actual_price:,.2f}")
#                     logger.info(f"   Error: {error_pct:.2f}%")
#                     logger.info(f"   Direction: {'✅ Correct' if direction_correct else '❌ Wrong'}")
                    
#                 except Exception as e:
#                     logger.error(f"Error validating {record['symbol']}: {e}")
#                     still_pending.append(record)
#             else:
#                 still_pending.append(record)
        
#         # Update pending file
#         with open(pending_file, 'w') as f:
#             json.dump(still_pending, f, indent=2)
        
#         # Save validated results
#         if validated:
#             validated_file = self.results_dir / f"validated_{datetime.now().strftime('%Y%m%d')}.json"
            
#             existing = []
#             if validated_file.exists():
#                 with open(validated_file, 'r') as f:
#                     existing = json.load(f)
            
#             existing.extend(validated)
            
#             with open(validated_file, 'w') as f:
#                 json.dump(existing, f, indent=2)
        
#         return validated
    
#     def _save_pending_validation(self, record):
#         """Save pending validation to file"""
#         pending_file = self.results_dir / "pending_validations.json"
        
#         existing = []
#         if pending_file.exists():
#             with open(pending_file, 'r') as f:
#                 existing = json.load(f)
        
#         existing.append(record)
        
#         with open(pending_file, 'w') as f:
#             json.dump(existing, f, indent=2)
    
#     def get_validation_summary(self):
#         """Get summary of all validation results"""
#         validated_files = list(self.results_dir.glob("validated_*.json"))
        
#         all_results = []
#         for file in validated_files:
#             with open(file, 'r') as f:
#                 all_results.extend(json.load(f))
        
#         if not all_results:
#             return {"message": "No validation results yet"}
        
#         # Calculate aggregate metrics
#         errors = [r['error_pct'] for r in all_results if 'error_pct' in r]
#         directions = [r['direction_correct'] for r in all_results if 'direction_correct' in r]
        
#         summary = {
#             'total_validations': len(all_results),
#             'avg_error_pct': np.mean(errors) if errors else 0,
#             'median_error_pct': np.median(errors) if errors else 0,
#             'directional_accuracy_pct': (sum(directions) / len(directions) * 100) if directions else 0,
#             'predictions_within_5_pct': (sum(1 for e in errors if e < 5) / len(errors) * 100) if errors else 0,
#         }
        
#         return summary


# # Add missing import
# import torch


# # ========================
# # CLI for validation
# # ========================
# if __name__ == "__main__":
#     import sys
#     from binance.client import Client
#     from dotenv import load_dotenv
#     import os
    
#     load_dotenv()
    
#     api_key = os.getenv("BINANCE_API_KEY")
#     api_secret = os.getenv("BINANCE_SECRET_KEY")
    
#     if api_key:
#         binance_client = Client(api_key, api_secret)
#     else:
#         binance_client = Client()
    
#     validator = ModelValidator(binance_client)
    
#     if len(sys.argv) > 1:
#         command = sys.argv[1]
        
#         if command == "backtest":
#             # python model_validator.py backtest BTCUSDT 30
#             symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
#             days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            
#             validator.backtest(symbol, days)
        
#         elif command == "live":
#             # python model_validator.py live BTCUSDT
#             symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
#             validator.live_validation(symbol)
        
#         elif command == "check":
#             # python model_validator.py check
#             validator.check_pending_validations()
        
#         elif command == "summary":
#             # python model_validator.py summary
#             summary = validator.get_validation_summary()
#             print(json.dumps(summary, indent=2))
    
#     else:
#         print("Usage:")
#         print("  python model_validator.py backtest BTCUSDT 30")
#         print("  python model_validator.py live BTCUSDT")
#         print("  python model_validator.py check")
#         print("  python model_validator.py summary")

"""
Model Validator: Test prediction accuracy and track performance
ENHANCED: 6-hour predictions, confidence filtering, directional focus
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path
import torch

import config
from data_collector import DataCollector
from ml_predictor import CryptoPredictor

class ModelValidator:
    """Validate and track model prediction accuracy"""
    
    def __init__(self, binance_client=None):
        self.data_collector = DataCollector(binance_client)
        self.predictor = CryptoPredictor(binance_client)
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)

    def backtest(self, symbol, days=30, prediction_horizon=6, use_confidence_filter=True):
        """
        Backtest model on historical data with 6-hour predictions
        
        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            days (int): Days of historical data to test
            prediction_horizon (int): Hours ahead to predict (default: 6)
            use_confidence_filter (bool): Only count high-confidence predictions
            
        Returns:
            dict: Backtesting results with metrics
        """
        logger.info(f"Backtesting {symbol} over {days} days (prediction horizon: {prediction_horizon}h)...")
        
        # Fetch historical data (need extra for context)
        total_days = days + 10  # Extra for context window
        df = self.data_collector.fetch_historical_data(symbol, days=total_days)
        
        if df is None or len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Store predictions vs actual
        predictions = []
        actuals = []
        timestamps = []
        confidences = []
        signals = []
        
        # Context length needed for predictions
        context_length = config.PREDICTION_CONFIG['context_length']
        
        # Walk forward through history
        start_idx = context_length
        end_idx = len(df) - prediction_horizon
        
        # Test every prediction_horizon hours (e.g., every 6 hours)
        step = prediction_horizon
        
        # Ensure model is loaded
        if self.predictor.model is None:
            self.predictor.load_model()
        
        for i in range(start_idx, end_idx, step):
            try:
                # Get context prices
                context_data = df.iloc[i-context_length:i]
                context_prices = torch.tensor(context_data['close'].values, dtype=torch.float32)
                
                # Make prediction
                forecast = self.predictor.model.predict(
                    context=context_prices,
                    prediction_length=prediction_horizon,
                    num_samples=50  # More samples for better confidence
                )
                
                # Process forecast
                if isinstance(forecast, torch.Tensor):
                    forecast_np = forecast.cpu().numpy()
                else:
                    forecast_np = np.array(forecast)
                
                if forecast_np.ndim == 1:
                    forecast_np = forecast_np.reshape(1, -1)
                
                # Get median prediction
                forecast_median = np.median(forecast_np, axis=(0,1)).flatten()
                forecast_std = np.std(forecast_np, axis=(0,1)).flatten()
                
                predicted_price = float(forecast_median[-1])
                
                # Calculate confidence based on prediction stability
                # Lower std = higher confidence
                prediction_std_pct = (forecast_std[-1] / predicted_price) * 100
                confidence = max(0.3, 1.0 - (prediction_std_pct / 10))  # 0.3 to 1.0 range
                
                # Get actual price (prediction_horizon hours later)
                actual_price = float(df.iloc[i + prediction_horizon - 1]['close'])
                current_price = float(context_data['close'].iloc[-1])
                
                # Determine signal
                price_change_pct = ((predicted_price - current_price) / current_price) * 100
                
                if price_change_pct > 1.0:
                    signal = "BUY"
                elif price_change_pct < -1.0:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                predictions.append(predicted_price)
                actuals.append(actual_price)
                timestamps.append(df.index[i])
                confidences.append(confidence)
                signals.append(signal)
                
                if len(predictions) % 10 == 0:
                    logger.info(f"Progress: {len(predictions)} predictions made")
                    logger.info(f"Latest - Date: {df.index[i].strftime('%Y-%m-%d %H:%M')}, "
                            f"Predicted: ${predicted_price:,.4f}, "
                            f"Actual: ${actual_price:,.4f}, "
                            f"Confidence: {confidence:.2%}, "
                            f"Signal: {signal}")
                
            except Exception as e:
                logger.warning(f"Error at index {i}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        if len(predictions) == 0:
            raise ValueError("No successful predictions made during backtest")
        
        # Calculate metrics (all predictions)
        metrics = self._calculate_metrics(predictions, actuals)
        
        # Calculate high-confidence metrics
        if use_confidence_filter:
            high_conf_metrics = self._calculate_filtered_metrics(
                predictions, actuals, confidences, min_confidence=0.65
            )
        else:
            high_conf_metrics = None
        
        # Save results
        results = {
            'symbol': symbol,
            'test_period_days': days,
            'prediction_horizon_hours': prediction_horizon,
            'num_predictions': len(predictions),
            'metrics': metrics,
            'high_confidence_metrics': high_conf_metrics,
            'predictions': predictions,
            'actuals': actuals,
            'confidences': confidences,
            'signals': signals,
            'timestamps': [str(t) for t in timestamps]
        }
        
        self._save_results(symbol, results, prediction_horizon)
        
        # Print results
        logger.info(f"\n{'='*70}")
        logger.info(f"Backtest Results for {symbol} ({prediction_horizon}h predictions):")
        logger.info(f"{'='*70}")
        logger.info(f"  Total Predictions: {len(predictions)}")
        logger.info(f"\n  ALL PREDICTIONS:")
        logger.info(f"    MAPE: {metrics['mape']:.2f}%")
        logger.info(f"    MAE: ${metrics['mae']:.4f}")
        logger.info(f"    RMSE: ${metrics['rmse']:.4f}")
        logger.info(f"    Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        logger.info(f"    Within 5% of Actual: {metrics['within_5_pct']:.2f}%")
        logger.info(f"    R² Score: {metrics['r2_score']:.4f}")
        
        if high_conf_metrics:
            logger.info(f"\n  HIGH CONFIDENCE ONLY (≥65%):")
            logger.info(f"    Count: {high_conf_metrics['count']}")
            logger.info(f"    Percentage: {high_conf_metrics['percentage']:.1f}%")
            logger.info(f"    MAPE: {high_conf_metrics['mape']:.2f}%")
            logger.info(f"    Directional Accuracy: {high_conf_metrics['directional_accuracy']:.2f}%")
            logger.info(f"    Within 5%: {high_conf_metrics['within_5_pct']:.2f}%")
        
        logger.info(f"{'='*70}\n")
        
        return results

    def backtest_multiple_horizons(self, symbol, days=30, horizons=[1, 3, 6, 12]):
        """
        Test multiple prediction horizons to find optimal timeframe
        
        Args:
            symbol (str): Trading pair
            days (int): Days to test
            horizons (list): List of hours to predict ahead
            
        Returns:
            dict: Results for each horizon
        """
        logger.info(f"Testing multiple horizons for {symbol}: {horizons}")
        
        all_results = {}
        
        for horizon in horizons:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {horizon}-hour predictions")
            logger.info(f"{'='*50}")
            
            try:
                results = self.backtest(
                    symbol=symbol,
                    days=days,
                    prediction_horizon=horizon,
                    use_confidence_filter=True
                )
                all_results[f'{horizon}h'] = results
            except Exception as e:
                logger.error(f"Error testing {horizon}h: {e}")
                continue
        
        # Compare results
        logger.info(f"\n{'='*70}")
        logger.info("COMPARISON ACROSS HORIZONS:")
        logger.info(f"{'='*70}")
        logger.info(f"{'Horizon':<10} {'Dir. Acc.':<12} {'High-Conf Acc.':<15} {'MAPE':<10}")
        logger.info("-" * 70)
        
        for horizon, results in all_results.items():
            dir_acc = results['metrics']['directional_accuracy']
            high_conf_acc = results['high_confidence_metrics']['directional_accuracy'] if results['high_confidence_metrics'] else 0
            mape = results['metrics']['mape']
            
            logger.info(f"{horizon:<10} {dir_acc:>10.2f}%  {high_conf_acc:>13.2f}%  {mape:>8.2f}%")
        
        logger.info(f"{'='*70}\n")
        
        return all_results
    
    def _calculate_metrics(self, predictions, actuals):
        """Calculate prediction accuracy metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - actuals))
        
        # Directional Accuracy
        if len(actuals) > 1:
            actual_directions = np.diff(actuals) > 0
            predicted_directions = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_directions == predicted_directions) * 100
        else:
            directional_accuracy = 0
        
        # R² Score
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Within 5% accuracy
        within_5_pct = np.mean(np.abs((actuals - predictions) / actuals) < 0.05) * 100
        
        return {
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'directional_accuracy': float(directional_accuracy),
            'r2_score': float(r2_score),
            'within_5_pct': float(within_5_pct)
        }
    
    def _calculate_filtered_metrics(self, predictions, actuals, confidences, min_confidence=0.65):
        """Calculate metrics only for high-confidence predictions"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)
        
        # Filter by confidence
        high_conf_mask = confidences >= min_confidence
        
        if high_conf_mask.sum() == 0:
            return {
                'count': 0,
                'percentage': 0,
                'mape': 0,
                'directional_accuracy': 0,
                'within_5_pct': 0
            }
        
        filtered_preds = predictions[high_conf_mask]
        filtered_actuals = actuals[high_conf_mask]
        
        # Calculate metrics
        mape = np.mean(np.abs((filtered_actuals - filtered_preds) / filtered_actuals)) * 100
        
        # Directional accuracy (need consecutive pairs)
        if len(filtered_actuals) > 1:
            actual_dirs = np.diff(filtered_actuals) > 0
            pred_dirs = np.diff(filtered_preds) > 0
            dir_acc = np.mean(actual_dirs == pred_dirs) * 100
        else:
            dir_acc = 0
        
        within_5 = np.mean(np.abs((filtered_actuals - filtered_preds) / filtered_actuals) < 0.05) * 100
        
        return {
            'count': int(high_conf_mask.sum()),
            'percentage': float(high_conf_mask.sum() / len(confidences) * 100),
            'mape': float(mape),
            'directional_accuracy': float(dir_acc),
            'within_5_pct': float(within_5)
        }
    
    def _save_results(self, symbol, results, prediction_horizon):
        """Save validation results to file"""
        filename = f"{symbol}_backtest_{prediction_horizon}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def live_validation(self, symbol, prediction_horizon=6):
        """
        Make a prediction and wait to validate against actual price
        
        Args:
            symbol (str): Trading pair
            prediction_horizon (int): Hours to wait before checking actual
            
        Returns:
            dict: Prediction and actual comparison
        """
        crypto_name = symbol.replace("USDT", "")
        
        # Make prediction
        logger.info(f"Making {prediction_horizon}h prediction for {crypto_name}...")
        
        # Use day trade method for 6h predictions
        if prediction_horizon == 6:
            prediction_result = self.predictor.predict_day_trade(symbol, crypto_name)
        else:
            prediction_result = self.predictor.predict(symbol, crypto_name)
        
        # Get prediction for specified horizon
        pred_key = f'{prediction_horizon}h'
        if pred_key in prediction_result['predictions']:
            predicted_price = prediction_result['predictions'][pred_key]
        else:
            predicted_price = prediction_result['predictions']['24h']
        
        current_price = prediction_result['current_price']
        prediction_time = datetime.now()
        
        logger.info(f"Prediction made at {prediction_time}")
        logger.info(f"Current: ${current_price:,.4f}")
        logger.info(f"Predicted {prediction_horizon}h: ${predicted_price:,.4f}")
        logger.info(f"Expected change: {((predicted_price - current_price) / current_price * 100):.2f}%")
        logger.info(f"Signal: {prediction_result['signal']} (Confidence: {prediction_result['confidence']:.1%})")
        
        # Save prediction for later validation
        validation_record = {
            'symbol': symbol,
            'prediction_time': prediction_time.isoformat(),
            'prediction_horizon_hours': prediction_horizon,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'signal': prediction_result['signal'],
            'confidence': prediction_result['confidence'],
            'validation_time': (prediction_time + timedelta(hours=prediction_horizon)).isoformat(),
            'status': 'pending'
        }
        
        self._save_pending_validation(validation_record)
        
        logger.info(f"\nCheck back in {prediction_horizon} hours to validate!")
        
        return validation_record
    
    def check_pending_validations(self):
        """Check all pending predictions and validate if time has passed"""
        pending_file = self.results_dir / "pending_validations.json"
        
        if not pending_file.exists():
            logger.info("No pending validations")
            return []
        
        with open(pending_file, 'r') as f:
            pending = json.load(f)
        
        validated = []
        still_pending = []
        
        for record in pending:
            validation_time = datetime.fromisoformat(record['validation_time'])
            
            if datetime.now() >= validation_time:
                try:
                    # Get current price
                    df = self.data_collector.get_realtime_data(record['symbol'], limit=1)
                    actual_price = float(df['close'].iloc[-1])
                    
                    # Calculate metrics
                    predicted = record['predicted_price']
                    current = record['current_price']
                    error_pct = abs((actual_price - predicted) / actual_price) * 100
                    
                    # Direction check
                    predicted_direction = "up" if predicted > current else "down"
                    actual_direction = "up" if actual_price > current else "down"
                    direction_correct = predicted_direction == actual_direction
                    
                    # Update record
                    record['actual_price'] = actual_price
                    record['error_pct'] = error_pct
                    record['direction_correct'] = direction_correct
                    record['status'] = 'validated'
                    record['validated_at'] = datetime.now().isoformat()
                    
                    validated.append(record)
                    
                    logger.info(f"\nValidated: {record['symbol']}")
                    logger.info(f"   Horizon: {record['prediction_horizon_hours']}h")
                    logger.info(f"   Current was: ${current:,.4f}")
                    logger.info(f"   Predicted: ${predicted:,.4f}")
                    logger.info(f"   Actual: ${actual_price:,.4f}")
                    logger.info(f"   Error: {error_pct:.2f}%")
                    logger.info(f"   Direction: {'✓ Correct' if direction_correct else '✗ Wrong'}")
                    logger.info(f"   Signal was: {record['signal']} (conf: {record['confidence']:.1%})")
                    
                except Exception as e:
                    logger.error(f"Error validating {record['symbol']}: {e}")
                    still_pending.append(record)
            else:
                hours_remaining = (validation_time - datetime.now()).total_seconds() / 3600
                logger.info(f"Pending: {record['symbol']} - {hours_remaining:.1f}h remaining")
                still_pending.append(record)
        
        # Update files
        with open(pending_file, 'w') as f:
            json.dump(still_pending, f, indent=2)
        
        if validated:
            validated_file = self.results_dir / f"validated_{datetime.now().strftime('%Y%m%d')}.json"
            
            existing = []
            if validated_file.exists():
                with open(validated_file, 'r') as f:
                    existing = json.load(f)
            
            existing.extend(validated)
            
            with open(validated_file, 'w') as f:
                json.dump(existing, f, indent=2)
        
        return validated
    
    def _save_pending_validation(self, record):
        """Save pending validation to file"""
        pending_file = self.results_dir / "pending_validations.json"
        
        existing = []
        if pending_file.exists():
            with open(pending_file, 'r') as f:
                existing = json.load(f)
        
        existing.append(record)
        
        with open(pending_file, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def get_validation_summary(self):
        """Get summary of all validation results"""
        validated_files = list(self.results_dir.glob("validated_*.json"))
        
        all_results = []
        for file in validated_files:
            with open(file, 'r') as f:
                all_results.extend(json.load(f))
        
        if not all_results:
            return {"message": "No validation results yet"}
        
        # Calculate aggregate metrics
        errors = [r['error_pct'] for r in all_results if 'error_pct' in r]
        directions = [r['direction_correct'] for r in all_results if 'direction_correct' in r]
        confidences = [r['confidence'] for r in all_results if 'confidence' in r]
        
        # Filter high confidence
        high_conf_results = [r for r in all_results if r.get('confidence', 0) >= 0.65]
        high_conf_directions = [r['direction_correct'] for r in high_conf_results if 'direction_correct' in r]
        
        summary = {
            'total_validations': len(all_results),
            'avg_error_pct': np.mean(errors) if errors else 0,
            'median_error_pct': np.median(errors) if errors else 0,
            'directional_accuracy_pct': (sum(directions) / len(directions) * 100) if directions else 0,
            'high_confidence_directional_accuracy': (sum(high_conf_directions) / len(high_conf_directions) * 100) if high_conf_directions else 0,
            'predictions_within_5_pct': (sum(1 for e in errors if e < 5) / len(errors) * 100) if errors else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
        }
        
        return summary


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
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "backtest":
            # python model_validator.py backtest BTCUSDT 30 6
            symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            horizon = int(sys.argv[4]) if len(sys.argv) > 4 else 6  # DEFAULT: 6 hours
            
            validator.backtest(symbol, days, prediction_horizon=horizon)
        
        elif command == "multi":
            # python model_validator.py multi DOGEUSDT 30
            symbol = sys.argv[2] if len(sys.argv) > 2 else "DOGEUSDT"
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            
            validator.backtest_multiple_horizons(symbol, days, horizons=[1, 3, 6, 12])
        
        elif command == "live":
            # python model_validator.py live BTCUSDT 6
            symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
            horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 6
            
            validator.live_validation(symbol, prediction_horizon=horizon)
        
        elif command == "check":
            # python model_validator.py check
            validator.check_pending_validations()
        
        elif command == "summary":
            # python model_validator.py summary
            summary = validator.get_validation_summary()
            print(json.dumps(summary, indent=2))
    
    else:
        print("Enhanced Model Validator - Usage:")
        print("  python model_validator.py backtest BTCUSDT 30 6    # 6-hour predictions")
        print("  python model_validator.py multi DOGEUSDT 30         # Test multiple horizons")
        print("  python model_validator.py live BTCUSDT 6            # Live 6h prediction")
        print("  python model_validator.py check                     # Check pending validations")
        print("  python model_validator.py summary                   # Get summary statistics")