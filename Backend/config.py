"""
Configuration for Chronos-based Crypto Prediction System
"""
import os
from pathlib import Path

# ========================
# BASE PATHS
# ========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR, 
                 DATA_DIR / "raw", DATA_DIR / "processed"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================
# SUPPORTED CRYPTOCURRENCIES
# ========================
SUPPORTED_CRYPTOS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT",
    "SOL": "SOLUSDT",
    "DOT": "DOTUSDT",
    "LINK": "LINKUSDT",
    "LTC": "LTCUSDT",
}

# ========================
# CHRONOS MODEL CONFIGURATION
# ========================
# Chronos model variants:
# - "amazon/chronos-t5-tiny"   (8M params,  fastest, less accurate)
# - "amazon/chronos-t5-mini"   (20M params, fast, good)
# - "amazon/chronos-t5-small"  (46M params, balanced) ⭐ RECOMMENDED
# - "amazon/chronos-t5-base"   (200M params, slow, best accuracy)
# - "amazon/chronos-t5-large"  (710M params, very slow, best)

MODEL_CONFIG = {
    "model_name": "amazon/chronos-t5-small",  # Change to tiny/mini for faster inference
    "device": "cpu",  # Change to "cuda" if you have GPU
    "torch_dtype": "float32",  # Use "bfloat16" for GPU
}

# ========================
# PREDICTION SETTINGS
# ========================

# For day trading, use shorter context
PREDICTION_CONFIG = {
    "context_length": 72,  # 3 days instead of 7
    "prediction_horizons": {
        "short": 1,    # 1 hour for scalping
        "medium": 6,   # 6 hours for day trading
        "long": 24,    # 24 hours max
    },
    "num_samples": 50,  # More samples = better confidence intervals
    "cache_duration_minutes": 5,  # Update more frequently
}
# PREDICTION_CONFIG = {
#     # How many historical hours to use as context
#     "context_length": 168,  # 7 days of hourly data (168 hours)
    
#     # Prediction horizons (hours ahead)
#     "prediction_horizons": {
#         "short": 6,    # 6 hours ahead
#         "medium": 24,  # 24 hours (1 day)
#         "long": 168,   # 168 hours (7 days)
#     },
    
#     # Number of prediction samples (higher = more robust but slower)
#     "num_samples": 20,
    
#     # Confidence thresholds for trading signals
#     "confidence_thresholds": {
#         "strong_buy": 0.75,
#         "buy": 0.60,
#         "neutral": 0.45,
#         "sell": 0.40,
#         "strong_sell": 0.25,
#     },
    
#     # Cache predictions for N minutes
#     "cache_duration_minutes": 15,
# }

# ========================
# DATA COLLECTION SETTINGS
# ========================
DATA_CONFIG = {
    # How much historical data to fetch for prediction
    "historical_days": 30,  # Fetch last 30 days
    
    # Binance API settings
    "interval": "1h",  # Hourly data
    "limit": 1000,  # Max per request
    
    # For training/fine-tuning (if needed)
    "training_days": 21,  # 1 year of data for training
}

# ========================
# FEATURE ENGINEERING
# ========================
TECHNICAL_INDICATORS = {
    "rsi_period": 14,
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "std": 2},
    "moving_averages": [7, 14, 25, 50],
    "ema_periods": [12, 26],
    "atr_period": 14,
}

# ========================
# WHATSAPP RESPONSE FORMAT
# ========================
WHATSAPP_CONFIG = {
    "show_confidence": True,
    "show_technical_indicators": True,
    "show_price_targets": True,
    "include_disclaimer": True,
}

# ========================
# FINE-TUNING SETTINGS (Optional)
# ========================
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "validation_split": 0.2,
    "early_stopping_patience": 3,
}

# ========================
# CACHE SETTINGS
# ========================
CACHE_CONFIG = {
    "enabled": True,
    "max_cache_size_mb": 100,
    "cleanup_interval_hours": 24,
}

# ========================
# LOGGING
# ========================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "predictions.log",
}

