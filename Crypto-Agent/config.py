from pathlib import Path

# BASE PATHS
BASE_DIR = Path(__file__).parent

directory_paths = [
    BASE_DIR / "models",
    BASE_DIR / "metrics",
]
# Ensure directories exist
for directory in directory_paths:
    directory.mkdir(parents=True, exist_ok=True)

# SUPPORTED CRYPTOCURRENCIES
SUPPORTED_CRYPTOS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "LINK": "LINKUSDT",
    # "USDC": "USDCUSDT",
}


MODEL_CONFIG = {
    "model_name": "amazon/chronos-t5-small",  # Change to tiny/mini for faster inference
    "device": "cpu",  # Change to "cuda" if you have GPU
    "torch_dtype": "float32",  # Use "bfloat16" for GPU
}

# PREDICTION SETTINGS

# For day trading, use shorter context
PREDICTION_CONFIG = {
    "context_length": 72,  # 3 days instead of 7
    "prediction_horizons": {
        "short": 1,  # 1 hour for scalping
        "medium": 6,  # 6 hours for day trading
        "long": 24,  # 24 hours max
    },
    "num_samples": 50,  # More samples = better confidence intervals
    "cache_duration_minutes": 5,  # Update more frequently
}

# DATA COLLECTION SETTINGS
DATA_CONFIG = {
    # How much historical data to fetch for prediction
    "historical_days": 30,  # Fetch last 30 days
    # Binance API settings
    "interval": "1h",  # Hourly data
    "limit": 1000,  # Max per request
    # For training/fine-tuning (if needed)
    "training_days": 21,  # 1 year of data for training
}

# FEATURE ENGINEERING
TECHNICAL_INDICATORS = {
    "rsi_period": 14,
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "std": 2},
    "moving_averages": [7, 14, 25, 50],
    "ema_periods": [12, 26],
    "atr_period": 14,
}

# WHATSAPP RESPONSE FORMAT
WHATSAPP_CONFIG = {
    "show_confidence": True,
    "show_price_targets": True,
    "include_disclaimer": True,
    "show_technical_indicators": True,
}

# FINE-TUNING SETTINGS (Optional)
TRAINING_CONFIG = {
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "validation_split": 0.2,
    "early_stopping_patience": 3,
}

# CACHE SETTINGS
CACHE_CONFIG = {
    "enabled": True,
    "max_cache_size_mb": 100,
    "cleanup_interval_hours": 24,
}
