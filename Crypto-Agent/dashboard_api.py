"""
Flask Backend for Crypto Trading Bot Dashboard
Provides API endpoints for predictions, price data, and wallet information
"""

import os
import sys
import json
import threading
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, Response
from flask_cors import CORS
from loguru import logger
from collections import deque

# Load environment variables from project root (one level above this package)
project_root = Path(__file__).resolve().parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)

# Add Crypto-Agent to path so we can import its modules
crypto_agent_path = str(project_root / "Crypto-Agent")
if crypto_agent_path not in sys.path:
    sys.path.insert(0, crypto_agent_path)

from config import SUPPORTED_CRYPTOS
from binance.client import Client
from data_collector import DataCollector
from xg_predict import SignalPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

# Suppress verbose data_collector logs
logger.disable("data_collector")

# Initialize Binance client
testing = os.getenv("TEST_MODE", "false") == "true"

print(f"Running in {'TESTNET' if testing else 'MAINNET'} mode.")

if testing:
    BINANCE_API_KEY = os.getenv("TESTNET_API_KEY")
    BINANCE_API_SECRET = os.getenv("TESTNET_API_SECRET")
else:
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=testing)

# Initialize data collector
data_collector = DataCollector(binance_client)

# Price streaming data: store last 60 1-minute candles per symbol
price_streams = {symbol: deque(maxlen=60) for symbol in SUPPORTED_CRYPTOS.values()}
stream_lock = threading.Lock()
last_price_update = {}
predictions_cache = {}
predictions_mtime = None


def _build_prices_snapshot():
    """Build prices payload from cached candles"""
    prices_data = {}

    with stream_lock:
        for symbol in SUPPORTED_CRYPTOS.values():
            candles = list(price_streams[symbol])

            if len(candles) > 0:
                closes = [c["close"] for c in candles]
                highs = [c["high"] for c in candles]
                lows = [c["low"] for c in candles]

                current_price = closes[-1]
                price_change = (
                    ((current_price - closes[0]) / closes[0] * 100)
                    if closes[0] != 0
                    else 0
                )

                prices_data[symbol] = {
                    "current_price": current_price,
                    "24h_high": max(highs),
                    "24h_low": min(lows),
                    "24h_change": price_change,
                    "history": candles,
                    "source": "periodic_cache",
                    "last_update": last_price_update.get(symbol, "unknown"),
                }
            else:
                prices_data[symbol] = {"error": f"No cached data for {symbol} yet"}

    return prices_data


def _load_predictions_cache():
    """Load predictions from disk only when file changes."""
    global predictions_cache, predictions_mtime

    predictions_file = project_root / "Crypto-Agent" / "cache" / "predictions.json"

    if not predictions_file.exists():
        predictions_cache = {}
        predictions_mtime = None
        return predictions_cache

    mtime = predictions_file.stat().st_mtime
    if predictions_mtime is None or mtime != predictions_mtime:
        with open(predictions_file, "r") as f:
            predictions_cache = json.load(f)
        predictions_mtime = mtime

    return predictions_cache


def _save_predictions_cache(pred_map):
    """Persist predictions to cache file and memory."""
    global predictions_cache, predictions_mtime

    predictions_cache = pred_map

    predictions_file = project_root / "Crypto-Agent" / "cache" / "predictions.json"
    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    with open(predictions_file, "w") as f:
        json.dump(pred_map, f, indent=2)

    predictions_mtime = predictions_file.stat().st_mtime


def _predictions_list_to_map(predictions_list):
    """Convert list of prediction dicts to keyed map for frontend."""
    pred_map = {}
    for p in predictions_list or []:
        symbol = p.get("symbol", "UNKNOWN")
        interval = p.get("interval", "")
        horizon = p.get("horizon_minutes")
        key_parts = [symbol]
        if interval:
            key_parts.append(interval)
        if horizon:
            key_parts.append(f"{horizon}m")
        key = "_".join(key_parts)

        pred_map[key] = {
            "symbol": symbol,
            "interval": interval,
            "horizon_minutes": horizon,
            "signal": p.get("signal"),
            "confidence": p.get("confidence"),
            "predicted_direction": p.get("predicted_direction"),
            "prediction_time": p.get("prediction_time"),
            "valid_until": p.get("valid_until"),
            "current_price": p.get("current_price"),
            "price_estimate": p.get("price_estimate"),
            "prob_up": p.get("prob_up"),
            "prob_down": p.get("prob_down"),
            "reason": p.get("reason"),
        }

    return pred_map


def update_prices_periodically():
    """Periodically fetch and update price data (simulates streaming)"""
    while True:
        try:
            for symbol in SUPPORTED_CRYPTOS.values():
                try:
                    df = data_collector.get_realtime_data(
                        symbol=symbol,
                        days=1,
                        include_ongoing=True,
                        interval="1m",
                    )

                    if df is not None and len(df) > 0:
                        # Get last 60 candles
                        df_1h = df.tail(60)

                        candles = [
                            {
                                "timestamp": idx.isoformat(),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": float(row["volume"]),
                            }
                            for idx, row in df_1h.iterrows()
                        ]

                        with stream_lock:
                            price_streams[symbol] = deque(candles, maxlen=60)
                            last_price_update[symbol] = datetime.now().isoformat()

                except Exception as e:
                    logger.error(f"{symbol}: {e}")

            # Update frequently for near-real-time UI (every 2 seconds)
            threading.Event().wait(2)

        except Exception as e:
            logger.error(f"Price update loop error: {e}")
            threading.Event().wait(10)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/api/cryptos", methods=["GET"])
def get_cryptos():
    """Get list of configured cryptocurrencies"""
    try:
        cryptos = [{"symbol": v, "name": k} for k, v in SUPPORTED_CRYPTOS.items()]
        return jsonify({"success": True, "data": cryptos})
    except Exception as e:
        logger.error(f"Error getting cryptos: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prices", methods=["GET"])
def get_prices():
    """
    Get current and recent price data from cached periodic updates
    Returns price data with 1-minute intervals for the last 60 candles
    """
    try:
        prices_data = _build_prices_snapshot()

        # Maintain backward compatibility with previous "data" key
        return jsonify(
            {
                "success": True,
                "data": prices_data,
                "prices": prices_data,
                "source": "periodic_cache",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in get_prices: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/stream/prices")
def stream_prices():
    """Server-Sent Events stream for prices"""

    def event_stream():
        while True:
            try:
                snapshot = _build_prices_snapshot()
                payload = {
                    "success": True,
                    "data": snapshot,
                    "prices": snapshot,
                    "source": "periodic_cache",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"Error building SSE payload: {e}")
                payload = {"success": False, "error": str(e)}

            yield f"data: {json.dumps(payload)}\n\n"
            threading.Event().wait(5)

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    return Response(event_stream(), headers=headers)


@app.route("/api/wallet", methods=["GET"])
def get_wallet_info():
    """
    Get Binance wallet information
    Returns balances, total value, and asset breakdown
    """
    try:
        account_info = binance_client.get_account()
        balances_data = []
        total_usdt_value = 0.0
        stable_assets = {"USDT", "BUSD", "FDUSD"}

        prices_snapshot = _build_prices_snapshot()

        for balance in account_info.get("balances", []):
            asset = balance.get("asset")
            free = float(balance.get("free", 0))
            locked = float(balance.get("locked", 0))
            total = free + locked

            if total <= 0:
                continue

            if asset in stable_assets:
                price = 1.0
            else:
                symbol = f"{asset}USDT"
                price = prices_snapshot.get(symbol, {}).get("current_price")

                if price is None:
                    try:
                        ticker = binance_client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker.get("price", 0))
                    except Exception:
                        price = 0.0

            usdt_value = total * price if price else 0.0

            balances_data.append(
                {
                    "asset": asset,
                    "free": free,
                    "locked": locked,
                    "total": total,
                    "price_usdt": price if price else None,
                    "value_usdt": usdt_value,
                }
            )

            total_usdt_value += usdt_value

        balances_data.sort(key=lambda x: x["value_usdt"], reverse=True)

        return jsonify(
            {
                "success": True,
                "data": {
                    "balances": balances_data,
                    "total_value_usdt": round(total_usdt_value, 2),
                    "asset_count": len(balances_data),
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting wallet info: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """
    Get all bot predictions from latest models
    Returns predictions for all symbols across different intervals
    """
    try:
        predictions = _load_predictions_cache()

        return jsonify(
            {
                "success": True,
                "data": predictions,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predictions/refresh", methods=["POST"])
def refresh_predictions():
    """Run predictions on-demand and update cache/file."""
    try:
        predictor = SignalPredictor(binance_client)
        results = predictor.predict_all_models(estimate_price=True, days=10)

        if not results or "predictions" not in results:
            return jsonify({"success": False, "error": "No predictions generated"}), 500

        pred_map = _predictions_list_to_map(results.get("predictions", []))

        _save_predictions_cache(pred_map)

        return jsonify(
            {
                "success": True,
                "data": pred_map,
                "metadata": {
                    "batch_timestamp": results.get("batch_timestamp"),
                    "total_models": results.get("total_models"),
                    "successful_predictions": results.get("successful_predictions"),
                    "failed_predictions": results.get("failed_predictions"),
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error refreshing predictions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predictions/<symbol>", methods=["GET"])
def get_symbol_predictions(symbol):
    """Get predictions for a specific symbol"""
    try:
        all_predictions = _load_predictions_cache()

        symbol_predictions = {k: v for k, v in all_predictions.items() if symbol in k}

        return jsonify(
            {
                "success": True,
                "symbol": symbol,
                "data": symbol_predictions,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting predictions for {symbol}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Dashboard API starting on http://0.0.0.0:5000")

    # Start periodic price update in background thread
    price_thread = threading.Thread(target=update_prices_periodically, daemon=True)
    price_thread.start()

    # Give it a moment to fetch initial data
    import time

    time.sleep(2)

    app.run(debug=False, host="0.0.0.0", port=5000)
