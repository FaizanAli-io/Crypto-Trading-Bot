"""
Optimized Flask + WebSocket Backend for Crypto Trading Bot Dashboard
Real-time streaming of prices and predictions with minimal logging
"""

# Enable true WebSocket support via eventlet
try:
    import eventlet

    eventlet.monkey_patch()
except Exception:
    # If eventlet isn't available, server will still run, but WS may fallback
    pass

import os
import sys
import json
import threading
import time
import math
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
from collections import deque

# Setup paths
project_root = Path(__file__).resolve().parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)

crypto_agent_path = str(project_root / "Crypto-Agent")
if crypto_agent_path not in sys.path:
    sys.path.insert(0, crypto_agent_path)

from config import SUPPORTED_CRYPTOS
from binance.client import Client
from data_collector import DataCollector

# from xg_predict import SignalPredictor  # OLD - has caching issues
from xg_predict import predict_all_models_simple  # NEW - simple direct fetching

# Disable all library logging
import logging

logging.getLogger("flask").setLevel(logging.CRITICAL)
logging.getLogger("flask_cors").setLevel(logging.CRITICAL)
logging.getLogger("flask_socketio").setLevel(logging.CRITICAL)
logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

# Initialize Flask + SocketIO
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Monkey-patch json module to handle NaN/Infinity
original_dumps = json.dumps


def safe_json_dumps(*args, **kwargs):
    kwargs.setdefault("allow_nan", False)
    return original_dumps(*args, **kwargs)


json.dumps = safe_json_dumps

CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# Initialize services
testing = os.getenv("TEST_MODE", "false") == "true"
print(
    f"🚀 Starting in {'TESTNET' if testing else 'MAINNET'} mode with WebSocket support"
)

BINANCE_API_KEY = os.getenv("TESTNET_API_KEY" if testing else "BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv(
    "TESTNET_API_SECRET" if testing else "BINANCE_API_SECRET"
)
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=testing)

data_collector = DataCollector(binance_client)
# predictor = SignalPredictor(binance_client)  # OLD - removed, using simple function now

# Global state
price_streams = {symbol: deque(maxlen=60) for symbol in SUPPORTED_CRYPTOS.values()}
stream_lock = threading.Lock()
last_price_update = {}
last_predictions = {}
last_predictions_timestamp = None
connected_clients = set()


def _sanitize_value(value):
    """Replace NaN/Infinity with None for safe JSON serialization"""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    elif isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    return value


def _build_prices_snapshot():
    """Build price data from cached candles"""
    prices_data = {}
    with stream_lock:
        for symbol in SUPPORTED_CRYPTOS.values():
            candles = list(price_streams[symbol])
            if candles:
                closes = [c["close"] for c in candles]
                change_pct = 0
                if len(closes) > 1 and closes[0] != 0:
                    change_pct = (closes[-1] - closes[0]) / closes[0] * 100
                    if math.isnan(change_pct) or math.isinf(change_pct):
                        change_pct = 0

                prices_data[symbol] = {
                    "current_price": closes[-1],
                    "24h_high": max(c["high"] for c in candles),
                    "24h_low": min(c["low"] for c in candles),
                    "24h_change": change_pct,
                    "history": candles,
                    "last_update": last_price_update.get(symbol),
                }
    return _sanitize_value(prices_data)


def _get_predictions_snapshot():
    """Return latest in-memory predictions"""
    return last_predictions or {}


def update_prices_continuously():
    """Fetch prices and broadcast via WebSocket"""
    while True:
        try:
            for symbol in SUPPORTED_CRYPTOS.values():
                try:
                    df = data_collector.get_realtime_data(
                        symbol=symbol, days=1, interval="1m", include_ongoing=True
                    )

                    if df is not None and len(df) > 0:
                        candles = [
                            {
                                "timestamp": idx.isoformat(),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": float(row["volume"]),
                            }
                            for idx, row in df.tail(60).iterrows()
                        ]

                        with stream_lock:
                            price_streams[symbol] = deque(candles, maxlen=60)
                            last_price_update[symbol] = datetime.now().isoformat()

                except Exception:
                    pass

            # Broadcast prices to all clients (no gating)
            prices = _build_prices_snapshot()
            socketio.emit(
                "prices_update",
                {"prices": prices, "timestamp": datetime.now().isoformat()},
                to=None,
            )

            time.sleep(2)
        except Exception:
            time.sleep(10)


def refresh_predictions_background():
    """Continuously generate predictions in-memory and broadcast"""
    try:
        interval = int(
            os.getenv("PREDICTIONS_INTERVAL_SECONDS", "60")
        )  # Default to 60s
        global last_predictions, last_predictions_timestamp

        print(f"📊 Predictions thread started (interval: {interval}s)")
        print("🔮 Generating FIRST predictions immediately...")

        while True:
            try:
                print(
                    f"🔮 [{datetime.now().strftime('%H:%M:%S')}] Generating predictions..."
                )
                # Use simple non-cached predictor
                results = predict_all_models_simple()
                print(
                    f"📋 Raw results: {type(results)}, keys: {results.keys() if isinstance(results, dict) else 'N/A'}"
                )

                predictions_list = results.get("predictions", [])
                print(f"📋 Predictions list length: {len(predictions_list)}")

                if len(predictions_list) > 0:
                    print(
                        f"   First prediction keys: {list(predictions_list[0].keys()) if predictions_list else 'N/A'}"
                    )

                preds = _convert_predictions(predictions_list)
                last_predictions = preds
                last_predictions_timestamp = datetime.now().isoformat()

                print(f"✅ Generated {len(preds)} predictions")
                if len(preds) > 0:
                    print(f"   Sample keys: {list(preds.keys())[:3]}")
                else:
                    print("   ⚠️ WARNING: Empty predictions dict!")

                # Broadcast to all clients (even if none connected, emit is cheap)
                socketio.emit(
                    "predictions_update",
                    {"predictions": preds, "timestamp": last_predictions_timestamp},
                    to=None,
                )
                print(f"📡 Broadcasted predictions update")
            except Exception as e:
                # Log error but keep trying
                print(f"❌ Predictions error: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
            finally:
                print(f"💤 Sleeping for {interval} seconds...")
                time.sleep(max(1, interval))
    except Exception as e:
        print(f"❌ FATAL: Predictions thread crashed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")


def _convert_predictions(pred_list):
    """Convert predictions list to keyed dict"""
    result = {}
    try:
        for p in pred_list or []:
            # Handle both 'horizon_minutes' and 'horizon' fields
            horizon = p.get("horizon_minutes") or p.get("horizon", 0)
            key = f"{p['symbol']}_{p['interval']}_{horizon}m"
            result[key] = p
        print(
            f"🔄 Converted {len(pred_list)} predictions to {len(result)} dict entries"
        )
    except Exception as e:
        print(f"❌ Error converting predictions: {e}")
        print(f"   Sample prediction: {pred_list[0] if pred_list else 'empty'}")
    return result


# REST Endpoints (backward compatibility)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/cryptos", methods=["GET"])
def get_cryptos():
    return jsonify(
        {
            "success": True,
            "data": [{"symbol": v, "name": k} for k, v in SUPPORTED_CRYPTOS.items()],
        }
    )


@app.route("/api/prices", methods=["GET"])
def get_prices():
    return jsonify(
        {
            "success": True,
            "prices": _build_prices_snapshot(),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/wallet", methods=["GET"])
def get_wallet():
    try:
        account = binance_client.get_account()
        prices = _build_prices_snapshot()
        balances = []
        total_value = 0

        for bal in account.get("balances", []):
            asset = bal["asset"]
            total = float(bal["free"]) + float(bal["locked"])
            if total <= 0:
                continue

            price = (
                1.0
                if asset in {"USDT", "BUSD", "FDUSD"}
                else prices.get(f"{asset}USDT", {}).get("current_price", 0)
            )
            value = total * price
            total_value += value

            balances.append(
                {
                    "asset": asset,
                    "free": float(bal["free"]),
                    "locked": float(bal["locked"]),
                    "total": total,
                    "price_usdt": price,
                    "value_usdt": value,
                }
            )

        balances.sort(key=lambda x: x["value_usdt"], reverse=True)
        return jsonify(
            {
                "success": True,
                "data": {
                    "balances": balances,
                    "total_value_usdt": round(total_value, 2),
                    "asset_count": len(balances),
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    return jsonify({"success": True, "data": _get_predictions_snapshot()})


@app.route("/api/predictions/refresh", methods=["POST"])
def refresh_predictions():
    try:
        print("🔄 Manual prediction refresh triggered")
        results = predict_all_models_simple()
        print(
            f"📋 Results: {type(results)}, predictions count: {len(results.get('predictions', []))}"
        )
        preds = _convert_predictions(results.get("predictions", []))
        print(f"📋 Converted predictions count: {len(preds)}")

        # Update in-memory cache and broadcast
        global last_predictions, last_predictions_timestamp
        last_predictions = preds
        last_predictions_timestamp = datetime.now().isoformat()
        socketio.emit(
            "predictions_update",
            {"predictions": preds, "timestamp": last_predictions_timestamp},
            to=None,
        )
        print(f"✅ Manual refresh complete: {len(preds)} predictions")

        return jsonify(
            {
                "success": True,
                "data": preds,
                "metadata": {
                    "total": results.get("total_models"),
                    "success": results.get("successful_predictions"),
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# WebSocket Events
@socketio.on("connect")
def handle_connect():
    """Client connected"""
    connected_clients.add(request.sid)
    print(f"📱 Client connected | Total: {len(connected_clients)}")

    # Send initial data
    prices = _build_prices_snapshot()
    preds = _get_predictions_snapshot()
    emit(
        "initial_data",
        {
            "prices": prices,
            "predictions": preds,
            "timestamp": datetime.now().isoformat(),
        },
    )


@socketio.on("disconnect")
def handle_disconnect():
    """Client disconnected"""
    connected_clients.discard(request.sid)
    print(f"📱 Client disconnected | Total: {len(connected_clients)}")


@socketio.on("request_prices")
def handle_price_request():
    """Send prices on demand"""
    emit(
        "prices_update",
        {"prices": _build_prices_snapshot(), "timestamp": datetime.now().isoformat()},
    )


@socketio.on("request_predictions")
def handle_prediction_request():
    """Send predictions on demand"""
    emit(
        "predictions_update",
        {
            "predictions": _get_predictions_snapshot(),
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    print("=" * 60)
    print("CRYPTO TRADING BOT - WEBSOCKET BACKEND")
    print("=" * 60)

    # Start background threads
    print("🔄 Starting price update thread...")
    price_thread = threading.Thread(target=update_prices_continuously, daemon=True)
    price_thread.start()
    print("✅ Price update thread started")

    print("🔄 Starting predictions thread...")
    pred_thread = threading.Thread(target=refresh_predictions_background, daemon=True)
    pred_thread.start()
    print("✅ Predictions thread started")

    time.sleep(3)

    # Check if threads are alive
    print(f"📊 Price thread alive: {price_thread.is_alive()}")
    print(f"📊 Predictions thread alive: {pred_thread.is_alive()}")

    print(f"✅ WebSocket server running on http://0.0.0.0:5000")
    print(f"📡 WebSocket endpoint: ws://0.0.0.0:5000/socket.io/")
    # Run with threading (most compatible mode on Windows)
    socketio.run(
        app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
    )
