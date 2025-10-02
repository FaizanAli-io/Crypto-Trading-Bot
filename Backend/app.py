

import os
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from flask import Flask, jsonify, request
from binance.exceptions import BinanceAPIException

"""
Crypto Trading Signals Flask App with ML Predictions
Integrates Chronos ML model for price predictions
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from flask import Flask, jsonify, request
from binance.exceptions import BinanceAPIException
from loguru import logger
# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)


# ========================
# ADD THESE IMPORTS AT THE TOP OF app.py
# ========================
from flask_swagger_ui import get_swaggerui_blueprint
from flask import send_from_directory
import json

# ========================
# ADD AFTER Flask app initialization
# ========================

# Swagger configuration
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI
API_URL = '/api/swagger.json'  # Path to swagger spec

# Create swagger blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Crypto Trading Bot API",
        'validatorUrl': None,
        'docExpansion': 'list',
        'defaultModelsExpandDepth': 3
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# ========================
# ADD SWAGGER SPEC ROUTE
# ========================

@app.route('/api/swagger.json')
def swagger_spec():
    """Serve Swagger/OpenAPI specification"""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Crypto Trading Bot API",
            "description": "AI-powered crypto trading signals with ML predictions using Amazon Chronos",
            "version": "2.0.0",
            "contact": {
                "name": "API Support",
                "email": "support@cryptobot.com"
            }
        },
        "servers": [
            {
                "url": "http://127.0.0.1:8000",
                "description": "Development server"
            }
        ],
        "tags": [
            {"name": "Health", "description": "Health check endpoints"},
            {"name": "Predictions", "description": "ML price prediction endpoints"},
            {"name": "Market Data", "description": "Real-time market data"},
            {"name": "Reports", "description": "Crypto market reports"},
            {"name": "WhatsApp", "description": "WhatsApp integration"},
            {"name": "Validation", "description": "Model accuracy validation and testing"}
        ],
        "paths": {
            "/health": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Health check",
                    "description": "Verify API is running and Binance connection status",
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "healthy"},
                                            "message": {"type": "string"},
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "binance_connected": {"type": "boolean"},
                                            "api_mode": {"type": "string", "enum": ["public", "read-only", "full"]}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict": {
                "get": {
                    "tags": ["Predictions"],
                    "summary": "Get crypto price prediction",
                    "description": "Predict cryptocurrency price for next 24 hours using AI",
                    "parameters": [
                        {
                            "name": "symbol",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "default": "BTC"},
                            "description": "Crypto symbol (BTC, ETH, BNB, etc.)"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Prediction generated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PredictionResponse"}
                                }
                            }
                        },
                        "400": {"description": "Invalid crypto symbol"},
                        "500": {"description": "Prediction failed"}
                    }
                },
                "post": {
                    "tags": ["Predictions"],
                    "summary": "Get crypto price prediction (POST)",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "symbol": {"type": "string", "example": "BTC"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Prediction generated"},
                        "400": {"description": "Invalid request"}
                    }
                }
            },
            "/predict/batch": {
                "post": {
                    "tags": ["Predictions"],
                    "summary": "Predict multiple cryptos at once",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "symbols": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "example": ["BTC", "ETH", "BNB"]
                                        }
                                    },
                                    "required": ["symbols"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Batch predictions generated",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "predictions": {"type": "object"},
                                            "errors": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict/model-info": {
                "get": {
                    "tags": ["Predictions"],
                    "summary": "Get ML model information",
                    "responses": {
                        "200": {
                            "description": "Model information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "model_info": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/ohlcv": {
                "get": {
                    "tags": ["Market Data"],
                    "summary": "Get OHLCV data",
                    "description": "Fetch historical OHLCV (Open, High, Low, Close, Volume) data",
                    "parameters": [
                        {
                            "name": "symbol",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "default": "BTCUSDT"},
                            "description": "Trading pair symbol"
                        },
                        {
                            "name": "interval",
                            "in": "query",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "default": "1h",
                                "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]
                            }
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer", "default": 100, "maximum": 1000}
                        }
                    ],
                    "responses": {
                        "200": {"description": "OHLCV data retrieved"},
                        "400": {"description": "Invalid parameters"}
                    }
                }
            },
            "/crypto-report": {
                "get": {
                    "tags": ["Reports"],
                    "summary": "Generate crypto market report",
                    "parameters": [
                        {
                            "name": "period",
                            "in": "query",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly"],
                                "default": "daily"
                            }
                        }
                    ],
                    "responses": {
                        "200": {"description": "Report generated"}
                    }
                }
            },
            "/top-cryptos": {
                "get": {
                    "tags": ["Market Data"],
                    "summary": "Get top 10 cryptocurrencies",
                    "description": "Current prices for top 10 cryptocurrencies by market cap",
                    "responses": {
                        "200": {
                            "description": "Top crypto prices",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "cryptos": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "symbol": {"type": "string"},
                                                        "price": {"type": "number"},
                                                        "pair": {"type": "string"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/webhook": {
                "get": {
                    "tags": ["WhatsApp"],
                    "summary": "Verify WhatsApp webhook",
                    "parameters": [
                        {"name": "hub.mode", "in": "query", "schema": {"type": "string"}},
                        {"name": "hub.verify_token", "in": "query", "schema": {"type": "string"}},
                        {"name": "hub.challenge", "in": "query", "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Webhook verified"},
                        "403": {"description": "Verification failed"}
                    }
                },
                "post": {
                    "tags": ["WhatsApp"],
                    "summary": "Handle WhatsApp messages",
                    "description": "Process incoming WhatsApp messages and commands",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Message processed"}
                    }
                }
            },
            "/validate/backtest": {
                "post": {
                    "tags": ["Validation"],
                    "summary": "Backtest model on historical data",
                    "description": "Test model accuracy by running predictions on past data and comparing with actual prices",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "symbol": {
                                            "type": "string",
                                            "example": "BTC",
                                            "description": "Crypto symbol to backtest"
                                        },
                                        "days": {
                                            "type": "integer",
                                            "default": 30,
                                            "example": 30,
                                            "description": "Number of days to backtest"
                                        },
                                        "prediction_horizon": {
                                            "type": "integer",
                                            "default": 24,
                                            "example": 24,
                                            "description": "Hours ahead to predict"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Backtest completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/BacktestResponse"}
                                }
                            }
                        },
                        "500": {"description": "Backtest failed"}
                    }
                }
            },
            "/validate/live": {
                "post": {
                    "tags": ["Validation"],
                    "summary": "Start live prediction validation",
                    "description": "Make a prediction and track it for validation in 24 hours",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "symbol": {
                                            "type": "string",
                                            "example": "BTC",
                                            "description": "Crypto symbol to predict"
                                        }
                                    },
                                    "required": ["symbol"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction recorded for validation",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "message": {"type": "string"},
                                            "validation_record": {
                                                "type": "object",
                                                "properties": {
                                                    "symbol": {"type": "string"},
                                                    "current_price": {"type": "number"},
                                                    "predicted_24h": {"type": "number"},
                                                    "prediction_time": {"type": "string"},
                                                    "validation_time": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/validate/check": {
                "get": {
                    "tags": ["Validation"],
                    "summary": "Check pending validations",
                    "description": "Validate predictions where 24 hours have passed",
                    "responses": {
                        "200": {
                            "description": "Validations checked",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "newly_validated": {"type": "integer"},
                                            "results": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "symbol": {"type": "string"},
                                                        "predicted_24h": {"type": "number"},
                                                        "actual_price": {"type": "number"},
                                                        "error_pct": {"type": "number"},
                                                        "direction_correct": {"type": "boolean"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/validate/summary": {
                "get": {
                    "tags": ["Validation"],
                    "summary": "Get validation summary",
                    "description": "Overall model accuracy metrics across all validated predictions",
                    "responses": {
                        "200": {
                            "description": "Validation summary",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ValidationSummary"}
                                }
                            }
                        }
                    }
                }
            },
            "/validate/accuracy-report": {
                "get": {
                    "tags": ["Validation"],
                    "summary": "Get detailed accuracy report",
                    "description": "Formatted report with model performance metrics",
                    "responses": {
                        "200": {
                            "description": "Accuracy report generated",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "report": {"type": "string"},
                                            "metrics": {"$ref": "#/components/schemas/ValidationSummary"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "PredictionResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "success"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "example": "BTC"},
                                "current_price": {"type": "number", "example": 65432.10},
                                "timestamp": {"type": "string", "format": "date-time"},
                                "predictions": {
                                    "type": "object",
                                    "properties": {
                                        "1h": {"type": "number"},
                                        "6h": {"type": "number"},
                                        "12h": {"type": "number"},
                                        "24h": {"type": "number"}
                                    }
                                },
                                "price_range": {
                                    "type": "object",
                                    "properties": {
                                        "24h_low": {"type": "number"},
                                        "24h_high": {"type": "number"}
                                    }
                                },
                                "analysis": {
                                    "type": "object",
                                    "properties": {
                                        "trend": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                                        "price_change_24h": {"type": "number"},
                                        "volatility_24h": {"type": "number"},
                                        "signal": {"type": "string", "enum": ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
                                    }
                                },
                                "technical_indicators": {
                                    "type": "object",
                                    "properties": {
                                        "rsi": {"type": "number"},
                                        "macd": {"type": "number"},
                                        "bb_position": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                },
                "BacktestResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "symbol": {"type": "string"},
                        "test_period_days": {"type": "integer"},
                        "total_predictions": {"type": "integer"},
                        "metrics": {"$ref": "#/components/schemas/ValidationSummary"}
                    }
                },
                "ValidationSummary": {
                    "type": "object",
                    "properties": {
                        "total_validations": {"type": "integer"},
                        "avg_error_pct": {"type": "number"},
                        "median_error_pct": {"type": "number"},
                        "predictions_within_5_pct": {"type": "number"},
                        "directional_accuracy_pct": {"type": "number"}
                    }
                }
            }
        }
    }
    return jsonify(spec)


# ========================
# ML PREDICTION SETUP
# ========================
from ml_predictor import CryptoPredictor
from whatsapp_handler import WhatsAppHandler
import config
# Add these imports to your existing app.py file
from whatsapp_handler import WhatsAppHandler
from flask import request
import requests







# Configure Binance client with API keys from environment variables
try:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")  # Optional for read-only operations

    if api_key:
        if api_secret:
            # Full client with both keys
            binance_client = Client(
                api_key=api_key, api_secret=api_secret, testnet=False
            )
            print("✅ Binance client initialized with API key and secret")
        else:
            # Read-only client with just API key
            binance_client = Client(api_key=api_key, testnet=False)
            print("✅ Binance client initialized with API key only (read-only mode)")
    else:
        # Public client (no authentication)
        binance_client = Client()
        print("⚠️  Binance client initialized in public mode (no API key)")

except Exception as e:
    print(f"❌ Error initializing Binance client: {e}")
    binance_client = None


# Initialize WhatsApp handler (add this after your existing binance_client initialization)
whatsapp_handler = WhatsAppHandler()
# Initialize ML predictor
ml_predictor = CryptoPredictor(binance_client)

# Route 1: Health Check
@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the application is running.
    Returns basic application status and timestamp.
    """
    return jsonify(
        {
            "status": "healthy",
            "message": "Crypto Trading Signals API is running",
            "timestamp": datetime.now().isoformat(),
            "binance_connected": binance_client is not None,
            "api_mode": (
                "read-only"
                if os.getenv("BINANCE_API_KEY") and not os.getenv("BINANCE_SECRET_KEY")
                else "full" if os.getenv("BINANCE_SECRET_KEY") else "public"
            ),
        }
    )


# Route 2: Binance Connection Test
@app.route("/binance/ping", methods=["GET"])
def binance_ping():
    """
    Test Binance API connection by pinging their servers.
    This endpoint verifies that our API connection is working.
    """
    if not binance_client:
        return (
            jsonify({"error": "Binance client not initialized. Check your API keys."}),
            500,
        )

    try:
        # Ping Binance servers
        ping_result = binance_client.ping()

        # Get server time to verify connection
        server_time = binance_client.get_server_time()

        return jsonify(
            {
                "status": "success",
                "message": "Successfully connected to Binance API",
                "ping_result": ping_result,
                "server_time": datetime.fromtimestamp(
                    server_time["serverTime"] / 1000
                ).isoformat(),
            }
        )

    except BinanceAPIException as e:
        return jsonify({"error": "Binance API Error", "message": str(e)}), 400

    except Exception as e:
        return jsonify({"error": "Connection Error", "message": str(e)}), 500


# Route 3: OHLCV Data Fetcher
@app.route("/ohlcv", methods=["GET"])
def get_ohlcv_data():
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a cryptocurrency pair.

    Query Parameters:
    - symbol: Trading pair symbol (e.g., BTCUSDT, ETHUSDT) - Required
    - interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d) - Default: 1h
    - limit: Number of data points to return (max 1000) - Default: 100

    Example: /ohlcv?symbol=BTCUSDT&interval=1h&limit=50
    """
    if not binance_client:
        return (
            jsonify({"error": "Binance client not initialized. Check your API keys."}),
            500,
        )

    # Get query parameters
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    interval = request.args.get("interval", "1h")
    limit = int(request.args.get("limit", 100))

    # Validate limit parameter
    if limit > 1000:
        return jsonify({"error": "Limit cannot exceed 1000"}), 400

    # Valid intervals for Binance
    valid_intervals = [
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    ]
    if interval not in valid_intervals:
        return (
            jsonify(
                {
                    "error": f'Invalid interval. Valid intervals: {", ".join(valid_intervals)}'
                }
            ),
            400,
        )

    try:
        # Fetch OHLCV data from Binance
        klines = binance_client.get_klines(
            symbol=symbol, interval=interval, limit=limit
        )

        # Format the data into a more readable structure
        formatted_data = []
        for kline in klines:
            formatted_data.append(
                {
                    "timestamp": datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "close_time": datetime.fromtimestamp(kline[6] / 1000).isoformat(),
                    "quote_volume": float(kline[7]),
                    "trades_count": kline[8],
                }
            )

        return jsonify(
            {
                "status": "success",
                "symbol": symbol,
                "interval": interval,
                "data_points": len(formatted_data),
                "data": formatted_data,
            }
        )

    except BinanceAPIException as e:
        return (
            jsonify(
                {"error": "Binance API Error", "message": str(e), "symbol": symbol}
            ),
            400,
        )

    except Exception as e:
        return (
            jsonify({"error": "Data Fetch Error", "message": str(e), "symbol": symbol}),
            500,
        )


# Route 4: Available Trading Pairs
@app.route("/symbols", methods=["GET"])
def get_trading_symbols():
    """
    Get list of available trading symbols from Binance.
    Returns popular USDT pairs for easy testing.
    """
    if not binance_client:
        return (
            jsonify({"error": "Binance client not initialized. Check your API keys."}),
            500,
        )

    try:
        # Get exchange info
        exchange_info = binance_client.get_exchange_info()

        # Filter for USDT pairs that are actively trading
        usdt_pairs = []
        for symbol in exchange_info["symbols"]:
            if (
                symbol["quoteAsset"] == "USDT"
                and symbol["status"] == "TRADING"
                and symbol["baseAsset"]
                in [
                    "BTC",
                    "ETH",
                    "BNB",
                    "ADA",
                    "DOT",
                    "LINK",
                    "LTC",
                    "XRP",
                    "MATIC",
                    "SOL",
                ]
            ):
                usdt_pairs.append(
                    {
                        "symbol": symbol["symbol"],
                        "baseAsset": symbol["baseAsset"],
                        "quoteAsset": symbol["quoteAsset"],
                    }
                )

        return jsonify(
            {
                "status": "success",
                "total_pairs": len(usdt_pairs),
                "popular_usdt_pairs": usdt_pairs,
            }
        )

    except Exception as e:
        return jsonify({"error": "Error fetching symbols", "message": str(e)}), 500


# Error handler for 404 (page not found)
@app.errorhandler(404)
def not_found(error):
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "message": "Please check the URL and try again",
                "available_endpoints": [
                    "/health",
                    "/binance/ping",
                    "/ohlcv?symbol=BTCUSDT&interval=1h&limit=100",
                    "/symbols",
                ],
            }
        ),
        404,
    )


##############################################################################################


# Top 10 cryptocurrencies by market cap
TOP_10_CRYPTOS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "SOLUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "LTCUSDT",
]

# NEW ML PREDICTION ROUTES
# ========================

@app.route("/predict", methods=["GET", "POST"])
def predict_crypto():
    """
    Predict crypto price for next 24 hours
    
    GET params: ?symbol=BTC or ?symbol=BTCUSDT
    POST body: {"symbol": "BTC"} or {"symbol": "BTCUSDT"}
    
    Returns: Full prediction with analysis
    """
    try:
        # Get symbol from query or body
        if request.method == "GET":
            symbol = request.args.get("symbol", "BTC").upper()
        else:
            data = request.get_json()
            symbol = data.get("symbol", "BTC").upper()
        
        # Convert short name to trading pair
        if symbol in config.SUPPORTED_CRYPTOS:
            trading_pair = config.SUPPORTED_CRYPTOS[symbol]
            crypto_name = symbol
        elif symbol.endswith("USDT"):
            trading_pair = symbol
            crypto_name = symbol.replace("USDT", "")
        else:
            return jsonify({"error": f"Unsupported crypto: {symbol}"}), 400
        
        # Make prediction
        logger.info(f"Prediction request for {crypto_name}")
        prediction = ml_predictor.predict(trading_pair, crypto_name)
        
        return jsonify({
            "status": "success",
            "data": prediction
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict multiple cryptos at once
    
    POST body: {"symbols": ["BTC", "ETH", "BNB"]}
    """
    try:
        data = request.get_json()
        symbols = data.get("symbols", [])
        
        if not symbols:
            return jsonify({"error": "No symbols provided"}), 400
        
        predictions = {}
        errors = {}
        
        for symbol in symbols:
            try:
                # Convert to trading pair
                if symbol in config.SUPPORTED_CRYPTOS:
                    trading_pair = config.SUPPORTED_CRYPTOS[symbol]
                    crypto_name = symbol
                else:
                    trading_pair = symbol
                    crypto_name = symbol.replace("USDT", "")
                
                prediction = ml_predictor.predict(trading_pair, crypto_name)
                predictions[symbol] = prediction
                
            except Exception as e:
                errors[symbol] = str(e)
                logger.error(f"Error predicting {symbol}: {e}")
        
        return jsonify({
            "status": "success" if predictions else "error",
            "predictions": predictions,
            "errors": errors if errors else None
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict/model-info", methods=["GET"])
def model_info():
    """Get ML model information and status"""
    try:
        info = ml_predictor.get_model_info()
        return jsonify({
            "status": "success",
            "model_info": info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict/clear-cache", methods=["POST"])
def clear_cache():
    """Clear prediction cache"""
    try:
        ml_predictor.cache = {}
        ml_predictor._save_cache()
        return jsonify({
            "status": "success",
            "message": "Cache cleared successfully"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ========================
# WHATSAPP WEBHOOK (Updated with Predictions)
# ========================

@app.route("/webhook", methods=["GET"])
def webhook_verify():
    """Verify WhatsApp webhook"""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    
    result = whatsapp_handler.verify_webhook(mode, token, challenge)
    if result:
        return result, 200
    else:
        return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook_handler():
    """
    Handle WhatsApp messages - now with ML predictions!
    
    Commands:
    - daily/weekly/monthly: Get report (existing)
    - predict BTC: Get ML price prediction
    - predict: Get prediction for BTC (default)
    """
    try:
        data = request.get_json()
        message_data = whatsapp_handler.process_webhook(data)
        
        if not message_data or not message_data.get("text"):
            return jsonify({"status": "ignored"}), 200
        
        message_id = message_data["id"]
        user_phone = message_data["from"]
        message_text = message_data["text"].lower().strip()
        
        # Deduplication
        if message_id in processed_messages:
            logger.info(f"⏭️  Skipping duplicate message ID: {message_id}")
            return jsonify({"status": "duplicate_ignored"}), 200
        
        processed_messages.add(message_id)
        if len(processed_messages) > 1000:
            processed_messages.clear()
        
        logger.info(f"📱 Received: {message_text} from {user_phone}")
        
        # ========================
        # NEW: Handle prediction commands
        # ========================
        if message_text.startswith("predict"):
            try:
                # Parse command: "predict BTC" or just "predict"
                parts = message_text.split()
                if len(parts) > 1:
                    crypto_symbol = parts[1].upper()
                else:
                    crypto_symbol = "BTC"  # Default to BTC
                
                # Validate symbol
                if crypto_symbol not in config.SUPPORTED_CRYPTOS:
                    error_msg = f"❌ Sorry, {crypto_symbol} is not supported.\n\n"
                    error_msg += "Supported cryptos:\n"
                    error_msg += ", ".join(config.SUPPORTED_CRYPTOS.keys())
                    whatsapp_handler.send_message(user_phone, error_msg)
                    return jsonify({"status": "unsupported_crypto"}), 200
                
                # Send "analyzing" message
                analyzing_msg = f"🔮 Analyzing {crypto_symbol}...\n⏳ This may take 10-15 seconds..."
                whatsapp_handler.send_message(user_phone, analyzing_msg)
                
                # Make prediction
                trading_pair = config.SUPPORTED_CRYPTOS[crypto_symbol]
                prediction = ml_predictor.predict(trading_pair, crypto_symbol)
                
                # Format for WhatsApp
                formatted_message = ml_predictor.format_prediction_for_whatsapp(prediction)
                
                # Send prediction
                success = whatsapp_handler.send_message(user_phone, formatted_message)
                
                if success:
                    return jsonify({
                        "status": "success",
                        "message": f"Prediction sent for {crypto_symbol}"
                    }), 200
                else:
                    return jsonify({"status": "error", "message": "Failed to send"}), 500
                    
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                error_msg = f"❌ Sorry, I couldn't generate prediction for {crypto_symbol}.\n\n"
                error_msg += "Please try again in a few moments."
                whatsapp_handler.send_message(user_phone, error_msg)
                return jsonify({"status": "error", "message": str(e)}), 500
        
        # ========================
        # EXISTING: Handle report commands
        # ========================
        elif whatsapp_handler.is_valid_command(message_text):
            try:
                from app import generate_crypto_report  # Your existing function
                report_data = generate_crypto_report(message_text)
                formatted_message = whatsapp_handler.format_crypto_report(report_data, message_text)
                success = whatsapp_handler.send_message(user_phone, formatted_message)
                
                if success:
                    return jsonify({
                        "status": "success",
                        "message": f"{message_text.title()} report sent"
                    }), 200
                else:
                    return jsonify({"status": "error"}), 500
                    
            except Exception as e:
                logger.error(f"Report error: {e}")
                error_msg = "❌ Sorry, couldn't generate report. Please try again."
                whatsapp_handler.send_message(user_phone, error_msg)
                return jsonify({"status": "error", "message": str(e)}), 500
        
        # ========================
        # Help message for invalid commands
        # ========================
        else:
            help_message = whatsapp_handler.get_help_message_with_predictions()
            whatsapp_handler.send_message(user_phone, help_message)
            return jsonify({"status": "help_sent"}), 200
            
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Route 7: Generate Crypto Report API
@app.route("/crypto-report", methods=["GET"])
def crypto_report_api():
    """
    Generate crypto report for specified period
    Query Parameters:
    - period: daily, weekly, or monthly (default: daily)
    """
    period = request.args.get("period", "daily").lower()

    if period not in ["daily", "weekly", "monthly"]:
        return jsonify({"error": "Invalid period. Use daily, weekly, or monthly"}), 400

    try:
        report_data = generate_crypto_report(period)
        return jsonify(report_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Updated generate_crypto_report function in app.py
def generate_crypto_report(period):
    """
    Generate enhanced crypto report with additional metrics
    """
    if not binance_client:
        raise Exception("Binance client not initialized")

    # Determine the interval and lookback based on period
    period_config = {
        "daily": {"interval": "1h", "lookback_hours": 25, "rsi_period": 14},
        "weekly": {"interval": "1d", "lookback_days": 8, "rsi_period": 7},
        "monthly": {"interval": "1d", "lookback_days": 31, "rsi_period": 14},
    }

    config = period_config.get(period)
    if not config:
        raise Exception("Invalid period specified")

    currencies_data = []

    # Get global market data (simplified)
    try:
        global_stats = get_global_market_data()
    except:
        global_stats = None

    for symbol in TOP_10_CRYPTOS:
        try:
            # Get historical data with more candles for calculations
            limit = config.get("lookback_days", config.get("lookback_hours", 25))
            klines = binance_client.get_klines(
                symbol=symbol,
                interval=config["interval"],
                limit=limit + config["rsi_period"],  # Extra data for RSI calculation
            )

            if len(klines) < 2:
                continue

            # Get current and previous prices
            current_candle = klines[-1]  # Most recent
            if period == "daily":
                previous_candle = klines[-25]  # 24 hours ago (hourly data)
            else:
                previous_candle = klines[-limit] if len(klines) >= limit else klines[0]

            current_price = float(current_candle[4])  # Close price
            previous_price = float(previous_candle[4])  # Close price
            high_price = float(current_candle[2])  # High price
            low_price = float(current_candle[3])  # Low price

            # Calculate percentage change
            change_percent = ((current_price - previous_price) / previous_price) * 100

            # Calculate volatility (high-low range as percentage of current price)
            volatility = ((high_price - low_price) / current_price) * 100

            # Calculate RSI
            rsi = calculate_rsi(
                [float(k[4]) for k in klines[-config["rsi_period"] - 1 :]],
                config["rsi_period"],
            )

            # Get 24h high/low for better context
            ticker_24h = binance_client.get_ticker(symbol=symbol)
            high_24h = float(ticker_24h["highPrice"])
            low_24h = float(ticker_24h["lowPrice"])

            currencies_data.append(
                {
                    "symbol": symbol.replace("USDT", ""),
                    "current_price": current_price,
                    "previous_price": previous_price,
                    "high_price": high_24h,  # Use 24h high for consistency
                    "low_price": low_24h,  # Use 24h low for consistency
                    "change_percent": change_percent,
                    "volatility": volatility,
                    "rsi": rsi,
                    "timestamp": datetime.fromtimestamp(
                        current_candle[0] / 1000
                    ).isoformat(),
                }
            )

        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            continue

    # Sort by percentage change in ASCENDING order (lowest to highest)
    currencies_data.sort(key=lambda x: x["change_percent"])

    return {
        "status": "success",
        "period": period,
        "generated_at": datetime.now().isoformat(),
        "total_currencies": len(currencies_data),
        "global_data": global_stats,
        "currencies": currencies_data[:10],  # Top 10
    }


# Helper function to calculate RSI
def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    """
    try:
        if len(prices) < period + 1:
            return 50  # Default neutral RSI

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    except:
        return 50  # Default neutral RSI


# Helper function to get global market data (simplified)
def get_global_market_data():
    """
    Get global cryptocurrency market data
    """
    try:
        # This is a simplified version - in production, you'd use CoinGecko API
        # For now, we'll calculate approximate values from our top currencies

        total_prices = []
        for symbol in TOP_10_CRYPTOS[:5]:  # Use top 5 for approximation
            try:
                ticker = binance_client.get_ticker(symbol=symbol)
                current_price = float(ticker["lastPrice"])
                price_change = float(ticker["priceChangePercent"])
                total_prices.append({"price": current_price, "change": price_change})
            except:
                continue

        if total_prices:
            avg_change = sum([p["change"] for p in total_prices]) / len(total_prices)
            return {
                "market_cap": "2.13T",  # Static for now
                "volume_24h": "89.2B",  # Static for now
                "change_24h": f"{avg_change:+.1f}%",
            }

        return None

    except:
        return None


# Route 8: Send Test WhatsApp Message
@app.route("/test-whatsapp", methods=["POST"])
def test_whatsapp():
    """
    Test endpoint to send a WhatsApp message
    Body: {"phone": "+1234567890", "message": "Test message"}
    """
    try:
        data = request.get_json()
        phone = data.get("phone")
        message = data.get("message", "Test message from Crypto Bot!")

        if not phone:
            return jsonify({"error": "Phone number is required"}), 400

        success = whatsapp_handler.send_message(phone, message)

        if success:
            return jsonify(
                {"status": "success", "message": "Test message sent successfully"}
            )
        else:
            return (
                jsonify({"status": "error", "message": "Failed to send test message"}),
                500,
            )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Route 9: Get Top Cryptocurrencies
@app.route("/top-cryptos", methods=["GET"])
def get_top_cryptos():
    """
    Get current prices for top 10 cryptocurrencies
    """
    if not binance_client:
        return jsonify({"error": "Binance client not initialized"}), 500

    try:
        crypto_prices = []

        for symbol in TOP_10_CRYPTOS:
            try:
                ticker = binance_client.get_symbol_ticker(symbol=symbol)
                crypto_prices.append(
                    {
                        "symbol": symbol.replace("USDT", ""),
                        "price": float(ticker["price"]),
                        "pair": symbol,
                    }
                )
            except Exception as e:
                print(f"❌ Error fetching price for {symbol}: {e}")
                continue

        return jsonify(
            {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "total_cryptos": len(crypto_prices),
                "cryptos": crypto_prices,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

"""
Add these routes to your app.py for model validation
"""

from model_validator import ModelValidator

# Initialize validator (add after ml_predictor initialization)
model_validator = ModelValidator(binance_client)

# ========================
# VALIDATION ROUTES
# ========================

@app.route("/validate/backtest", methods=["POST"])
def backtest_model():
    """
    Backtest model on historical data
    
    POST body: {
        "symbol": "BTC",
        "days": 30,
        "prediction_horizon": 24
    }
    
    Returns: Detailed backtest results with accuracy metrics
    """
    try:
        data = request.get_json()
        symbol = data.get("symbol", "BTC").upper()
        days = data.get("days", 30)
        horizon = data.get("prediction_horizon", 24)
        
        # Convert to trading pair
        if symbol in config.SUPPORTED_CRYPTOS:
            trading_pair = config.SUPPORTED_CRYPTOS[symbol]
        else:
            trading_pair = symbol
        
        logger.info(f"Starting backtest for {symbol}...")
        
        # Run backtest
        results = model_validator.backtest(
            symbol=trading_pair,
            days=days,
            prediction_horizon=horizon
        )
        
        return jsonify({
            "status": "success",
            "backtest_results": results
        })
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/validate/live", methods=["POST"])
def start_live_validation():
    """
    Make a prediction and track it for validation
    
    POST body: {
        "symbol": "BTC"
    }
    
    Returns: Prediction that will be validated in 24 hours
    """
    try:
        data = request.get_json()
        symbol = data.get("symbol", "BTC").upper()
        
        # Convert to trading pair
        if symbol in config.SUPPORTED_CRYPTOS:
            trading_pair = config.SUPPORTED_CRYPTOS[symbol]
        else:
            trading_pair = symbol
        
        # Start live validation
        validation_record = model_validator.live_validation(trading_pair)
        
        return jsonify({
            "status": "success",
            "message": "Prediction recorded. Check back in 24 hours!",
            "validation_record": validation_record
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/validate/check", methods=["GET"])
def check_validations():
    """
    Check pending validations and validate if time has passed
    
    Returns: List of newly validated predictions
    """
    try:
        validated = model_validator.check_pending_validations()
        
        return jsonify({
            "status": "success",
            "newly_validated": len(validated),
            "results": validated
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/validate/summary", methods=["GET"])
def validation_summary():
    """
    Get overall validation summary with accuracy metrics
    
    Returns: Aggregate statistics of all validations
    """
    try:
        summary = model_validator.get_validation_summary()
        
        return jsonify({
            "status": "success",
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/validate/accuracy-report", methods=["GET"])
def accuracy_report():
    """
    Get detailed accuracy report with visualizations
    
    Returns: HTML report with charts (optional)
    """
    try:
        summary = model_validator.get_validation_summary()
        
        # Format as readable report
        if summary.get('message'):
            return jsonify({
                "status": "info",
                "message": summary['message']
            })
        
        report = f"""
        📊 MODEL ACCURACY REPORT
        ========================
        
        Total Predictions Validated: {summary['total_validations']}
        
        Price Accuracy:
        - Average Error: {summary['avg_error_pct']:.2f}%
        - Median Error: {summary['median_error_pct']:.2f}%
        - Within 5% of Actual: {summary['predictions_within_5_pct']:.1f}%
        
        Direction Accuracy:
        - Correct Direction: {summary['directional_accuracy_pct']:.1f}%
        
        Model Performance: {'🟢 GOOD' if summary['directional_accuracy_pct'] > 60 else '🟡 FAIR' if summary['directional_accuracy_pct'] > 50 else '🔴 POOR'}
        """
        
        return jsonify({
            "status": "success",
            "report": report,
            "metrics": summary
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Crypto Trading Bot with ML Predictions...")
    print("\n📊 Available endpoints:")
    print("   • GET  /health - Health check")
    print("   • GET  /predict?symbol=BTC - Predict crypto price")
    print("   • POST /predict - Predict with JSON body")
    print("   • POST /predict/batch - Predict multiple cryptos")
    print("   • GET  /predict/model-info - Get model info")
    print("   • POST /predict/clear-cache - Clear prediction cache")
    print("\n📚 API Documentation:")
    print("   • Swagger UI: http://127.0.0.1:8000/api/docs")
    print("   • OpenAPI Spec: http://127.0.0.1:8000/api/swagger.json")
    print("\n💬 WhatsApp commands:")
    print("   • predict BTC - Get BTC price prediction")
    print("   • predict ETH - Get ETH price prediction")
    print("   • daily/weekly/monthly - Get market report")
    print("\n🔗 Try: http://127.0.0.1:8000/predict?symbol=BTC")
    print()
    
    app.run(debug=True, host="127.0.0.1", port=8000)
