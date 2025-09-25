"""
Crypto Trading Signals Flask App
A beginner-friendly Flask application that connects to Binance API
to fetch cryptocurrency market data and trading signals.
"""

from flask import Flask, jsonify, request
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure Binance client with API keys from environment variables
try:
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET_KEY')  # Optional for read-only operations
    
    if api_key:
        if api_secret:
            # Full client with both keys
            binance_client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=False
            )
            print("✅ Binance client initialized with API key and secret")
        else:
            # Read-only client with just API key
            binance_client = Client(
                api_key=api_key,
                testnet=False
            )
            print("✅ Binance client initialized with API key only (read-only mode)")
    else:
        # Public client (no authentication)
        binance_client = Client()
        print("⚠️  Binance client initialized in public mode (no API key)")
        
except Exception as e:
    print(f"❌ Error initializing Binance client: {e}")
    binance_client = None


# Route 1: Health Check
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the application is running.
    Returns basic application status and timestamp.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Crypto Trading Signals API is running',
        'timestamp': datetime.now().isoformat(),
        'binance_connected': binance_client is not None,
        'api_mode': 'read-only' if os.getenv('BINANCE_API_KEY') and not os.getenv('BINANCE_SECRET_KEY') else 'full' if os.getenv('BINANCE_SECRET_KEY') else 'public'
    })


# Route 2: Binance Connection Test
@app.route('/binance/ping', methods=['GET'])
def binance_ping():
    """
    Test Binance API connection by pinging their servers.
    This endpoint verifies that our API connection is working.
    """
    if not binance_client:
        return jsonify({
            'error': 'Binance client not initialized. Check your API keys.'
        }), 500
    
    try:
        # Ping Binance servers
        ping_result = binance_client.ping()
        
        # Get server time to verify connection
        server_time = binance_client.get_server_time()
        
        return jsonify({
            'status': 'success',
            'message': 'Successfully connected to Binance API',
            'ping_result': ping_result,
            'server_time': datetime.fromtimestamp(server_time['serverTime'] / 1000).isoformat()
        })
    
    except BinanceAPIException as e:
        return jsonify({
            'error': 'Binance API Error',
            'message': str(e)
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Connection Error',
            'message': str(e)
        }), 500


# Route 3: OHLCV Data Fetcher
@app.route('/ohlcv', methods=['GET'])
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
        return jsonify({
            'error': 'Binance client not initialized. Check your API keys.'
        }), 500
    
    # Get query parameters
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    interval = request.args.get('interval', '1h')
    limit = int(request.args.get('limit', 100))
    
    # Validate limit parameter
    if limit > 1000:
        return jsonify({
            'error': 'Limit cannot exceed 1000'
        }), 400
    
    # Valid intervals for Binance
    valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if interval not in valid_intervals:
        return jsonify({
            'error': f'Invalid interval. Valid intervals: {", ".join(valid_intervals)}'
        }), 400
    
    try:
        # Fetch OHLCV data from Binance
        klines = binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        # Format the data into a more readable structure
        formatted_data = []
        for kline in klines:
            formatted_data.append({
                'timestamp': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': datetime.fromtimestamp(kline[6] / 1000).isoformat(),
                'quote_volume': float(kline[7]),
                'trades_count': kline[8]
            })
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'interval': interval,
            'data_points': len(formatted_data),
            'data': formatted_data
        })
    
    except BinanceAPIException as e:
        return jsonify({
            'error': 'Binance API Error',
            'message': str(e),
            'symbol': symbol
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Data Fetch Error',
            'message': str(e),
            'symbol': symbol
        }), 500


# Route 4: Available Trading Pairs
@app.route('/symbols', methods=['GET'])
def get_trading_symbols():
    """
    Get list of available trading symbols from Binance.
    Returns popular USDT pairs for easy testing.
    """
    if not binance_client:
        return jsonify({
            'error': 'Binance client not initialized. Check your API keys.'
        }), 500
    
    try:
        # Get exchange info
        exchange_info = binance_client.get_exchange_info()
        
        # Filter for USDT pairs that are actively trading
        usdt_pairs = []
        for symbol in exchange_info['symbols']:
            if (symbol['quoteAsset'] == 'USDT' and 
                symbol['status'] == 'TRADING' and
                symbol['baseAsset'] in ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP', 'MATIC', 'SOL']):
                usdt_pairs.append({
                    'symbol': symbol['symbol'],
                    'baseAsset': symbol['baseAsset'],
                    'quoteAsset': symbol['quoteAsset']
                })
        
        return jsonify({
            'status': 'success',
            'total_pairs': len(usdt_pairs),
            'popular_usdt_pairs': usdt_pairs
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Error fetching symbols',
            'message': str(e)
        }), 500


# Error handler for 404 (page not found)
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the URL and try again',
        'available_endpoints': [
            '/health',
            '/binance/ping', 
            '/ohlcv?symbol=BTCUSDT&interval=1h&limit=100',
            '/symbols'
        ]
    }), 404


# Main execution
if __name__ == '__main__':
    print("🚀 Starting Crypto Trading Signals API...")
    print("📊 Available endpoints:")
    print("   • GET /health - Health check")
    print("   • GET /binance/ping - Test Binance connection") 
    print("   • GET /ohlcv - Get OHLCV data")
    print("   • GET /symbols - Get available trading pairs")
    print()
    print("🔗 Example URLs:")
    print("   • http://127.0.0.1:5000/health")
    print("   • http://127.0.0.1:5000/binance/ping")
    print("   • http://127.0.0.1:5000/ohlcv?symbol=BTCUSDT&interval=1h&limit=10")
    print("   • http://127.0.0.1:5000/ohlcv?symbol=ETHUSDT&interval=15m&limit=50")
    print("   • http://127.0.0.1:5000/symbols")
    print()
    
    # Run the Flask app
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000
    )