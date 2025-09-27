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



##############################################################################################
# Add these imports to your existing app.py file
from whatsapp_handler import WhatsAppHandler
from flask import request
import requests

# Initialize WhatsApp handler (add this after your existing binance_client initialization)
whatsapp_handler = WhatsAppHandler()

# Top 10 cryptocurrencies by market cap
TOP_10_CRYPTOS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
                  'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT']

# Route 5: WhatsApp Webhook Verification
@app.route('/webhook', methods=['GET'])
def webhook_verify():
    """
    Verify WhatsApp webhook with Meta
    """
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token') 
    challenge = request.args.get('hub.challenge')
    
    result = whatsapp_handler.verify_webhook(mode, token, challenge)
    if result:
        return result, 200
    else:
        return 'Forbidden', 403


# Route 6: WhatsApp Webhook Handler
@app.route('/webhook', methods=['POST'])
def webhook_handler():
    """
    Handle incoming WhatsApp messages
    """
    try:
        data = request.get_json()
        
        # Process the webhook data
        message_data = whatsapp_handler.process_webhook(data)
        
        if not message_data or not message_data.get('text'):
            return jsonify({'status': 'ignored'}), 200
        
        user_phone = message_data['from']
        message_text = message_data['text']
        
        print(f"📱 Received message from {user_phone}: {message_text}")
        
        # Check if it's a valid command
        if whatsapp_handler.is_valid_command(message_text):
            # Generate crypto report
            try:
                report_data = generate_crypto_report(message_text)
                formatted_message = whatsapp_handler.format_crypto_report(report_data, message_text)
                
                # Send the report back to user
                success = whatsapp_handler.send_message(user_phone, formatted_message)
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'{message_text.title()} report sent successfully'
                    }), 200
                else:
                    return jsonify({
                        'status': 'error', 
                        'message': 'Failed to send report'
                    }), 500
                    
            except Exception as e:
                error_msg = "❌ Sorry, I couldn't generate the crypto report right now. Please try again later."
                whatsapp_handler.send_message(user_phone, error_msg)
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        else:
            # Send help message for invalid commands
            help_message = whatsapp_handler.get_help_message()
            whatsapp_handler.send_message(user_phone, help_message)
            
            return jsonify({
                'status': 'help_sent',
                'message': 'Help message sent for invalid command'
            }), 200
            
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Webhook processing failed'
        }), 500


# Route 7: Generate Crypto Report API
@app.route('/crypto-report', methods=['GET'])
def crypto_report_api():
    """
    Generate crypto report for specified period
    Query Parameters:
    - period: daily, weekly, or monthly (default: daily)
    """
    period = request.args.get('period', 'daily').lower()
    
    if period not in ['daily', 'weekly', 'monthly']:
        return jsonify({
            'error': 'Invalid period. Use daily, weekly, or monthly'
        }), 400
    
    try:
        report_data = generate_crypto_report(period)
        return jsonify(report_data)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Updated generate_crypto_report function in app.py
def generate_crypto_report(period):
    """
    Generate enhanced crypto report with additional metrics
    """
    if not binance_client:
        raise Exception('Binance client not initialized')
    
    # Determine the interval and lookback based on period
    period_config = {
        'daily': {'interval': '1h', 'lookback_hours': 25, 'rsi_period': 14},
        'weekly': {'interval': '1d', 'lookback_days': 8, 'rsi_period': 7}, 
        'monthly': {'interval': '1d', 'lookback_days': 31, 'rsi_period': 14}
    }
    
    config = period_config.get(period)
    if not config:
        raise Exception('Invalid period specified')
    
    currencies_data = []
    
    # Get global market data (simplified)
    try:
        global_stats = get_global_market_data()
    except:
        global_stats = None
    
    for symbol in TOP_10_CRYPTOS:
        try:
            # Get historical data with more candles for calculations
            limit = config.get('lookback_days', config.get('lookback_hours', 25))
            klines = binance_client.get_klines(
                symbol=symbol,
                interval=config['interval'],
                limit=limit + config['rsi_period']  # Extra data for RSI calculation
            )
            
            if len(klines) < 2:
                continue
            
            # Get current and previous prices
            current_candle = klines[-1]  # Most recent
            if period == 'daily':
                previous_candle = klines[-25]  # 24 hours ago (hourly data)
            else:
                previous_candle = klines[-limit] if len(klines) >= limit else klines[0]
            
            current_price = float(current_candle[4])  # Close price
            previous_price = float(previous_candle[4])  # Close price
            high_price = float(current_candle[2])  # High price
            low_price = float(current_candle[3])   # Low price
            
            # Calculate percentage change
            change_percent = ((current_price - previous_price) / previous_price) * 100
            
            # Calculate volatility (high-low range as percentage of current price)
            volatility = ((high_price - low_price) / current_price) * 100
            
            # Calculate RSI
            rsi = calculate_rsi([float(k[4]) for k in klines[-config['rsi_period']-1:]], config['rsi_period'])
            
            # Get 24h high/low for better context
            ticker_24h = binance_client.get_ticker(symbol=symbol)
            high_24h = float(ticker_24h['highPrice'])
            low_24h = float(ticker_24h['lowPrice'])
            
            currencies_data.append({
                'symbol': symbol.replace('USDT', ''),
                'current_price': current_price,
                'previous_price': previous_price,
                'high_price': high_24h,  # Use 24h high for consistency
                'low_price': low_24h,    # Use 24h low for consistency
                'change_percent': change_percent,
                'volatility': volatility,
                'rsi': rsi,
                'timestamp': datetime.fromtimestamp(current_candle[0] / 1000).isoformat()
            })
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            continue
    
    # Sort by percentage change in ASCENDING order (lowest to highest)
    currencies_data.sort(key=lambda x: x['change_percent'])
    
    return {
        'status': 'success',
        'period': period,
        'generated_at': datetime.now().isoformat(),
        'total_currencies': len(currencies_data),
        'global_data': global_stats,
        'currencies': currencies_data[:10]  # Top 10
    }


# Helper function to calculate RSI
def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    """
    try:
        if len(prices) < period + 1:
            return 50  # Default neutral RSI
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
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
                current_price = float(ticker['lastPrice'])
                price_change = float(ticker['priceChangePercent'])
                total_prices.append({'price': current_price, 'change': price_change})
            except:
                continue
        
        if total_prices:
            avg_change = sum([p['change'] for p in total_prices]) / len(total_prices)
            return {
                'market_cap': '2.13T',  # Static for now
                'volume_24h': '89.2B',  # Static for now
                'change_24h': f"{avg_change:+.1f}%"
            }
        
        return None
        
    except:
        return None


# Route 8: Send Test WhatsApp Message
@app.route('/test-whatsapp', methods=['POST'])
def test_whatsapp():
    """
    Test endpoint to send a WhatsApp message
    Body: {"phone": "+1234567890", "message": "Test message"}
    """
    try:
        data = request.get_json()
        phone = data.get('phone')
        message = data.get('message', 'Test message from Crypto Bot!')
        
        if not phone:
            return jsonify({
                'error': 'Phone number is required'
            }), 400
        
        success = whatsapp_handler.send_message(phone, message)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Test message sent successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to send test message'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Route 9: Get Top Cryptocurrencies
@app.route('/top-cryptos', methods=['GET'])
def get_top_cryptos():
    """
    Get current prices for top 10 cryptocurrencies
    """
    if not binance_client:
        return jsonify({
            'error': 'Binance client not initialized'
        }), 500
    
    try:
        crypto_prices = []
        
        for symbol in TOP_10_CRYPTOS:
            try:
                ticker = binance_client.get_symbol_ticker(symbol=symbol)
                crypto_prices.append({
                    'symbol': symbol.replace('USDT', ''),
                    'price': float(ticker['price']),
                    'pair': symbol
                })
            except Exception as e:
                print(f"❌ Error fetching price for {symbol}: {e}")
                continue
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_cryptos': len(crypto_prices),
            'cryptos': crypto_prices
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500






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
    print("   • http://127.0.0.1:8000/health")
    print("   • http://127.0.0.1:8000/binance/ping")
    print("   • http://127.0.0.1:8000/ohlcv?symbol=BTCUSDT&interval=1h&limit=10")
    print("   • http://127.0.0.1:8000/ohlcv?symbol=ETHUSDT&interval=15m&limit=50")
    print("   • http://127.0.0.1:8000/symbols")
    print()
    
    # Run the Flask app
    app.run(
        debug=True,
        host='127.0.0.1',
        port=8000
    )