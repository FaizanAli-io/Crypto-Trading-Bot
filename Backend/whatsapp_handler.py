"""
WhatsApp Business API Handler for Crypto Trading Bot
Handles incoming WhatsApp messages and sends crypto reports
"""

import requests
import json
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhatsAppHandler:
    def __init__(self):
        self.access_token = os.getenv('META_ACCESS_TOKEN')
        self.business_id = os.getenv('META_BUSINESS_ID')
        self.phone_number_id = os.getenv('META_PHONE_NUMBER_ID')
        self.webhook_verify_token = os.getenv('WEBHOOK_VERIFY_TOKEN')
        
        # WhatsApp API URLs
        self.base_url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}"
        self.messages_url = f"{self.base_url}/messages"
        
        if not all([self.access_token, self.business_id, self.phone_number_id]):
            logger.warning("⚠️  WhatsApp credentials not fully configured. Check your environment variables.")
    
    def verify_webhook(self, mode, token, challenge):
        """
        Verify webhook subscription with Meta
        """
        if mode == "subscribe" and token == self.webhook_verify_token:
            logger.info("✅ Webhook verified successfully")
            return challenge
        else:
            logger.warning("❌ Webhook verification failed")
            return None
    
    def process_webhook(self, data):
        """
        Process incoming WhatsApp webhook data
        Returns the message details if it's a valid text message
        """
        try:
            # Extract message data
            entry = data.get('entry', [])
            if not entry:
                return None
            
            changes = entry[0].get('changes', [])
            if not changes:
                return None
            
            value = changes[0].get('value', {})
            messages = value.get('messages', [])
            
            if not messages:
                return None
            
            message = messages[0]
            
            # Extract message details
            message_data = {
                'from': message.get('from'),
                'id': message.get('id'),
                'timestamp': message.get('timestamp'),
                'type': message.get('type'),
                'text': None
            }
            
            # Handle text messages
            if message.get('type') == 'text':
                message_data['text'] = message.get('text', {}).get('body', '').lower().strip()
            
            return message_data
            
        except Exception as e:
            logger.error(f"❌ Error processing webhook: {e}")
            return None
    
    def send_message(self, to_phone, message_text):
        """
        Send a text message to a WhatsApp number
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'messaging_product': 'whatsapp',
                'to': to_phone,
                'type': 'text',
                'text': {
                    'body': message_text
                }
            }
            
            response = requests.post(
                self.messages_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Message sent successfully to {to_phone}")
                return True
            else:
                logger.error(f"❌ Failed to send message. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            return False
    
    
    # Updated format_crypto_report function in whatsapp_handler.py
    def format_crypto_report(self, report_data, period):
        """
        Format crypto report data into a WhatsApp message with enhanced details
        """
        try:
            period_emoji = {
                'daily': '📅',
                'weekly': '📊', 
                'monthly': '📈'
            }
            
            message = f"{period_emoji.get(period, '📈')} *{period.upper()} CRYPTO REPORT*\n"
            message += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
            
            # Add global market data if available
            if report_data.get('global_data'):
                global_data = report_data['global_data']
                message += f"🌍 Global Market Cap: ${global_data.get('market_cap', 'N/A')}\n"
                message += f"💸 24h Volume: ${global_data.get('volume_24h', 'N/A')}\n"
            
            message += "\n*Cryptocurrencies*\n"
            message += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            if report_data.get('status') != 'success':
                message += "❌ Unable to fetch crypto data. Please try again later."
                return message
            
            currencies = report_data.get('currencies', [])
            if not currencies:
                message += "❌ No currency data available."
                return message
            
            for i, currency in enumerate(currencies, 1):
                symbol = currency.get('symbol', 'N/A')
                current_price = currency.get('current_price', 0)
                previous_price = currency.get('previous_price', 0)
                high_price = currency.get('high_price', 0)
                low_price = currency.get('low_price', 0)
                change_percent = currency.get('change_percent', 0)
                volatility = currency.get('volatility', 0)
                rsi = currency.get('rsi', 50)
                
                # Format prices
                if current_price >= 1:
                    current_str = f"{current_price:,.2f}"
                    previous_str = f"{previous_price:,.2f}"
                    high_str = f"{high_price:,.2f}"
                    low_str = f"{low_price:,.2f}"
                else:
                    current_str = f"{current_price:.6f}"
                    previous_str = f"{previous_price:.6f}"
                    high_str = f"{high_price:.6f}"
                    low_str = f"{low_price:.6f}"
                
                # Determine trend emoji
                if change_percent > 0:
                    trend = "🟢"
                    change_str = f"+{change_percent:.2f}%"
                elif change_percent < 0:
                    trend = "🔴"
                    change_str = f"{change_percent:.2f}%"
                else:
                    trend = "⚪"
                    change_str = "0.00%"
                
                # RSI Status
                if rsi >= 70:
                    rsi_status = "Overbought"
                elif rsi <= 30:
                    rsi_status = "Oversold"
                else:
                    rsi_status = "Neutral"
                
                message += f"{i}. *{symbol}*\n"
                message += f"   Current: ${current_str}\n"
                message += f"   Previous: ${previous_str}\n"
                message += f"   Change: {trend} {change_str}\n"
                message += f"   High: ${high_str} | Low: ${low_str}\n"
                message += f"   Volatility: ±{volatility:.1f}%\n"
                message += f"   RSI: {rsi:.0f} ({rsi_status})\n\n"
            
            message += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += "💡 *Note:* Prices are in USD\n"
            message += "🤖 Powered by Crypto Trading Bot"
            
            return message
            
        except Exception as e:
            logger.error(f"❌ Error formatting report: {e}")
            return "❌ Error generating crypto report. Please try again later."


    def is_valid_command(self, text):
        """
        Check if the received text is a valid command
        """
        valid_commands = ['daily', 'weekly', 'monthly']
        return text in valid_commands