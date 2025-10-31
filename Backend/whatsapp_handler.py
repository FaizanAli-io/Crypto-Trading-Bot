"""
WhatsApp Business API Handler for Crypto Trading Bot
Handles incoming WhatsApp messages and sends crypto reports
"""

import os
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhatsAppHandler:
    def __init__(self):
        self.access_token = os.getenv("META_ACCESS_TOKEN")
        self.business_id = os.getenv("META_BUSINESS_ID")
        self.phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
        self.webhook_verify_token = os.getenv("WEBHOOK_VERIFY_TOKEN")

        # WhatsApp API URLs
        self.base_url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}"
        self.messages_url = f"{self.base_url}/messages"

        if not all([self.access_token, self.business_id, self.phone_number_id]):
            logger.warning(
                "⚠️  WhatsApp credentials not fully configured. Check your environment variables."
            )

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
            entry = data.get("entry", [])
            if not entry:
                return None

            changes = entry[0].get("changes", [])
            if not changes:
                return None

            value = changes[0].get("value", {})
            messages = value.get("messages", [])

            if not messages:
                return None

            message = messages[0]

            # Extract message details
            message_data = {
                "from": message.get("from"),
                "id": message.get("id"),
                "timestamp": message.get("timestamp"),
                "type": message.get("type"),
                "text": None,
            }

            # Handle text messages
            if message.get("type") == "text":
                message_data["text"] = (
                    message.get("text", {}).get("body", "").lower().strip()
                )

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
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "messaging_product": "whatsapp",
                "to": to_phone,
                "type": "text",
                "text": {"body": message_text},
            }

            response = requests.post(
                self.messages_url, headers=headers, json=payload, timeout=30
            )

            if response.status_code == 200:
                logger.info(f"✅ Message sent successfully to {to_phone}")
                return True
            else:
                logger.error(
                    f"❌ Failed to send message. Status: {response.status_code}, Response: {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            return False

    def get_next_prediction_schedule(self):
        """
        Calculate and return when the next predictions will run for each interval
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        current_minute = now.minute
        current_hour = now.hour
        
        # Calculate next run times
        # 5-minute predictions - every 5 minutes
        next_5min = 5 - (current_minute % 5)
        if next_5min == 0:
            next_5min = 5
        
        # 15-minute predictions - at :00, :15, :30, :45
        next_15min_options = [0, 15, 30, 45]
        next_15min = min([m for m in next_15min_options if m > current_minute] or [60])
        if next_15min == 60:
            next_15min = 60 - current_minute
        else:
            next_15min = next_15min - current_minute
        
        # 30-minute predictions - at :00 and :30
        next_30min_options = [0, 30]
        next_30min = min([m for m in next_30min_options if m > current_minute] or [60])
        if next_30min == 60:
            next_30min = 60 - current_minute
        else:
            next_30min = next_30min - current_minute
        
        # 1-hour predictions - every hour at :00
        next_1hour = 60 - current_minute
        
        # Format message
        message = "👋 *Welcome to Crypto Trading Bot!*\n\n"
        message += "🤖 I'm running automated predictions and will send you high-confidence signals!\n\n"
        message += "⏰ *Next Predictions:*\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        message += f"🕐 *5-min interval:* {next_5min} min\n"
        message += f"   • ETHUSDT\n\n"
        
        message += f"🕐 *15-min interval:* {next_15min} min\n"
        message += f"   • BTC, ETH, BNB, XRP, ADA\n"
        message += f"   • DOGE, SOL, DOT, LINK, LTC\n\n"
        
        message += f"🕐 *30-min interval:* {next_30min} min\n"
        message += f"   • All 10 cryptocurrencies\n\n"
        
        message += f"🕐 *1-hour interval:* {next_1hour} min\n"
        message += f"   • All 10 cryptocurrencies\n\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        message += "📊 *What I do:*\n"
        message += "• Monitor 10 major cryptocurrencies\n"
        message += "• Run AI predictions every 5-60 minutes\n"
        message += "• Send you alerts when confidence ≥ 80%\n"
        message += "• Include BUY/SELL signals with targets\n\n"
        
        message += f"🕐 *Current Time:* {now.strftime('%H:%M:%S')}\n\n"
        message += "💡 Just sit back and wait for high-confidence signals!\n\n"
        message += "⚠️ *Disclaimer:* Not financial advice. Trade at your own risk."
        
        return message
