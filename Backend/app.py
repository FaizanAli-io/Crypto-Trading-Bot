"""
Crypto Trading Bot - Main Application
Runs scheduled predictions with WhatsApp alerts
"""
import os
import threading
from flask import Flask, request, jsonify
from loguru import logger
from dotenv import load_dotenv

from scheduled_predictor import ScheduledPredictor
from whatsapp_handler import WhatsAppHandler
import config

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize WhatsApp handler
whatsapp_handler = WhatsAppHandler()

# Message deduplication
processed_messages = set()

# Global scheduler instance
scheduler = None


@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Crypto Trading Bot is running",
        "scheduler_active": scheduler is not None
    }

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
    Handle WhatsApp messages - Tell users when next prediction will run
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
        
        # Get next prediction schedule info
        if scheduler:
            schedule_info = whatsapp_handler.get_next_prediction_schedule()
        else:
            schedule_info = "⚠️ Scheduler is not running yet. Please wait..."
        
        # Send schedule information
        success = whatsapp_handler.send_message(user_phone, schedule_info)
        
        if success:
            return jsonify({"status": "success", "message": "Schedule info sent"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to send"}), 500
            
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def run_scheduler():
    """Run the scheduled predictor in a separate thread"""
    global scheduler
    try:
        logger.info("🚀 Starting scheduled predictor...")
        scheduler = ScheduledPredictor()
        scheduler.run()
    except Exception as e:
        logger.error(f"❌ Scheduler error: {e}")


if __name__ == "__main__":
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    logger.info("="*60)
    logger.info("🤖 CRYPTO TRADING BOT STARTED")
    logger.info("="*60)
    logger.info("📊 Scheduler: Running in background")
    logger.info("🌐 Flask API: http://127.0.0.1:8000")
    logger.info("="*60)
    
    # Run Flask app
    app.run(host="0.0.0.0", port=8000, debug=False)
