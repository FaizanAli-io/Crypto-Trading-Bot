"""
Scheduled Crypto Predictor with WhatsApp Alerts
Runs predictions at specific intervals and sends high-confidence signals via WhatsApp
"""
import os
import time
import schedule
from datetime import datetime, timedelta
from loguru import logger
from binance.client import Client

from xg_predict import SignalPredictor
from whatsapp_handler import WhatsAppHandler

# Configuration
# WhatsApp numbers to send alerts to (can be a single number or list)
WHATSAPP_NUMBERS = [
    "+923312844594",
    "+923332275445",
    "+966560771267",
    "+923132680496"
    # Add more numbers here:
    # "+1234567890",
    # "+9876543210",
]

HIGH_CONFIDENCE_THRESHOLD = 0.5  # 75% confidence for WhatsApp alerts

# Symbols to monitor
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", 
           "DOGEUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT"]

class ScheduledPredictor:
    def __init__(self):
        """Initialize predictor with Binance client and WhatsApp handler"""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")
        
        if api_key and api_secret:
            self.binance_client = Client(api_key, api_secret)
            logger.info("✅ Binance client initialized with API keys")
        else:
            self.binance_client = Client()
            logger.warning("⚠️ Using public Binance client (no API keys)")
        
        self.predictor = SignalPredictor(self.binance_client)
        self.whatsapp = WhatsAppHandler()
        
        # Track last predictions to avoid duplicates
        self.last_predictions = {}
        
        # Record bot start time to ignore old signals
        self.bot_start_time = datetime.now()
        logger.info(f"🕐 Bot started at: {self.bot_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def predict_and_alert(self, symbol, interval, horizon_minutes, days=10):
        """
        Run prediction for a symbol and send WhatsApp alert if confidence is high
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Prediction horizon in minutes
            days: Days of historical data to use
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔮 Predicting {symbol} - {interval} interval, {horizon_minutes}min horizon")
            logger.info(f"{'='*60}")
            
            # Run prediction
            result = self.predictor.predict_signal(
                symbol=symbol,
                interval=interval,
                horizon_minutes=horizon_minutes,
                custom_confidence=HIGH_CONFIDENCE_THRESHOLD,
                days=days
            )
            
            # Check if we should send WhatsApp alert
            if result['confidence'] >= HIGH_CONFIDENCE_THRESHOLD :
                self._send_whatsapp_alert(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error predicting {symbol} ({interval}): {e}")
            return None
    
    def _send_whatsapp_alert(self, result):
        """Send WhatsApp alert for high-confidence signal to all numbers"""
        try:
            # Check if signal is fresh (generated within last 2 minutes)
            signal_time = datetime.fromisoformat(result['timestamp'])
            current_time = datetime.now()
            signal_age_minutes = (current_time - signal_time).total_seconds() / 60
            
            # Ignore signals generated before bot started
            if signal_time < self.bot_start_time:
                logger.warning(f"⏭️ Skipping signal generated before bot start ({signal_time.strftime('%H:%M:%S')})")
                return
            
            # Ignore signals older than 2 minutes
            if signal_age_minutes > 2:
                logger.warning(f"⏭️ Skipping old signal (generated {signal_age_minutes:.1f} minutes ago)")
                return
            
            # Create unique key to avoid duplicate alerts
            alert_key = f"{result['symbol']}_{result['interval']}_{result['horizon_minutes']}_{result['signal']}"
            
            # Check if we already sent this alert recently (within horizon time)
            if alert_key in self.last_predictions:
                last_time = self.last_predictions[alert_key]
                time_diff = (current_time - last_time).total_seconds() / 60
                
                if time_diff < result['horizon_minutes']:
                    logger.info(f"⏭️ Skipping duplicate alert (sent {time_diff:.0f}min ago)")
                    return
            
            # Format WhatsApp message
            message = self._format_alert_message(result)
            
            # Send WhatsApp message to all numbers
            success_count = 0
            failed_count = 0
            
            for phone_number in WHATSAPP_NUMBERS:
                try:
                    success = self.whatsapp.send_message(phone_number, message)
                    if success:
                        success_count += 1
                        logger.info(f"✅ Alert sent to {phone_number}")
                    else:
                        failed_count += 1
                        logger.error(f"❌ Failed to send alert to {phone_number}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Error sending to {phone_number}: {e}")
            
            # Log summary
            logger.info(f"📊 Alert delivery: {success_count} sent, {failed_count} failed for {result['symbol']} {result['signal']}")
            
            # Mark as sent if at least one succeeded
            if success_count > 0:
                self.last_predictions[alert_key] = current_time
                
        except Exception as e:
            logger.error(f"❌ Error sending WhatsApp alert: {e}")
    
    def _format_alert_message(self, result):
        """Format prediction result into WhatsApp message"""
        signal_emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "HOLD": "⚪"
        }
        
        emoji = signal_emoji.get(result['signal'], "⚪")
        
        message = f"🚨 *HIGH CONFIDENCE SIGNAL* 🚨\n\n"
        message += f"{emoji} *{result['signal']}* {result['symbol']}\n"
        message += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        message += f"💰 *Current Price:* ${result['current_price']:,.2f}\n"
        message += f"📊 *Confidence:* {result['confidence']:.1%}\n"
        message += f"📈 *Direction:* {result['predicted_direction']}\n"
        message += f"⏰ *Timeframe:* {result['interval']} ({result['horizon_minutes']}min ahead)\n\n"
        
        message += f"💡 *Reason:*\n{result['reason']}\n\n"
        
        # Add SMC context if available
        if result.get('smc_enabled') and result.get('smc_context'):
            ctx = result['smc_context']
            message += f"📊 *SMC Analysis:*\n"
            message += f"Bias: {ctx['bias']} (Strength: {ctx['strength']})\n"
            
            if result.get('smc_adjustment'):
                message += f"{result['smc_adjustment']}\n"
            
            message += "\n"
        
        # Add price targets if available
        if result.get('smc_targets') and result['signal'] != "HOLD":
            targets = result['smc_targets']
            if targets.get('targets'):
                message += f"🎯 *Targets:*\n"
                for i, target in enumerate(targets['targets'][:2], 1):
                    message += f"  T{i}: ${target['level']:,.2f}\n"
                
                if targets.get('stop_loss'):
                    message += f"🛑 *Stop Loss:* ${targets['stop_loss']:,.2f}\n"
                
                if targets.get('risk_reward'):
                    message += f"⚖️ *R/R:* 1:{targets['risk_reward']}\n"
                
                message += "\n"
        
        message += f"⏳ *Valid Until:* {datetime.fromisoformat(result['valid_until']).strftime('%H:%M:%S')}\n"
        message += f"🕐 *Generated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "⚠️ *Not financial advice. Trade at your own risk.*"
        
        return message
    
    # ========== SCHEDULED JOBS ==========
    
    def predict_5min_models(self):
        """Run predictions for 5-minute interval models"""
        logger.info("\n🕐 Running 5-minute interval predictions...")
        
        # Only ETHUSDT has 5min model based on the files
        for symbol in ["ETHUSDT"]:
            self.predict_and_alert(symbol, "5m", 5, days=7)
    
    def predict_15min_models(self):
        """Run predictions for 15-minute interval models"""
        logger.info("\n🕐 Running 15-minute interval predictions...")
        
        for symbol in SYMBOLS:
            self.predict_and_alert(symbol, "15m", 15, days=10)
    
    def predict_30min_models(self):
        """Run predictions for 30-minute interval models"""
        logger.info("\n🕐 Running 30-minute interval predictions...")
        
        for symbol in SYMBOLS:
            self.predict_and_alert(symbol, "30m", 30, days=10)
    
    def predict_1hour_models(self):
        """Run predictions for 1-hour interval models"""
        logger.info("\n🕐 Running 1-hour interval predictions...")
        
        for symbol in SYMBOLS:
            self.predict_and_alert(symbol, "1h", 60, days=15)
    
    def setup_schedule(self):
        """Setup all scheduled jobs"""
        logger.info("⚙️ Setting up prediction schedule...")
        
        # 5-minute predictions - every 5 minutes
        schedule.every(5).minutes.do(self.predict_5min_models)
        logger.info("✅ 5-min predictions: Every 5 minutes")
        
        # 15-minute predictions - at :00, :15, :30, :45
        schedule.every().hour.at(":00").do(self.predict_15min_models)
        schedule.every().hour.at(":15").do(self.predict_15min_models)
        schedule.every().hour.at(":30").do(self.predict_15min_models)
        schedule.every().hour.at(":45").do(self.predict_15min_models)
        logger.info("✅ 15-min predictions: At :00, :15, :30, :45")
        
        # 30-minute predictions - at :00 and :30
        schedule.every().hour.at(":00").do(self.predict_30min_models)
        schedule.every().hour.at(":30").do(self.predict_30min_models)
        logger.info("✅ 30-min predictions: At :00 and :30")
        
        # 1-hour predictions - every hour at :00
        schedule.every().hour.at(":00").do(self.predict_1hour_models)
        logger.info("✅ 1-hour predictions: Every hour at :00")
        
        logger.info(f"\n📱 WhatsApp alerts enabled for confidence >= {HIGH_CONFIDENCE_THRESHOLD:.0%}")
        logger.info(f"📞 Alert numbers: {len(WHATSAPP_NUMBERS)} recipient(s)")
        for i, number in enumerate(WHATSAPP_NUMBERS, 1):
            logger.info(f"   {i}. {number}")
        logger.info(f"💱 Monitoring {len(SYMBOLS)} symbols")
        logger.info("\n🚀 Scheduler ready! Waiting for next scheduled time...")
    
    def run(self):
        """Start the scheduler"""
        self.setup_schedule()
        
        # Send startup notification to all numbers
        startup_msg = f"🤖 *Crypto Predictor Started*\n\n"
        startup_msg += f"Monitoring {len(SYMBOLS)} symbols\n"
        startup_msg += f"Alert threshold: {HIGH_CONFIDENCE_THRESHOLD:.0%}\n"
        startup_msg += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        logger.info("📤 Sending startup notifications...")
        for phone_number in WHATSAPP_NUMBERS:
            try:
                self.whatsapp.send_message(phone_number, startup_msg)
                logger.info(f"✅ Startup notification sent to {phone_number}")
            except Exception as e:
                logger.error(f"❌ Failed to send startup notification to {phone_number}: {e}")
        
        # Run scheduler loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                logger.info("\n⏹️ Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("🚀 SCHEDULED CRYPTO PREDICTOR WITH WHATSAPP ALERTS")
    logger.info("="*60)
    
    predictor = ScheduledPredictor()
    predictor.run()


if __name__ == "__main__":
    main()
