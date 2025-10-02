"""
Data Collector: Fetch historical crypto data from Binance
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
import config

class DataCollector:
    def __init__(self, binance_client=None):
        """
        Initialize data collector with Binance client
        """
        self.client = binance_client
        if not self.client:
            # Initialize public client if not provided
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_SECRET_KEY")
            if api_key:
                self.client = Client(api_key, api_secret)
            else:
                self.client = Client()  # Public client
                logger.warning("Using public Binance client (no API keys)")
    
    def fetch_historical_data(self, symbol, days=30, interval="1h"):
        """
        Fetch historical OHLCV data from Binance
        
        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            days (int): Number of days of historical data
            interval (str): Candle interval (1h, 4h, 1d, etc.)
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            logger.info(f"Fetching {days} days of {interval} data for {symbol}")
            
            # Calculate how many candles we need
            hours_per_candle = self._interval_to_hours(interval)
            total_candles = int((days * 24) / hours_per_candle)
            
            # Binance limit is 1000 per request
            all_klines = []
            remaining = total_candles
            end_time = None
            
            while remaining > 0:
                limit = min(remaining, 1000)
                
                # Fetch klines
                if end_time:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        endTime=end_time
                    )
                else:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit
                    )
                
                if not klines:
                    break
                
                all_klines = klines + all_klines  # Prepend older data
                remaining -= len(klines)
                end_time = klines[0][0] - 1  # Go back in time
                
                logger.info(f"Fetched {len(klines)} candles, {remaining} remaining")
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(all_klines)
            
            # Save raw data
            self._save_raw_data(df, symbol)
            
            logger.info(f"✅ Collected {len(df)} data points for {symbol}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def _interval_to_hours(self, interval):
        """Convert interval string to hours"""
        mapping = {
            "1m": 1/60, "3m": 3/60, "5m": 5/60, "15m": 15/60, "30m": 30/60,
            "1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12,
            "1d": 24, "3d": 72, "1w": 168
        }
        return mapping.get(interval, 1)
    
    def _klines_to_dataframe(self, klines):
        """
        Convert Binance klines to pandas DataFrame
        """
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in price_cols:
            df[col] = df[col].astype(float)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Keep only essential columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def _save_raw_data(self, df, symbol):
        """Save raw data to CSV"""
        try:
            filepath = config.DATA_DIR / "raw" / f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filepath)
            logger.info(f"Saved raw data to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save raw data: {e}")
    
    def get_latest_price(self, symbol):
        """
        Get current price for a symbol
        
        Returns:
            float: Current price
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            return None
    
    def get_realtime_data(self, symbol, limit=168):
        """
        Get most recent data for prediction (last 168 hours = 7 days)
        
        Args:
            symbol (str): Trading pair
            limit (int): Number of recent candles
            
        Returns:
            pd.DataFrame: Recent price data
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval="1h",
                limit=limit
            )
            df = self._klines_to_dataframe(klines)
            return df
        except Exception as e:
            logger.error(f"Error fetching realtime data: {e}")
            raise