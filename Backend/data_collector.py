"""
Data Collector: Fetch historical crypto data from Binance
Enhanced with ongoing candle data
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
import config
import time

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
        
        # Convert types - convert UTC timestamps to local timezone
        from datetime import datetime as dt
        local_tz = dt.now().astimezone().tzinfo
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(local_tz).dt.tz_localize(None)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True).dt.tz_convert(local_tz).dt.tz_localize(None)
        
        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in price_cols:
            df[col] = df[col].astype(float)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Keep only essential columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    

    
    def get_historical_data_by_date_range(self, symbol, start_date, days, interval="1h"):
        """
        Get historical data between specific dates (going backward from start_date)
        
        Args:
            symbol (str): Trading pair (e.g., "ETHUSDT")
            start_date (str or datetime): End date - format "YYYY-MM-DD" or datetime object
                                        Data collection ends at this date
            days (int): Number of days to go backward from start_date
                    Example: start_date="2025-08-01", days=365 
                    -> fetches from 2024-08-01 to 2025-08-01
            interval (str): Candle interval (default "1h")
            
        Returns:
            pd.DataFrame: Historical price data for the date range
            
        Example:
            # Get data from Aug 1, 2024 to Aug 1, 2025
            df = collector.get_historical_data_by_date_range(
                symbol="ETHUSDT",
                start_date="2025-08-01",
                days=365
            )
        """
        try:
            # Parse start_date
            if isinstance(start_date, str):
                end_date = datetime.strptime(start_date, "%Y-%m-%d")
            elif isinstance(start_date, datetime):
                end_date = start_date
            else:
                raise ValueError("start_date must be string 'YYYY-MM-DD' or datetime object")
            
            # Calculate begin_date (going backward)
            begin_date = end_date - timedelta(days=days)
            
            logger.info(f"Fetching {symbol} data from {begin_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"Period: {days} days")
            
            # Convert to milliseconds timestamp for Binance API
            start_timestamp = int(begin_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            # Determine interval duration in milliseconds
            interval_map = {
                "1m": 60 * 1000,
                "3m": 3 * 60 * 1000,
                "5m": 5 * 60 * 1000,
                "15m": 15 * 60 * 1000,
                "30m": 30 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "2h": 2 * 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "6h": 6 * 60 * 60 * 1000,
                "8h": 8 * 60 * 60 * 1000,
                "12h": 12 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000,
                "3d": 3 * 24 * 60 * 60 * 1000,
                "1w": 7 * 24 * 60 * 60 * 1000,
            }
            
            interval_ms = interval_map.get(interval)
            if not interval_ms:
                raise ValueError(f"Unsupported interval: {interval}")
            
            # Calculate expected number of candles
            total_ms = end_timestamp - start_timestamp
            expected_candles = total_ms // interval_ms
            
            logger.info(f"Expected ~{expected_candles:,} candles")
            
            # Binance API limit is 1000 candles per request
            # We need to fetch in batches
            all_klines = []
            current_start = start_timestamp
            batch_count = 0
            
            while current_start < end_timestamp:
                batch_count += 1
                
                # Fetch batch
                try:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=current_start,
                        endTime=end_timestamp,
                        limit=1000  # Binance max limit
                    )
                    
                    if not klines:
                        logger.warning(f"No more data available after {datetime.fromtimestamp(current_start/1000)}")
                        break
                    
                    all_klines.extend(klines)
                    
                    # Update start time for next batch (last candle timestamp + 1 interval)
                    last_timestamp = klines[-1][0]
                    current_start = last_timestamp + interval_ms
                    
                    logger.info(f"Batch {batch_count}: Fetched {len(klines)} candles | "
                            f"Total: {len(all_klines):,} | "
                            f"Last: {datetime.fromtimestamp(last_timestamp/1000).strftime('%Y-%m-%d %H:%M')}")
                    
                    # Rate limiting - avoid hitting Binance API limits
                    if batch_count % 3 == 0:
                        import time
                        time.sleep(0.5)  # Small delay every 3 requests
                    
                except Exception as e:
                    logger.error(f"Error fetching batch {batch_count}: {e}")
                    break
            
            if not all_klines:
                logger.error(f"No data retrieved for {symbol}")
                return None
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(all_klines)
            
            # Filter to exact date range (in case we got extra data)
            df = df[(df.index >= begin_date) & (df.index <= end_date)]
            

            
            logger.info(f"✅ Successfully collected {len(df):,} candles for {symbol}")
            logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data by date range: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def get_realtime_data(self, symbol, days=7, include_ongoing=False, interval="1h"):
        """
        Get most recent data for prediction INCLUDING ongoing candle
        Supports fetching more than 1000 candles using batch logic
        
        Args:
            symbol (str): Trading pair
            days (int): Number of days of historical data to fetch
            include_ongoing (bool): Include current incomplete candle
            interval (str): Candle interval (default "1h")
            
        Returns:
            pd.DataFrame: Recent price data (calculated candles + 1 row if ongoing included)
        """
        try:
            # Calculate interval duration in minutes
            interval_minutes_map = {
                "1m": 1,
                "3m": 3,
                "5m": 5,
                "15m": 15,
                "30m": 30,
                "1h": 60,
                "2h": 120,
                "4h": 240,
                "6h": 360,
                "8h": 480,
                "12h": 720,
                "1d": 1440,
                "3d": 4320,
                "1w": 10080,
            }
            
            interval_minutes = interval_minutes_map.get(interval, 60)  # Default to 1h
            
            # Calculate total number of candles needed
            total_minutes = days * 24 * 60
            limit = total_minutes // interval_minutes
            
            logger.info(f"Fetching {days} days of {interval} candles for {symbol} (total: {limit} candles)")
            
            # If limit <= 1000, use simple single request
            if limit <= 1000:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                df = self._klines_to_dataframe(klines)
            
            # If limit > 1000, use batch fetching
            else:
                logger.info(f"Limit > 1000, using batch fetching...")
                all_klines = []
                remaining = limit
                batch_count = 0
                
                # Calculate interval duration in milliseconds
                interval_map = {
                    "1m": 60 * 1000,
                    "3m": 3 * 60 * 1000,
                    "5m": 5 * 60 * 1000,
                    "15m": 15 * 60 * 1000,
                    "30m": 30 * 60 * 1000,
                    "1h": 60 * 60 * 1000,
                    "2h": 2 * 60 * 60 * 1000,
                    "4h": 4 * 60 * 60 * 1000,
                    "6h": 6 * 60 * 60 * 1000,
                    "8h": 8 * 60 * 60 * 1000,
                    "12h": 12 * 60 * 60 * 1000,
                    "1d": 24 * 60 * 60 * 1000,
                    "3d": 3 * 24 * 60 * 60 * 1000,
                    "1w": 7 * 24 * 60 * 60 * 1000,
                }
                
                interval_ms = interval_map.get(interval, 60 * 60 * 1000)  # Default to 1h
                
                # Get current timestamp
                current_time = int(datetime.now().timestamp() * 1000)
                
                # Calculate start timestamp (go backward from now)
                start_time = current_time - (limit * interval_ms)
                
                # Fetch in batches
                current_end = current_time
                
                while remaining > 0:
                    batch_count += 1
                    batch_limit = min(remaining, 1000)
                    
                    # Calculate start time for this batch
                    batch_start = current_end - (batch_limit * interval_ms)
                    
                    try:
                        klines = self.client.get_klines(
                            symbol=symbol,
                            interval=interval,
                            startTime=batch_start,
                            endTime=current_end,
                            limit=batch_limit
                        )
                        
                        if not klines:
                            logger.warning(f"No more data available at batch {batch_count}")
                            break
                        
                        # Insert at beginning (we're going backward in time)
                        all_klines = klines + all_klines
                        
                        remaining -= len(klines)
                        
                        # Update end time for next batch (earliest timestamp from this batch)
                        current_end = klines[0][0] - interval_ms
                        
                        logger.info(f"Batch {batch_count}: Fetched {len(klines)} candles | "
                                f"Remaining: {remaining} | "
                                f"Total: {len(all_klines)}")
                        
                        # If we got fewer candles than requested, we've hit the data limit
                        if len(klines) < batch_limit:
                            logger.warning(f"Reached data limit, got {len(klines)} < {batch_limit}")
                            break
                        
                        # Rate limiting
                        if batch_count % 3 == 0:
                            import time
                            time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error in batch {batch_count}: {e}")
                        break
                
                if not all_klines:
                    raise ValueError(f"No data retrieved for {symbol}")
                
                df = self._klines_to_dataframe(all_klines)
                
                # Keep only the requested number of most recent candles
                df = df.tail(limit)
                
                logger.info(f"Fetched {len(df)} candles across {batch_count} batches")
            
            # Add ongoing candle if requested
            if include_ongoing:
                ongoing_candle = self._get_ongoing_candle(symbol)
                if ongoing_candle is not None:
                    # Check if ongoing candle timestamp already exists in df
                    if ongoing_candle.index[0] not in df.index:
                        # Append ongoing candle
                        df = pd.concat([df, ongoing_candle])
                        logger.info(f"Added ongoing candle at {ongoing_candle.index[0]}")
            
            logger.info(f"✅ Collected {len(df)} data points for {symbol}")
            logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching realtime data: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise

    def _get_ongoing_candle(self, symbol):
        """
        Construct ongoing (incomplete) hourly candle from recent trades
        
        Returns:
            pd.DataFrame: Single row with ongoing candle data
        """
        try:
            # Get current time
            now = datetime.now()
            current_hour_start = now.replace(minute=0, second=0, microsecond=0)
            
            # Method 1: Use Binance aggregated trades (more accurate)
            # Get all trades since the start of current hour
            start_time_ms = int(current_hour_start.timestamp() * 1000)
            
            trades = self.client.get_aggregate_trades(
                symbol=symbol,
                startTime=start_time_ms,
                limit=1000
            )
            
            if not trades:
                logger.warning("No trades found for ongoing candle")
                return None
            
            # Extract prices and volumes
            prices = [float(t['p']) for t in trades]
            volumes = [float(t['q']) for t in trades]
            
            # Construct OHLCV
            open_price = prices[0]  # First trade price
            high_price = max(prices)
            low_price = min(prices)
            close_price = prices[-1]  # Last trade price (current price)
            volume = sum(volumes)
            
            # Create DataFrame row
            ongoing_df = pd.DataFrame({
                'open': [open_price],
                'high': [high_price],
                'low': [low_price],
                'close': [close_price],
                'volume': [volume]
            }, index=[current_hour_start])
            
            ongoing_df.index.name = 'timestamp'
            
            return ongoing_df
            
        except Exception as e:
            logger.warning(f"Could not fetch ongoing candle: {e}")
            return None
