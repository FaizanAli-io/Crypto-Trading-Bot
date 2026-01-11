"""
Data Collector: Fetch historical crypto data from Binance
Enhanced with ongoing candle data
"""

import os
import time
import pandas as pd
from loguru import logger
from binance.client import Client
from datetime import datetime, timedelta


class DataCollector:
    def __init__(self, binance_client=None):
        """
        Initialize data collector with Binance client
        """
        self.client = binance_client
        if not self.client:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_SECRET_KEY")
            self.client = Client(api_key, api_secret) if api_key else Client()
            if not api_key:
                logger.warning("Using public Binance client (no API keys)")

    def _interval_to_hours(self, interval):
        """Convert interval string to hours"""
        mapping = {
            "1m": 1 / 60,
            "3m": 3 / 60,
            "5m": 5 / 60,
            "15m": 15 / 60,
            "30m": 30 / 60,
            "1h": 1,
            "2h": 2,
            "4h": 4,
            "6h": 6,
            "8h": 8,
            "12h": 12,
            "1d": 24,
            "3d": 72,
            "1w": 168,
        }
        return mapping.get(interval, 1)

    def _klines_to_dataframe(self, klines):
        """Convert Binance klines to pandas DataFrame"""
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        local_tz = datetime.now().astimezone().tzinfo
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            .dt.tz_convert(local_tz)
            .dt.tz_localize(None)
        )
        df["close_time"] = (
            pd.to_datetime(df["close_time"], unit="ms", utc=True)
            .dt.tz_convert(local_tz)
            .dt.tz_localize(None)
        )

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    def get_historical_data_by_date_range(
        self, symbol, start_date, days, interval="1h"
    ):
        """Fetch historical data for a symbol between start_date and start_date + days"""
        try:
            if isinstance(start_date, str):
                begin_date = datetime.strptime(start_date, "%Y-%m-%d")
            elif isinstance(start_date, datetime):
                begin_date = start_date
            else:
                raise ValueError(
                    "start_date must be string 'YYYY-MM-DD' or datetime object"
                )

            end_date = begin_date + timedelta(days=days)
            start_ts = int(begin_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

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

            all_klines, current_start, batch_count = [], start_ts, 0
            while current_start < end_ts:
                batch_count += 1
                try:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=current_start,
                        endTime=end_ts,
                        limit=1000,
                    )
                    if not klines:
                        break
                    all_klines.extend(klines)
                    current_start = klines[-1][0] + interval_ms
                    if batch_count % 3 == 0:
                        time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error fetching batch {batch_count}: {e}")
                    break

            if not all_klines:
                return None

            df = self._klines_to_dataframe(all_klines)
            return df[(df.index >= begin_date) & (df.index <= end_date)]
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None

    def get_realtime_data(self, symbol, days=7, include_ongoing=False, interval="1h"):
        """Fetch most recent candles for prediction, optionally including ongoing candle"""
        try:
            interval_min_map = {
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
            interval_minutes = interval_min_map.get(interval, 60)
            limit = (days * 24 * 60) // interval_minutes

            df = None  # ⚠ Ensure df exists

            # Fetch from Binance
            if limit <= 1000:
                klines = self.client.get_klines(
                    symbol=symbol, interval=interval, limit=limit
                )
                df = self._klines_to_dataframe(klines)
            else:
                df = self._fetch_batches(symbol, interval, limit)

            # Add ongoing candle if requested
            if include_ongoing:
                ongoing = self._get_ongoing_candle(symbol)
                if ongoing is not None and ongoing.index[0] not in df.index:
                    df = pd.concat([df, ongoing])

            return df

        except Exception as e:
            logger.error(f"Error fetching realtime data: {e}")
            raise

    def _fetch_batches(self, symbol, interval, limit):
        """Fetch more than 1000 candles in batches"""
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
        interval_ms = interval_map.get(interval, 60 * 60 * 1000)
        current_time = int(datetime.now().timestamp() * 1000)
        remaining = limit
        current_end = current_time
        all_klines, batch_count = [], 0

        while remaining > 0:
            batch_count += 1
            batch_limit = min(remaining, 1000)
            batch_start = current_end - batch_limit * interval_ms
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=batch_start,
                    endTime=current_end,
                    limit=batch_limit,
                )
                if not klines:
                    break
                all_klines = klines + all_klines
                remaining -= len(klines)
                current_end = klines[0][0] - interval_ms
                if len(klines) < batch_limit:
                    break
                if batch_count % 3 == 0:
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in batch {batch_count}: {e}")
                break

        if not all_klines:
            raise ValueError(f"No data retrieved for {symbol}")

        df = self._klines_to_dataframe(all_klines)

        return df.tail(limit)

    def _get_ongoing_candle(self, symbol):
        """Construct ongoing candle from recent trades"""
        try:
            now = datetime.now()
            current_hour_start = now.replace(minute=0, second=0, microsecond=0)
            start_ms = int(current_hour_start.timestamp() * 1000)

            trades = self.client.get_aggregate_trades(
                symbol=symbol, startTime=start_ms, limit=1000
            )
            if not trades:
                return None

            prices = [float(t["p"]) for t in trades]
            volumes = [float(t["q"]) for t in trades]

            ongoing_df = pd.DataFrame(
                {
                    "open": [prices[0]],
                    "high": [max(prices)],
                    "low": [min(prices)],
                    "close": [prices[-1]],
                    "volume": [sum(volumes)],
                },
                index=[current_hour_start],
            )
            ongoing_df.index.name = "timestamp"
            return ongoing_df
        except Exception as e:
            logger.warning(f"Could not fetch ongoing candle: {e}")
            return None
