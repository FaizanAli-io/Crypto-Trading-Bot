from datetime import datetime
from binance.client import Client
from loguru import logger
client = Client()

# Get server time from Binance
server_time = client.get_server_time()
binance_time = datetime.fromtimestamp(server_time['serverTime'] / 1000)

print(f"Your system time: {datetime.now()}")
print(f"Binance UTC time: {binance_time}")


# def get_latest_candles(self, symbol, limit=24):
#     """
#     Get the absolute latest candles (no caching)
    
#     Args:
#         symbol: Trading pair
#         limit: Number of most recent candles
#     """
#     try:
#         logger.info(f"Fetching latest {limit} candles for {symbol}")
        
#         # Force fresh fetch from Binance
#         klines = self.client.get_klines(
#             symbol=symbol,
#             interval="1h",
#             limit=limit
#         )
        
#         df = self._klines_to_dataframe(klines)
        
#         logger.info(f"Latest candle: {df.index[-1]}")
#         logger.info(f"Latest close price: {df.iloc[-1]['close']}")
        
#         return df
        
#     except Exception as e:
#         logger.error(f"Error fetching latest candles: {e}")
#         raise


# get_latest_candles()