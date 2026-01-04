from datetime import datetime
from binance.client import Client

client = Client()

server_time = client.get_server_time()
binance_time = datetime.fromtimestamp(server_time["serverTime"] / 1000)

print(f"Binance UTC time: {binance_time}")
print(f"Your system time: {datetime.now()}")
