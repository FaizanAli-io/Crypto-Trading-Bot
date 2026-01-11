import os
from dotenv import load_dotenv
from binance.client import Client

SUPPORTED_CRYPTOS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "LINK": "LINKUSDT",
}

TRACKED_ASSETS = set(SUPPORTED_CRYPTOS.keys()) | {"USDT"}

load_dotenv()

client = Client(
    os.getenv("TESTNET_API_KEY"), os.getenv("TESTNET_API_SECRET"), testnet=True
)

account = client.get_account()

print("\n--- TESTNET SPOT WALLET (FILTERED) ---")
for a in account["balances"]:
    asset = a["asset"]
    if asset not in TRACKED_ASSETS:
        continue

    free = float(a["free"])
    locked = float(a["locked"])

    if free > 0 or locked > 0:
        print(f"{asset}: free={free}, locked={locked}")
