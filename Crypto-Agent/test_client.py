import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

testing = os.getenv("TEST_MODE", "false") == "true"

print(f"Running in {'TESTNET' if testing else 'MAINNET'} mode.")

if testing:
    BINANCE_API_KEY = os.getenv("TESTNET_API_KEY")
    BINANCE_API_SECRET = os.getenv("TESTNET_API_SECRET")
else:
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=testing)


def show_spot_wallet():
    print("\n--- SPOT WALLET ---")
    for a in binance_client.get_account()["balances"]:
        if float(a["free"]) > 0 or float(a["locked"]) > 0:
            print(a)


show_spot_wallet()
