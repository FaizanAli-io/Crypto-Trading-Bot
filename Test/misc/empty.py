import os
from dotenv import load_dotenv
from binance.client import Client

TARGET_USDT = 1000.0
GARBAGE_PAIR = "TRXUSDT"

load_dotenv()

client = Client(
    os.getenv("TESTNET_API_KEY"), os.getenv("TESTNET_API_SECRET"), testnet=True
)

account = client.get_account()
balances = {a["asset"]: float(a["free"]) for a in account["balances"]}

print("\n--- TRASHING TESTNET WALLET ---")

# 1. Sell everything except USDT
for asset, free in balances.items():
    if asset in ("USDT",) or free <= 0:
        continue

    symbol = f"{asset}USDT"

    try:
        client.create_order(
            symbol=symbol, side="SELL", type="MARKET", quantity=round(free, 6)
        )
        print(f"Sold {free:.6f} {asset} → USDT")
    except Exception:
        print(f"Skipping {asset} (no USDT pair)")

# 2. Re-fetch USDT balance
account = client.get_account()
usdt = float(next(a["free"] for a in account["balances"] if a["asset"] == "USDT"))

excess = usdt - TARGET_USDT

# 3. Dump excess USDT into garbage coin
if excess > 10:
    client.create_order(
        symbol=GARBAGE_PAIR, side="BUY", type="MARKET", quoteOrderQty=round(excess, 2)
    )
    print(f"Burned {excess:.2f} USDT → {GARBAGE_PAIR}")
else:
    print("USDT already within limit")

print("\n✓ Wallet now effectively capped at 1000 USDT")
