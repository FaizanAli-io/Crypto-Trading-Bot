import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load JSON ---
with open("simulator.json", "r") as f:
    data = json.load(f)

sim = data.get("simulation_params", {})
summary = data.get("trading_summary", {})
trades = pd.DataFrame(data.get("trades", []))

# --- Basic cleanup ---
# Parse datetimes
if "entry_time" in trades.columns:
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
if "exit_time" in trades.columns:
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])

# Sort by exit_time to build equity curve
trades = trades.sort_values("exit_time").reset_index(drop=True)

# Some safety: fill missing numeric fields with 0
for col in ["pnl", "pnl_pct", "gross_pnl", "fee", "confidence", "position_amount"]:
    if col in trades.columns:
        trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0.0)

# --- Equity curve ---
initial_capital = float(summary.get("initial_capital", sim.get("initial_capital", 1000)))
# If your JSON already has capital_after per trade, prefer that
if "capital_after" in trades.columns and trades["capital_after"].notna().all():
    equity = trades[["exit_time", "capital_after"]].copy()
    equity = equity.rename(columns={"capital_after": "equity"})
else:
    # Rebuild equity from initial + cumulative pnl
    trades["equity"] = initial_capital + trades["pnl"].cumsum()
    equity = trades[["exit_time", "equity"]].copy()

# --- Drawdown (based on equity) ---
equity["peak"] = equity["equity"].cummax()
equity["drawdown"] = (equity["equity"] - equity["peak"]) / equity["peak"]

# --- Win flag ---
if "prediction_correct" in trades.columns:
    trades["win"] = trades["prediction_correct"].astype(bool)
else:
    # fallback: pnl > 0 treated as win
    trades["win"] = trades["pnl"] > 0

# --- Plot 1: Equity curve ---
plt.figure()
plt.plot(equity["exit_time"], equity["equity"])
plt.title("Equity Curve")
plt.xlabel("Time")
plt.ylabel("Equity (USDT)")
plt.tight_layout()

# --- Plot 2: Drawdown ---
plt.figure()
plt.plot(equity["exit_time"], equity["drawdown"] * 100.0)
plt.title("Drawdown (%)")
plt.xlabel("Time")
plt.ylabel("Drawdown %")
plt.tight_layout()

# --- Plot 3: PnL% distribution ---
plt.figure()
plt.hist(trades["pnl_pct"], bins=20)
plt.title("Trade PnL% Distribution")
plt.xlabel("PnL% (net)")
plt.ylabel("Count")
plt.tight_layout()

# --- Plot 4: Exit reasons ---
if "exit_reason" in trades.columns:
    exit_counts = trades["exit_reason"].value_counts()
    plt.figure()
    exit_counts.plot(kind="bar")
    plt.title("Exit Reasons")
    plt.xlabel("Reason")
    plt.ylabel("Trades")
    plt.tight_layout()

# --- Plot 5: Confidence vs PnL% (size ~ position) ---
if "confidence" in trades.columns:
    plt.figure()
    sizes = trades["position_amount"].abs()
    if sizes.max() > 0:
        sizes = 100 * (sizes / sizes.max())  # scale for visibility
    else:
        sizes = 50
    plt.scatter(trades["confidence"], trades["pnl_pct"], s=sizes)
    plt.title("Confidence vs PnL%")
    plt.xlabel("Model Confidence")
    plt.ylabel("PnL%")
    plt.tight_layout()

# --- Plot 6: Calibration (confidence bins vs win-rate) ---
if "confidence" in trades.columns:
    # Bin confidence into deciles
    bins = np.linspace(0.0, 1.0, 11)
    trades["conf_bin"] = pd.cut(trades["confidence"], bins=bins, include_lowest=True)
    calib = trades.groupby("conf_bin")["win"].mean().reset_index()
    # mid-point for x-axis
    calib["bin_mid"] = calib["conf_bin"].apply(lambda iv: (iv.left + iv.right) / 2 if hasattr(iv, "left") else np.nan)

    plt.figure()
    plt.plot(calib["bin_mid"], calib["win"] * 100.0, marker="o")
    plt.title("Calibration: Confidence vs Win Rate")
    plt.xlabel("Confidence (bin mid)")
    plt.ylabel("Win Rate %")
    plt.ylim(0, 100)
    plt.tight_layout()

# --- Plot 7: Cumulative fees ---
if "fee" in trades.columns:
    plt.figure()
    plt.plot(trades["exit_time"], trades["fee"].cumsum())
    plt.title("Cumulative Fees Paid")
    plt.xlabel("Time")
    plt.ylabel("USDT")
    plt.tight_layout()

# --- Quick text summary in console ---
num_trades = len(trades)
wins = int(trades["win"].sum())
win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
total_fees = trades["fee"].sum()
final_equity = float(summary.get("final_capital", equity["equity"].iloc[-1] if len(equity) > 0 else initial_capital))
profit_factor = float(summary.get("profit_factor", np.nan))

print("=== Summary ===")
print("Initial capital:", initial_capital)
print("Final capital:", final_equity)
print("Num trades:", num_trades)
print("Wins:", wins, f"({win_rate:.2f}% win rate)")
print("Total fees:", round(total_fees, 2))
if not math.isnan(profit_factor):
    print("Profit factor:", round(profit_factor, 3))

plt.show()