import sys, json, pandas as pd
from pathlib import Path

def main(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trades = data.get("trades", [])
    if not isinstance(trades, list) or not trades:
        print("No trades found (expected data['trades'] to be a non-empty list).")
        return

    # Build DataFrame (pandas auto-flattens simple objects)
    df = pd.DataFrame(trades)

    # Optional: parse timestamps if present
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Choose output names based on input name
    base = path.stem
    csv_out = path.with_name(f"{base}_trades.csv")
    xlsx_out = path.with_name(f"{base}_trades.xlsx")

    # Write CSV and XLSX
    df.to_csv(csv_out, index=False)
    with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="trades", index=False)

    print(f"Saved:\n- {csv_out}\n- {xlsx_out}")

if __name__ == "__main__":
    main("simulator.json")