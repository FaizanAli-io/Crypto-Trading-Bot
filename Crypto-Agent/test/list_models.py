#!/usr/bin/env python3
"""
Model Catalog - Display all saved XGBoost models in an organized way
Lists models grouped by symbol, with intervals and horizons shown clearly
"""

from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


def parse_model_filename(filename: str) -> dict:
    """
    Parse XGBoost model filename to extract metadata

    Format (after reorganization): [smc_]{symbol}_{interval}_{horizon}min_{timestamp}.pkl
    Examples:
        - BTCUSDT_15m_15min_20251008_195152.pkl
        - smc_ETHUSDT_1h_60min_20251008_194800.pkl
    """
    # Remove .pkl extension
    name = filename.replace(".pkl", "")

    # Check if SMC model (prefix smc_)
    is_smc = name.startswith("smc_")

    # Pattern for simple models: {symbol}_{interval}_{horizon}min_{timestamp}
    # Pattern for SMC models: smc_{symbol}_{interval}_{horizon}min_{timestamp}

    if is_smc:
        pattern = r"smc_([A-Z]+)_(\w+)_(\d+)min_(\d{8})_(\d{6})"
    else:
        pattern = r"([A-Z]+)_(\w+)_(\d+)min_(\d{8})_(\d{6})"

    match = re.match(pattern, name)

    if not match:
        return None

    symbol, interval, horizon, date, time = match.groups()

    return {
        "symbol": symbol,
        "interval": interval,
        "horizon_minutes": int(horizon),
        "date": date,
        "time": time,
        "is_smc": is_smc,
        "timestamp": f"{date}_{time}",
        "model_type": "SMC" if is_smc else "Simple",
    }


def format_timestamp(date_str: str, time_str: str) -> str:
    """Convert YYYYMMDD_HHMMSS to readable format"""
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return f"{date_str} {time_str}"


def main():
    model_dir = Path("models") / "xg_models"

    if not model_dir.exists():
        print("❌ Models directory not found!")
        return

    # Find all model files
    model_files = sorted(model_dir.glob("*.pkl"))

    if not model_files:
        print("📁 No models found in models/xg_models directory")
        return

    print("=" * 90)
    print("📊 CRYPTO TRADING BOT - MODEL CATALOG")
    print("=" * 90)
    print()

    # Parse all models
    models = []
    for file in model_files:
        data = parse_model_filename(file.name)
        if data:
            data["filename"] = file.name
            data["size_mb"] = file.stat().st_size / (1024 * 1024)
            models.append(data)

    # Group by symbol
    by_symbol = defaultdict(list)
    for model in models:
        by_symbol[model["symbol"]].append(model)

    # Sort symbols
    sorted_symbols = sorted(by_symbol.keys())

    # Statistics
    total_models = len(models)
    smc_count = sum(1 for m in models if m["is_smc"])
    simple_count = total_models - smc_count

    print(f"📈 TOTAL MODELS: {total_models}")
    print(f"   • SMC Models:    {smc_count}")
    print(f"   • Simple Models: {simple_count}")
    print()

    # Display by symbol
    for symbol in sorted_symbols:
        symbol_models = by_symbol[symbol]

        # Count by type for this symbol
        symbol_smc = sum(1 for m in symbol_models if m["is_smc"])
        symbol_simple = len(symbol_models) - symbol_smc

        print(f"\n{'─' * 90}")
        print(f"💱 {symbol} ({len(symbol_models)} models)")
        print(f"   SMC: {symbol_smc} | Simple: {symbol_simple}")
        print(f"{'─' * 90}")

        # Group by interval within symbol
        by_interval = defaultdict(list)
        for model in symbol_models:
            by_interval[model["interval"]].append(model)

        intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h"]
        sorted_intervals = sorted(
            by_interval.keys(),
            key=lambda x: (intervals.index(x) if x in intervals else 999),
        )

        for interval in sorted_intervals:
            interval_models = by_interval[interval]

            # Group by horizon
            by_horizon = defaultdict(list)
            for model in interval_models:
                by_horizon[model["horizon_minutes"]].append(model)

            sorted_horizons = sorted(by_horizon.keys())

            print(f"\n   ⏱️  {interval} interval:")

            for horizon in sorted_horizons:
                horizon_models = by_horizon[horizon]

                # Most recent of this config
                latest = max(horizon_models, key=lambda m: m["timestamp"])

                smc_available = any(m["is_smc"] for m in horizon_models)
                simple_available = any(not m["is_smc"] for m in horizon_models)

                types_str = ""
                if smc_available:
                    types_str += "🔵 SMC"
                if simple_available:
                    if types_str:
                        types_str += " + "
                    types_str += "⚪ Simple"

                print(f"       • {horizon:3d}min → {types_str}")
                print(
                    f"              Updated: {format_timestamp(latest['date'], latest['time'])}"
                )
                print(f"              File: {latest['filename']}")

    print()
    print("=" * 90)
    print(
        f"✅ Catalog complete: {total_models} models ready for prediction/backtesting"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
