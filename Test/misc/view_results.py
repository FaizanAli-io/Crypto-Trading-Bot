"""
View validation results in a clean table format
"""

import json
import pandas as pd
from pathlib import Path
import sys


def create_results_table(json_file):
    """Create a formatted table from validation results"""

    # Load results
    with open(json_file, "r") as f:
        results = json.load(f)

    # Extract data for table
    table_data = []
    for result in results:
        row = {
            "SYMBOL": result["symbol"],
            "HORIZON": result["horizon_minutes"],
            "TOTAL": result["total_predictions"],
            "CORRECT %": result["overall_accuracy"],
            "HIGH CONF %": result["accuracy_by_confidence"]["high_confidence_70plus"][
                "accuracy"
            ],
            "MED CONF %": result["accuracy_by_confidence"]["medium_confidence_60_70"][
                "accuracy"
            ],
            "LOW CONF %": result["accuracy_by_confidence"]["low_confidence_under_60"][
                "accuracy"
            ],
        }
        table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Sort by symbol, then interval, then horizon
    df = df.sort_values(["SYMBOL", "HORIZON"])

    # Display table
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS TABLE")
    print("=" * 100 + "\n")
    print(df.to_string(index=False))
    print("\n" + "=" * 100)

    # Summary statistics
    print(f"\nAVERAGE ACCURACY: {df['CORRECT %'].mean():.2f}%")
    print(
        f"BEST MODEL: {df.loc[df['CORRECT %'].idxmax()]['SYMBOL']} {df.loc[df['CORRECT %'].idxmax()]['HORIZON']} ({df['CORRECT %'].max():.2f}%)"
    )
    print(
        f"WORST MODEL: {df.loc[df['CORRECT %'].idxmin()]['SYMBOL']} {df.loc[df['CORRECT %'].idxmin()]['HORIZON']} ({df['CORRECT %'].min():.2f}%)"
    )
    print("=" * 100 + "\n")

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Find most recent validation results file
        json_files = list(Path(".").glob("validation_results_*.json"))
        if not json_files:
            print("❌ No validation results found")
            print("Usage: python view_results.py [validation_results.json]")
            sys.exit(1)
        json_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Using: {json_file}\n")

    create_results_table(json_file)
