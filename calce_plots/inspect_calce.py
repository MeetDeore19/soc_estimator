import os
import glob
import pandas as pd

FOLDER = "calce"

files = sorted(
    glob.glob(os.path.join(FOLDER, "**", "*.xls"), recursive=True)
    + glob.glob(os.path.join(FOLDER, "**", "*.xlsx"), recursive=True)
)

if not files:
    print(f"No Excel files found under '{FOLDER}'. Check the zips were unzipped there.")

for path in files:
    print("=" * 80)
    print(f"FILE: {path}")
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        print(f"  Could not read: {e}")
        continue

    for name, df in sheets.items():
        print(f"\n  SHEET: '{name}'  ({df.shape[0]} rows x {df.shape[1]} columns)")
        print("  Columns:")
        for col in df.columns:
            print(f"    - {col}")
        if df.shape[0] > 0:
            print("  First 3 rows:")
            print(df.head(3).to_string(max_cols=12, max_colwidth=14))
print("=" * 80)