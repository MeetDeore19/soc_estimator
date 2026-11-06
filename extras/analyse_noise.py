import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Usage: python3 analyse_noise.py <file.csv> [start_seconds] [end_seconds]
FILE = sys.argv[1] if len(sys.argv) > 1 else "noise_test_caps.csv"
START_SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
END_SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else None

base = os.path.splitext(FILE)[0]
if END_SECONDS is not None:
    PLOT_FILE = f"{base}_{int(START_SECONDS)}-{int(END_SECONDS)}s_plot.png"
else:
    PLOT_FILE = f"{base}_plot.png"

df = pd.read_csv(FILE)
df["t"] = (df["esp_timestamp_ms"] - df["esp_timestamp_ms"].iloc[0]) / 1000.0
df = df[df["t"] >= START_SECONDS]
if END_SECONDS is not None:
    df = df[df["t"] <= END_SECONDS]
df = df.reset_index(drop=True)

if len(df) < 10:
    print("Too few samples in that time window. Check the start and end times.")
    sys.exit(1)

cells = ["cell1", "cell2", "cell3", "cell4"]
df["avg_cell"] = df[cells].mean(axis=1)

window = f"{START_SECONDS:.0f} s to " + (f"{END_SECONDS:.0f} s" if END_SECONDS is not None else "end")
print(f"File: {FILE}")
print(f"Window: {window}")
print(f"Samples analysed: {len(df)} over {df['t'].iloc[-1] - df['t'].iloc[0]:.0f} s\n")

print("Voltage channels (mV):")
print(f"{'channel':<10}{'mean V':>10}{'std':>10}{'p2p':>10}{'drift':>10}")
for col in cells + ["avg_cell"]:
    s = df[col]
    third = len(s) // 3
    drift_mv = (s.iloc[-third:].mean() - s.iloc[:third].mean()) * 1000
    print(f"{col:<10}{s.mean():>10.4f}{s.std() * 1000:>10.2f}"
          f"{(s.max() - s.min()) * 1000:>10.2f}{drift_mv:>10.2f}")

cur = df["current"]
print("\nCurrent (should be about 0 A, nothing flowing):")
print(f"  mean offset: {cur.mean() * 1000:.1f} mA")
print(f"  std (noise): {cur.std() * 1000:.1f} mA")
print(f"  peak-to-peak: {(cur.max() - cur.min()) * 1000:.1f} mA")

for col in ["temp1", "temp2"]:
    t = df[col]
    print(f"\n{col}: mean {t.mean():.2f} C, std {t.std():.2f} C, range {t.min():.2f} to {t.max():.2f} C")

r_volt = df["avg_cell"].var()
print("\nSuggested Kalman filter measurement noise (EKF on average cell voltage):")
print(f"  R = {r_volt:.3e} V^2   (std {df['avg_cell'].std() * 1000:.2f} mV)")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for col in cells:
    axes[0].plot(df["t"], (df[col] - df[col].mean()) * 1000, label=col, linewidth=0.8)
axes[0].set_ylabel("Cell deviation (mV)")
axes[0].legend(loc="upper right")
axes[0].grid(True)
axes[1].plot(df["t"], cur * 1000, color="red", linewidth=0.8)
axes[1].set_ylabel("Current (mA)")
axes[1].grid(True)
axes[2].plot(df["t"], df["temp1"], label="temp1")
axes[2].plot(df["t"], df["temp2"], label="temp2")
axes[2].set_ylabel("Temp (C)")
axes[2].set_xlabel("Time (s)")
axes[2].legend(loc="upper right")
axes[2].grid(True)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
print(f"\nSaved {PLOT_FILE}")