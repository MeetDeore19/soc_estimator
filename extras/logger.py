"""
BMS Monitor logger
------------------
Polls the ESP32's /data endpoint and saves every new reading to a CSV file.
When you stop it (Ctrl+C), it prints a summary: averages, noise, and the
amp-hours it counted, which are the numbers you need for calibration.

Setup:
    1. Join the ESP32's WiFi network "BMS-Monitor" (password: bms12345).
    2. Run this script from a terminal.

Usage:
    python logger.py <label> [esp32_ip]

Examples:
    python logger.py zero_offset
    python logger.py discharge_0p35A
    python logger.py charge_1A 192.168.4.1

No extra packages needed: it only uses Python's standard library.
"""

import csv
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime

DEFAULT_IP = "192.168.4.1"   # ESP32 access-point address (printed in Serial Monitor)
POLL_INTERVAL_S = 0.5        # ask twice a second; repeated readings are skipped
TIMEOUT_S = 3.0              # the ESP32 can be slow to answer while it samples
GAP_WARNING_S = 5.0          # warn if readings stop arriving for this long


def fetch(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode())


def stats_line(name, values, unit, scale=1.0, decimals=4):
    if not values:
        return f"  {name}: no data"
    mean = statistics.mean(values) * scale
    if len(values) >= 2:
        sd = statistics.stdev(values) * scale
    else:
        sd = 0.0
    lo = min(values) * scale
    hi = max(values) * scale
    return (f"  {name}: mean {mean:.{decimals}f} {unit} | noise (std dev) {sd:.{decimals}f} {unit}"
            f" | min {lo:.{decimals}f} | max {hi:.{decimals}f}")


def print_summary(rows, first_ms, last_ms, errors, currents, cells, counted_ah, filename):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if rows == 0:
        print("  No readings were saved. Check you're on the BMS-Monitor WiFi.")
        return

    duration_s = (last_ms - first_ms) / 1000 if last_ms is not None and first_ms is not None else 0
    print(f"  File: {filename}")
    print(f"  Readings saved: {rows}   Duration: {duration_s:.0f} s   Missed replies: {errors}")
    print()
    print(stats_line("Current", currents, "A", decimals=4))
    for i in range(4):
        print(stats_line(f"Cell {i + 1} ", cells[i], "mV", scale=1000, decimals=1))
    print()
    print(f"  Counted charge: {counted_ah:+.5f} Ah  ({counted_ah * 1000:+.1f} mAh)")
    print()
    print("  Calibration notes:")
    print("  - Zero-offset test (no current): new offset = old offset + mean current x sensitivity")
    print("  - Gain test: new sensitivity = old sensitivity x (mean ESP32 current / multimeter current)")
    print("  - Noise: the EKF's R and Q use the variance, which is (std dev) squared")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    label = sys.argv[1]
    ip = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IP
    url = f"http://{ip}/data"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.csv"

    header = ["pc_time", "esp_ms", "c1_v", "c2_v", "c3_v", "c4_v", "pack_v",
              "current_a", "temp1_c", "temp2_c", "counted_ah"]

    currents = []
    cells = [[], [], [], []]
    counted_ah = 0.0
    first_ms = None
    last_ms = None
    last_arrival = time.time()
    rows = 0
    errors = 0

    print(f"Logging {url} to {filename}")
    print("Press Ctrl+C to stop and see the summary.\n")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        try:
            while True:
                try:
                    d = fetch(url)
                except Exception as e:
                    errors += 1
                    print(f"[no reply: {e}] retrying...")
                    time.sleep(1.0)
                    continue

                esp_ms = d["timestamp_ms"]

                # Same reading as last time: the ESP32 hasn't measured again yet
                if esp_ms == last_ms:
                    if time.time() - last_arrival > GAP_WARNING_S:
                        print(f"[warning: no new reading for {GAP_WARNING_S:.0f}+ s]")
                        last_arrival = time.time()
                    time.sleep(POLL_INTERVAL_S)
                    continue

                c = d["cells"]
                current = d["current"]
                t1 = d["temp1"]
                t2 = d["temp2"]

                # Coulomb counting using the ESP32's own clock.
                # Skipped on the first reading, or if the ESP32 restarted (clock went backwards).
                if last_ms is not None and esp_ms > last_ms:
                    dt_s = (esp_ms - last_ms) / 1000
                    if dt_s > GAP_WARNING_S:
                        print(f"[warning: {dt_s:.1f} s gap between readings]")
                    counted_ah += current * dt_s / 3600
                elif last_ms is not None and esp_ms < last_ms:
                    print("[warning: ESP32 clock reset - did it restart?]")

                if first_ms is None:
                    first_ms = esp_ms
                last_ms = esp_ms
                last_arrival = time.time()

                pack_v = sum(c)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    esp_ms,
                    f"{c[0]:.4f}", f"{c[1]:.4f}", f"{c[2]:.4f}", f"{c[3]:.4f}",
                    f"{pack_v:.4f}",
                    f"{current:.4f}",
                    f"{t1:.2f}", f"{t2:.2f}",
                    f"{counted_ah:.6f}",
                ])
                f.flush()   # save each row immediately, so nothing is lost if it crashes

                currents.append(current)
                for i in range(4):
                    cells[i].append(c[i])
                rows += 1

                print(f"{esp_ms / 1000:9.1f} s | C1 {c[0]:.3f}  C2 {c[1]:.3f}  C3 {c[2]:.3f}  C4 {c[3]:.3f}"
                      f" | Pack {pack_v:.3f} V | I {current:+.3f} A | {counted_ah * 1000:+.1f} mAh"
                      f" | T {t1:.1f}/{t2:.1f} C")

                time.sleep(POLL_INTERVAL_S)

        except KeyboardInterrupt:
            pass

    print_summary(rows, first_ms, last_ms, errors, currents, cells, counted_ah, filename)


if __name__ == "__main__":
    main()