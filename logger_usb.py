"""
BMS Monitor USB logger
----------------------
Reads the ESP32's "DATA,..." lines over the USB cable and saves every reading
to a CSV with the same columns as logger.py, so characterise.py and
soc_estimate.py work on it unchanged. Press Ctrl+C to stop and see a summary.

Setup (once):
    pip3 install pyserial

Usage:
    python3 logger_usb.py <label> [port]

Examples:
    python3 logger_usb.py validation_v2
    python3 logger_usb.py validation_v2 /dev/cu.usbmodem101

IMPORTANT: close the Arduino Serial Monitor first. Only one program can use
the USB port at a time.
"""

import csv
import glob
import statistics
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    print("pyserial is missing. Install it with:  pip3 install pyserial")
    sys.exit(1)

BAUD = 115200
GAP_WARNING_S = 5.0


def find_port():
    ports = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*") +
                   glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/ttyUSB*"))
    return ports[0] if ports else None


def stats_line(name, values, unit, scale=1.0, decimals=4):
    if not values:
        return f"  {name}: no data"
    mean = statistics.mean(values) * scale
    sd = statistics.stdev(values) * scale if len(values) >= 2 else 0.0
    return (f"  {name}: mean {mean:.{decimals}f} {unit} | spread (std dev) {sd:.{decimals}f} {unit}"
            f" | min {min(values) * scale:.{decimals}f} | max {max(values) * scale:.{decimals}f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    label = sys.argv[1]
    port = sys.argv[2] if len(sys.argv) > 2 else find_port()
    if port is None:
        print("No ESP32 USB port found. Plug it in, or give the port, e.g. /dev/cu.usbmodem101")
        sys.exit(1)

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.csv"
    header = ["pc_time", "esp_ms", "c1_v", "c2_v", "c3_v", "c4_v", "pack_v",
              "current_a", "temp1_c", "temp2_c", "counted_ah"]

    ser = serial.Serial(port, BAUD, timeout=2)
    print(f"Logging {port} to {filename}")
    print("Press Ctrl+C to stop and see the summary.\n")

    currents, cells = [], [[], [], [], []]
    counted_ah = 0.0
    first_ms = last_ms = None
    last_arrival = time.time()
    rows = bad_lines = restarts = 0

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        try:
            while True:
                raw = ser.readline()
                if not raw:
                    if time.time() - last_arrival > GAP_WARNING_S:
                        print(f"[warning: no data for {GAP_WARNING_S:.0f}+ s - is the ESP32 running?]")
                        last_arrival = time.time()
                    continue
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("DATA,"):
                    if line:
                        print(f"[ESP32] {line}")
                    continue
                parts = line.split(",")
                if len(parts) != 13:
                    bad_lines += 1
                    continue
                try:
                    esp_ms = int(parts[1])
                    c = [float(x) for x in parts[2:6]]
                    current = float(parts[6])
                    t1, t2 = float(parts[11]), float(parts[12])
                except ValueError:
                    bad_lines += 1
                    continue

                if last_ms is not None and esp_ms > last_ms:
                    dt_s = (esp_ms - last_ms) / 1000
                    if dt_s > GAP_WARNING_S:
                        print(f"[warning: {dt_s:.1f} s gap between readings]")
                    counted_ah += current * dt_s / 3600
                elif last_ms is not None and esp_ms < last_ms:
                    restarts += 1
                    print("[warning: ESP32 clock reset - did it restart?]")
                if first_ms is None:
                    first_ms = esp_ms
                last_ms = esp_ms
                last_arrival = time.time()

                pack_v = sum(c)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"), esp_ms,
                    f"{c[0]:.4f}", f"{c[1]:.4f}", f"{c[2]:.4f}", f"{c[3]:.4f}", f"{pack_v:.4f}",
                    f"{current:.4f}", f"{t1:.2f}", f"{t2:.2f}", f"{counted_ah:.6f}",
                ])
                f.flush()
                currents.append(current)
                for i in range(4):
                    cells[i].append(c[i])
                rows += 1
                print(f"{esp_ms / 1000:9.1f} s | C1 {c[0]:.3f}  C2 {c[1]:.3f}  C3 {c[2]:.3f}  C4 {c[3]:.3f}"
                      f" | Pack {pack_v:.3f} V | I {current:+.3f} A | {counted_ah * 1000:+.1f} mAh")
        except KeyboardInterrupt:
            pass
    ser.close()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if rows == 0:
        print("  No readings saved. Is the Serial Monitor closed, and is the USB firmware uploaded?")
        return
    duration = (last_ms - first_ms) / 1000
    print(f"  File: {filename}")
    print(f"  Readings saved: {rows}   Duration: {duration:.0f} s   Bad lines: {bad_lines}   ESP32 restarts: {restarts}")
    print()
    print(stats_line("Current", currents, "A"))
    for i in range(4):
        print(stats_line(f"Cell {i + 1} ", cells[i], "mV", scale=1000, decimals=1))
    print()
    print(f"  Counted charge: {counted_ah:+.5f} Ah  ({counted_ah * 1000:+.1f} mAh)  (raw, before offset correction)")
    print("  For a rest-only run, the 'spread' figures are the noise.")
    print("=" * 70)


if __name__ == "__main__":
    main()