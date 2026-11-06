import sys
import os
import requests
import csv
import time
from datetime import datetime

ESP32_IP = "192.168.4.1"
URL = f"http://{ESP32_IP}/data"
POLL_INTERVAL_SEC = 1

def pick_output_file():
    # Use the name given on the command line, otherwise a timestamped one
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.csv")
    if os.path.exists(name):
        print(f"{name} already exists. Choose a different name so it isn't overwritten.")
        sys.exit(1)
    return name

def main():
    output_file = pick_output_file()
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wall_time", "esp_timestamp_ms",
            "cell1", "cell2", "cell3", "cell4",
            "current", "temp1", "temp2"
        ])
        f.flush()

        print(f"Logging to {output_file}. Press Ctrl+C to stop.")
        try:
            while True:
                try:
                    resp = requests.get(URL, timeout=2)
                    data = resp.json()
                    row = [
                        datetime.now().isoformat(),
                        data["timestamp_ms"],
                        data["cells"][0], data["cells"][1],
                        data["cells"][2], data["cells"][3],
                        data["current"], data["temp1"], data["temp2"]
                    ]
                    writer.writerow(row)
                    f.flush()
                    print(row)
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}, retrying...")
                time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\nStopped. Data saved to", output_file)

if __name__ == "__main__":
    main()