"""
Live battery dashboard: SOC from the EKF, in real time
======================================================
Reads the ESP32 over USB (or replays a saved test), runs the same 2-RC EKF as
soc_estimate.py one reading at a time, and shows:

    big SOC number + uncertainty | SOC over time
    cell voltages                | current

Usage:
    Live (also records a CSV, so you don't need logger_usb.py at the same time):
        python3 live_dashboard.py --port /dev/cu.usbmodem101
    Replay a recorded test, e.g. 60x faster (great for the demo video):
        python3 live_dashboard.py --replay validation_v2_soc.csv --speed 60

Options:
    --params ekf_parameters.json   your pack's model (from characterise.py)
    --soc0 0.70                    start the EKF at this SOC instead of the voltage guess
    --snapshot out.png             save the final frame instead of opening a window

Needs: numpy, pandas, matplotlib, pyserial (live mode), and soc_estimate.py in
the same folder. Close the Arduino Serial Monitor before running live.
"""

import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

import soc_estimate as se   # same model, constants and noise settings as the offline EKF

CELLS = ["c1_v", "c2_v", "c3_v", "c4_v"]
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- EKF, one step at a time
class OnlineEKF:
    """Exactly the equations of soc_estimate.run_ekf, but fed one reading at a time."""

    def __init__(self, model, soc0, p_soc0):
        self.m = model
        self.x = np.array([soc0, 0.0, 0.0])
        self.P = np.diag([p_soc0, 1e-6, 1e-6])
        self.offset = 0.0
        self.rest_time = 0.0
        self.prev_i = 0.0
        self.prev_t = None

    def step(self, t, i_meas, v_meas):
        m = self.m
        dt = 0.0 if self.prev_t is None else max(t - self.prev_t, 0.0)
        self.prev_t = t

        if abs(i_meas - self.offset) < se.REST_THRESHOLD_A:
            self.rest_time += dt
            if self.rest_time >= se.REST_LEARN_AFTER_S and dt > 0:
                self.offset += min(dt / se.OFFSET_LEARN_TIME_S, 1.0) * (i_meas - self.offset)
        else:
            self.rest_time = 0.0
        i_k = i_meas - self.offset

        x, P = self.x, self.P
        if dt > 0:
            a1, a2 = np.exp(-dt / m.tau1), np.exp(-dt / m.tau2)
            pi = self.prev_i
            x = np.array([x[0] - pi * dt / (3600.0 * m.capacity_ah),
                          a1 * x[1] + m.R1 * (1 - a1) * pi,
                          a2 * x[2] + m.R2 * (1 - a2) * pi])
            A = np.diag([1.0, a1, a2])
            q_soc = ((se.SIGMA_I_A + se.SIGMA_CAP_FRAC * abs(pi)) * dt / (3600.0 * m.capacity_ah)) ** 2 \
                + se.Q_SOC_PER_S * dt
            s_rc = se.SIGMA_RC_V_REST + se.SIGMA_RC_V_LOAD * min(abs(pi) / se.LOAD_REF_A, 1.0)
            P = A @ P @ A.T + np.diag([q_soc, s_rc ** 2, s_rc ** 2])
        x[0] = min(max(x[0], 0.0), 1.0)

        v_pred = m.ocv(x[0]) - x[1] - x[2] - m.r0(x[0]) * i_k
        if np.isfinite(v_meas):
            C = np.array([m.docv_dsoc(x[0]), -1.0, -1.0])
            K = P @ C / (C @ P @ C + se.SIGMA_V_MEAS ** 2)
            x = x + K * (v_meas - v_pred)
            P = (np.eye(3) - np.outer(K, C)) @ P
            P = 0.5 * (P + P.T)
        x[0] = min(max(x[0], 0.0), 1.0)

        self.x, self.P, self.prev_i = x, P, i_k
        return x[0], float(np.sqrt(max(P[0, 0], 0.0))), i_k


# ---------------------------------------------------------------- data sources
def serial_source(port, record_name):
    """Yield (t_s, cells[4], current) from the ESP32's DATA lines, recording a CSV."""
    import serial
    ser = serial.Serial(port, 115200, timeout=2)
    f = open(record_name, "w", newline="")
    w = csv.writer(f)
    w.writerow(["pc_time", "esp_ms", "c1_v", "c2_v", "c3_v", "c4_v", "pack_v",
                "current_a", "temp1_c", "temp2_c", "counted_ah"])
    t0, counted, last_ms = None, 0.0, None
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line.startswith("DATA,"):
                continue
            p = line.split(",")
            if len(p) != 13:
                continue
            try:
                ms, c, cur = int(p[1]), [float(v) for v in p[2:6]], float(p[6])
            except ValueError:
                continue
            if last_ms is not None and ms > last_ms:
                counted += cur * (ms - last_ms) / 3.6e6
            last_ms = ms
            t0 = ms if t0 is None else t0
            w.writerow([datetime.now().isoformat(timespec="seconds"), ms, *[f"{v:.4f}" for v in c],
                        f"{sum(c):.4f}", f"{cur:.4f}", p[11], p[12], f"{counted:.6f}"])
            f.flush()
            yield (ms - t0) / 1000.0, c, cur, None
    finally:
        f.close()
        ser.close()


def replay_source(path, speed):
    """Yield readings from a saved CSV, paced at `speed` x real time."""
    df = se.load_log(path)
    ref = df["soc_reference"].to_numpy() if "soc_reference" in df else None
    start = time.time()
    for k in range(len(df)):
        if speed > 0:
            wait = df["t"].iloc[k] / speed - (time.time() - start)
            if wait > 0:
                time.sleep(wait)
        yield (float(df["t"].iloc[k]), [float(df[c].iloc[k]) for c in CELLS],
               float(df["current_a"].iloc[k]), None if ref is None else float(ref[k]))


# ---------------------------------------------------------------- dashboard
def main():
    ap = argparse.ArgumentParser(description="Live EKF battery dashboard")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port")
    src.add_argument("--replay")
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--params", default="ekf_parameters.json")
    ap.add_argument("--soc0", type=float, default=None)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    if args.snapshot:
        plt.switch_backend("Agg")
    else:
        # soc_estimate.py switches matplotlib to "Agg" (save-to-file only) when it
        # is imported, so switch back to a backend that can open a window.
        for backend in ("MacOSX", "QtAgg", "TkAgg"):
            try:
                plt.switch_backend(backend)
                break
            except Exception:
                continue
        else:
            print("Could not open a plot window; use --snapshot out.png instead.")
            return

    with open(args.params) as f:
        model = se.CellModel(json.load(f))

    if args.port:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_dashboard_live.csv"
        source = serial_source(args.port, name)
        print(f"Live from {args.port}, recording to {name}. Close the window or press Ctrl+C to stop.")
    else:
        source = replay_source(args.replay, 0 if args.snapshot else args.speed)

    hist = {k: deque() for k in ["t", "soc", "lo", "hi", "ref", "i", "c1", "c2", "c3", "c4"]}
    ekf = None

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(12, 7))
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title("Battery monitor - EKF SOC")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.6, 1.6], height_ratios=[1, 1])
    ax_num = fig.add_subplot(gs[0, 0])
    ax_soc = fig.add_subplot(gs[0, 1:])
    ax_v = fig.add_subplot(gs[1, :2])
    ax_i = fig.add_subplot(gs[1, 2])

    ax_num.axis("off")
    txt_soc = ax_num.text(0.5, 0.62, "--%", ha="center", va="center", fontsize=44, color=INK, weight="bold")
    txt_sig = ax_num.text(0.5, 0.40, "", ha="center", va="center", fontsize=11, color=INK2)
    txt_info = ax_num.text(0.5, 0.18, "", ha="center", va="center", fontsize=10, color=INK2)
    ax_num.set_title("State of charge (EKF)", loc="left", color=INK)

    (l_ref,) = ax_soc.plot([], [], color=INK2, lw=2.5, label="Reference (hindsight)")
    (l_soc,) = ax_soc.plot([], [], color=BLUE, lw=2, label="EKF estimate")
    band: list = [None]   # holds the shaded +/-1 sigma area so it can be redrawn
    ax_soc.set_ylim(0, 100)
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_title("SOC over time", loc="left", color=INK)

    cell_lines = [ax_v.plot([], [], color=c, lw=1.2, label=f"Cell {n + 1}")[0]
                  for n, c in enumerate([BLUE, ORANGE, AQUA, VIOLET])]
    ax_v.set_ylabel("Cell voltage (V)")
    ax_v.set_xlabel("Time (min)")
    ax_v.set_title("Cell voltages", loc="left", color=INK)
    ax_v.legend(frameon=False, ncol=4, loc="lower left", fontsize=9)

    (l_i,) = ax_i.plot([], [], color=BLUE, lw=1.2)
    ax_i.set_ylabel("Current (A, discharge +)")
    ax_i.set_xlabel("Time (min)")
    ax_i.set_title("Current", loc="left", color=INK)

    for ax in (ax_soc, ax_v, ax_i):
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    window = {"closed": False}
    fig.canvas.mpl_connect("close_event", lambda event: window.update(closed=True))

    def redraw():
        t_min = np.array(hist["t"]) / 60
        if len(t_min) < 2:
            return
        l_soc.set_data(t_min, hist["soc"])
        if any(r is not None for r in hist["ref"]):
            ref = np.array([np.nan if r is None else r for r in hist["ref"]], dtype=float)
            l_ref.set_data(t_min, ref)
            if not ax_soc.get_legend():
                ax_soc.legend(frameon=False, fontsize=9, loc="upper right")
        if band[0] is not None:
            band[0].remove()
        band[0] = ax_soc.fill_between(t_min, hist["lo"], hist["hi"], color=BLUE, alpha=0.15, lw=0)
        for k, line in enumerate(cell_lines):
            line.set_data(t_min, hist[f"c{k + 1}"])
        l_i.set_data(t_min, hist["i"])
        for ax in (ax_soc, ax_v, ax_i):
            ax.set_xlim(0, max(t_min[-1], 1))
        allv = np.concatenate([np.asarray(hist[f"c{k}"]) for k in range(1, 5)])
        ax_v.set_ylim(np.nanmin(allv) - 0.02, np.nanmax(allv) + 0.02)
        ax_i.set_ylim(min(-0.2, np.min(hist["i"]) - 0.2), max(0.5, np.max(hist["i"]) + 0.3))
        soc, lo, hi = hist["soc"][-1], hist["lo"][-1], hist["hi"][-1]
        txt_soc.set_text(f"{soc:.1f}%")
        txt_sig.set_text(f"+/- {(hi - lo) / 2:.1f}% (1 sigma)")
        lowest = min(hist[f"c{k}"][-1] for k in range(1, 5))
        txt_info.set_text(f"{hist['i'][-1]:+.2f} A  |  lowest cell {lowest:.3f} V")

    last_draw = 0.0
    try:
        for t, cells, cur, ref in source:
            v = float(np.mean(cells))
            if ekf is None:
                soc0 = args.soc0 if args.soc0 is not None else model.soc_from_ocv(v)
                ekf = OnlineEKF(model, soc0, 0.2 ** 2 if args.soc0 is not None else 0.1 ** 2)
            soc, sig, i_k = ekf.step(t, cur, v)
            hist["t"].append(t)
            hist["soc"].append(100 * soc)
            hist["lo"].append(100 * max(soc - sig, 0))
            hist["hi"].append(100 * min(soc + sig, 1))
            hist["ref"].append(None if ref is None else 100 * ref)
            hist["i"].append(i_k)
            for k in range(4):
                hist[f"c{k + 1}"].append(cells[k])
            if not args.snapshot and time.time() - last_draw > 0.3:
                redraw()
                plt.pause(0.001)
                last_draw = time.time()
                if window["closed"]:
                    break
    except KeyboardInterrupt:
        pass

    redraw()
    if args.snapshot:
        fig.savefig(args.snapshot, dpi=130)
        print(f"Saved {args.snapshot}  (final EKF SOC {hist['soc'][-1]:.1f}%)")
    else:
        print("Stopped.")
        plt.show()


if __name__ == "__main__":
    main()