"""
SOC estimator: Extended Kalman Filter with a 2-RC equivalent circuit model
=========================================================================
Usage:
    python3 soc_estimate.py <log.csv> [--params ekf_parameters.json]
                            [--soc0 0.70] [--reference-full]

  <log.csv>          a CSV from logger.py
  --params           the ekf_parameters.json written by characterise.py
                     (capacity, OCV table, R0, R1/C1, R2/C2 for YOUR pack)
  --soc0             start the filter at this SOC (0-1) instead of reading it
                     from the first voltage. Use a WRONG value (e.g. 0.70 on a
                     full pack) to show the filter correcting itself.
  --reference-full   the test started from a FULL, RESTED pack: also compute
                     the hindsight reference SOC and report the EKF's error.

What the filter does every second
---------------------------------
State x = [SOC, v1, v2]
    SOC : state of charge (0-1)
    v1  : voltage across the fast RC branch (tau ~ 30 s)
    v2  : voltage across the slow RC branch (tau ~ 16 min)

Predict (coulomb counting + RC dynamics), current I > 0 = discharge:
    SOC <- SOC - I*dt / (3600*Q)
    v1  <- a1*v1 + R1*(1-a1)*I        a1 = exp(-dt/tau1)
    v2  <- a2*v2 + R2*(1-a2)*I        a2 = exp(-dt/tau2)

Predicted terminal voltage of the average cell:
    V = OCV(SOC) - v1 - v2 - R0*I

Correct:
    C = [dOCV/dSOC, -1, -1]           (the "Extended" part: re-slope every step)
    K = P C' / (C P C' + R)
    x <- x + K * (V_measured - V_predicted)
    P <- (I - K C) P

On LFP's flat middle, dOCV/dSOC is tiny, so K for SOC is tiny and the filter
rides on coulomb counting. Near full and empty the curve is steep and the
voltage pulls the SOC back to the truth.

Current offset: the ACS712's zero drifts by tens of mA over days. Whenever the
pack has been resting for a minute, the filter slowly learns the reading it
sees as the new zero (the true current at rest is 0 A).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
# Spike rejection on cell voltages (Hampel filter)
HAMPEL_WINDOW = 7
HAMPEL_SIGMAS = 4.0
SPIKE_FLOOR_CELL_V = 0.010

# Noise levels (from your rest measurements, with margin)
SIGMA_I_A = 0.010           # current noise per reading (measured ~0.007 A)
SIGMA_CAP_FRAC = 0.02       # +/-2% uncertainty in capacity / sensor gain
SIGMA_V_MEAS = 0.006        # measured average-cell voltage noise + model error (V)
# How much the RC voltages may wander per step (V). Under load the fixed
# R1/C1/R2/C2 values are only approximate, so the filter may adjust them; at
# rest the RC voltages simply decay, so they get almost no freedom (otherwise
# they could "explain away" a voltage that really means a different SOC).
SIGMA_RC_V_LOAD = 0.0005    # at about 2.5 A
SIGMA_RC_V_REST = 0.00002   # at 0 A
LOAD_REF_A = 2.5
# Extra SOC uncertainty per second for slow, systematic errors (offset drift,
# capacity error, temperature). Without it the filter grows over-confident and
# stops listening to the voltage. 1e-8 per s = about 0.6% per sqrt(hour).
Q_SOC_PER_S = 1e-8

# Rest detection and offset learning
REST_THRESHOLD_A = 0.10     # |current| below this = resting
REST_LEARN_AFTER_S = 60     # start learning the zero after this long at rest
OFFSET_LEARN_TIME_S = 120   # time constant of the zero-learning average

MAX_DT_S = 30               # gaps longer than this are reported as dropouts

SLOPE_WINDOW = 0.05         # OCV slope measured over +/-5% SOC
MIN_SLOPE_V = 0.01          # never treat the OCV curve as perfectly flat (0.1 mV per %)

CHART = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
         "violet": "#6250d6", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}


# ---------------------------------------------------------------- model
class CellModel:
    """Your pack's average cell, built from ekf_parameters.json."""

    def __init__(self, params):
        self.capacity_ah = params["capacity_ah"]
        table = params["ocv_soc_table"]
        soc = np.array(table["soc_pct"]) / 100.0
        ocv = np.array(table.get("mean_cell_ocv_v_monotonic", table["mean_cell_ocv_v"]))
        order = np.argsort(soc)
        self.soc_pts = np.clip(soc[order], 0.0, 1.0)
        self.ocv_pts = ocv[order]

        steps = params["per_step_mean_cell"]
        r0_soc = np.array(steps["soc_after_pct"]) / 100.0
        r0 = np.array(steps["R0_ohm"])
        order = np.argsort(r0_soc)
        self.r0_soc, self.r0_vals = r0_soc[order], r0[order]

        two = params["two_rc_mean_cell"]["median"]
        self.R1, self.C1 = two["R1_ohm"], two["C1_F"]
        self.R2, self.C2 = two["R2_ohm"], two["C2_F"]
        self.tau1, self.tau2 = self.R1 * self.C1, self.R2 * self.C2

    def ocv(self, soc):
        return float(np.interp(soc, self.soc_pts, self.ocv_pts))

    def docv_dsoc(self, soc):
        """Slope of the OCV curve at this SOC (V per unit SOC), measured across
        a +/-5% SOC window. A table segment can be exactly flat (two neighbouring
        points with the same voltage); a slope of zero there would make the
        filter ignore the voltage completely and get stuck, so the slope is
        taken over a wider window and never allowed below a tiny floor."""
        lo, hi = max(soc - SLOPE_WINDOW, 0.0), min(soc + SLOPE_WINDOW, 1.0)
        if hi - lo < 1e-6:
            return MIN_SLOPE_V
        slope = (self.ocv(hi) - self.ocv(lo)) / (hi - lo)
        return float(max(slope, MIN_SLOPE_V))

    def soc_from_ocv(self, v):
        return float(np.interp(v, self.ocv_pts, self.soc_pts))

    def r0(self, soc):
        return float(np.interp(soc, self.r0_soc, self.r0_vals))


# ---------------------------------------------------------------- data
def load_log(path):
    """Read a logger CSV (new or old column names) into a standard frame."""
    df = pd.read_csv(path)
    rename = {"esp_timestamp_ms": "esp_ms", "cell1": "c1_v", "cell2": "c2_v",
              "cell3": "c3_v", "cell4": "c4_v", "current": "current_a"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    ms = df["esp_ms"].to_numpy(dtype=float)
    for j in np.where(np.diff(ms) < 0)[0]:          # stitch an ESP32 restart
        ms[j + 1:] += ms[j] - ms[j + 1] + 1000.0
    df["t"] = (ms - ms[0]) / 1000.0
    return df


def hampel(series, window, n_sigmas, floor):
    """Replace single-reading spikes with the local median."""
    med = series.rolling(window, center=True, min_periods=1).median()
    dev = (series - med).abs()
    mad = dev.rolling(window, center=True, min_periods=1).median()
    threshold = np.maximum(n_sigmas * 1.4826 * mad, floor)
    spikes = dev > threshold
    return series.where(~spikes, med), int(spikes.sum())



def run_ekf(t, i_meas, v_meas, model, soc0, p_soc0):
    n = len(t)
    soc_out = np.zeros(n)
    sig_out = np.zeros(n)
    vpred_out = np.full(n, np.nan)
    offset_out = np.zeros(n)
    gain_out = np.zeros(n)

    x = np.array([soc0, 0.0, 0.0])
    P = np.diag([p_soc0, 1e-6, 1e-6])
    Rv = SIGMA_V_MEAS ** 2
    offset = 0.0
    rest_time = 0.0
    q_ah = model.capacity_ah
    prev_i = 0.0

    for k in range(n):
        dt = t[k] - t[k - 1] if k > 0 else 0.0

        
        if abs(i_meas[k] - offset) < REST_THRESHOLD_A:
            rest_time += dt
            if rest_time >= REST_LEARN_AFTER_S and dt > 0:
                alpha = min(dt / OFFSET_LEARN_TIME_S, 1.0)
                offset += alpha * (i_meas[k] - offset)
        else:
            rest_time = 0.0
        i_k = i_meas[k] - offset

        
        if dt > 0:
            a1 = np.exp(-dt / model.tau1)
            a2 = np.exp(-dt / model.tau2)
            x = np.array([
                x[0] - prev_i * dt / (3600.0 * q_ah),
                a1 * x[1] + model.R1 * (1 - a1) * prev_i,
                a2 * x[2] + model.R2 * (1 - a2) * prev_i,
            ])
            A = np.diag([1.0, a1, a2])
            q_soc = ((SIGMA_I_A + SIGMA_CAP_FRAC * abs(prev_i)) * dt / (3600.0 * q_ah)) ** 2 \
                + Q_SOC_PER_S * dt
            s_rc = SIGMA_RC_V_REST + SIGMA_RC_V_LOAD * min(abs(prev_i) / LOAD_REF_A, 1.0)
            Q = np.diag([q_soc, s_rc ** 2, s_rc ** 2])
            P = A @ P @ A.T + Q
        x[0] = min(max(x[0], 0.0), 1.0)

        
        v_pred = model.ocv(x[0]) - x[1] - x[2] - model.r0(x[0]) * i_k
        vpred_out[k] = v_pred
        if np.isfinite(v_meas[k]):
            C = np.array([model.docv_dsoc(x[0]), -1.0, -1.0])
            S = C @ P @ C + Rv
            K = P @ C / S
            x = x + K * (v_meas[k] - v_pred)
            P = (np.eye(3) - np.outer(K, C)) @ P
            P = 0.5 * (P + P.T)
            gain_out[k] = K[0]
        x[0] = min(max(x[0], 0.0), 1.0)

        soc_out[k] = x[0]
        sig_out[k] = np.sqrt(max(P[0, 0], 0.0))
        offset_out[k] = offset
        prev_i = i_k

    return soc_out, sig_out, vpred_out, offset_out, gain_out


def coulomb_only(t, i_meas, soc0, capacity_ah):
    """Plain coulomb counting with the raw current and the same start: what the
    EKF would be without voltage corrections or offset learning."""
    dq = np.concatenate([[0.0], i_meas[:-1] * np.diff(t)]) / 3600.0
    return np.clip(soc0 - np.cumsum(dq) / capacity_ah, 0.0, 1.0)


def reference_soc(t, i_meas, capacity_ah):
    """Hindsight 'answer key' for a test that starts FULL and rested:
    the current offset is measured in every rest (true current = 0 A there) and
    removed, then the corrected current is integrated from 100%."""
    resting = np.abs(i_meas) < REST_THRESHOLD_A
    edges = np.flatnonzero(np.diff(resting.astype(int))) + 1
    bounds = np.concatenate([[0], edges, [len(t)]])
    offsets = np.zeros(len(t))
    last = 0.0
    segs = list(zip(bounds[:-1], bounds[1:]))
    # offset of each rest (skip its first 30 s)
    rest_off = {}
    for a, b in segs:
        if resting[a]:
            m = t[a:b] >= t[a] + 30
            vals = i_meas[a:b][m]
            rest_off[a] = float(np.median(vals)) if len(vals) else None
    for a, b in segs:
        if resting[a] and rest_off.get(a) is not None:
            last = rest_off[a]
        offsets[a:b] = last              # load periods use the rest before them
    i_corr = i_meas - offsets
    dq = np.concatenate([[0.0], 0.5 * (i_corr[1:] + i_corr[:-1]) * np.diff(t)]) / 3600.0
    return 1.0 - np.cumsum(dq) / capacity_ah, float(np.sum(dq))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="EKF SOC estimator (2-RC model)")
    ap.add_argument("log")
    ap.add_argument("--params", default="ekf_parameters.json")
    ap.add_argument("--soc0", type=float, default=None)
    ap.add_argument("--reference-full", action="store_true")
    args = ap.parse_args()

    with open(args.params) as f:
        model = CellModel(json.load(f))

    df = load_log(args.log)
    cells = ["c1_v", "c2_v", "c3_v", "c4_v"]
    spikes = 0
    for c in cells:
        df[c], n = hampel(df[c], HAMPEL_WINDOW, HAMPEL_SIGMAS, SPIKE_FLOOR_CELL_V)
        spikes += n
    df["avg_cell"] = df[cells].mean(axis=1)

    t = df["t"].to_numpy()
    i_meas = df["current_a"].to_numpy(dtype=float)
    v_meas = df["avg_cell"].to_numpy(dtype=float)
    gaps = int(np.sum(np.diff(t) > MAX_DT_S))

    if args.soc0 is not None:
        soc0, p_soc0 = args.soc0, 0.2 ** 2          # unsure: +/-20%
    else:
        soc0, p_soc0 = model.soc_from_ocv(v_meas[0]), 0.1 ** 2

    soc, sig, vpred, offset, gain = run_ekf(t, i_meas, v_meas, model, soc0, p_soc0)
    df["soc_ekf"] = soc
    df["soc_sigma"] = sig
    df["v_pred"] = vpred
    df["offset_learned"] = offset
    df["soc_gain"] = gain
    df["soc_coulomb_only"] = coulomb_only(t, i_meas, soc0, model.capacity_ah)

    base = os.path.splitext(args.log)[0]
    lines = [
        f"File: {args.log}",
        f"Rows: {len(df)}, duration {t[-1] / 3600:.2f} h, dropouts > {MAX_DT_S} s: {gaps}",
        f"Voltage spikes removed: {spikes}",
        f"Model: capacity {model.capacity_ah:.3f} Ah, R1 {model.R1 * 1000:.1f} mOhm / tau1 {model.tau1:.0f} s, "
        f"R2 {model.R2 * 1000:.1f} mOhm / tau2 {model.tau2:.0f} s",
        f"Start SOC (EKF): {soc0 * 100:.1f}%   End SOC: {soc[-1] * 100:.1f}%",
        f"Learned current offset at end: {offset[-1]:+.4f} A",
        f"Voltage model error (rms, measured - predicted): "
        f"{1000 * np.sqrt(np.nanmean((v_meas - vpred) ** 2)):.1f} mV",
    ]

    if args.reference_full:
        ref, total_ah = reference_soc(t, i_meas, model.capacity_ah)
        df["soc_reference"] = ref
        err = (soc - ref) * 100
        err_cc = (df["soc_coulomb_only"].to_numpy() - ref) * 100
        settle = t > t[0] + 1800                    # judge after a 30-min settling period
        flat = settle & (ref > 0.3) & (ref < 0.8)
        ends = settle & ~((ref > 0.3) & (ref < 0.8))
        rms = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
        lines += [
            "",
            "ACCURACY vs hindsight reference (after the first 30 min)",
            f"  Charge removed (corrected): {total_ah:.3f} Ah, reference end SOC {ref[-1] * 100:.1f}%",
            f"  EKF RMS error:              {rms(err[settle]):.2f}%   max {np.max(np.abs(err[settle])):.2f}%",
            f"  EKF RMS, flat zone 30-80%:  {rms(err[flat]):.2f}%",
            f"  EKF RMS, near full/empty:   {rms(err[ends]):.2f}%",
            f"  Coulomb-only RMS error:     {rms(err_cc[settle]):.2f}%   max {np.max(np.abs(err_cc[settle])):.2f}%",
        ]

    df.to_csv(f"{base}_soc.csv", index=False)
    with open(f"{base}_soc_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- plots: SOC, voltage check, current (three charts, one axis each)
    hours = t / 3600
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": CHART["grid"]})
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1]})
    ax = axes[0]
    if "soc_reference" in df:
        ax.plot(hours, df["soc_reference"] * 100, color=CHART["ink2"], lw=2, label="Reference (hindsight)")
    ax.plot(hours, soc * 100, color=CHART["blue"], lw=2, label="EKF estimate")
    ax.fill_between(hours, (soc - sig) * 100, (soc + sig) * 100, color=CHART["blue"], alpha=0.15,
                    lw=0, label="EKF +/-1 sigma")
    ax.plot(hours, df["soc_coulomb_only"] * 100, "--", color=CHART["orange"], lw=1.5,
            label="Coulomb counting only")
    ax.set_ylabel("SOC (%)")
    ax.set_title("State of charge", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(hours, v_meas, color=CHART["blue"], lw=1, label="Measured (average cell)")
    ax.plot(hours, vpred, color=CHART["orange"], lw=1, label="Model prediction")
    ax.set_ylabel("Cell voltage (V)")
    ax.set_title("Model check: predicted vs measured voltage", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(hours, i_meas - offset, color=CHART["blue"], lw=1)
    ax.set_ylabel("Current (A)")
    ax.set_xlabel("Time (h)")
    ax.set_title("Current (offset-corrected, discharge positive)", loc="left")

    for ax in axes:
        ax.grid(color=CHART["grid"], lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{base}_soc_plot.png", dpi=150)
    print(f"Saved {base}_soc.csv, {base}_soc_summary.txt and {base}_soc_plot.png")


if __name__ == "__main__":
    main()