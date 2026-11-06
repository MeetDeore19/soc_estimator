"""
Battery characterisation analysis
=================================
Turns the logger's CSV from the step-discharge test (load ON for ~10% of
capacity, then rest ~1 hour, repeated until the BMS cuts off) into the
numbers the EKF needs for THIS pack:

  * capacity (Ah)            - total charge removed, offset-corrected
  * OCV-SOC table (per cell) - rested voltage at the end of every rest
  * R0  (per cell, per step) - instant voltage jump at each switch-off / current
  * R1, C1 (per cell, step)  - slow voltage recovery during each rest (1-RC fit)

Usage:
    python characterise.py <logger_csv> [output_folder]

Needs: pandas, numpy, matplotlib  (pip install pandas numpy matplotlib)

How it works, step by step
--------------------------
1. Load the CSV and build a clean time axis from the ESP32's own clock.
2. Remove glitches: single-second spikes in a cell voltage (e.g. a loose
   balance wire) and impossible currents while the pack was resting.
3. Split the test into ON blocks (bulbs on) and REST blocks (bulbs off).
4. Re-zero the current: during every rest the true current is 0 A, so the
   average reading there IS the sensor's offset at that time. Subtract it.
5. Integrate the corrected current from the start to the BMS cut-off ->
   capacity, and SOC at every second.
6. At the end of each rest, the cells have settled -> one OCV-SOC point.
7. At each switch-off, the instant jump in voltage / current -> R0.
8. During each rest, fit  V(t) = V_end - A*exp(-t/tau)  -> R1 = A/I, C1 = tau/R1.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = ["c1_v", "c2_v", "c3_v", "c4_v"]

# ---- thresholds (amps, volts, seconds) ----
ON_CURRENT_A = 1.0          # above this, the load is on
REST_CURRENT_A = 0.2        # below this (in size), the pack is resting
SPIKE_V = 0.08              # a cell reading this far from its local median is a glitch
MERGE_GAP_S = 10            # ON samples separated by less than this belong to one block
OCV_WINDOW_S = 120          # average the last 2 minutes of each rest for the OCV point
OFFSET_SKIP_S = 30          # ignore the first 30 s of a rest when estimating the offset
MIN_REST_S = 600            # rests shorter than 10 min are not used for OCV / RC fits
MIN_ON_S = 60               # an ON "block" shorter than this is a glitch, not a load step

# chart colours (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------- 1. load
def load(path):
    df = pd.read_csv(path)
    ms = df["esp_ms"].to_numpy(dtype=float)
    # handle an ESP32 restart (clock jumps backwards): stitch the clock
    jumps = np.where(np.diff(ms) < 0)[0]
    for j in jumps:
        ms[j + 1:] += ms[j] - ms[j + 1] + 1000.0
    df["t"] = (ms - ms[0]) / 1000.0
    return df, len(jumps)


# ---------------------------------------------------------------- 2. clean
def clean(df):
    df = df.copy()
    glitch_v = pd.Series(False, index=df.index)
    for c in CELLS:
        med = df[c].rolling(9, center=True, min_periods=1).median()
        bad = (df[c] - med).abs() > SPIKE_V
        glitch_v |= bad
    df.loc[glitch_v, CELLS] = np.nan

    # Currents between rest and on, or clearly negative, that are NOT next to
    # an ON sample are spurious (nothing was charging the pack).
    cur = df["current_a"]
    is_on = cur > ON_CURRENT_A
    near_on = is_on | is_on.shift(1, fill_value=False) | is_on.shift(-1, fill_value=False)
    odd = (cur.abs() >= REST_CURRENT_A) & ~is_on & ~near_on
    df["current_clean"] = cur.where(~odd, np.nan)
    return df, int(glitch_v.sum()), int(odd.sum())


# ---------------------------------------------------------------- 3. segment
def find_on_blocks(df):
    on_t = df.loc[df["current_clean"] > ON_CURRENT_A, "t"].to_numpy()
    if len(on_t) == 0:
        return []
    blocks, start, prev = [], on_t[0], on_t[0]
    for t in on_t[1:]:
        if t - prev > MERGE_GAP_S:
            blocks.append((start, prev))
            start = t
        prev = t
    blocks.append((start, prev))
    real = [b for b in blocks if b[1] - b[0] >= MIN_ON_S]
    # wipe the glitch spikes so they are treated as rest
    for a, b in blocks:
        if b - a < MIN_ON_S:
            df.loc[(df["t"] >= a - 1) & (df["t"] <= b + 1), "current_clean"] = np.nan
    return real


def rest_windows(df, blocks):
    """Rest periods: before the first block, between blocks, after the last."""
    edges = [df["t"].iloc[0]] + [x for b in blocks for x in b] + [df["t"].iloc[-1]]
    rests = []
    for i in range(0, len(edges), 2):
        a, b = edges[i], edges[i + 1]
        if i > 0:
            a += 1.0          # first second after switch-off is a transition
        if i + 1 < len(edges) - 1:
            b -= 1.0          # last second before switch-on is a transition
        rests.append((a, b))
    return rests


# ---------------------------------------------------------------- 4. offset
def rest_offset(df, a, b):
    m = (df["t"] >= a + OFFSET_SKIP_S) & (df["t"] <= b)
    vals = df.loc[m, "current_clean"].dropna()
    vals = vals[vals.abs() < REST_CURRENT_A]
    return float(vals.median()) if len(vals) else 0.0


def corrected_current(df, blocks, rests):
    """Subtract the offset seen in each rest. An ON block uses the offset of the
    rest just before it (the rest after the last block may follow a BMS trip,
    which can change the sensor's ground reference)."""
    offsets = [rest_offset(df, a, b) for a, b in rests]
    off = np.zeros(len(df))
    t = df["t"].to_numpy()
    for k, (a, b) in enumerate(rests):
        off[(t >= a - 1.0) & (t <= b + 1.0)] = offsets[k]
    for k, (a, b) in enumerate(blocks):
        off[(t >= a - 1.0) & (t <= b + 1.0)] = offsets[k]
    cur = df["current_clean"].to_numpy()
    # spurious samples during rests: the true current was zero
    cur = np.where(np.isnan(cur), off, cur)
    return cur - off, offsets


# ---------------------------------------------------------------- 5. capacity
def integrate(t, i):
    dq = np.concatenate([[0.0], 0.5 * (i[1:] + i[:-1]) * np.diff(t)])  # A*s
    return np.cumsum(dq) / 3600.0  # Ah


# ---------------------------------------------------------------- 8. RC fit
def fit_one_rc(tt, vv):
    """Fit v = v_end - A*exp(-tt/tau) by scanning tau and solving the rest
    by linear least squares. Returns (v_end, A, tau, rmse)."""
    best = None
    for tau in np.geomspace(5, 3000, 200):
        X = np.column_stack([np.ones_like(tt), -np.exp(-tt / tau)])
        coef, *_ = np.linalg.lstsq(X, vv, rcond=None)
        err = vv - X @ coef
        sse = float(err @ err)
        if best is None or sse < best[0]:
            best = (sse, coef[0], coef[1], tau)
    sse, v_end, a, tau = best
    return v_end, a, tau, (sse / len(tt)) ** 0.5


def monotonic_ocv(ocv):
    """OCV values made non-decreasing with SOC (listed in the table's order)."""
    order = np.argsort(ocv["soc_pct"].to_numpy())             # 0% -> 100%
    v = ocv["mean_cell_ocv"].to_numpy()[order]
    v = np.maximum.accumulate(v)
    out = np.empty_like(v)
    out[order] = v
    return [round(float(x), 4) for x in out]


def fit_two_rc(tt, vv):
    """Fit v = v_end - A1*exp(-tt/tau1) - A2*exp(-tt/tau2): a fast branch
    (seconds to a couple of minutes) and a slow branch (tens of minutes).
    LFP needs both: the first ~2 minutes recover much faster than the rest."""
    best = None
    for tau1 in np.geomspace(3, 200, 40):
        for tau2 in np.geomspace(200, 3000, 40):
            X = np.column_stack([np.ones_like(tt), -np.exp(-tt / tau1), -np.exp(-tt / tau2)])
            coef, *_ = np.linalg.lstsq(X, vv, rcond=None)
            err = vv - X @ coef
            sse = float(err @ err)
            if best is None or sse < best[0]:
                best = (sse, coef, tau1, tau2)
    sse, coef, tau1, tau2 = best
    return coef[0], coef[1], tau1, coef[2], tau2, (sse / len(tt)) ** 0.5


# ---------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "characterisation_results"
    os.makedirs(out, exist_ok=True)

    df, restarts = load(path)
    df, n_vglitch, n_iglitch = clean(df)
    blocks = find_on_blocks(df)
    rests = rest_windows(df, blocks)
    i_corr, offsets = corrected_current(df, blocks, rests)
    t = df["t"].to_numpy()
    df["i_corr"] = i_corr
    df["mean_cell_v"] = df[CELLS].mean(axis=1, skipna=False)
    df["q_ah"] = integrate(t, i_corr)

    cutoff_t = blocks[-1][1]
    capacity = float(np.interp(cutoff_t + 1.0, t, df["q_ah"]))
    df["soc"] = 1.0 - df["q_ah"] / capacity
    raw_counted = float(df["counted_ah"].iloc[-1])

    # ---- 6. OCV points (end of each rest)
    ocv_rows = []
    for k, (a, b) in enumerate(rests):
        if k > 0 and b - a < MIN_REST_S:
            continue
        m = (df["t"] >= b - OCV_WINDOW_S) & (df["t"] <= b)
        row = {"rest": k, "time_h": round(b / 3600, 3),
               "soc_pct": round(100 * float(df.loc[m, "soc"].mean()), 2),
               "rest_min": round((b - a) / 60, 1)}
        for c in CELLS:
            row[c.replace("_v", "_ocv")] = round(float(df.loc[m, c].median()), 4)
        row["mean_cell_ocv"] = round(np.mean([row[c.replace("_v", "_ocv")] for c in CELLS]), 4)
        ocv_rows.append(row)
    ocv = pd.DataFrame(ocv_rows)
    ocv.loc[0, "soc_pct"] = 100.0     # full and rested by definition
    ocv.to_csv(os.path.join(out, "ocv_table.csv"), index=False)

    # ---- 7 + 8. R0 at each switch-off, R1/C1 from each following rest
    step_rows = []
    for k, (a_on, b_on) in enumerate(blocks):
        rest_a, rest_b = rests[k + 1]
        on_m = (df["t"] >= b_on - 30) & (df["t"] <= b_on - 1)       # settled ON samples
        i_on = float(df.loc[on_m, "i_corr"].median())
        on_rows = df[(df["t"] <= b_on) & (df["i_corr"] > 0.9 * i_on)]
        before = on_rows.iloc[-1]                                   # last fully-ON second
        off_rows = df[(df["t"] > b_on) & (df["i_corr"].abs() < 0.1)]
        after = off_rows.iloc[0]                                    # first fully-OFF second
        row = {"step": k + 1,
               "on_min": round((b_on - a_on) / 60, 1),
               "i_on_a": round(i_on, 3),
               "soc_after_pct": round(100 * float(df.loc[df["t"] >= rest_a, "soc"].iloc[0]), 1),
               "rest_min": round((rest_b - rest_a) / 60, 1),
               "bms_cutoff": k == len(blocks) - 1}
        for c in CELLS:
            name = c.replace("_v", "")
            dv = after[c] - before[c]
            row[f"{name}_R0_mohm"] = round(1000 * dv / i_on, 1) if np.isfinite(dv) else np.nan
            if rest_b - rest_a >= MIN_REST_S:
                m = (df["t"] >= rest_a) & (df["t"] <= rest_b) & df[c].notna()
                tt = df.loc[m, "t"].to_numpy() - rest_a
                vv = df.loc[m, c].to_numpy()
                v_end, amp, tau, rmse = fit_one_rc(tt, vv)
                r1 = amp / i_on
                row[f"{name}_R1_mohm"] = round(1000 * r1, 1)
                row[f"{name}_tau_s"] = round(tau, 0)
                row[f"{name}_C1_F"] = round(tau / r1, 0) if r1 > 0 else np.nan
                row[f"{name}_fit_rmse_mv"] = round(1000 * rmse, 2)
        # mean-cell fit: averaging four cells halves the noise, so this is the
        # most reliable estimate and the one the EKF should use
        mc = "mean_cell_v"
        row["mean_R0_mohm"] = round(1000 * (after[mc] - before[mc]) / i_on, 1)
        if rest_b - rest_a >= MIN_REST_S:
            m = (df["t"] >= rest_a) & (df["t"] <= rest_b) & df[mc].notna()
            tt = df.loc[m, "t"].to_numpy() - rest_a
            v_end, amp, tau, rmse = fit_one_rc(tt, df.loc[m, mc].to_numpy())
            r1 = amp / i_on
            row["mean_R1_mohm"] = round(1000 * r1, 1)
            row["mean_tau_s"] = round(tau, 0)
            row["mean_C1_F"] = round(tau / r1, 0) if r1 > 0 else np.nan
            row["mean_fit_rmse_mv"] = round(1000 * rmse, 2)
            v_end2, a1, t1, a2, t2, rmse2 = fit_two_rc(tt, df.loc[m, mc].to_numpy())
            row["mean2_R1_mohm"] = round(1000 * a1 / i_on, 1)
            row["mean2_tau1_s"] = round(t1, 1)
            row["mean2_C1_F"] = round(t1 / (a1 / i_on), 0) if a1 > 0 else np.nan
            row["mean2_R2_mohm"] = round(1000 * a2 / i_on, 1)
            row["mean2_tau2_s"] = round(t2, 0)
            row["mean2_C2_F"] = round(t2 / (a2 / i_on), 0) if a2 > 0 else np.nan
            row["mean2_fit_rmse_mv"] = round(1000 * rmse2, 2)
        step_rows.append(row)
    steps = pd.DataFrame(step_rows)
    steps.to_csv(os.path.join(out, "step_parameters.csv"), index=False)

    # ---- EKF-ready parameter file (mean cell, steps before the BMS cut-off)
    good = steps[~steps["bms_cutoff"]]
    def mean_cols(suffix):
        return good[f"mean_{suffix}"]
    ekf = {
        "capacity_ah": round(capacity, 4),
        "ocv_soc_table": {
            "soc_pct": ocv["soc_pct"].tolist(),
            "mean_cell_ocv_v": ocv["mean_cell_ocv"].tolist(),
            # The EKF needs OCV to rise with SOC everywhere (its slope is the
            # Kalman gain's C). On LFP's flat plateau, 1-2 mV of noise can make a
            # point dip; this version removes those dips (running max from 0% up).
            "mean_cell_ocv_v_monotonic": monotonic_ocv(ocv),
        },
        "per_step_mean_cell": {
            "soc_after_pct": good["soc_after_pct"].tolist(),
            "R0_ohm": [round(x / 1000, 5) for x in mean_cols("R0_mohm")],
            "R1_ohm": [round(x / 1000, 5) for x in mean_cols("R1_mohm")],
            "C1_F": [round(x, 0) for x in mean_cols("C1_F")],
            "tau_s": [round(x, 0) for x in mean_cols("tau_s")],
        },
        "two_rc_mean_cell": {
            "soc_after_pct": good["soc_after_pct"].tolist(),
            "R1_ohm": [round(x / 1000, 5) for x in good["mean2_R1_mohm"]],
            "tau1_s": good["mean2_tau1_s"].tolist(),
            "R2_ohm": [round(x / 1000, 5) for x in good["mean2_R2_mohm"]],
            "tau2_s": good["mean2_tau2_s"].tolist(),
            "median": {
                "R1_ohm": round(float(good["mean2_R1_mohm"].median()) / 1000, 5),
                "C1_F": round(float(good["mean2_C1_F"].median()), 0),
                "R2_ohm": round(float(good["mean2_R2_mohm"].median()) / 1000, 5),
                "C2_F": round(float(good["mean2_C2_F"].median()), 0),
            },
        },
        "suggested_constants": {
            "R0_ohm": round(float(mean_cols("R0_mohm").median()) / 1000, 5),
            "R1_ohm": round(float(mean_cols("R1_mohm").median()) / 1000, 5),
            "C1_F": round(float(mean_cols("C1_F").median()), 0),
        },
        "notes": "Discharge-only OCV (no hysteresis). R0 includes ~1 s of fast "
                 "relaxation because readings are 1-second averages.",
    }
    with open(os.path.join(out, "ekf_parameters.json"), "w") as f:
        json.dump(ekf, f, indent=2)

    # ---- summary
    lines = [
        "CHARACTERISATION SUMMARY",
        "=" * 60,
        f"File:                     {os.path.basename(path)}",
        f"Duration:                 {t[-1] / 3600:.2f} h, {len(df)} readings, ESP32 restarts: {restarts}",
        f"Load steps found:         {len(blocks)} (last one ended by the BMS cut-off)",
        f"Glitches removed:         {n_vglitch} voltage rows, {n_iglitch} current rows",
        f"Rest offsets (A):         " + ", ".join(f"{o:+.3f}" for o in offsets),
        f"Raw counted charge:       {raw_counted:.3f} Ah (logger, no offset correction)",
        f"CAPACITY (corrected):     {capacity:.3f} Ah to BMS cut-off",
        f"Mean-cell OCV at 100%:    {ocv['mean_cell_ocv'].iloc[0]:.4f} V",
        f"Mean-cell OCV after trip: {ocv['mean_cell_ocv'].iloc[-1]:.4f} V",
        f"Typical R0 (mean cell):   {ekf['suggested_constants']['R0_ohm'] * 1000:.1f} mOhm",
        f"Typical R1 (mean cell):   {ekf['suggested_constants']['R1_ohm'] * 1000:.1f} mOhm",
        f"Typical C1 (mean cell):   {ekf['suggested_constants']['C1_F']:.0f} F",
        f"1-RC fit error (median):  {good['mean_fit_rmse_mv'].median():.2f} mV",
        f"2-RC fit (median):        R1 {ekf['two_rc_mean_cell']['median']['R1_ohm'] * 1000:.1f} mOhm, "
        f"tau1 {good['mean2_tau1_s'].median():.0f} s | R2 {ekf['two_rc_mean_cell']['median']['R2_ohm'] * 1000:.1f} mOhm, "
        f"tau2 {good['mean2_tau2_s'].median():.0f} s | error {good['mean2_fit_rmse_mv'].median():.2f} mV",
    ]
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nOCV table:\n", ocv.to_string(index=False))
    print("\nPer-step parameters:\n", steps.to_string(index=False))

    # ---- plots
    style = dict(color=INK_2)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2})

    # overview: current and cell voltages as two stacked charts (one axis each)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    hours = t / 3600
    ax1.plot(hours, df["i_corr"], color=SERIES[0], lw=1)
    ax1.set_ylabel("Current (A, corrected)")
    ax1.set_title("Characterisation test overview", color=INK, loc="left")
    for i, c in enumerate(CELLS):
        ax2.plot(hours, df[c], color=SERIES[i], lw=1, label=f"Cell {i + 1}")
    ax2.set_ylabel("Cell voltage (V)")
    ax2.set_xlabel("Time (h)")
    ax2.legend(frameon=False, ncol=4, loc="lower left")
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "overview.png"), dpi=150)
    plt.close(fig)

    # OCV-SOC curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, c in enumerate(CELLS):
        col = c.replace("_v", "_ocv")
        ax.plot(ocv["soc_pct"], ocv[col], "-o", color=SERIES[i], lw=2, ms=5, label=f"Cell {i + 1}")
    ax.set_xlabel("State of charge (%)")
    ax.set_ylabel("Rested voltage, OCV (V)")
    ax.set_title("OCV-SOC curve (discharge)", color=INK, loc="left")
    ax.invert_xaxis()
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ocv_soc.png"), dpi=150)
    plt.close(fig)

    # R0 and R1 vs SOC for the mean cell (two separate charts, one series each).
    # Per-cell values are in step_parameters.csv, but each cell is the
    # difference of two taps, so its single-second jump is noisier than the
    # four-cell average used here.
    for key, label, title, fname in [
            ("mean_R0_mohm", "R0 (mOhm)", "R0 by state of charge (mean cell)", "r0_vs_soc.png"),
            ("mean_R1_mohm", "R1 (mOhm)", "R1 by state of charge (mean cell)", "r1_vs_soc.png")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(good["soc_after_pct"], good[key], "-o", color=SERIES[0], lw=2, ms=5)
        ax.set_xlabel("State of charge after the step (%)")
        ax.set_ylabel(label)
        ax.set_ylim(0, max(good[key].max() * 1.2, 1))
        ax.set_title(title, color=INK, loc="left")
        ax.invert_xaxis()
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=150)
        plt.close(fig)

    # example relaxation fit: a mid-SOC rest, mean cell
    k = len(blocks) // 2
    rest_a, rest_b = rests[k + 1]
    m = (df["t"] >= rest_a) & (df["t"] <= rest_b) & df["mean_cell_v"].notna()
    tt = df.loc[m, "t"].to_numpy() - rest_a
    vv = df.loc[m, "mean_cell_v"].to_numpy()
    v_end, amp, tau, rmse = fit_one_rc(tt, vv)
    v2, a1, t1, a2, t2, rmse2 = fit_two_rc(tt, vv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tt / 60, vv, color=SERIES[0], lw=1, label="Measured (mean cell)")
    ax.plot(tt / 60, v_end - amp * np.exp(-tt / tau), color=SERIES[1], lw=2,
            label=f"1-RC fit (tau = {tau:.0f} s), error {1000 * rmse:.1f} mV")
    ax.plot(tt / 60, v2 - a1 * np.exp(-tt / t1) - a2 * np.exp(-tt / t2), color=SERIES[3], lw=2,
            label=f"2-RC fit (tau = {t1:.0f} s + {t2:.0f} s), error {1000 * rmse2:.1f} mV")
    ax.set_xlabel("Minutes after switch-off")
    ax.set_ylabel("Mean cell voltage (V)")
    ax.set_title(f"Voltage recovery after step {k + 1}", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "relaxation_fit_example.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved results to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
