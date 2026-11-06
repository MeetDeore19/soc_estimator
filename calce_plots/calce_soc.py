import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Usage: python3 calce_soc.py [CELL]
#   CELL is A1-007 (default) or A1-008
# ============================================================
CELL = sys.argv[1] if len(sys.argv) > 1 else "A1-007"

# ---- Simulated hardware problems (to test the filter) ----
SIM_CURRENT_OFFSET_A = 0.010   # current-sensor offset, like your ACS712's residual
INITIAL_SOC_ERROR = -0.20      # estimators start 20% too low

# ---- Kalman filter settings (same logic as soc_estimate.py) ----
P0 = 0.04                      # initial uncertainty: about +/-20% SOC
Q_PER_S = 2e-7                 # uncertainty added per second of coulomb counting
R_REST = 4e-4                  # trust in voltage when settled (about +/-2% SOC)
R_ACTIVE = 5e-2                # trust in voltage under load (mostly ignored)
REST_THRESHOLD_A = 0.02        # below this counts as no current (lab sensor is precise)
REST_SETTLE_S = 600            # voltage trusted only after this long at rest
MAX_DT_S = 60                  # longer gaps treated as logging dropouts

# ---- Detection thresholds ----
LOW_CURRENT_MIN_A = 0.01       # C/20 for a 1.1Ah cell is about 0.055 A
LOW_CURRENT_MAX_A = 0.10
FULL_VOLTAGE_V = 3.55          # LFP full-charge voltage region


def arr(series):
    """Plain float NumPy array from a pandas column."""
    return series.to_numpy(dtype=float)


def find_file(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"Could not find a file matching: {pattern}")
        sys.exit(1)
    return matches[0]


def load_channel(path):
    print(f"Loading {path} (this can take a minute)...")
    sheets = pd.read_excel(path, sheet_name=None)
    name = next(k for k in sheets if str(k).startswith("Channel"))
    df = sheets[name]
    df = df.rename(columns={
        "Test_Time(s)": "t",
        "Step_Index": "step",
        "Current(A)": "I",
        "Voltage(V)": "V",
        "Temperature (C)_1": "T",
    })
    df = df[["t", "step", "I", "V", "T"]].dropna().reset_index(drop=True)
    t = arr(df["t"])
    dt = np.clip(np.diff(t, prepend=t[0]), 0, MAX_DT_S)
    df["dt"] = dt
    # Lab convention: positive current = charging
    df["ah_charged"] = np.cumsum(arr(df["I"]) * dt) / 3600.0
    return df


def build_ocv(df):
    rows = []
    for step, g in df.groupby("step"):
        i = arr(g["I"])
        ah = abs(float(np.sum(i * arr(g["dt"]))) / 3600.0)
        rows.append((step, float(np.median(i)), ah, len(g)))
    steps = pd.DataFrame(rows, columns=["step", "median_I", "ah", "rows"])

    low = steps[(steps["median_I"].abs() >= LOW_CURRENT_MIN_A) &
                (steps["median_I"].abs() <= LOW_CURRENT_MAX_A)]
    dis = low[low["median_I"] < 0].sort_values("ah", ascending=False)
    chg = low[low["median_I"] > 0].sort_values("ah", ascending=False)

    print("\nLow-current steps found in the OCV test:")
    print(low.to_string(index=False))
    if dis.empty or chg.empty:
        print("Could not find both a low-current discharge and charge step.")
        sys.exit(1)

    d_step = dis.iloc[0]["step"]
    c_step = chg.iloc[0]["step"]

    d = df[df["step"] == d_step]
    q_d = -np.cumsum(arr(d["I"]) * arr(d["dt"])) / 3600.0
    Q_dis = float(q_d[-1])
    soc_d = 1.0 - q_d / Q_dis
    v_d = arr(d["V"])

    c = df[df["step"] == c_step]
    q_c = np.cumsum(arr(c["I"]) * arr(c["dt"])) / 3600.0
    Q_chg = float(q_c[-1])
    soc_c = q_c / Q_chg
    v_c = arr(c["V"])

    grid = np.linspace(0.0, 1.0, 201)
    order_d = np.argsort(soc_d)
    v_dis = np.interp(grid, soc_d[order_d], v_d[order_d])
    v_chg = np.interp(grid, soc_c, v_c)
    v_avg = (v_dis + v_chg) / 2.0

    print(f"\nUsing discharge step {d_step} ({Q_dis:.4f} Ah) and charge step {c_step} ({Q_chg:.4f} Ah)")
    return grid, v_dis, v_chg, v_avg, Q_dis


def main():
    ocv_path = find_file(f"calce/OCV25-*/{CELL}-OCV-25-*.xlsx")
    dyn_path = find_file(f"calce/DST-US06-FUDS-25/{CELL}-DST-US06-FUDS-25-*.xlsx")

    # ---------------- OCV curve ----------------
    ocv_df = load_channel(ocv_path)
    grid, v_dis, v_chg, v_avg, Q = build_ocv(ocv_df)

    mid = (grid >= 0.1) & (grid <= 0.9)
    hyst_mv = float(np.mean(v_chg[mid] - v_dis[mid])) * 1000
    print(f"Measured capacity (C/20 discharge): {Q:.4f} Ah")
    print(f"Average hysteresis gap (10-90% SOC): {hyst_mv:.1f} mV")

    pd.DataFrame({"soc": grid, "v_discharge": v_dis, "v_charge": v_chg,
                  "v_average": v_avg}).to_csv(f"calce_{CELL}_ocv.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(grid * 100, v_dis, label="Discharge (C/20)")
    plt.plot(grid * 100, v_chg, label="Charge (C/20)")
    plt.plot(grid * 100, v_avg, "k--", label="Average (used as OCV)")
    plt.xlabel("SOC (%)")
    plt.ylabel("Voltage (V)")
    plt.title(f"A123 LFP OCV curve, 25C, cell {CELL}  (hysteresis about {hyst_mv:.0f} mV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"calce_{CELL}_ocv.png", dpi=150)

    # Monotonic OCV for voltage-to-SOC lookup
    v_lookup = np.maximum.accumulate(v_avg) + np.arange(len(v_avg)) * 1e-7

    def ocv_to_soc(v):
        return np.interp(v, v_lookup, grid)

    # ---------------- Dynamic profiles ----------------
    dyn = load_channel(dyn_path)
    v_all = arr(dyn["V"])
    i_all = arr(dyn["I"])

    full_idx = np.where(v_all >= FULL_VOLTAGE_V)[0]
    if len(full_idx) == 0:
        anchor = int(np.argmax(v_all))
    else:
        after = np.where((np.arange(len(dyn)) > full_idx[0]) & (i_all < 0.05))[0]
        anchor = int(after[0]) if len(after) else int(full_idx[0])

    d = dyn.iloc[anchor:].reset_index(drop=True)
    t = arr(d["t"])
    hours = (t - t[0]) / 3600.0
    ah = arr(d["ah_charged"])
    soc_true = np.clip(1.0 + (ah - ah[0]) / Q, 0, 1)
    i_lab = arr(d["I"])
    v = arr(d["V"])
    dt = arr(d["dt"])

    # Our convention: positive = discharge, plus the simulated sensor offset
    i_meas = -i_lab + SIM_CURRENT_OFFSET_A

    start = 1.0 + INITIAL_SOC_ERROR

    # Estimator 1: coulomb counting only
    soc_cc = start - np.cumsum(i_meas * dt) / (3600.0 * Q)

    # Estimator 2: voltage lookup only
    soc_v = ocv_to_soc(v)

    # Estimator 3: Kalman filter
    soc_kf = np.zeros(len(d))
    soc, P, rest_time = start, P0, 0.0
    for k in range(len(d)):
        soc = soc - i_meas[k] * dt[k] / (3600.0 * Q)
        soc = min(max(soc, 0.0), 1.0)
        P = P + Q_PER_S * dt[k]
        if abs(i_lab[k]) < REST_THRESHOLD_A:
            rest_time += dt[k]
        else:
            rest_time = 0.0
        R = R_REST if rest_time >= REST_SETTLE_S else R_ACTIVE
        K = P / (P + R)
        soc = soc + K * (soc_v[k] - soc)
        soc = min(max(soc, 0.0), 1.0)
        P = (1.0 - K) * P
        soc_kf[k] = soc

    out = pd.DataFrame({
        "hours": hours, "current_A": i_lab, "voltage_V": v,
        "soc_true": soc_true, "soc_coulomb": soc_cc,
        "soc_voltage": soc_v, "soc_kalman": soc_kf,
    })
    out.to_csv(f"calce_{CELL}_soc.csv", index=False)

    def err(est):
        e = (est - soc_true) * 100
        return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))), e

    print(f"\nDynamic test: {hours[-1]:.1f} h analysed after the first full charge")
    print(f"Simulated: current offset {SIM_CURRENT_OFFSET_A * 1000:.0f} mA, "
          f"starting SOC error {INITIAL_SOC_ERROR * 100:.0f}%")
    print(f"\n{'Estimator':<22}{'RMSE (%)':>10}{'Max error (%)':>16}")
    errors = {}
    for label, est in [("Coulomb counting", soc_cc), ("Voltage lookup", soc_v), ("Kalman filter", soc_kf)]:
        rmse, mx, e = err(est)
        errors[label] = e
        print(f"{label:<22}{rmse:>10.2f}{mx:>16.2f}")

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(hours, i_lab, linewidth=0.6)
    axes[0].set_ylabel("Current (A)\n(+ = charge)")
    axes[0].grid(True)
    axes[1].plot(hours, v, linewidth=0.6, color="purple")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True)
    axes[2].plot(hours, soc_true * 100, "k", linewidth=1.5, label="True SOC (lab)")
    axes[2].plot(hours, soc_cc * 100, "--", linewidth=1, label="Coulomb counting")
    axes[2].plot(hours, soc_v * 100, ".", markersize=1, alpha=0.3, label="Voltage lookup")
    axes[2].plot(hours, soc_kf * 100, linewidth=1.2, label="Kalman filter")
    axes[2].set_ylabel("SOC (%)")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True)
    for label in ["Coulomb counting", "Kalman filter"]:
        axes[3].plot(hours, errors[label], linewidth=1, label=label)
    axes[3].axhline(0, color="k", linewidth=0.5)
    axes[3].set_ylabel("Error (% SOC)")
    axes[3].set_xlabel("Time (hours)")
    axes[3].legend(loc="best", fontsize=8)
    axes[3].grid(True)
    fig.suptitle(f"A123 LFP cell {CELL}, DST/US06/FUDS at 25C")
    plt.tight_layout()
    plt.savefig(f"calce_{CELL}_soc.png", dpi=150)

    print(f"\nSaved calce_{CELL}_ocv.png, calce_{CELL}_soc.png and CSVs")


if __name__ == "__main__":
    main()