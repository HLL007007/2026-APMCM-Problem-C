"""P2 threshold rationale using the P3 behavioral fidelity bridge.

Run after P2.py and P3_bridge.py:
    python P2_threshold_bridge.py

Outputs in outputs_p2/:
  P2_threshold_rationale.csv
  P2_fig16_threshold_rationale.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
P2 = ROOT / "outputs_p2"
BR = ROOT / "outputs_bridge"
CALLS = ["flat", "mod", "tv"]
PALETTE = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}


def main():
    avg = pd.read_csv(P2 / "P2_model_averaged_range.csv")
    eff = pd.read_csv(P2 / "P2_effective_range.csv")
    thr = pd.read_csv(BR / "B01_fidelity_thresholds.csv").set_index("Quantity")

    f50 = float(thr.loc["F at response probability 0.5", "Estimate"])
    f70 = float(thr.loc["F at response probability 0.7", "Estimate"])
    f50_lo = float(thr.loc["F at response probability 0.5", "CI_Low"])
    f50_hi = float(thr.loc["F at response probability 0.5", "CI_High"])
    f70_lo = float(thr.loc["F at response probability 0.7", "CI_Low"])
    f70_hi = float(thr.loc["F at response probability 0.7", "CI_High"])

    def interp(tau):
        if not np.isfinite(f50) or not np.isfinite(f70):
            return "behavioral calibration not identified"
        if tau < f50:
            return "below behavioral F50 (conservative for 50% response)"
        if tau <= f70:
            return "between behavioral F50 and F70 (intermediate reliability)"
        return "above behavioral F70 (stricter than 70% response)"

    rows = []
    for level, tau in [("Potential", 0.50), ("Recognition", 0.60), ("Reliable", 0.70)]:
        main = eff[(eff["Level"] == level) & (np.isclose(eff["tau"], tau))]
        main_range = float(main["Range_m"].iloc[0]) if len(main) else np.nan
        ar = avg[np.isclose(avg["tau"], tau)]
        avg_txt = {ct: float(ar[ar["CallType"] == ct]["Range_avg"].iloc[0])
                   for ct in CALLS if len(ar[ar["CallType"] == ct])}
        rows.append({
            "Acoustic_threshold": tau, "P2_level": level,
            "P2_main_range_m": main_range,
            "P2_model_avg_flat": avg_txt.get("flat", np.nan),
            "P2_model_avg_mod": avg_txt.get("mod", np.nan),
            "P2_model_avg_tv": avg_txt.get("tv", np.nan),
            "Behavioral_F50": f50, "Behavioral_F70": f70,
            "F50_CI_Low": f50_lo, "F50_CI_High": f50_hi,
            "F70_CI_Low": f70_lo, "F70_CI_High": f70_hi,
            "Interpretation": interp(tau),
            "Note": ("Behavioral F50/F70 are cluster-robust logistic estimates on natural "
                     "calls; CIs are wide, so this is a consistency check, not a calibration.")
        })
    out = pd.DataFrame(rows)
    out.to_csv(P2 / "P2_threshold_rationale.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2),
                             gridspec_kw={"width_ratios": [1.4, 1.0]})
    for ct in CALLS:
        s = avg[avg["CallType"] == ct].sort_values("tau")
        axes[0].plot(s["tau"], s["Range_avg"], color=PALETTE[ct], lw=2, label=ct)
    for tau in [0.50, 0.60, 0.70]:
        axes[0].axvline(tau, color="gray", ls="--", lw=0.9, alpha=0.8)
    for val, color, label in [(f50, "#D62728", "Behavioral F50"),
                              (f70, "#8C2D04", "Behavioral F70")]:
        if np.isfinite(val):
            axes[0].axvline(val, color=color, ls=":", lw=1.4, label=label)
    axes[0].set_xlabel("Recognition threshold τ")
    axes[0].set_ylabel("Model-averaged effective range (m)")
    axes[0].set_title("P2 range–threshold curves vs P3 behavioral fidelity thresholds",
                      fontweight="bold")
    axes[0].legend(fontsize=8)

    y = [0, 1]
    vals = [f50, f70]
    lo = [f50_lo, f70_lo]
    hi = [f50_hi, f70_hi]
    axes[1].errorbar(vals, y, xerr=[np.array(vals) - np.array(lo),
                                    np.array(hi) - np.array(vals)],
                     fmt="o", capsize=5, color="#D62728", lw=1.4)
    for tau in [0.50, 0.60, 0.70]:
        axes[1].axvline(tau, color="gray", ls="--", lw=0.9, alpha=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(["F at response 0.5", "F at response 0.7"])
    axes[1].set_xlabel("Fidelity F")
    axes[1].set_xlim(-1.0, 1.2)
    axes[1].set_title("Behavioral fidelity thresholds (95% CI)", fontweight="bold")
    plt.suptitle("P2 Threshold Scheme vs Behavioral Evidence",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(P2 / "P2_fig16_threshold_rationale.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
