"""P3 part2 audit sensitivity: nonresponse with positive latency/intensity.

Run after P3_part2.py:
    python P3_part2_sensitivity.py

Outputs in outputs_p3_part2/:
  audit_sensitivity_counts.csv
  audit_sensitivity.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import P3_part1 as p1
import P3_part2 as p2


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_p3_part2"


def main():
    df = p1.load()
    flags = df[[b + "YesNo" for b in p2.BEHAVIORS]].to_numpy(bool)
    raw = df[[b + "Latency" for b in p2.BEHAVIORS]].to_numpy(float)
    marks = df[p2.MARKS].to_numpy(float)

    alt = flags | (np.isfinite(raw) & (raw >= 0)) | (marks > 0)
    t_mask = np.where(flags, raw, np.nan)
    t_alt = np.where(alt, raw, np.nan)
    y_mask = np.where(flags & (marks > 0), marks, np.nan)
    y_alt = np.where(alt & (marks > 0), marks, np.nan)

    def add_first(t3):
        first = np.min(np.where(np.isfinite(t3), t3, np.inf), axis=1)
        first = np.where(np.isfinite(first), first, np.nan)
        return np.column_stack([t3, first])

    t_mask = add_first(t_mask)
    t_alt = add_first(t_alt)

    counts = []
    for k, b in enumerate(p2.BEHAVIORS):
        counts.append({
            "Behavior": b,
            "Flagged_responses": int(flags[:, k].sum()),
            "Alt_responses": int(alt[:, k].sum()),
            "Nonresponse_with_latency": int((~flags[:, k] & np.isfinite(raw[:, k]) & (raw[:, k] >= 0)).sum()),
            "Nonresponse_with_positive_intensity": int((~flags[:, k] & (marks[:, k] > 0)).sum()),
        })
    pd.DataFrame(counts).to_csv(OUT / "audit_sensitivity_counts.csv",
                                index=False, encoding="utf-8-sig")

    x = p1.design(df)[:, :8]
    pre = np.log1p(df.VocalNcallsBefore.to_numpy(float))
    settings = pd.read_csv(OUT / "settings.csv")
    lam = float(settings.penalty.iloc[0])

    grid = pd.DataFrame({
        "CallType": p1.GROUP, "Distance": 100, "Context": "commuting",
        "Npenguins": df.Npenguins.median()
    })
    gx = p1.design(grid)
    xg = gx[:, :8]
    pre_grid = np.full(len(grid), np.median(pre))

    rows = []
    for label, t, y in [("masked", t_mask, y_mask), ("as_response", t_alt, y_alt)]:
        tm = p2.fit_time(x, t, p2.MODELS[1], lam)
        par = p2.time_parameters(tm, xg)
        med = p2.quantile(*par, .5)
        mk = p2.fit_marks(x, t, y, pre, False)
        inten = p2.predict_marks(mk, xg, pre_grid, par, False)
        for i, ct in enumerate(p1.GROUP):
            for k, name in enumerate(p2.TASKS):
                rows.append({"Scenario": label, "CallType": ct, "Endpoint": name,
                             "Latency_median_s": float(med[i, k]),
                             "Intensity_mean": np.nan})
            for k, name in enumerate(p2.MARKS):
                rows.append({"Scenario": label, "CallType": ct, "Endpoint": name,
                             "Latency_median_s": np.nan,
                             "Intensity_mean": float(inten[i, k])})
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["CallType", "Endpoint"], columns="Scenario",
                          values=["Latency_median_s", "Intensity_mean"]).reset_index()
    piv.columns = ["_".join([str(x) for x in c if x != ""]).strip("_")
                   if isinstance(c, tuple) else c for c in piv.columns]
    for metric in ["Latency_median_s", "Intensity_mean"]:
        a, b = metric + "_masked", metric + "_as_response"
        if a in piv and b in piv:
            piv[metric + "_diff"] = piv[b] - piv[a]
    piv.to_csv(OUT / "audit_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(counts).to_string(index=False))
    print(piv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
