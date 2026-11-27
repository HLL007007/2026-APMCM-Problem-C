"""P3 part1 supplement: no-skill baseline and model comparison tests.

Run after P3_part1.py:
    python P3_part1_supplement.py

Outputs in outputs_p3_part1/:
  P3_part1_baseline.csv
  P3_part1_model_tests.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import P3_part1 as p1


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_p3_part1"


def main():
    df = p1.load()
    counts = df.state.value_counts(normalize=True)
    y = df.state.to_numpy()
    p_marg = counts.reindex(range(8), fill_value=0).to_numpy()
    p_marg = np.clip(p_marg, 1e-12, None)
    marginal_ll = float(-np.mean(np.log(p_marg[y])))
    majority_ll = float(-np.log(float(counts.max())))
    baseline = pd.DataFrame([
        {"Model": "Marginal state distribution (no-skill)",
         "LogLoss": marginal_ll, "Note": "Global state frequencies from Data02"},
        {"Model": "Majority state (no-skill)", "LogLoss": majority_ll,
         "Note": "Degenerate baseline: zero probability for non-None states; not a usable probabilistic forecast"},
        {"Model": "Hurdle pairwise (CV, expID)", "LogLoss": 1.1151466058090613,
         "Note": "From P3_part1 validation.csv"},
        {"Model": "Multinomial (CV, expID)", "LogLoss": 1.1614972834867465,
         "Note": "From P3_part1 validation.csv"},
    ])
    baseline.to_csv(OUT / "P3_part1_baseline.csv", index=False,
                    encoding="utf-8-sig")

    folds = pd.read_csv(OUT / "validation_folds.csv")
    piv = folds.pivot_table(index=["split", "repeat", "fold"],
                            columns="model", values="log_loss").reset_index()
    rows = []
    for split, g in piv.groupby("split"):
        for a, b in [("Hurdle pairwise", "Multinomial"),
                     ("Hurdle pairwise", "Hurdle additive"),
                     ("Hurdle additive", "Multinomial")]:
            if a not in g.columns or b not in g.columns:
                continue
            d = (g[a] - g[b]).dropna().to_numpy()
            if len(d) < 3:
                continue
            t, p = stats.ttest_rel(g[a].dropna(), g[b].dropna())
            se = d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
            rows.append({"split": split, "Comparison": f"{a} - {b}",
                         "N_pairs": len(d), "Mean_logloss_diff": float(d.mean()),
                         "CI_Low": float(lo), "CI_High": float(hi),
                         "t": float(t), "p_value": float(p)})
    tests = pd.DataFrame(rows)
    if len(tests):
        # Holm within each split
        for split, idx in tests.groupby("split").groups.items():
            ps = tests.loc[idx, "p_value"].values
            order = np.argsort(ps)
            adj = np.empty(len(ps))
            running = 0.0
            for rank, j in enumerate(order):
                running = max(running, (len(ps) - rank) * ps[j])
                adj[j] = min(1.0, running)
            tests.loc[idx, "p_holm"] = adj
    tests.to_csv(OUT / "P3_part1_model_tests.csv", index=False,
                 encoding="utf-8-sig")
    print(baseline.round(4).to_string(index=False))
    print()
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
