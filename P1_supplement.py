"""P1 supplement: main-factor evidence, rule classifier and key-measure hand-off.

Run after P1.py:
    python P1_supplement.py

Outputs in outputs_p1/:
  P1_main_factor_evidence.csv
  P1_rule_classifier.csv
  P1_key_measure_handoff.csv
  P1_fig14_main_factors.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             precision_recall_fscore_support)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_p1"
CALLS = ["flat", "mod", "tv"]
FEATURES = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75",
            "SpecBandwidth", "HarmonicRatio", "SpecSkew",
            "PeakAtFundamental", "F02_present", "PulseRate_present"]


def main():
    df = pd.read_csv(OUT / "P1_analysis_dataset.csv")
    df["CallType"] = df["CallType"].astype(str)
    y = df["CallType"].values
    X = df[FEATURES].astype(float).values
    mi = mutual_info_classif(X, y, random_state=42)

    aucs = {}
    for j, feat in enumerate(FEATURES):
        aucs[feat] = {}
        for c in CALLS:
            yb = (y == c).astype(int)
            if len(np.unique(yb)) < 2:
                aucs[feat][c] = np.nan
            else:
                aucs[feat][c] = float(roc_auc_score(yb, X[:, j]))

    kw = pd.read_csv(OUT / "P1_nonparam_kruskal_holm.csv")
    kw = kw.set_index("Feature")
    rows = []
    for j, feat in enumerate(FEATURES):
        if feat in kw.index:
            p_raw = float(kw.loc[feat, "p_raw"])
            p_holm = float(kw.loc[feat, "p_holm"])
            p_date = float(kw.loc[feat, "p_holm_date_adjusted"])
            eff = float(kw.loc[feat, "EffectSize"])
        else:
            groups = [X[y == c, j] for c in CALLS]
            p_raw = float(stats.kruskal(*groups).pvalue)
            p_holm = p_raw
            p_date = np.nan
            eff = np.nan
        if feat in ("F02_present", "PulseRate_present"):
            status = "definitional / structural"
        elif feat == "HarmonicRatio":
            status = "artifact (bimodal Fpeak/F01)"
        elif feat == "SpecBandwidth":
            status = "redundant with quartiles"
        elif feat == "F01":
            status = "date-robust discriminative"
        elif feat == "Fpeak":
            status = "naive significant, not date-robust"
        elif feat == "PeakAtFundamental":
            status = "regime indicator"
        elif feat == "Q50":
            status = "date-driven (not date-robust)"
        else:
            status = "weak / not significant"
        rows.append({
            "Feature": feat, "MutualInfo": float(mi[j]),
            "AUC_flat": aucs[feat]["flat"], "AUC_mod": aucs[feat]["mod"],
            "AUC_tv": aucs[feat]["tv"],
            "AUC_max": float(np.nanmax(list(aucs[feat].values()))),
            "KW_p_raw": p_raw, "KW_p_holm": p_holm,
            "KW_p_holm_date": p_date, "EffectSize_eps2": eff,
            "Status": status
        })
    ev = pd.DataFrame(rows).sort_values(
        ["Status", "AUC_max"], ascending=[True, False])
    ev.to_csv(OUT / "P1_main_factor_evidence.csv", index=False,
              encoding="utf-8-sig")

    # Definitional rule classifier (explicitly structural, not an acoustic learning result)
    pred = np.where(df["F02_present"].values == 1, "tv",
                    np.where(df["PulseRate_present"].values == 1, "mod", "flat"))
    cm = confusion_matrix(y, pred, labels=CALLS)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=CALLS, zero_division=0)
    acc = float(np.mean(y == pred))
    rule = pd.DataFrame([
        {"CallType": c, "Precision": float(prec[i]), "Recall": float(rec[i]),
         "F1": float(f1[i]), "Support": int(sup[i])}
        for i, c in enumerate(CALLS)
    ])
    rule["Overall_Accuracy"] = acc
    rule["Rule"] = "F02 present -> tv; else PulseRate present -> mod; else flat"
    rule["Note"] = ("Definitional rule based on measurement availability; it is not a "
                    "learned acoustic classifier and should not be reported as such.")
    rule.to_csv(OUT / "P1_rule_classifier.csv", index=False, encoding="utf-8-sig")

    handoff = pd.DataFrame([
        ("F01", "First fundamental (carrier)", "date-robust",
         "P2 carrier/absorption; P4 source level", "not used directly (call type)"),
        ("F02", "Second fundamental (two-voice)", "definitional for tv; F02/F01~1.11; ~30 Hz beat",
         "P2 two-voice robustness; P4 signature", "call type tv"),
        ("Fpeak", "Peak frequency", "naive significant; regime-dependent",
         "P2 spectrogram/amplitude-spectrum fidelity", "not used directly"),
        ("Q25", "25th-percentile frequency", "not significant", "P2 spectral envelope", "not used"),
        ("Q50", "Median frequency", "date-driven", "P2 spectral envelope", "not used"),
        ("Q75", "75th-percentile frequency", "not significant", "P2 spectral envelope", "not used"),
        ("Duration", "Call duration", "not significant", "P2 envelope correlation; P3 exposure", "not used"),
        ("PulseRate", "Pulse rate (mod/tv only)", "mod vs tv not significant",
         "P2 envelope correlation; P3 latency", "not used directly"),
        ("SpecBandwidth", "Q75-Q25", "redundant", "P2 spectral fidelity", "not used"),
        ("SpecSkew", "Spectral skewness", "not significant", "P2 spectral shape", "not used"),
        ("HarmonicRatio", "Fpeak/F01", "bimodal artifact; do not use as biology",
         "not recommended", "not recommended"),
        ("PeakAtFundamental", "Fpeak/F01<1.5", "regime indicator",
         "P2 spectral peak interpretation", "not used"),
    ], columns=["Feature", "Role", "Evidence", "P2_Use", "P3_Use"])
    handoff["P2_Use"] = handoff["P2_Use"].apply(
        lambda s: s if str(s).startswith("not") else "recommended: " + str(s))
    handoff["P3_Use"] = handoff["P3_Use"].apply(
        lambda s: s if str(s).startswith("not") else "recommended: " + str(s))
    handoff["Use_Status"] = ("recommended for interpretation/P4; not an input "
                             "to the current P2/P3 models")
    handoff.to_csv(OUT / "P1_key_measure_handoff.csv", index=False,
                   encoding="utf-8-sig")

    # Figure: mutual information + per-class AUC for the key features
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    e = ev.sort_values("MutualInfo")
    axes[0].barh(e["Feature"], e["MutualInfo"], color="#4C72B0")
    axes[0].set_xlabel("Mutual information with CallType")
    axes[0].set_title("Mutual information (univariate)", fontweight="bold")
    key = ["F01", "Fpeak", "PeakAtFundamental", "F02_present"]
    x = np.arange(len(key))
    width = 0.26
    for k, c in enumerate(CALLS):
        vals = [aucs[f][c] for f in key]
        axes[1].bar(x + (k - 1) * width, vals, width=width, label=c)
    axes[1].axhline(0.5, color="gray", ls="--", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(key)
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_ylabel("One-vs-rest AUC")
    axes[1].set_title("Per-class AUC of key features", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.suptitle("P1 Main-Factor Evidence: Structural vs Continuous",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "P1_fig14_main_factors.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(ev.round(3).to_string(index=False))
    print("\nDefinitional rule accuracy:", round(acc, 4))
    print(rule.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
