"""Cross-question bridge: P2 acoustic fidelity -> P3 playback response/latency.

Run after P1/P2/P3 outputs exist:
    python P3_bridge.py

Outputs in outputs_bridge/:
  B01_response_bridge.csv       fidelity vs distance response models (cluster robust)
  B01_fidelity_thresholds.csv   behavioral F_50/F_70 (bootstrap)
  B01_threshold_rationale.csv   comparison with P2 acoustic thresholds 0.50/0.60/0.70
  B01_fidelity_response.png
  B02_cascade.csv               P(V|H), P(A|H,V), P(A|H) with bootstrap CIs
  B02_cascade.png
  B03_latency_bridge.csv        AFT latency: distance vs fidelity (natural calls)
  B03_latency_bridge.png
  B04_call_type_evidence.csv    call type x P2 range x P3 response states
  B05_summary.json
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from statsmodels.discrete.discrete_model import Logit

import P3_part1 as p1
import P3_part2 as p2


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_bridge"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

GROUP = p1.GROUP
CONTEXTS = p1.CONTEXT
NAMES = p1.NAMES
SEED = 20261129
N_BOOT = 300


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def load_fidelity():
    """Per-call-type isotonic fidelity from the P2 analysis dataset."""
    path = ROOT / "outputs_p2/P2_analysis_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Run P2.py first: " + str(path))
    d = pd.read_csv(path)
    iso = {}
    for ct in GROUP[1:]:
        s = d[d.CallType == ct].dropna(subset=["F_main"])
        iso[ct] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            s.Distance.values.astype(float), s.F_main.values.astype(float))
    return iso


def add_fidelity(df, iso):
    f = []
    for ct, dist in zip(df.CallType, df.Distance):
        f.append(float(iso[ct].predict([float(dist)])[0]) if ct in iso else np.nan)
    df = df.copy()
    df["F_model"] = f
    return df


def fit_cluster_logit(X, y, groups):
    return Logit(y, X).fit(disp=0, cov_type="cluster",
                           cov_kwds={"groups": groups})


def cv_logloss(X, y, groups, n_splits=5, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    folds = np.array_split(rng.permutation(uniq), n_splits)
    losses = []
    for hold in folds:
        te = np.isin(groups, hold)
        tr = ~te
        if te.sum() < 3 or tr.sum() < 10:
            continue
        try:
            m = Logit(y[tr], X[tr]).fit(disp=0)
            p = np.clip(m.predict(X[te]), 1e-6, 1 - 1e-6)
            losses.append(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
        except Exception:
            continue
    return float(np.mean(losses)) if losses else np.nan


def response_bridge(df):
    """Does acoustic fidelity predict any behavioral response among natural calls?"""
    nat = df[df.CallType != "control"].copy()
    nat["search"] = nat.Context.eq("searching").astype(float)
    nat["after"] = nat.Context.eq("after").astype(float)
    nat["logsize"] = np.log2(nat.Npenguins / 8.0)
    nat["dist50"] = (nat.Distance - 100.0) / 50.0
    y = nat.ResponseYesNo.values.astype(float)
    groups = nat.expID.values

    Xf = np.column_stack([np.ones(len(nat)), nat.F_model.values,
                          nat.search.values, nat.after.values, nat.logsize.values])
    Xd = np.column_stack([np.ones(len(nat)), nat.dist50.values,
                          nat.search.values, nat.after.values, nat.logsize.values])
    mf = fit_cluster_logit(Xf, y, groups)
    md = fit_cluster_logit(Xd, y, groups)

    rows, comp = [], []
    for label, model, X, term, coef_i, scale in [
        ("Fidelity model", mf, Xf, "F per 0.1", 1, 0.1),
        ("Distance model", md, Xd, "Distance per 50 m", 1, 1.0),
    ]:
        beta = float(model.params[coef_i])
        se = float(model.bse[coef_i])
        rows.append({
            "Model": label, "Term": term,
            "OR": float(np.exp(beta * scale)),
            "OR_Low": float(np.exp((beta - 1.96 * se) * scale)),
            "OR_High": float(np.exp((beta + 1.96 * se) * scale)),
            "p_value": float(model.pvalues[coef_i]),
            "loglik": float(model.llf), "AIC": float(model.aic),
            "BIC": float(model.bic),
            "CV_logloss": cv_logloss(X, y, groups)
        })
    # Context and group size rows (same in both; report from F model)
    for i, term in [(2, "Searching vs commuting"), (3, "After feeding vs commuting"), (4, "Group size (per doubling)")]:
        beta = float(mf.params[i]); se = float(mf.bse[i])
        rows.append({"Model": "Fidelity model", "Term": term,
                     "OR": float(np.exp(beta)),
                     "OR_Low": float(np.exp(beta - 1.96 * se)),
                     "OR_High": float(np.exp(beta + 1.96 * se)),
                     "p_value": float(mf.pvalues[i]),
                     "loglik": np.nan, "AIC": np.nan, "BIC": np.nan,
                     "CV_logloss": np.nan})
    bridge = pd.DataFrame(rows)
    save(bridge, "B01_response_bridge")
    comp = pd.DataFrame([
        {"Model": "Fidelity model", "loglik": mf.llf, "AIC": mf.aic,
         "BIC": mf.bic, "CV_logloss": bridge.loc[0, "CV_logloss"]},
        {"Model": "Distance model", "loglik": md.llf, "AIC": md.aic,
         "BIC": md.bic, "CV_logloss": bridge.loc[1, "CV_logloss"]},
    ])
    save(comp, "B01_response_model_comparison")

    # Behavioral F at response 0.5/0.7 on the natural-call fidelity model.
    med_log = float(nat.logsize.median())
    b0, bF = float(mf.params[0]), float(mf.params[1])
    offset = b0 + float(mf.params[2]) * 0.0 + float(mf.params[3]) * 0.0 + float(mf.params[4]) * med_log

    def solve_f(level, b0_, bF_):
        if bF_ <= 0:
            return np.nan
        return float((np.log(level / (1 - level)) - b0_) / bF_)

    point = {lv: solve_f(lv, offset, bF) for lv in [0.50, 0.70]}

    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    boot = []
    for _ in range(N_BOOT):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        try:
            mb = fit_cluster_logit(Xf[idx], y[idx], groups[idx])
            b0b, bFb = float(mb.params[0]), float(mb.params[1])
            offb = b0b + float(mb.params[4]) * med_log
            boot.append([solve_f(lv, offb, bFb) for lv in [0.50, 0.70]])
        except Exception:
            continue
    boot = np.asarray(boot, float)
    ci = (np.nanpercentile(boot, [2.5, 97.5], axis=0) if len(boot) >= 10
          else np.full((2, 2), np.nan))
    thresholds = pd.DataFrame([
        {"Quantity": "F at response probability 0.5", "Estimate": point[0.50],
         "CI_Low": float(ci[0, 0]), "CI_High": float(ci[1, 0]), "N": len(nat)},
        {"Quantity": "F at response probability 0.7", "Estimate": point[0.70],
         "CI_Low": float(ci[0, 1]), "CI_High": float(ci[1, 1]), "N": len(nat)},
    ])
    thresholds["F_model_coef"] = bF
    thresholds["F_model_p"] = float(mf.pvalues[1])
    thresholds["Note"] = ("Cluster-robust logistic on natural calls; Context and "
                          "log2(Npenguins/8) held at reference/median. "
                          "Wide CI means the playback design does not identify a threshold.")
    save(thresholds, "B01_fidelity_thresholds")

    rationale = pd.DataFrame([
        {"Acoustic_threshold": tau,
         "P2_level": p1_label,
         "Behavioral_F_50": point[0.50], "Behavioral_F_70": point[0.70],
         "F50_CI_Low": float(ci[0, 0]), "F50_CI_High": float(ci[1, 0]),
         "F70_CI_Low": float(ci[0, 1]), "F70_CI_High": float(ci[1, 1]),
         "Interpretation": ("acoustic threshold below behavioral F50 -> conservative"
                            if tau < point[0.50] else
                            "acoustic threshold above behavioral F50 -> stricter than 50% response")}
        for tau, p1_label in [(0.50, "Potential"), (0.60, "Recognition"), (0.70, "Reliable")]
    ])
    save(rationale, "B01_threshold_rationale")

    # Figure: observed binned response vs F + fitted curve.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    nat["F_bin"] = pd.qcut(nat.F_model, 5, duplicates="drop")
    obs = nat.groupby("F_bin", observed=True).agg(F=("F_model", "mean"),
                                                 P=("ResponseYesNo", "mean"),
                                                 N=("ResponseYesNo", "size")).reset_index()
    ax.scatter(obs.F, obs.P, s=55, color="#4C72B0", edgecolors="black", zorder=3,
               label="Observed (quintile bins)")
    grid = np.linspace(0.0, 1.0, 200)
    p = 1.0 / (1.0 + np.exp(-(offset + bF * grid)))
    ax.plot(grid, p, color="#4C72B0", lw=2, label="Cluster-robust logistic")
    for tau in [0.50, 0.60, 0.70]:
        ax.axvline(tau, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.axhline(0.50, color="gray", ls=":", lw=0.8)
    ax.axhline(0.70, color="gray", ls=":", lw=0.8)
    if np.isfinite(point[0.50]):
        ax.scatter([point[0.50]], [0.50], color="#D62728", marker="X", s=90, zorder=5)
        ax.annotate(f"F50={point[0.50]:.2f}", (point[0.50], 0.50),
                    textcoords="offset points", xytext=(8, -18), fontsize=8)
    if np.isfinite(point[0.70]):
        ax.scatter([point[0.70]], [0.70], color="#D62728", marker="X", s=90, zorder=5)
        ax.annotate(f"F70={point[0.70]:.2f}", (point[0.70], 0.70),
                    textcoords="offset points", xytext=(8, -18), fontsize=8)
    ax.set_xlabel("P2 composite fidelity at playback distance")
    ax.set_ylabel("P(any response) among natural calls")
    ax.set_ylim(0, 1.05)
    ax.set_title("P2 -> P3 Bridge: Fidelity vs Any Behavioral Response\n"
                 "Vertical dashed = acoustic thresholds 0.50/0.60/0.70", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "B01_fidelity_response.png", dpi=300, bbox_inches="tight")
    plt.close()
    return bridge, thresholds, rationale


def cascade_bridge(df):
    """P(V|H), P(A|H,V), P(A|H) by stimulus, with trial bootstrap."""
    x = p1.design(df)
    y = df.state.to_numpy()
    model = p1.fit(x, y, p1.MODELS[-1])
    grids = [p1.design(df.assign(CallType=g)) for g in GROUP]
    rates = p1.standardize(model, grids)  # 4 x 8

    def casc(r):
        h = r[:, 1] + r[:, 4] + r[:, 5] + r[:, 7]
        hv = r[:, 4] + r[:, 7]
        ha = r[:, 5] + r[:, 7]
        with np.errstate(divide="ignore", invalid="ignore"):
            pvh = np.where(h > 0, hv / h, np.nan)
            pav = np.where(hv > 0, r[:, 7] / hv, np.nan)
            pah = np.where(h > 0, ha / h, np.nan)
        return pvh, pav, pah

    point = casc(rates)
    rng = np.random.default_rng(SEED)
    pairs = np.array(list(df.groupby("expID").indices.values()))
    draws = []
    for _ in range(N_BOOT):
        sample = pairs[rng.integers(0, len(pairs), len(pairs))].ravel()
        try:
            mb = p1.fit(x[sample], y[sample], p1.MODELS[-1])
            rb = p1.standardize(mb, grids)
            draws.append(casc(rb))
        except Exception:
            continue
    draws = np.array(draws)  # B x 3 x 4
    rows = []
    for i, g in enumerate(GROUP):
        row = {"Group": g}
        for j, (name, vals) in enumerate([
            ("P(V|H)", draws[:, 0, i]),
            ("P(A|H,V)", draws[:, 1, i]),
            ("P(A|H)", draws[:, 2, i]),
        ]):
            vals = vals[np.isfinite(vals)]
            lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) >= 10
                      else (np.nan, np.nan))
            row[name] = float(point[j][i])
            row[name + "_CI_Low"] = float(lo)
            row[name + "_CI_High"] = float(hi)
        rows.append(row)
    out = pd.DataFrame(rows)
    save(out, "B02_cascade")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = p1.COLORS
    for ax, name in zip(axes, ["P(V|H)", "P(A|H,V)"]):
        vals = out[name].values
        lo = out[name + "_CI_Low"].values
        hi = out[name + "_CI_High"].values
        pos = np.arange(len(out))
        ax.bar(pos, vals, color=colors, width=0.6)
        ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none",
                    ecolor="black", capsize=3, lw=1)
        ax.set_xticks(pos)
        ax.set_xticklabels(out.Group)
        ax.set_ylim(0, 1.05)
        ax.set_title(name, fontweight="bold")
    axes[0].set_ylabel("Conditional probability")
    plt.suptitle("Behavioral Cascade: Head-up -> Vocalization -> Approach",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "B02_cascade.png", dpi=300, bbox_inches="tight")
    plt.close()
    return out


def latency_bridge(df):
    st = ROOT / "outputs_p3_part2/settings.csv"
    LAM = float(pd.read_csv(st).penalty.iloc[0]) if st.exists() else 64.0

    """AFT latency: distance vs fidelity among natural calls."""
    df2, t, y = p2.prepare()
    x = p1.design(df2)[:, :8]
    nat = (df2.CallType != "control").values
    fcol = p1_fidelity = add_fidelity(df2, load_fidelity())["F_model"].values
    x_dist = x[nat].copy()
    x_fid = x[nat].copy()
    x_fid[:, 6] = fcol[nat]
    t_nat = t[nat]
    groups = df2.expID.values[nat]

    rows = []
    for label, xx in [("Distance", x_dist), ("Fidelity", x_fid)]:
        fold_metrics = {k: [] for k in range(4)}
        for tr, te in p2.splits(groups, 5, SEED):
            try:
                par = p2.time_parameters(p2.fit_time(xx[tr], t_nat[tr], p2.MODELS[1], LAM), xx[te])
                pred = p2.quantile(*par, .5)
                low = p2.quantile(*par, .05)
                high = p2.quantile(*par, .95)
                for k in range(4):
                    ok = np.isfinite(t_nat[te, k])
                    if not ok.any():
                        continue
                    err = pred[ok, k] - t_nat[te[ok], k]
                    fold_metrics[k].append((np.abs(err).sum(), (err ** 2).sum(),
                                            ((t_nat[te[ok], k] >= low[ok, k]) &
                                             (t_nat[te[ok], k] <= high[ok, k])).sum(),
                                            ok.sum()))
            except Exception:
                continue
        for k, name in enumerate(p2.TASKS):
            if not fold_metrics[k]:
                continue
            arr = np.array(fold_metrics[k], float)
            rows.append({"Model": label, "Endpoint": name,
                         "MAE_s": arr[:, 0].sum() / arr[:, 3].sum(),
                         "RMSE_s": np.sqrt(arr[:, 1].sum() / arr[:, 3].sum()),
                         "Coverage90": arr[:, 2].sum() / arr[:, 3].sum()})
    out = pd.DataFrame(rows)
    save(out, "B03_latency_bridge")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    endpoints = list(p2.TASKS)
    width = 0.36
    for j, (label, color) in enumerate([("Distance", "#4C72B0"), ("Fidelity", "#DD8452")]):
        s = out[out.Model == label].set_index("Endpoint").reindex(endpoints)
        ax.bar(np.arange(len(endpoints)) + (j - 0.5) * width, s.MAE_s,
               width=width, color=color, label=label)
    ax.set_xticks(np.arange(len(endpoints)))
    ax.set_xticklabels(endpoints, rotation=15)
    ax.set_ylabel("Held-out MAE (s)")
    ax.set_title("P2 -> P3 Bridge: Latency Predicted by Distance vs Fidelity\n"
                 "Natural calls only; shared AFT; penalised slope", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "B03_latency_bridge.png", dpi=300, bbox_inches="tight")
    plt.close()
    return out


def call_type_evidence(df, iso):
    """One row per call type: P2 range + P3 adjusted state probabilities."""
    p2r = pd.read_csv(ROOT / "outputs_p2/P2_effective_range.csv")
    p2r = p2r[p2r.Level.isin(["Potential", "Reliable"])]
    p3p = pd.read_csv(ROOT / "outputs_p3_part1/adjusted_probabilities.csv")
    p3p["state"] = p3p["state"].fillna("None")
    rows = []
    for ct in GROUP:
        d = {"CallType": ct}
        if ct in GROUP[1:]:
            d["P2_d50"] = float(p2r[(p2r.CallType == ct) & (p2r.Level == "Potential")].Range_m.iloc[0])
            d["P2_d70"] = float(p2r[(p2r.CallType == ct) & (p2r.Level == "Reliable")].Range_m.iloc[0])
            d["P2_F02_signature"] = "yes" if ct == "tv" else "no"
        else:
            d["P2_d50"] = np.nan
            d["P2_d70"] = np.nan
            d["P2_F02_signature"] = "n/a"
        s = p3p[p3p.CallType == ct].set_index("state")
        d["P3_P_any"] = float(1 - s.loc["None", "probability"])
        for st in ["H", "V", "HV", "HVA"]:
            d["P3_P_" + st] = float(s.loc[st, "probability"])
        rows.append(d)
    out = pd.DataFrame(rows)
    save(out, "B04_call_type_evidence")
    return out


def run():
    iso = load_fidelity()
    df = add_fidelity(p1.load(), iso)
    bridge, thresholds, rationale = response_bridge(df)
    cascade = cascade_bridge(df)
    latency = latency_bridge(df)
    evidence = call_type_evidence(df, iso)
    summary = {
        "fidelity_model_coef": thresholds.F_model_coef.tolist(),
        "fidelity_model_p": thresholds.F_model_p.tolist(),
        "F50": float(thresholds.Estimate.iloc[0]),
        "F70": float(thresholds.Estimate.iloc[1]),
        "latency_MAE": {f"{r.Model}|{r.Endpoint}": float(r.MAE_s) for r in latency.itertuples()},
        "cascade": cascade.set_index("Group")[["P(V|H)", "P(A|H,V)"]].to_dict("index"),
    }
    (OUT / "B05_summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    run()
