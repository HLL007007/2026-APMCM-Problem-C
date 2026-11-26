# -*- coding: utf-8 -*-
"""
Problem 1: Acoustic Characteristics, Statistical Testing, and Classification of Sea-Surface Calls
------------------------------------------------------------------------------------------------
1. Data preprocessing, quality auditing, and bioacoustic feature engineering
2. Descriptive statistics and distribution visualization
3. Nonparametric hypothesis testing (Kruskal-Wallis H + Holm post-hoc; PERMANOVA + PCA biplot)
4. Three-stage feature ablation and classification (LASSO Multinomial Logit vs. Random Forest + SHAP)
5. Preliminary linkage with sound-propagation degradation and marine playback-response data
"""

import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

from scipy import stats
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ==============================================================================
# 0. Configuration & Data Loading Utilities
# ==============================================================================
DATA_DIR = "data"
OUT_DIR = "outputs_p1"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=plt.rcParams["font.sans-serif"][0])

CALL_ORDER = ["flat", "mod", "tv"]
PALETTE = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}

RAW_ACOUSTIC = ["Duration", "F01", "F02", "Fpeak", "Q25", "Q50", "Q75", "PulseRate"]
RAW_SHARED = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
ENG_FEATS = ["SpecBandwidth", "HarmonicRatio", "SpecSkew"]
SHARED_9D = RAW_SHARED + ENG_FEATS
ALL_FEATS_DESC = RAW_ACOUSTIC + ENG_FEATS


def smart_read_table(pattern):
    """Load CSV or Excel tables from the relative data directory."""
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found in '{DATA_DIR}/'.")
    fpath = matches[0]
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(fpath, na_values=["NA", "na", "NaN", "null", "", " "])
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(fpath, na_values=["NA", "na", "NaN", "null", "", " "])
    else:
        raise ValueError(f"Unsupported file format: {fpath}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    df = df.dropna(how="all").reset_index(drop=True)
    return df, os.path.basename(fpath)


def find_col(cols, prefix):
    cands = [c for c in cols if c.lower().startswith(prefix.lower())]
    return cands[0] if cands else None


# ==============================================================================
# 1. Data Preprocessing, Cleaning Audit & Bioacoustic Feature Engineering
# ==============================================================================
print("=" * 75)
print("Step 1: Data Cleaning, Auditing, and Bioacoustic Feature Engineering")
print("=" * 75)

cleaning_logs = []

# --- 1.1 Acoustic Dataset (Data01) ---
df1_raw, f1_name = smart_read_table("*Data01*")
n_raw_1 = len(df1_raw)

for col in df1_raw.select_dtypes(include=["object", "string"]).columns:
    df1_raw[col] = df1_raw[col].astype(str).str.strip()
df1_raw["CallType"] = df1_raw["CallType"].str.lower().replace({
    "modulated": "mod", "two-voice": "tv", "twovoice": "tv", "two_voice": "tv"
})

df1 = df1_raw[df1_raw["CallType"].isin(CALL_ORDER)].copy()
n_invalid_lbl = n_raw_1 - len(df1)

for col in RAW_ACOUSTIC:
    df1[col] = pd.to_numeric(df1[col], errors="coerce")

n_dup_1 = int(df1.duplicated().sum())
if n_dup_1 > 0:
    df1 = df1.drop_duplicates().reset_index(drop=True)

# Physical validity checks: positive measurements and monotonic spectral quartiles
bad_phys = pd.Series(False, index=df1.index)
for col in RAW_ACOUSTIC:
    bad_phys |= (df1[col].notna() & (df1[col] <= 0))
n_bad_phys = int(bad_phys.sum())
if n_bad_phys > 0:
    df1 = df1[~bad_phys].reset_index(drop=True)

q_valid = df1[["Q25", "Q50", "Q75"]].notna().all(axis=1)
q_bad = q_valid & ((df1["Q25"] > df1["Q50"]) | (df1["Q50"] > df1["Q75"]))
n_q_bad = int(q_bad.sum())
if n_q_bad > 0:
    df1 = df1[~q_bad].reset_index(drop=True)

n_unexpected_na = int(df1[RAW_SHARED].isna().any(axis=1).sum())
df1 = df1.dropna(subset=RAW_SHARED).reset_index(drop=True)
df1["CallType"] = pd.Categorical(df1["CallType"], categories=CALL_ORDER, ordered=True)

# Bioacoustic derived features
bw = df1["Q75"] - df1["Q25"]
df1["SpecBandwidth"] = bw
df1["HarmonicRatio"] = df1["Fpeak"] / df1["F01"]
df1["SpecSkew"] = np.where(
    bw > 1e-6,
    ((df1["Q75"] - df1["Q50"]) - (df1["Q50"] - df1["Q25"])) / bw,
    0.0
)

# Outlier audit (3 x IQR rule; retained as valid biological variation)
outlier_records = []
for ctype in CALL_ORDER:
    sub = df1[df1["CallType"] == ctype]
    for col in RAW_ACOUSTIC:
        vals = sub[col].dropna()
        if len(vals) >= 4:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            low, high = q1 - 3.0 * iqr, q3 + 3.0 * iqr
            n_out = int(((vals < low) | (vals > high)).sum())
            if n_out > 0:
                outlier_records.append({
                    "CallType": ctype, "Feature": col,
                    "ExtremeOutliers_3IQR": n_out, "LowerBound": low, "UpperBound": high
                })

if outlier_records:
    pd.DataFrame(outlier_records).to_csv(
        os.path.join(OUT_DIR, "P1_outlier_audit.csv"), index=False, encoding="utf-8-sig"
    )

cleaning_logs.append({
    "Dataset": f1_name, "Raw_Rows": n_raw_1,
    "Dropped_Invalid": n_invalid_lbl + n_dup_1 + n_bad_phys + n_q_bad + n_unexpected_na,
    "Clean_Rows": len(df1)
})

print(f"Data01 clean sample size: {len(df1)} "
      f"(flat={(df1['CallType']=='flat').sum()}, mod={(df1['CallType']=='mod').sum()}, tv={(df1['CallType']=='tv').sum()})")
print("\nStructural missingness audit (valid / total):")
print(df1.groupby("CallType", observed=False)[RAW_ACOUSTIC].agg(lambda s: f"{s.notna().sum()}/{len(s)}"))

# --- 1.2 Playback Dataset (Data02) ---
df2_raw, f2_name = smart_read_table("*Data02*")
df2 = df2_raw.drop_duplicates().copy()
for col in ["Context", "Type", "CallType"]:
    if col in df2.columns:
        df2[col] = df2[col].astype(str).str.strip().str.lower()

c_resp = find_col(df2.columns, "Response")
c_head = find_col(df2.columns, "HeadsUpY")
c_appr = find_col(df2.columns, "ApproachY")
c_voc = find_col(df2.columns, "VocalY")
pb_cols = [c for c in [c_resp, c_head, c_appr, c_voc] if c is not None]
for c in pb_cols:
    df2[c] = pd.to_numeric(df2[c], errors="coerce")

if all(c is not None for c in [c_resp, c_head, c_appr, c_voc]):
    any_beh = (df2[[c_head, c_appr, c_voc]].fillna(0).max(axis=1) > 0).astype(int)
    mismatch = (any_beh == 1) & (df2[c_resp] == 0)
    if mismatch.sum() > 0:
        df2.loc[mismatch, c_resp] = 1

cleaning_logs.append({
    "Dataset": f2_name, "Raw_Rows": len(df2_raw),
    "Dropped_Invalid": len(df2_raw) - len(df2), "Clean_Rows": len(df2)
})

# --- 1.3 Propagation Datasets (Data03-11) ---
deg_specs = [
    ("Data03", "flat", "Spectrogram"), ("Data04", "mod", "Spectrogram"), ("Data05", "tv", "Spectrogram"),
    ("Data06", "flat", "AmplitudeSpectrum"), ("Data07", "mod", "AmplitudeSpectrum"), ("Data08", "tv", "AmplitudeSpectrum"),
    ("Data09", "flat", "Envelope"), ("Data10", "mod", "Envelope"), ("Data11", "tv", "Envelope"),
]
deg_long_list = []
for prefix, ctype, metric in deg_specs:
    d_raw, d_name = smart_read_table(f"*{prefix}*")
    dist_map = {
        col: int(re.match(r"^\s*(\d+)\s*m?\s*$", col, re.IGNORECASE).group(1))
        for col in d_raw.columns if re.match(r"^\s*(\d+)\s*m?\s*$", col, re.IGNORECASE)
    }
    d_sub = d_raw[list(dist_map.keys())].apply(pd.to_numeric, errors="coerce")
    out_range = int(((d_sub < -1.0) | (d_sub > 1.0)).sum().sum())
    if out_range > 0:
        d_sub = d_sub.mask((d_sub < -1.0) | (d_sub > 1.0))

    cleaning_logs.append({
        "Dataset": d_name, "Raw_Rows": len(d_raw),
        "Dropped_Invalid": out_range, "Clean_Rows": len(d_sub.dropna(how="all"))
    })

    long_d = d_sub.melt(var_name="DistLabel", value_name="Correlation")
    long_d["Distance_m"] = long_d["DistLabel"].map(dist_map)
    long_d["CallType"] = ctype
    long_d["Metric"] = metric
    deg_long_list.append(long_d.dropna(subset=["Correlation"]))

pd.DataFrame(cleaning_logs).to_csv(
    os.path.join(OUT_DIR, "P1_data_cleaning_audit.csv"), index=False, encoding="utf-8-sig"
)


# ==============================================================================
# 2. Descriptive Statistics & Feature Visualizations
# ==============================================================================
print("\n" + "=" * 75)
print("Step 2: Descriptive Statistics and Visualizations")
print("=" * 75)

desc_rows = []
for ctype in CALL_ORDER:
    sub = df1[df1["CallType"] == ctype]
    for feat in ALL_FEATS_DESC:
        vals = sub[feat].dropna()
        if len(vals) > 0:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            desc_rows.append({
                "CallType": ctype, "Feature": feat,
                "N_valid": len(vals), "N_missing": len(sub) - len(vals),
                "Mean": vals.mean(), "SD": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "Median": med, "IQR": q3 - q1, "Q25": q1, "Q75": q3,
                "Min": vals.min(), "Max": vals.max()
            })
        else:
            desc_rows.append({
                "CallType": ctype, "Feature": feat, "N_valid": 0, "N_missing": len(sub),
                "Mean": np.nan, "SD": np.nan, "Median": np.nan, "IQR": np.nan,
                "Q25": np.nan, "Q75": np.nan, "Min": np.nan, "Max": np.nan
            })

pd.DataFrame(desc_rows).to_csv(
    os.path.join(OUT_DIR, "P1_descriptive_statistics.csv"), index=False, encoding="utf-8-sig"
)

# Figure 1: Raw acoustic feature distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 8.8))
for ax, feat in zip(axes.ravel(), RAW_ACOUSTIC):
    pdata = df1[["CallType", feat]].dropna()
    sns.violinplot(data=pdata, x="CallType", y=feat, order=CALL_ORDER, palette=PALETTE,
                   inner=None, alpha=0.35, cut=0, ax=ax)
    sns.boxplot(data=pdata, x="CallType", y=feat, order=CALL_ORDER, width=0.28,
                boxprops={"facecolor": "none", "zorder": 3}, whiskerprops={"zorder": 3},
                medianprops={"color": "black", "linewidth": 1.8}, showfliers=False, ax=ax)
    sns.stripplot(data=pdata, x="CallType", y=feat, order=CALL_ORDER, palette=PALETTE,
                  size=3.3, alpha=0.65, jitter=0.14, ax=ax)
    ax.set_title(f"{feat} (n={len(pdata)})", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("")
plt.suptitle("Raw Acoustic Feature Distributions across Three Call Types", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "P1_fig1_raw_distributions.png"), dpi=300, bbox_inches="tight")
plt.close()

# Figure 2: Engineered bioacoustic feature distributions
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
titles_eng = {
    "SpecBandwidth": "Spectral Bandwidth (Q75 - Q25)",
    "HarmonicRatio": "Harmonic Ratio (Fpeak / F01)",
    "SpecSkew": "Spectral Skewness"
}
for ax, feat in zip(axes, ENG_FEATS):
    sns.violinplot(data=df1, x="CallType", y=feat, order=CALL_ORDER, palette=PALETTE,
                   inner=None, alpha=0.35, cut=0, ax=ax)
    sns.boxplot(data=df1, x="CallType", y=feat, order=CALL_ORDER, width=0.28,
                boxprops={"facecolor": "none", "zorder": 3}, whiskerprops={"zorder": 3},
                medianprops={"color": "black", "linewidth": 1.8}, showfliers=False, ax=ax)
    sns.stripplot(data=df1, x="CallType", y=feat, order=CALL_ORDER, palette=PALETTE,
                  size=3.6, alpha=0.65, jitter=0.14, ax=ax)
    ax.set_title(titles_eng[feat], fontsize=11.5, fontweight="bold")
    ax.set_xlabel("")
plt.suptitle("Bioacoustic Engineered Feature Distributions", fontsize=13.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "P1_fig2_engineered_distributions.png"), dpi=300, bbox_inches="tight")
plt.close()

# Figure 3: Spearman correlation matrix
plt.figure(figsize=(8.8, 7.2))
sns.heatmap(df1[SHARED_9D].corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Spearman Correlation Matrix (Shared + Engineered Features)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "P1_fig3_spearman_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()


# ==============================================================================
# 3. Nonparametric Statistical Tests (Kruskal-Wallis + PERMANOVA)
# ==============================================================================
print("\n" + "=" * 75)
print("Step 3: Nonparametric Tests (Kruskal-Wallis + Holm & PERMANOVA)")
print("=" * 75)

kw_rows, pw_rows = [], []
for feat in SHARED_9D:
    groups = [df1.loc[df1["CallType"] == c, feat].dropna().values for c in CALL_ORDER]
    H_stat, p_kw = stats.kruskal(*groups)
    eps2 = max(0.0, (H_stat - 2.0) / (len(df1) - 3.0))
    kw_rows.append({
        "Feature": feat, "Test": "Kruskal-Wallis (3 groups)",
        "Statistic": H_stat, "p_value": p_kw, "EffectSize": eps2
    })

    raw_p, tmp = [], []
    for g1, g2 in [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]:
        x1 = df1.loc[df1["CallType"] == g1, feat].values
        x2 = df1.loc[df1["CallType"] == g2, feat].values
        u_s, p_u = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        r_rb = 1.0 - (2.0 * u_s) / (len(x1) * len(x2))
        raw_p.append(p_u)
        tmp.append((g1, g2, u_s, p_u, r_rb))
    _, adj_p, _, _ = multipletests(raw_p, method="holm")
    for (g1, g2, u_s, p_u, r_rb), p_h in zip(tmp, adj_p):
        pw_rows.append({
            "Feature": feat, "Comparison": f"{g1} vs {g2}",
            "U_stat": u_s, "p_raw": p_u, "p_holm": p_h,
            "Significant_0.05": p_h < 0.05, "RankBiserial_r": r_rb
        })

# PulseRate (mod vs tv)
pr_m = df1.loc[df1["CallType"] == "mod", "PulseRate"].dropna().values
pr_t = df1.loc[df1["CallType"] == "tv", "PulseRate"].dropna().values
u_pr, p_pr = stats.mannwhitneyu(pr_m, pr_t, alternative="two-sided")
r_pr = 1.0 - (2.0 * u_pr) / (len(pr_m) * len(pr_t))
kw_rows.append({
    "Feature": "PulseRate (mod vs tv)", "Test": "Mann-Whitney U (2 groups)",
    "Statistic": u_pr, "p_value": p_pr, "EffectSize": abs(r_pr)
})
pw_rows.append({
    "Feature": "PulseRate", "Comparison": "mod vs tv",
    "U_stat": u_pr, "p_raw": p_pr, "p_holm": p_pr,
    "Significant_0.05": p_pr < 0.05, "RankBiserial_r": r_pr
})

kw_df = pd.DataFrame(kw_rows)
pw_df = pd.DataFrame(pw_rows)
kw_df.to_csv(os.path.join(OUT_DIR, "P1_nonparam_kruskal_mwu.csv"), index=False, encoding="utf-8-sig")
pw_df.to_csv(os.path.join(OUT_DIR, "P1_nonparam_pairwise_holm.csv"), index=False, encoding="utf-8-sig")
print("[Test 1: Univariate Kruskal-Wallis H / Mann-Whitney U]:")
print(kw_df.round(4).to_string(index=False))


def run_permanova(X_mat, labels, n_perm=4999, seed=42):
    """PERMANOVA (Anderson, 2001) on standardized Euclidean distances."""
    rng = np.random.default_rng(seed)
    X_std = StandardScaler().fit_transform(X_mat)
    D2 = squareform(pdist(X_std, metric="euclidean")) ** 2
    n = len(labels)
    ug = np.unique(labels)
    k = len(ug)
    SS_T = np.sum(D2) / (2.0 * n)

    def calc_f(lbls):
        ss_w = sum(
            np.sum(D2[np.ix_(np.where(lbls == g)[0], np.where(lbls == g)[0])]) / (2.0 * np.sum(lbls == g))
            for g in ug
        )
        ss_a = SS_T - ss_w
        return (ss_a / (k - 1)) / (ss_w / (n - k)), ss_a / SS_T

    arr = np.asarray(labels)
    f_obs, r2_obs = calc_f(arr)
    ge = sum(1 for _ in range(n_perm) if calc_f(rng.permutation(arr))[0] >= f_obs)
    return f_obs, r2_obs, (ge + 1) / (n_perm + 1)


X_9d = df1[SHARED_9D].values
y_str = df1["CallType"].astype(str).values
f_all, r2_all, p_all = run_permanova(X_9d, y_str)

perm_rows = [{
    "Comparison": "Overall (flat vs mod vs tv)",
    "Pseudo_F": f_all, "R2": r2_all, "p_value": p_all, "p_holm": p_all
}]
p_r, tmp_pm = [], []
for g1, g2 in [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]:
    m = np.isin(y_str, [g1, g2])
    fp, r2p, pp = run_permanova(X_9d[m], y_str[m])
    p_r.append(pp)
    tmp_pm.append((f"{g1} vs {g2}", fp, r2p, pp))
_, p_h_pm, _, _ = multipletests(p_r, method="holm")
for (comp, fp, r2p, pp), ph in zip(tmp_pm, p_h_pm):
    perm_rows.append({"Comparison": comp, "Pseudo_F": fp, "R2": r2p, "p_value": pp, "p_holm": ph})

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(os.path.join(OUT_DIR, "P1_nonparam_permanova.csv"), index=False, encoding="utf-8-sig")
print("\n[Test 2: Multivariate PERMANOVA (9D Acoustic Space)]:")
print(perm_df.round(4).to_string(index=False))

# Figure 4: PCA Biplot paired with PERMANOVA (non-overlapping label offsets)
X_std9 = StandardScaler().fit_transform(X_9d)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std9)
var_r = pca.explained_variance_ratio_
loads = pca.components_.T * np.sqrt(pca.explained_variance_)

label_offsets = {
    "Duration": (0.00, 0.16),
    "F01": (0.20, 0.02),
    "Fpeak": (0.30, 0.10),
    "Q25": (0.22, 0.08),
    "Q50": (0.22, 0.05),
    "Q75": (0.14, 0.16),
    "SpecBandwidth": (-0.36, 0.12),
    "HarmonicRatio": (-0.08, -0.22),
    "SpecSkew": (0.04, 0.16),
}

fig, ax = plt.subplots(figsize=(9.2, 7.2))
for ctype in CALL_ORDER:
    pts = X_pca[y_str == ctype]
    ax.scatter(pts[:, 0], pts[:, 1], label=ctype, color=PALETTE[ctype], alpha=0.72, edgecolors="w", s=55)
    cov = np.cov(pts, rowvar=False) + 1e-9 * np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    scale = np.sqrt(chi2.ppf(0.68, df=2))
    w, h = 2 * scale * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=theta,
                         edgecolor=PALETTE[ctype], facecolor=PALETTE[ctype], alpha=0.12, lw=2))
    ax.scatter(*pts.mean(axis=0), color=PALETTE[ctype], marker="X", s=140, edgecolors="black", zorder=5)

for i, feat in enumerate(SHARED_9D):
    vx, vy = loads[i, 0] * 2.7, loads[i, 1] * 2.7
    dx, dy = label_offsets.get(feat, (0.12, 0.12))
    ax.arrow(0, 0, vx, vy, color="#222222", alpha=0.82, head_width=0.09, length_includes_head=True, lw=1.2, zorder=6)
    ax.text(vx + dx, vy + dy, feat, color="#111111", fontsize=9.8, fontweight="bold",
            ha="center", va="center", zorder=7)

ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
ax.set_xlim(X_pca[:, 0].min() - 0.6, X_pca[:, 0].max() + 0.6)
ax.set_ylim(X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5)
ax.set_xlabel(f"PC1 ({var_r[0]*100:.1f}% variance)", fontsize=11.5)
ax.set_ylabel(f"PC2 ({var_r[1]*100:.1f}% variance)", fontsize=11.5)
ax.set_title(f"PCA Biplot with 68% Core Ellipses (PERMANOVA Pseudo-F={f_all:.2f}, p={p_all:.4f})",
             fontsize=12.5, fontweight="bold")
ax.legend(title="CallType (X = Centroid)", loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "P1_fig4_PCA_PERMANOVA_biplot.png"), dpi=300, bbox_inches="tight")
plt.close()


# ==============================================================================
# 4. Call-Type Classification, Feature Ablation & SHAP Interpretation
# ==============================================================================
print("\n" + "=" * 75)
print("Step 4: Classification Modeling, Ablation, SHAP & Stability Analysis")
print("=" * 75)

le = LabelEncoder().fit(CALL_ORDER)
y = le.transform(y_str)

X_set1 = df1[RAW_SHARED].copy()
X_set2 = df1[SHARED_9D].copy()

df_s3 = df1.copy()
df_s3["Has_F02"] = df_s3["F02"].notna().astype(int)
df_s3["Has_PulseRate"] = df_s3["PulseRate"].notna().astype(int)
df_s3["F02_filled"] = df_s3["F02"].fillna(0.0)
df_s3["PulseRate_filled"] = df_s3["PulseRate"].fillna(0.0)
FEATS_S3 = SHARED_9D + ["F02_filled", "PulseRate_filled", "Has_F02", "Has_PulseRate"]
X_set3 = df_s3[FEATS_S3].copy()

models = {
    "LASSO_Logit (Baseline)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=42))
    ]),
    "RandomForest (Primary)": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42
    )
}

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"acc": "accuracy", "bal_acc": "balanced_accuracy", "macro_f1": "f1_macro", "auc": "roc_auc_ovr"}

feature_sets = [
    ("Set 1: Raw Shared (6D)", X_set1),
    ("Set 2: +Engineered (9D)", X_set2),
    ("Set 3: Full + Structural", X_set3)
]

ablation_rows, fold_rows = [], []
oof_dict = {}

for fs_name, X_curr in feature_sets:
    for m_name, mdl in models.items():
        res = cross_validate(mdl, X_curr, y, cv=rskf, scoring=scoring, n_jobs=1)
        ablation_rows.append({
            "FeatureSet": fs_name, "Model": m_name,
            "Acc_Mean": res["test_acc"].mean(), "Acc_SD": res["test_acc"].std(),
            "BalAcc_Mean": res["test_bal_acc"].mean(),
            "MacroF1_Mean": res["test_macro_f1"].mean(), "MacroF1_SD": res["test_macro_f1"].std(),
            "AUC_OvR_Mean": res["test_auc"].mean()
        })
        for a, f1, auc in zip(res["test_acc"], res["test_macro_f1"], res["test_auc"]):
            fold_rows.append({"FeatureSet": fs_name, "Model": m_name, "Accuracy": a, "Macro_F1": f1, "AUC_OvR": auc})
        oof_dict[(fs_name, m_name)] = cross_val_predict(mdl, X_curr, y, cv=skf)

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(os.path.join(OUT_DIR, "P1_classification_ablation.csv"), index=False, encoding="utf-8-sig")
print("[10x5 Repeated Stratified CV Benchmark]:")
print(ablation_df.round(4).to_string(index=False))

fold_df = pd.DataFrame(fold_rows)

# Figure 5: Classification performance comparison across feature sets
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
for ax, met in zip(axes, ["Accuracy", "Macro_F1", "AUC_OvR"]):
    sns.boxplot(
        data=fold_df, x="FeatureSet", y=met, hue="Model",
        palette={"LASSO_Logit (Baseline)": "#8DA0CB", "RandomForest (Primary)": "#FC8D62"},
        width=0.52, ax=ax
    )
    ax.set_title(f"{met} (10x5 Repeated CV)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)
    ax.set_ylim(0.15, 1.05)
    if met != "AUC_OvR":
        ax.get_legend().remove()
    else:
        ax.legend(title="Model", loc="lower right")
plt.suptitle("Classification Ablation: Raw vs. Engineered vs. Full Structural Features", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "P1_fig5_ablation_model_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# Figure 6: Confusion matrices for Set 2 and Set 3
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
cm_layout = [
    ("Set 2: +Engineered (9D)", "LASSO_Logit (Baseline)"),
    ("Set 2: +Engineered (9D)", "RandomForest (Primary)"),
    ("Set 3: Full + Structural", "LASSO_Logit (Baseline)"),
    ("Set 3: Full + Structural", "RandomForest (Primary)")
]
for ax, (fs_name, m_name) in zip(axes.ravel(), cm_layout):
    yp = oof_dict[(fs_name, m_name)]
    ConfusionMatrixDisplay(confusion_matrix(y, yp), display_labels=CALL_ORDER).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{fs_name}\n{m_name} | Acc={accuracy_score(y, yp):.3f}, F1={f1_score(y, yp, average='macro'):.3f}",
                 fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "P1_fig6_confusion_matrices.png"), dpi=300, bbox_inches="tight")
plt.close()

# Feature attribution (Random Forest + SHAP + LASSO coefficients on 9D shared space)
rf_s2 = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42
).fit(X_set2, y)

perm_s2 = permutation_importance(rf_s2, X_set2, y, n_repeats=30, random_state=42)
imp_s2 = pd.DataFrame({
    "Feature": SHARED_9D,
    "RF_Gini": rf_s2.feature_importances_,
    "PermImp_Mean": perm_s2.importances_mean,
    "PermImp_SD": perm_s2.importances_std
}).sort_values("PermImp_Mean", ascending=False)
imp_s2.to_csv(os.path.join(OUT_DIR, "P1_feature_importance_9D.csv"), index=False, encoding="utf-8-sig")

lasso_s2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=42))
]).fit(X_set2, y)
pd.DataFrame(
    lasso_s2.named_steps["clf"].coef_,
    index=[f"Class_{c}" for c in CALL_ORDER],
    columns=SHARED_9D
).to_csv(os.path.join(OUT_DIR, "P1_lasso_coefficients_9D.csv"), encoding="utf-8-sig")

if HAS_SHAP:
    explainer = shap.TreeExplainer(rf_s2)
    sv = explainer.shap_values(X_set2)
    if isinstance(sv, list):
        mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
        sv_list = sv
    else:
        mean_shap = np.abs(sv).mean(axis=(0, 2))
        sv_list = [sv[:, :, i] for i in range(len(CALL_ORDER))]

    sh_df = pd.DataFrame({"Feature": SHARED_9D, "Mean_Abs_SHAP": mean_shap}).sort_values("Mean_Abs_SHAP")
    plt.figure(figsize=(8.2, 5.2))
    colors_bar = ["#DD8452" if f in ENG_FEATS else "#4C72B0" for f in sh_df["Feature"]]
    plt.barh(sh_df["Feature"], sh_df["Mean_Abs_SHAP"], color=colors_bar)
    plt.xlabel("Mean |SHAP value| (Orange = Engineered Features)")
    plt.title("Global Feature Importance via SHAP (Random Forest, 9D Acoustic Features)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig7_SHAP_global.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for i, ctype in enumerate(CALL_ORDER):
        plt.sca(axes[i])
        shap.summary_plot(sv_list[i], X_set2, show=False, plot_size=None)
        axes[i].set_title(f"SHAP Impact: '{ctype}'", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig7b_SHAP_by_class.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    imp_s = imp_s2.sort_values("PermImp_Mean")
    plt.figure(figsize=(8.2, 5.2))
    plt.barh(imp_s["Feature"], imp_s["PermImp_Mean"], xerr=imp_s["PermImp_SD"], color="#4C72B0")
    plt.xlabel("Permutation Importance (Accuracy Drop)")
    plt.title("Global Feature Importance (Random Forest, 9D Acoustic Features)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig7_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

# Hyperparameter sensitivity check for Random Forest
sens_rows = []
for n_est in [100, 300, 500]:
    for msl in [1, 2, 4]:
        mdl_tmp = RandomForestClassifier(
            n_estimators=n_est, min_samples_leaf=msl, class_weight="balanced", random_state=42
        )
        r_s2 = cross_validate(mdl_tmp, X_set2, y, cv=skf, scoring=["accuracy", "f1_macro"], n_jobs=1)
        r_s3 = cross_validate(mdl_tmp, X_set3, y, cv=skf, scoring=["accuracy", "f1_macro"], n_jobs=1)
        sens_rows.append({
            "n_estimators": n_est, "min_samples_leaf": msl,
            "Set2_Acc": r_s2["test_accuracy"].mean(), "Set2_F1": r_s2["test_f1_macro"].mean(),
            "Set3_Acc": r_s3["test_accuracy"].mean(), "Set3_F1": r_s3["test_f1_macro"].mean()
        })
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(OUT_DIR, "P1_rf_hyperparam_sensitivity.csv"), index=False, encoding="utf-8-sig")
print("\n[Random Forest Hyperparameter Sensitivity]:")
print(sens_df.round(4).to_string(index=False))


# ==============================================================================
# 5. Linkage with Propagation Degradation & Playback Behavioral Data
# ==============================================================================
print("\n" + "=" * 75)
print("Step 5: Ecological Function & Transmission Robustness Synthesis")
print("=" * 75)

pb_sum = df2.groupby("CallType")[pb_cols].mean().reset_index()
pb_sum.to_csv(os.path.join(OUT_DIR, "P1_playback_summary.csv"), index=False, encoding="utf-8-sig")
print("[Playback Response Proportions by CallType]:")
print(pb_sum.round(3).to_string(index=False))

deg_all = pd.concat(deg_long_list, ignore_index=True)
deg_sum = deg_all.groupby(["Metric", "CallType", "Distance_m"])["Correlation"].agg(
    Mean="mean", SD="std", N="count"
).reset_index()
deg_sum.to_csv(os.path.join(OUT_DIR, "P1_propagation_summary.csv"), index=False, encoding="utf-8-sig")

piv_deg = deg_sum.pivot_table(index=["Metric", "Distance_m"], columns="CallType", values="Mean")[CALL_ORDER]
print("\n[Propagation Correlation Means by Distance]:")
print(piv_deg.round(3))

# Figure 8: Combined propagation decay and playback behavioral response synthesis
fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))
for ax, met in zip(axes[:3], ["Spectrogram", "AmplitudeSpectrum", "Envelope"]):
    sub_m = deg_sum[deg_sum["Metric"] == met].sort_values("Distance_m")
    for ctype in CALL_ORDER:
        sub_c = sub_m[sub_m["CallType"] == ctype]
        ax.errorbar(sub_c["Distance_m"], sub_c["Mean"], yerr=sub_c["SD"],
                    marker="o", capsize=3, label=ctype, color=PALETTE[ctype], lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{met} Decay", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Distance (m)")
    if met == "Spectrogram":
        ax.set_ylabel("Mean Correlation (± SD)")
        ax.legend(title="CallType")

pb_melt = pb_sum.melt(id_vars="CallType", var_name="Behavior", value_name="Rate")
pb_melt["Behavior"] = pb_melt["Behavior"].replace({
    c_resp: "Any Response", c_head: "Heads Up", c_appr: "Approach", c_voc: "Vocalize"
})
order_pb = [c for c in ["control", "flat", "mod", "tv"] if c in pb_sum["CallType"].values]
sns.barplot(data=pb_melt, x="Behavior", y="Rate", hue="CallType", hue_order=order_pb,
            palette={"control": "#999999", **PALETTE}, ax=axes[3])
axes[3].set_ylim(0, 0.85)
axes[3].set_title("Playback Behavioral Response", fontsize=11.5, fontweight="bold")
axes[3].set_xlabel("")
axes[3].set_ylabel("Proportion Responding")
axes[3].tick_params(axis="x", rotation=15)
axes[3].legend(title="Stimulus", fontsize=8.5)

plt.suptitle("Transmission Robustness & Ecological Function Synthesis across Call Types",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "P1_fig8_robustness_and_function_synthesis.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\nProblem 1 analysis completed. All tables and figures saved to ./{OUT_DIR}/")