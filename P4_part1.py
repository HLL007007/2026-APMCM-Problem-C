"""Part 4, first section: fidelity, behavioral utility and communication range.

Run from any directory: python P4_part1.py
Requires P2.py, P3_part1.py, the original data and saved P2/P3 reference CSVs.
Uses the same scientific Python environment as P2 and P3. Figures are in English.
"""
import argparse
from itertools import combinations, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import P2 as acoustic
import P3_part1 as behavior

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_p4_part1"
FIG = OUT / "figures"
TYPES = behavior.GROUP[1:]
CONTEXTS = behavior.CONTEXT
LAYERS = CONTEXTS + ["standardized"]
LABELS = {"flat":"Flat call", "mod":"Modulated call", "tv":"Two-voice call",
          "commuting":"Commuting", "searching":"Searching", "after":"After feeding",
          "standardized":"Context-standardized"}
DIST = np.arange(5., 201.)
AT = [50, 100, 150]
THRESHOLD = 15.
SEED = 41027
FEATURES = np.column_stack([behavior.BITS, behavior.BITS.sum(axis=1) >= 2])
WEIGHTS = {"Main": [.15,.30,.40,.15], "Equal behaviors": [1/3,1/3,1/3,0],
           "Vocal contact": [.15,.50,.25,.10], "Approach": [.15,.20,.55,.10],
           "No co-occurrence bonus": [3/17,6/17,8/17,0]}
UTILITY = FEATURES @ WEIGHTS["Main"]
COLORS = ["#197b70","#ad691b","#54569b"]

# AHP justification of the Main utility weights.
AHP_CRITERIA = ["Head-up (vigilance)", "Vocal (contact)",
                "Approach (recruitment)", "Co-occurrence (integration)"]
AHP_MATRIX = np.array([[1.0, 1/2, 3/8, 1.0],
                       [2.0, 1.0, 3/4, 2.0],
                       [8/3, 4/3, 1.0, 8/3],
                       [1.0, 1/2, 3/8, 1.0]])
RI = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41}

def ahp_weights(matrix):
    vals, vecs = np.linalg.eig(matrix)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    n = matrix.shape[0]
    cr = (lam - n) / (n - 1) / RI[n] if RI.get(n, 0) > 0 else 0.0
    return w, lam, cr

AHP_PRIORITIES, AHP_LAMBDA, AHP_CR = ahp_weights(AHP_MATRIX)
assert np.allclose(AHP_PRIORITIES, WEIGHTS["Main"], atol=1e-3)


def write(frame, name):
    frame.to_csv(OUT / (name+".csv"),index=False,encoding="utf-8-sig")


def input_data():
    long = pd.read_csv(ROOT/"outputs_p2/P2_analysis_dataset.csv")
    cubes = []
    for typ in TYPES:
        d = long[long.CallType.eq(typ)].sort_values(["TrialID","Distance"])
        assert not d.duplicated(["TrialID","Distance"]).any()
        assert np.array_equal(d.Distance.to_numpy().reshape(-1,7),
                              np.tile(acoustic.DISTANCES,(d.TrialID.nunique(),1)))
        cubes.append(d[["r_spec_pos","r_amp_pos","r_env_pos"]].to_numpy().reshape(-1,7,3))
    return cubes, behavior.load()


def fidelity(cubes, distances):
    # Vectorized version of P2.fuse_rows: renormalize the available metrics
    # per trial and distance, preserving completely missing rows as missing.
    weights = acoustic.pca_weights_from_matrix(np.concatenate(cubes).reshape(-1,3))[0]
    curves=[]
    for block in cubes:
        denom=np.isfinite(block) @ weights
        y=np.divide(np.nan_to_num(block) @ weights,denom,
                    out=np.full(denom.shape,np.nan),where=denom>0)
        ds=np.broadcast_to(acoustic.DISTANCES,y.shape)
        ok=np.isfinite(y)
        fit=acoustic.fit_isotonic(ds[ok],y[ok])
        if fit is None:
            raise ValueError("Cannot fit the propagation curve")
        curves.append(fit["predict"](distances))
    return np.array(curves),weights


def probability_grid(df):
    size=float(df.Npenguins.median())
    grid=pd.DataFrame(product(behavior.GROUP,CONTEXTS,DIST),
                      columns=["CallType","Context","Distance"])
    grid["Npenguins"]=size
    return size,behavior.design(grid)


def probabilities(model, design):
    p=behavior.predict(model,design).reshape(4,3,len(DIST),8)
    if not np.allclose(p.sum(axis=-1),1) or np.any(p<0):
        raise ValueError("Invalid behavior probabilities")
    return p


def include_average(values, context_weights):
    standardized=np.einsum("tcd,c->td",values,context_weights)
    return np.concatenate([values,standardized[:,None,:]],axis=1)


def index_components(f,p,context_weights,utility=UTILITY):
    expected=p @ utility
    excess=np.maximum(expected[1:]-expected[0],0.)
    score=100*f[:,None,:]*excess
    return (include_average(score,context_weights),include_average(excess,context_weights),
            include_average(1-p[1:,:,:,0],context_weights))


def get_domains(df):
    """Context-specific overlap with controls; overall type overlap for pooled estimates."""
    def overlap(typed,control):
        return max(typed.min(),control.min(),min(acoustic.DISTANCES)),min(typed.max(),control.max(),max(acoustic.DISTANCES))
    domains={}
    for typ in TYPES:
        for ctx in LAYERS:
            g=df if ctx=="standardized" else df[df.Context.eq(ctx)]
            domains[typ,ctx]=overlap(g[g.CallType.eq(typ)].Distance,g[g.CallType.eq("control")].Distance)
    return domains


def crossing(curve,domain,threshold):
    """First downward crossing from the lower support limit, with boundary flags."""
    a,b=domain
    mask=(DIST>=a)&(DIST<=b)
    d,s=DIST[mask],curve[mask]
    below=np.flatnonzero(s<threshold)
    if len(below)==0:
        return float(d[-1]),"at_least_upper_limit"
    j=below[0]
    if j==0:
        return float(d[0]),"below_lower_limit"
    fraction=(s[j-1]-threshold)/(s[j-1]-s[j])
    return float(d[j-1]+fraction*(d[j]-d[j-1])),"crossing"


def radii(score,domains,threshold):
    return [[crossing(score[i,j],domains[typ,ctx],threshold) for j,ctx in enumerate(LAYERS)]
            for i,typ in enumerate(TYPES)]


def input_checks(cubes,model,size):
    p2=pd.read_csv(ROOT/"outputs_p2/P2_effective_range.csv")
    f,w=fidelity(cubes,acoustic.GRID)
    err=[]
    for row in p2.itertuples():
        r=acoustic.solve_range(f[TYPES.index(row.CallType)],row.tau)[0]
        err.append(abs(r-row.Range_m))
    p3=pd.read_csv(ROOT/"outputs_p3_part1/scenario_probabilities.csv",keep_default_na=False)
    assert (p3.Npenguins==size).all()
    prob=behavior.predict(model,behavior.design(p3))
    gap=np.max(np.abs(prob-p3[behavior.NAMES].to_numpy(dtype=float)))
    check=pd.DataFrame({"quantity":["P2 maximum radius error (meters)","P3 maximum state probability error"],
                        "absolute_error":[max(err),gap],"tolerance":[.02,2e-4]})
    write(check,"upstream_reproduction")
    if (check.absolute_error>check.tolerance).any():
        raise ValueError("Reference reproduction failed: "+str(check))
    return w


def validate_behavior(df,x,y):
    """Held-out behavioral utility is observable. The composite has no direct label."""
    results=[]
    groups=df.expID.to_numpy()
    ids=np.unique(groups)
    rng=np.random.default_rng(SEED+1)
    for repeat in range(5):
        for hold in np.array_split(rng.permutation(ids),5):
            test=np.isin(groups,hold)
            fit=behavior.fit(x[~test],y[~test],behavior.MODELS[-1])
            pred=behavior.predict(fit,x[test]) @ UTILITY
            train=df.loc[~test,["CallType"]].assign(utility=UTILITY[y[~test]])
            simple=df.loc[test,"CallType"].map(train.groupby("CallType").utility.mean())
            for name,values in [("Type-only training mean",simple),("Eight-state model",pred)]:
                err=values-UTILITY[y[test]]
                results.append([name,len(err),np.abs(err).sum(),np.square(err).sum(),err.sum()])
    g=pd.DataFrame(results,columns=["model","n","absolute","squared","signed"]).groupby("model").sum()
    return pd.DataFrame({"MAE":g.absolute/g.n,"RMSE":np.sqrt(g.squared/g.n),"mean_error":g.signed/g.n}).reset_index()


def simulate(cubes,df,x,y,gx,context_weights,domains,nboot):
    rng=np.random.default_rng(SEED)
    pairs=np.stack(list(df.groupby("expID").indices.values()))
    scores=[];ranges=[];counts=[]
    for _ in range(nboot):
        # Different experiments for propagation and playback. Within each type
        # keep the full distance trajectory of each selected acoustic trial.
        fb,_=fidelity([block[rng.integers(len(block),size=len(block))] for block in cubes],DIST)
        idx=pairs[rng.integers(len(pairs),size=len(pairs))].ravel()
        pb=probabilities(behavior.fit(x[idx],y[idx],behavior.MODELS[-1]),gx)
        score,_,respond=index_components(fb,pb,context_weights)
        scores.append(score)
        ranges.append([[v[0] for v in group] for group in radii(score,domains,THRESHOLD)])
        counts.append(respond[:,:,np.isin(DIST,AT)])
    return np.asarray(scores),np.asarray(ranges),np.asarray(counts)


def make_figures(curves,ranges,sensitivity,counts,threshold,group_size):
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                         "axes.spines.right":False,"font.size":10,"savefig.dpi":190,
                         "grid.alpha":.4,"grid.linewidth":.6})
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(FIG/(name+".png"),bbox_inches="tight")
        plt.close(fig)
    g=curves[curves.context.eq("standardized")]
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.0))
    for i,typ in enumerate(TYPES):
        s=g[g.call_type.eq(typ)]
        for ax,col in zip(axes,["fidelity","excess_utility","index"]):
            ax.plot(s.distance,s[col],label=LABELS[typ],color=COLORS[i],linewidth=2.1)
        axes[2].fill_between(s.distance,s.low95,s.high95,color=COLORS[i],alpha=.12)
    for ax,title in zip(axes,["Signal fidelity","Behavioral utility above control","Communication effectiveness index"]):
        ax.set(xlabel="Playback distance (meters)",title=title)
    axes[0].set_ylabel("Fidelity score");axes[1].set_ylabel("Expected utility difference");axes[2].set_ylabel("Index points")
    axes[0].legend(frameon=False);axes[2].axhline(threshold,color="#65727b",ls="--",lw=1)
    finish(fig,"01_index_components")
    fig,axes=plt.subplots(1,3,sharey=True,figsize=(12.7,4.0))
    for ax,ctx in zip(axes,CONTEXTS):
        for i,typ in enumerate(TYPES):
            s=curves[curves.context.eq(ctx)&curves.call_type.eq(typ)]
            ax.plot(s.distance,s["index"],label=LABELS[typ],color=COLORS[i],lw=2.1)
        ax.axhline(threshold,color="#65727b",ls="--",lw=1)
        ax.set(title=LABELS[ctx],xlabel="Playback distance (meters)",xlim=(0,205))
    axes[0].set_ylabel("Communication effectiveness index");axes[2].legend(frameon=False)
    finish(fig,"02_behavioral_contexts")
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for i,typ in enumerate(TYPES):
        s=sensitivity[sensitivity.call_type.eq(typ)&sensitivity.weights.eq("Main")]
        axes[0].plot(s.threshold,s.radius,color=COLORS[i],label=LABELS[typ])
        cut=s[s.status.ne("crossing")]
        axes[0].scatter(cut.threshold,cut.radius,facecolors="white",edgecolors=COLORS[i],s=25)
        z=ranges[ranges.call_type.eq(typ)].set_index("context").loc[LAYERS]
        pos=np.arange(4)+.18*(i-1)
        axes[1].vlines(pos,z.low95_clipped,z.high95_clipped,color=COLORS[i],lw=2)
        axes[1].scatter(pos,z.radius,color=COLORS[i],label=LABELS[typ],zorder=3)
        cens=z.status.ne("crossing").to_numpy()
        axes[1].scatter(pos[cens],z.radius.to_numpy()[cens],color=COLORS[i],marker="^",s=65,zorder=4)
    axes[0].set(title="Index threshold sensitivity",xlabel="Index threshold (points)",ylabel="Radius within data support (meters)")
    axes[1].set(title="Effective radius at the main threshold",ylabel="Radius within data support (meters)",
                xticks=range(4),xticklabels=["Commuting","Searching","After feeding","Standardized"])
    axes[0].legend(frameon=False);axes[0].axvline(threshold,ls=":",color="#63717b")
    fig.text(.5,-.022,"Open circles mark support limits. Triangles mark boundary estimates. Intervals are clipped to observed distance ranges.",ha="center",fontsize=9)
    finish(fig,"03_effective_radius")
    fig,axes=plt.subplots(1,3,figsize=(12.3,3.8),sharey=True)
    for ax,d in zip(axes,AT):
        for i,typ in enumerate(TYPES):
            s=counts[counts.distance.eq(d)&counts.call_type.eq(typ)&counts.context.eq("standardized")]
            ax.plot(100*s.responding_fraction,s.expected_birds,color=COLORS[i],marker="o",label=LABELS[typ])
        ax.set(xlabel="Responding share in responsive groups (%)",title=f"Playback distance of {d} meters",ylim=(0,group_size),
               xticks=[12.5,25,50,75,100])
        ax.tick_params(axis="x",labelsize=9)
    axes[0].set_ylabel("Expected number of responding\npenguins per playback")
    axes[-1].legend(frameon=False)
    finish(fig,"04_participation_sensitivity")


def plot_group_size(gs):
    fig,axes=plt.subplots(1,3,figsize=(12.3,3.8),sharey=False)
    for ax,d in zip(axes,AT):
        for i,typ in enumerate(TYPES):
            s=gs[(gs.call_type==typ)&(gs.distance==d)].sort_values("group_size")
            ax.plot(s.group_size,s.expected_full,marker="o",color=COLORS[i],label=LABELS[typ])
        ax.set_xscale("log",base=2)
        ax.set_xticks([2,4,8,16,32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set(xlabel="Observed group size",title=f"Playback distance {d} m")
    axes[0].set_ylabel("Expected responders (all responsive-group members)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Group-Size Sensitivity of Expected Responders; P(N)=P(8)(N/8)^beta, beta from P3 GEE",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG/"05_group_size_responders.png",bbox_inches="tight")
    plt.close(fig)


def run(nboot):
    FIG.mkdir(parents=True,exist_ok=True)
    cubes,df=input_data()
    size,gx=probability_grid(df)
    x=behavior.design(df); y=df.state.to_numpy()
    model=behavior.fit(x,y,behavior.MODELS[-1])
    w=input_checks(cubes,model,size)
    ahp=pd.DataFrame({"criterion":AHP_CRITERIA,"priority":AHP_PRIORITIES})
    ahp["lambda_max"]=AHP_LAMBDA
    ahp["consistency_ratio"]=AHP_CR
    ahp["note"]="AHP priorities equal the Main utility weights; CR=0 for this consistent matrix"
    write(ahp,"weight_ahp")
    f,_=fidelity(cubes,DIST)
    p=probabilities(model,gx)
    ctx=df.Context.value_counts(normalize=True).loc[CONTEXTS].to_numpy()
    domains=get_domains(df)
    score,excess,respond=index_components(f,p,ctx)
    center=radii(score,domains,THRESHOLD)
    near_rows=[]
    for i,typ in enumerate(TYPES):
        a,_=domains[typ,"standardized"]
        k=int(np.argmin(np.abs(DIST-a)))
        near=float(score[i,-1,k])
        near_rows.append({"call_type":typ,"near_source_distance":float(DIST[k]),
                          "near_source_index":near,"threshold":THRESHOLD,
                          "threshold_over_near_source":THRESHOLD/near if near>0 else np.nan,
                          "note":"15 points is approximately half of the near-source index; sensitivity 8-24"})
    write(pd.DataFrame(near_rows),"threshold_rationale")
    bs,br,bp=simulate(cubes,df,x,y,gx,ctx,domains,nboot)
    comparison=[]
    for a,b in combinations(range(len(TYPES)),2):
        for d in AT:
            k=int(np.flatnonzero(DIST==d)[0])
            differences=bs[:,b,-1,k]-bs[:,a,-1,k]
            lo,hi=np.quantile(differences,[.025,.975])
            comparison.append([TYPES[b],TYPES[a],d,score[b,-1,k]-score[a,-1,k],lo,hi])
    write(pd.DataFrame(comparison,columns=["call_type","reference","distance","difference_points","low95","high95"]),
          "index_contrasts")
    ci=np.quantile(bs,[.025,.975],axis=0); ri=np.quantile(br,[.025,.975],axis=0)
    pc=np.quantile(bp,[.025,.975],axis=0)
    curves=[];rs=[];counts=[];sensitivity=[]
    for i,typ in enumerate(TYPES):
        strict=(max(domains[typ,c][0] for c in CONTEXTS),min(domains[typ,c][1] for c in CONTEXTS))
        for j,layer in enumerate(LAYERS):
            a,b=domains[typ,layer]
            r,status=center[i][j]
            report_as=(f">={r:.0f} m" if status=="at_least_upper_limit" else
                       f"<={r:.0f} m" if status=="below_lower_limit" else f"{r:.0f} m")
            rs.append([typ,layer,a,b,r,status,*ri[:,i,j],np.mean(br[:,i,j]==a),np.mean(br[:,i,j]==b),report_as])
            for k,d in enumerate(DIST):
                if a<=d<=b:
                    curves.append([typ,layer,d,f[i,k],excess[i,j,k],respond[i,j,k],score[i,j,k],
                                   *ci[:,i,j,k],strict[0]<=d<=strict[1]])
                if d in AT and a<=d<=b:
                    for fraction in [1/size,.25,.5,.75,1.]:
                        m=size*fraction
                        counts.append([typ,layer,d,size,fraction,respond[i,j,k],m*respond[i,j,k],
                                       *(m*pc[:,i,j,AT.index(d)])])
        for name,ww in WEIGHTS.items():
            alt=index_components(f,p,ctx,FEATURES @ ww)[0][i,-1]
            for threshold in np.arange(8.,24.01,1):
                radius,status=crossing(alt,domains[typ,"standardized"],threshold)
                sensitivity.append([typ,name,threshold,radius,status])
    curves=pd.DataFrame(curves,columns=["call_type","context","distance","fidelity","excess_utility","p_any","index","low95","high95","all_contexts_in_support"])
    rs=pd.DataFrame(rs,columns=["call_type","context","lower_support","upper_support","radius","status","low95_clipped","high95_clipped","below_fraction","above_fraction","report_as"])
    counts=pd.DataFrame(counts,columns=["call_type","context","distance","group_size","responding_fraction","p_any","expected_birds","low95","high95"])
    gee=pd.read_csv(ROOT/"outputs_p3_part1/overall_gee.csv")
    row=gee[gee.term.astype(str).str.contains("logsize",case=False)]
    beta=float(np.log(row.OR.iloc[0])) if len(row) else float(np.log(0.705))
    N_GRID=[2,4,8,16,32]
    group_rows=[]
    for i,typ in enumerate(TYPES):
        for d in AT:
            k=int(np.flatnonzero(DIST==d)[0])
            p8=float(respond[i,-1,k])
            for n in N_GRID:
                pn=float(np.clip(p8*(n/size)**beta,0,1))
                group_rows.append([typ,d,n,size,p8,pn,n*pn,n*.5*pn,.5*pn])
    group_size=pd.DataFrame(group_rows,columns=["call_type","distance","group_size","reference_size","p_at_reference","p_group_size","expected_full","expected_half","per_capita_half"])
    write(group_size,"group_size_responders")
    sensitivity=pd.DataFrame(sensitivity,columns=["call_type","weights","threshold","radius","status"])
    assert curves["index"].between(0,100).all() and counts.expected_birds.between(0,size).all()
    assert np.allclose(curves["index"],100*curves.fidelity*curves.excess_utility)
    write(curves,"index_curves");write(rs,"effective_radii");write(counts,"responding_birds");write(sensitivity,"radius_sensitivity")
    layers=[]
    for i,typ in enumerate(TYPES):
        a=max(domains[typ,c][0] for c in CONTEXTS);b=min(domains[typ,c][1] for c in CONTEXTS)
        r,status=crossing(score[i,-1],(a,b),THRESHOLD)
        report_as=(f">={r:.0f} m" if status=="at_least_upper_limit" else
                   f"<={r:.0f} m" if status=="below_lower_limit" else f"{r:.0f} m")
        layers.append([typ,a,b,r,status,report_as])
    write(pd.DataFrame(layers,columns=["call_type","common_lower","common_upper","radius","status","report_as"]),"strict_context_support")
    write(pd.DataFrame(FEATURES @ np.array(list(WEIGHTS.values())).T,index=behavior.NAMES,columns=list(WEIGHTS)).rename_axis("state").reset_index(),"combination_utilities")
    write(validate_behavior(df,x,y),"behavior_validation")
    make_figures(curves,rs,sensitivity,counts,THRESHOLD,size)
    plot_group_size(group_size)
    print(f"Pairs = {df.expID.nunique()}, typical group = {size:g}, replicates = {nboot}")
    print("Fidelity weights:", w.round(5),"Context weights:", ctx.round(5))
    print(rs.to_string(index=False,float_format=lambda v:f"{v:.2f}"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",type=int,default=1000)
    args=parser.parse_args()
    if args.bootstrap<2: parser.error("--bootstrap must be at least 2")
    run(args.bootstrap)
