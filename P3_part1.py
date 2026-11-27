"""African penguin playback responses. Run with: python P3_part1.py

Requires numpy, pandas, scipy, statsmodels and matplotlib. All paths are
relative to this file. Fits population-average models for a single playback.
"""
from pathlib import Path
from itertools import combinations
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import binomtest
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_p3_part1"
FIG = OUT / "figures"
GROUP = ["control", "flat", "mod", "tv"]
CONTEXT = ["commuting", "searching", "after"]
NAMES = ["None", "H", "V", "A", "HV", "HA", "VA", "HVA"]
BITS = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1],
                 [1,1,0], [1,0,1], [0,1,1], [1,1,1]])
COLORS = ["#616978", "#18786f", "#c47727", "#525aa1"]
SEED = 20261127
REPS = 1500
MODELS = ["Multinomial", "Hurdle additive", "Hurdle pairwise"]


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def load():
    df = pd.read_csv(ROOT / "data/Data02_AfricanPenguins_playbacks_data.csv")
    fields = ["HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]
    assert df[fields].isin([0, 1]).all().all()
    assert set(df.CallType) == set(GROUP) and set(df.Context) == set(CONTEXT)
    assert (df.Type.eq("control") == df.CallType.eq("control")).all()
    assert ((df[fields].sum(axis=1)>0) == df.ResponseYesNo).all()
    assert df[["Distance","Npenguins"]].gt(0).all().all()
    assert df.groupby(["expID","Type"]).size().unstack().eq(1).all().all()
    lookup = {tuple(bits): i for i, bits in enumerate(BITS)}
    df["state"] = [lookup[tuple(row)] for row in df[fields].to_numpy()]
    return df.sort_values(["expID","Type"]).reset_index(drop=True)


def design(df):
    natural = df.CallType.ne("control").to_numpy(dtype=float)
    # Orthonormal sum-to-zero subtype contrasts; control scores zero.
    a = (df.CallType.eq("flat").astype(float)-df.CallType.eq("mod"))/np.sqrt(2)
    b = (natural-3*df.CallType.eq("tv").astype(float))/np.sqrt(6)
    env = np.column_stack([df.Context.eq("searching"), df.Context.eq("after"),
                           (df.Distance-100)/50, np.log2(df.Npenguins/8)])
    return np.column_stack([np.ones(len(df)), natural, a, b, env, natural[:,None]*env])


def optimize(fun, shape):
    result = minimize(fun, np.zeros(np.prod(shape)), jac=True, method="L-BFGS-B",
                      options={"maxiter":500, "ftol":1e-11})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.reshape(shape)


def categorical_fit(x, y, basis, penalty):
    """Shared-feature conditional softmax. Rows of basis describe outcomes."""
    def objective(params):
        coef = params.reshape(penalty.shape)
        scores = x @ coef @ basis.T
        logp = scores - logsumexp(scores, axis=1, keepdims=True)
        loss = -logp[np.arange(len(x)), y].sum() + (penalty*coef**2).sum()/2
        grad = x.T @ (np.exp(logp) @ basis - basis[y]) + penalty*coef
        return loss, grad.ravel()
    return optimize(objective, penalty.shape)


def fit(x, y, mode, strength=1):
    if mode == "Multinomial":
        basis = np.eye(8)
        penalty = np.ones((x.shape[1], 8))*strength
        penalty[0] *= 0.2
        return None, categorical_fit(x,y,basis,penalty), basis
    gate_penalty = np.array([0.05,0.25,1,1,1,1,1,1,2,2,2,2])*strength
    event = (y>0).astype(float)
    def gate_loss(w):
        eta = x @ w
        return ((np.logaddexp(0,eta)-event*eta).sum()+np.dot(gate_penalty,w*w)/2,
                x.T @ (expit(eta)-event)+gate_penalty*w)
    gate = optimize(gate_loss, (x.shape[1],))
    bits = BITS[1:]
    basis = bits
    if mode == "Hurdle pairwise":
        basis = np.column_stack([bits]+[bits[:,i]*bits[:,j] for i,j in combinations(range(3),2)])
    # Environmental and subtype effects are shared across states via their
    # component behaviors. Co-occurrence terms use shared intercepts only.
    xc = x[event==1,:8]
    main_shape = (8,3)
    extra = basis.shape[1]-3
    def conditional_loss(params):
        w = params[:24].reshape(main_shape)
        score = xc @ w @ bits.T
        if extra:
            score += params[24:] @ basis[:,3:].T
        logp = score - logsumexp(score, axis=1, keepdims=True)
        labels = y[event==1]-1
        residual = np.exp(logp) @ basis - basis[labels]
        pen = np.full(main_shape, strength, dtype=float)
        pen[0] *= 0.2
        pen[2:4] *= 2
        loss = -logp[np.arange(len(labels)), labels].sum() + (pen*w*w).sum()/2
        grad = (xc.T @ residual[:,:3]+pen*w).ravel()
        if extra:
            loss += strength*np.dot(params[24:],params[24:])/2
            grad = np.r_[grad, residual[:,3:].sum(axis=0)+strength*params[24:]]
        return loss, grad
    return gate, optimize(conditional_loss, (24+extra,)), basis


def predict(model, x):
    gate, coef, basis = model
    if gate is None:
        score = x @ coef
    else:
        score = x[:,:8] @ coef[:24].reshape(8,3) @ basis[:,:3].T
        score += coef[24:] @ basis[:,3:].T
    probs = np.exp(score-logsumexp(score,axis=1,keepdims=True))
    if gate is not None:
        r = expit(x @ gate)
        probs = np.column_stack([1-r,r[:,None]*probs])
    return probs


def standardize(model, grids):
    return np.stack([predict(model,x).mean(axis=0) for x in grids])


def paired_tests(df, rng):
    paired = df.pivot(index="expID", columns="Type",values="state")
    delta = np.eye(8)[paired.call]-np.eye(8)[paired.control]
    obs = np.linalg.norm(delta.sum(axis=0))**2
    exceed = 0
    for _ in range(100):
        perm = rng.choice([-1,1],size=(1000,len(delta))) @ delta
        exceed += np.count_nonzero((perm**2).sum(axis=1)>=round(obs))
    a,b = paired.call.gt(0), paired.control.gt(0)
    table = pd.crosstab(a,b).reindex(index=[False,True],columns=[False,True],fill_value=0)
    save(table.rename_axis("call_response").reset_index(),"paired_counts")
    discord = int((a!=b).sum())
    return {"pairs":len(paired),"discordant":discord,
            "mcnemar_p":binomtest(int((a & ~b).sum()),discord).pvalue,
            "paired_swap_p":(1+exceed)/100001,
            "different_distances":int((df.groupby("expID").Distance.nunique()>1).sum())}


def gee(df):
    # Population-average odds ratios, sandwich covariance by experiment.
    import statsmodels.formula.api as smf
    d = df.assign(dist50=(df.Distance-100)/50,logsize=np.log2(df.Npenguins/8))
    model = smf.gee("ResponseYesNo ~ C(CallType, Treatment('control')) + "
                    "C(Context, Treatment('commuting')) + dist50 + logsize",
                    groups="expID",data=d,family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    assert model.converged
    bounds = np.exp(model.conf_int())
    results = pd.DataFrame({"term":model.params.index,"OR":np.exp(model.params).values,
                            "low":bounds.iloc[:,0].values,"high":bounds.iloc[:,1].values,
                            "p":model.pvalues.values})
    save(results,"overall_gee")
    # Test effect modification by context, distance and group size without
    # estimating a separate slope for each small natural-call subtype.
    d["natural"] = d.CallType.ne("control").astype(int)
    context_term = "C(Context, Treatment('commuting'))"
    interaction_model = smf.gee(
        "ResponseYesNo ~ C(CallType, Treatment('control')) + " + context_term
        + " + dist50 + logsize + natural:search + natural:afterfeeding"
        + " + natural:dist50 + natural:logsize",
        data=d.assign(search=d.Context.eq("searching").astype(int),
                      afterfeeding=d.Context.eq("after").astype(int)),
        groups="expID",family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable()).fit()
    assert interaction_model.converged
    tests=[]
    for factor,terms in [("Context",["search","afterfeeding"]),("Distance",["dist50"]),
                         ("Npenguins",["logsize"])]:
        idx=[interaction_model.params.index.get_loc("natural:"+t) for t in terms]
        test=interaction_model.wald_test(np.eye(len(interaction_model.params))[idx],scalar=True)
        tests.append([factor,len(idx),float(test.statistic),float(test.pvalue)])
    tests=pd.DataFrame(tests,columns=["modifier","df","wald","p"])
    tests["p_holm"]=sm.stats.multipletests(tests.p,method="holm")[1]
    save(tests,"effect_modification")
    return results


def validate(df,x,y,rng):
    scores, saved = [], []
    for grouping,repeats in [("expID",10),("Date",1)]:
        groups = df[grouping].to_numpy()
        unique = np.unique(groups)
        for run in range(repeats):
            folds = np.array_split(rng.permutation(unique),5 if grouping=="expID" else len(unique))
            for fold,hold in enumerate(folds):
                test = np.isin(groups,hold)
                for mode in MODELS:
                    probs = predict(fit(x[~test],y[~test],mode),x[test])
                    scores.append((grouping,run,fold,mode,test.sum(),
                                   -np.log(probs[np.arange(test.sum()),y[test]]).mean(),
                                   np.mean(((1-probs[:,0])-(y[test]>0))**2),
                                   np.mean(np.sum((probs-np.eye(8)[y[test]])**2,axis=1))))
                    if grouping=="expID" and run==0:
                        saved.append(pd.DataFrame({"row":np.flatnonzero(test),"model":mode,
                                                   "observed":y[test]>0,"predicted":1-probs[:,0]}))
    detail = pd.DataFrame(scores,columns=["split","repeat","fold","model","n","log_loss","binary_brier","joint_brier"])
    save(detail,"validation_folds")
    summary=[]
    for key,g in detail.groupby(["split","model"]):
        summary.append([*key,*np.average(g.iloc[:,-3:],axis=0,weights=g.n)])
    cv = pd.DataFrame(summary,columns=["split","model","log_loss","binary_brier","joint_brier"])
    save(cv,"validation")
    return cv,pd.concat(saved)


def plot_results(df, counts, rates, effects, cv, oof, model):
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":120,"font.size":10})
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(FIG/(name+".png"),dpi=200,bbox_inches="tight")
        plt.close(fig)
    overall = df.groupby("Type").ResponseYesNo.agg(["sum", "count"]).loc[["call", "control"]]
    fig, ax = plt.subplots(figsize=(8, 2.35))
    for pos, (row, color) in enumerate(zip(overall.itertuples(), [COLORS[1], COLORS[0]])):
        rate = 100 * row.sum / row.count
        ax.barh(pos, rate, height=.55, color=color)
        ax.text(rate + 1.5, pos, f"{rate:.1f}% ({row.sum}/{row.count})", va="center")
    ax.set(yticks=[0, 1], yticklabels=["Natural calls", "Control sounds"], xlim=(0, 100),
           xlabel="Playbacks with at least one response (%)",
           title="Overall response rates to natural calls and control sounds")
    ax.invert_yaxis()
    finish(fig,"01_overall_response")
    fig,ax=plt.subplots(figsize=(8,3.8))
    for j,(typ,col) in enumerate(zip(["control","call"],[COLORS[0],COLORS[1]])):
        g=df[df.Type==typ].groupby("Context").ResponseYesNo.agg(["sum","count"]).loc[CONTEXT]
        lo,hi=proportion_confint(g["sum"],g["count"],method="wilson")
        rate=g["sum"]/g["count"]
        pos=np.arange(3)+(j-.5)*.32
        ax.bar(pos,100*rate,width=.3,color=col,label=["Control","Natural calls"][j])
        ax.errorbar(pos,100*rate,yerr=100*np.array([rate-lo,hi-rate]),fmt="none",ecolor="#28333a",capsize=3)
    ax.set(xticks=range(3),xticklabels=["Commuting","Searching","After feeding"],ylabel="Playbacks with a response (%)",ylim=(0,105))
    ax.set_title("Observed response rates to natural calls and control sounds by behavioral context")
    ax.legend(frameon=False); finish(fig,"02_context")
    fig,axes=plt.subplots(1,2,figsize=(9,3.9),sharey=True)
    for ax,var,grid,label in [(axes[0],"Distance",np.arange(20,151,2),"Playback distance (m)"),
                              (axes[1],"Npenguins",np.arange(1,41),"Observed group size")]:
        for typ,color,name in zip(GROUP,COLORS,["Control","Flat","Modulated","Two-voice"]):
            grid_df=pd.DataFrame({"CallType":typ,"Context":"commuting","Distance":100,"Npenguins":8,var:grid})
            ax.plot(grid,100*(1-predict(model,design(grid_df))[:,0]),label=name,color=color)
        ax.set_xlabel(label);ax.set_ylim(0,100)
    axes[0].set_title("Commuting groups of 8 penguins",fontsize=10)
    axes[1].set_title("Commuting groups at 100 m",fontsize=10)
    axes[0].set_ylabel("Predicted response probability (%)");axes[1].legend(frameon=False)
    fig.suptitle("Predicted response probability by call type, playback distance and group size")
    finish(fig,"03_environment")
    fig,ax=plt.subplots(figsize=(8,3.8))
    for j,typ in enumerate(["control","call"]):
        vals=counts.loc[typ].to_numpy()
        bars=ax.bar(np.arange(8)+(j-.5)*.37,vals,.36,color=COLORS[j],label=["Control","Natural calls"][j])
        ax.bar_label(bars,fontsize=8,padding=2)
    ax.set(xticks=range(8),xticklabels=NAMES,ylabel="Number of playbacks",ylim=(0,140))
    ax.legend(frameon=False);finish(fig,"04_observed_states")
    fig,ax=plt.subplots(figsize=(8,3.3))
    im=ax.imshow(rates*100,aspect="auto",vmin=0,vmax=100,cmap="Blues")
    for i,j in np.ndindex(rates.shape):
        ax.text(j,i,f"{100*rates[i,j]:.1f}",ha="center",va="center",color="white" if rates[i,j]>.52 else "#193249")
    ax.set(xticks=range(8),xticklabels=NAMES,yticks=range(4),yticklabels=["Control","Flat","Modulated","Two-voice"])
    fig.colorbar(im,ax=ax,label="Adjusted probability (%)");finish(fig,"05_adjusted_states")
    fig,axes=plt.subplots(1,3,figsize=(10.4,4),sharey=True)
    for ax,group,color in zip(axes,GROUP[1:],COLORS[1:]):
        e=effects[effects.contrast.eq(group+" - control")].reset_index(drop=True)
        pos=np.arange(8)
        ax.hlines(pos,100*e.sim_low,100*e.sim_high,color=color)
        ax.scatter(100*e.difference,pos,color=color,s=22)
        ax.axvline(0,lw=.8,color="#77818b");ax.set_title(group.upper())
        ax.set(xlim=(-80,55),yticks=pos,yticklabels=NAMES)
    axes[0].invert_yaxis();fig.supxlabel("Adjusted probability difference (percentage points); simultaneous 95% intervals",fontsize=10)
    finish(fig,"06_combination_contrasts")
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for i,(split,title) in enumerate([("expID","Held-out experiments"),("Date","Held-out dates")]):
        g=cv[cv.split==split].set_index("model").loc[MODELS]
        axes[0].plot(range(3),g.log_loss,marker="o",label=title,color=COLORS[i])
    axes[0].set(xticks=range(3),xticklabels=["Multinomial","Additive","Pairwise"],ylabel="Eight-state log loss")
    axes[0].legend(frameon=False,fontsize=8)
    for name,color in zip([MODELS[0],MODELS[-1]],COLORS[:2]):
        g=oof[oof.model==name].copy()
        g["bin"]=pd.qcut(g.predicted,5,duplicates="drop")
        v=g.groupby("bin",observed=True)[["predicted","observed"]].mean()
        axes[1].plot(v.predicted,v.observed,marker="o",label=name,color=color)
    axes[1].plot([0,1],[0,1],"--",color="#888888",lw=1)
    axes[1].set(xlabel="Predicted response probability",ylabel="Observed response fraction")
    axes[1].legend(frameon=False,fontsize=8);finish(fig,"07_validation")


def run():
    FIG.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    df=load()
    x=design(df); y=df.state.to_numpy()
    summary=paired_tests(df,rng)
    counts=pd.crosstab(df.Type,df.state).reindex(columns=range(8),fill_value=0)
    counts.columns=NAMES
    save(counts.reset_index(),"observed_states")
    save(pd.crosstab([df.Context,df.CallType],df.state).rename(columns=dict(enumerate(NAMES))).reset_index(),"states_by_condition")
    gee(df)
    cv,oof=validate(df,x,y,rng)
    model=fit(x,y,MODELS[-1])
    grids=[design(df.assign(CallType=g)) for g in GROUP]
    rates=standardize(model,grids)
    weights=df.loc[df.Type=="call","CallType"].value_counts(normalize=True).loc[GROUP[1:]].values
    mixes=np.vstack([[-1,*weights]]+[np.eye(4)[j]-np.eye(4)[i] for i,j in combinations(range(4),2)])
    labels=["natural - control"]+[GROUP[j]+" - "+GROUP[i] for i,j in combinations(range(4),2)]
    effect=mixes@rates
    # Refit entire model to resampled experiment pairs; keep target covariate
    # distribution and observed natural-call mixture fixed for standardization.
    pairs=np.array(list(df.groupby("expID").indices.values()))
    draws=[]; raw_diffs=[]
    for _ in range(REPS):
        sample=pairs[rng.integers(0,len(pairs),len(pairs))].ravel()
        draws.append(standardize(fit(x[sample],y[sample],MODELS[-1]),grids))
        e=(y[sample]>0).astype(float).reshape(-1,2)
        raw_diffs.append((e[:,0]-e[:,1]).mean())
    summary["raw_diff_ci"]=np.quantile(raw_diffs,[.025,.975]).tolist()
    draws=np.array(draws)
    contrasts=mixes@draws
    se=contrasts.std(axis=0,ddof=1)
    # Single critical value for 7 contrasts x 8 outcomes.
    critical=np.quantile(np.abs((contrasts-effect)/se).max(axis=(1,2)),.95)
    summary["sim_critical"]=float(critical)
    records=[]
    for i,label in enumerate(labels):
        for j,state in enumerate(NAMES):
            lo,hi=np.quantile(contrasts[:,i,j],[.025,.975])
            records.append([label,state,effect[i,j],lo,hi,max(-1,effect[i,j]-critical*se[i,j]),min(1,effect[i,j]+critical*se[i,j])])
    effects=pd.DataFrame(records,columns=["contrast","state","difference","point_low","point_high","sim_low","sim_high"])
    save(effects,"adjusted_contrasts")
    records=[]
    for i,(typ,row) in enumerate(zip(GROUP,rates)):
        for j,state in enumerate(NAMES):
            lo,hi=np.quantile(draws[:,i,j],[.025,.975])
            records.append([typ,state,row[j],lo,hi])
    save(pd.DataFrame(records,columns=["CallType","state","probability","low","high"]),"adjusted_probabilities")
    checks=[]
    for lam in [.25,1,4]:
        p=standardize(fit(x,y,MODELS[-1],lam),grids)
        checks.append(pd.DataFrame(mixes@p,index=labels,columns=NAMES).rename_axis("contrast").reset_index().assign(penalty=lam))
    save(pd.concat(checks),"penalty_sensitivity")
    common=df.Distance.between(20,150).to_numpy()
    overlap=standardize(model,[g[common] for g in grids])
    save(pd.DataFrame(mixes@overlap,index=labels,columns=NAMES).rename_axis("contrast").reset_index(),
         "common_distance_standardization")
    grid=pd.MultiIndex.from_product([GROUP,CONTEXT,[50,100,150],[8]],names=["CallType","Context","Distance","Npenguins"]).to_frame(index=False)
    pred=predict(model,design(grid))
    assert np.allclose(pred.sum(axis=1),1) and np.all(pred>=0)
    save(pd.concat([grid,pd.DataFrame(pred,columns=NAMES)],axis=1),"scenario_probabilities")
    plot_results(df,counts,rates,effects,cv,oof,model)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); print(cv.to_string(index=False))


if __name__=="__main__":
    run()
