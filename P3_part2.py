"""Playback response latency and conditional intensity. Run: python P3_part2.py

Dependencies: the same environment as P3_part1.py (numpy, pandas, scipy,
statsmodels, matplotlib). No saved first-part estimates are needed.
"""
import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from numpy.polynomial.hermite import hermgauss

import P3_part1 as occurrence

OUT = Path(__file__).resolve().parent / "outputs_p3_part2"
FIG = OUT / "figures"
TASKS = ["Head-up", "Vocal", "Approach", "First response"]
BEHAVIORS = ["HeadsUp", "Vocal", "Approach"]
MARKS = ["HeadsUpDuration", "VocalNcalls", "ApproachDuration"]
TYPES = occurrence.GROUP
CONTEXTS = occurrence.CONTEXT
DISTANCES = [50, 100, 150]
MODELS = ["Separate AFT", "Separate ridge AFT", "Shared AFT"]
POOL_GRID = [1., 4., 16., 64., 256., 1024.]
SEED = 20261128
NODES, NODE_WEIGHTS = hermgauss(16)
NODE_WEIGHTS /= np.sqrt(np.pi)


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def prepare():
    df = occurrence.load()
    flags = df[[b+"YesNo" for b in BEHAVIORS]].to_numpy(bool)
    raw = df[[b+"Latency" for b in BEHAVIORS]].to_numpy(float)
    marks = df[MARKS].to_numpy(float)
    assert np.isfinite(raw[flags]).all() and (raw[flags] >= 0).all()
    assert np.isfinite(marks[flags]).all() and (marks[flags] > 0).all()
    assert np.all(marks[flags[:, 1], 1] == np.floor(marks[flags[:, 1], 1]))
    t = np.where(flags, raw, np.nan)
    first = np.min(np.where(flags, raw, np.inf), axis=1)
    times = np.column_stack([t, np.where(np.isfinite(first), first, np.nan)])
    audit = []
    for k,b in enumerate(BEHAVIORS):
        audit.append([b, flags[:,k].sum(), np.sum(flags[:,k] & (raw[:,k]==0)),
                      np.sum(~flags[:,k] & np.isfinite(raw[:,k])),
                      np.sum(~flags[:,k] & (marks[:,k]>0))])
    save(pd.DataFrame(audit, columns=["behavior","responses","immediate","nonresponse_with_latency",
                                      "nonresponse_with_positive_intensity"]), "audit")
    return df, times, np.where(flags, marks, np.nan)


def fit_time(x, t, mode, lam):
    """Estimate separate intercepts; shrink slopes toward their cross-task mean.

    The first-event outcome duplicates information from the component outcomes.
    Fit it in a separate block so it cannot influence the three behavior slopes.
    """
    p = x.shape[1]
    slopes = np.eye(p)
    slopes[0,0] = 0.
    coefs, scales = [], []
    for block in ([0,1,2], [3]):
        size = len(block)
        matrix = np.zeros((size*p, size*p))
        rhs = np.zeros(size*p)
        data = []
        for i,k in enumerate(block):
            ok = t[:,k] > 0
            xx, yy = x[ok], np.log(t[ok,k])
            if not len(yy):
                raise ValueError("No positive times for " + TASKS[k])
            sl = slice(i*p, (i+1)*p)
            matrix[sl,sl] = xx.T @ xx
            rhs[sl] = xx.T @ yy
            data.append((sl,xx,yy))
        penalty = np.zeros_like(matrix)
        if mode != MODELS[0]:
            center = np.eye(size)
            if mode == MODELS[2] and size > 1:
                center -= np.ones((size,size))/size
            # Derived first response uses the same modest standalone ridge in both models.
            reg = np.eye(size) + lam*center if size>1 else 2*np.eye(1)
            penalty = np.kron(reg, slopes)
        inverse = np.linalg.pinv(matrix + penalty, hermitian=True)
        estimate = inverse @ rhs
        for sl,xx,yy in data:
            beta = estimate[sl]
            effective_df = np.sum((xx @ inverse[sl,sl])*xx)
            sd = np.sqrt(np.sum((yy-xx@beta)**2)/max(len(yy)-effective_df,1.))
            coefs.append(beta); scales.append(max(sd,1e-3))
    # A point mass for documented immediate head-up and derived first response.
    n = np.isfinite(t).sum(axis=0)
    zero = np.where([True,False,False,True], ((t==0).sum(axis=0)+.5)/(n+1), 0.)
    return np.array(coefs).T, np.array(scales), zero


def time_parameters(model, x):
    beta, sd, zero = model
    return x @ beta, sd, zero


def quantile(mu, sd, zero, alpha):
    level = (alpha-zero)/(1-zero)
    value = np.exp(mu+sd*ndtri(np.clip(level,1e-10,1-1e-10)))
    return np.where(level<=0,0,value)


def nll(t, mu, sd, zero):
    positive = np.maximum(np.nan_to_num(t),1e-9)
    loss = (np.log(positive*sd*np.sqrt(2*np.pi))+
            .5*((np.log(positive)-mu)/sd)**2-np.log1p(-zero))
    return np.where(np.isnan(t),np.nan,np.where(t==0,-np.log(np.maximum(zero,1e-12)),loss))


def splits(groups, nfolds, seed):
    unique = np.unique(groups)
    for ids in np.array_split(np.random.default_rng(seed).permutation(unique),nfolds):
        test = np.isin(groups,ids)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def choose_penalty(x,t,groups,mode,seed):
    scores = []
    for lam in POOL_GRID:
        losses = []
        for tr,te in splits(groups,4,seed):
            par = time_parameters(fit_time(x[tr],t[tr],mode,lam),x[te])
            losses.append(nll(t[te],*par)[:,:3])
        # Equal weight for the three behaviors, including sparse approach events.
        scores.append(np.nanmean(np.nanmean(np.vstack(losses),axis=0)))
    return POOL_GRID[int(np.argmin(scores))]


def fit_marks(x,t,y,pre,coupled):
    """Conditional mean intensities. Timing is marginalized before prediction."""
    fits = []
    for k in range(3):
        valid = np.isfinite(y[:,k])
        features = x[valid]
        if k==1:
            features = np.column_stack([features,pre[valid]])
        if coupled:
            features = np.column_stack([features,np.log1p(t[valid,k])])
        target = y[valid,k]
        pen = np.ones(features.shape[1])*3.
        pen[0] = 0.
        if k==1:
            # Positive count = 1 + excess count, with a Poisson mean model.
            def objective(beta):
                linear = features@beta
                mean = np.exp(linear)
                return (np.sum(mean-(target-1)*linear)+.5*np.dot(pen,beta**2),
                        features.T@(mean-(target-1))+pen*beta)
            res = minimize(objective,np.zeros(features.shape[1]),jac=True,method="L-BFGS-B")
            if not res.success:
                raise ValueError(res.message)
            beta, smear = res.x, 1.
        else:
            beta = np.linalg.solve(features.T@features+np.diag(pen),features.T@np.log(target))
            smear = np.mean(target/np.exp(features@beta))
        fits.append((beta,smear))
    return fits


def predict_marks(fits,x,pre,par,coupled):
    mu,sd,zero = par
    output = []
    for k,(beta,smear) in enumerate(fits):
        features = np.column_stack([x,pre]) if k==1 else x
        base = np.exp(features@beta[:features.shape[1]])
        if coupled:
            logtime = mu[:,k,None]+np.sqrt(2)*sd[k]*NODES
            # E[(1+T)^s] integrates both immediate and positive times.
            factor = np.exp(beta[-1]*np.logaddexp(0,logtime)) @ NODE_WEIGHTS
            base *= zero[k]+(1-zero[k])*factor
        output.append((1+base) if k==1 else smear*base)
    return np.column_stack(output)


def validate(x,t,y,pre,groups):
    records, marks, folds = [], [], []
    for repeat in range(5):
        for fold,(tr,te) in enumerate(splits(groups,5,SEED+repeat)):
            for mode in MODELS:
                lam = (0 if mode==MODELS[0] else
                       choose_penalty(x[tr],t[tr],groups[tr],mode,SEED+111*repeat+fold))
                folds.append([repeat,fold,mode,lam])
                par = time_parameters(fit_time(x[tr],t[tr],mode,lam),x[te])
                pred = quantile(*par,.5)
                low,high = quantile(*par,.05),quantile(*par,.95)
                loss = nll(t[te],*par)
                for k,name in enumerate(TASKS):
                    ok = np.isfinite(t[te,k])
                    truth = t[te[ok],k]
                    error = pred[ok,k]-truth
                    inside = (truth>=low[ok,k]) & (truth<=high[ok,k])
                    records.append([mode,name,len(truth),np.abs(error).sum(),(error**2).sum(),
                                    loss[ok,k].sum(),inside.sum()])
                if mode==MODELS[-1]:
                    for coupled in [False,True]:
                        f = fit_marks(x[tr],t[tr],y[tr],pre[tr],coupled)
                        predicted = predict_marks(f,x[te],pre[te],par,coupled)
                        for k,name in enumerate(MARKS):
                            err = predicted[:,k]-y[te,k]
                            marks.append([coupled,name,np.isfinite(err).sum(),
                                          np.nansum(abs(err)),np.nansum(err**2)])
    agg=pd.DataFrame(records,columns=["model","endpoint","n","abs","squared","loss","covered"])
    agg=agg.groupby(["model","endpoint"],sort=False).sum().reset_index()
    result=agg[["model","endpoint"]].copy()
    result["events"]=agg.n//5
    result["MAE_s"]=agg["abs"]/agg.n
    result["RMSE_s"]=np.sqrt(agg.squared/agg.n)
    result["NLL"]=agg.loss/agg.n
    result["coverage90"]=agg.covered/agg.n
    save(result,"latency_validation")
    m=pd.DataFrame(marks,columns=["coupled","endpoint","n","abs","squared"])
    m=m.groupby(["coupled","endpoint"]).sum().reset_index()
    m["MAE"]=m["abs"]/m.n; m["RMSE"]=np.sqrt(m.squared/m.n)
    save(m[["coupled","endpoint","MAE","RMSE"]],"intensity_validation")
    save(pd.DataFrame(folds,columns=["repeat","fold","model","penalty"]),"fold_settings")
    return result,m


def make_grid(df):
    grid = pd.DataFrame(product(TYPES,DISTANCES,CONTEXTS),columns=["CallType","Distance","Context"])
    grid["Npenguins"] = df.Npenguins.median()
    return grid


def scenarios(gate,model,mark_models,gx,context_weight,pre):
    """Context-standardized medians among responders and mean observed intensities."""
    states = occurrence.predict(gate,gx)
    pr = np.column_stack([states @ occurrence.BITS,1-states[:,0]])
    w = pr.reshape(-1,3,4)*context_weight[None,:,None]
    w /= w.sum(axis=1,keepdims=True)
    x = gx[:,:8]
    mu,sd,zero = time_parameters(model,x)
    m = mu.reshape(-1,3,4)
    # Bisection on the log scale for a mixture of context-specific CDFs.
    lo=np.min(m-9*sd,axis=1); hi=np.max(m+9*sd,axis=1)
    for _ in range(44):
        mid=(hi+lo)/2
        cdf=zero+(1-zero)*ndtr((mid[:,None,:]-m)/sd)
        below=(w*cdf).sum(axis=1)<.5
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    time=np.where(zero>=.5,0,np.exp((lo+hi)/2))
    intens=predict_marks(mark_models,x,pre,(mu,sd,zero),False).reshape(-1,3,3)
    return time,(intens*w[:,:,:3]).sum(axis=1)


def summarize_grid(df,t,point,draws):
    rows=[]
    for i,(typ,dist) in enumerate(product(TYPES,DISTANCES)):
        group=(df.CallType==typ)&(df.Distance==dist)
        exact=group & df.Npenguins.eq(df.Npenguins.median())
        for k,name in enumerate(TASKS):
            n=int(group.sum()); events=np.isfinite(t[group,k]).sum()
            strict_events=int(np.isfinite(t[exact,k]).sum())
            for measure,idx in [("Latency",0),("Intensity",1)]:
                if idx==1 and k==3:
                    continue
                estimate=point[idx][i,k]
                low,high=np.quantile(draws[idx][:,i,k],[.025,.975])
                if not events:
                    estimate=low=high=np.nan
                rows.append([typ,dist,df.Npenguins.median(),name,measure,n,events,int(exact.sum()),
                             strict_events,"No observations" if not n else ("No response" if not events else "Estimated"),
                             estimate,low,high])
    return pd.DataFrame(rows,columns=["CallType","Distance","Npenguins","endpoint","measure",
                    "playbacks","events","exact_size_playbacks","exact_size_events","status","estimate","lo95","hi95"])


def table_figure(tab,measure,strict=False):
    data=tab[tab.measure==measure]
    endpoints=TASKS if measure=="Latency" else TASKS[:3]
    if measure=="Latency":
        title="Predicted median response latency"
        headers=["Time to first head\nraise (seconds)","Time to first\nvocalization (seconds)",
                 "Time to first\napproach (seconds)","Time to first response\nof any kind (seconds)"]
    else:
        title="Predicted mean intensity among responders"
        headers=["Head-raising duration\n(seconds)","Number of vocalizations\n(calls)",
                 "Approach duration\n(seconds)"]
    fig,ax=plt.subplots(figsize=(13,7.3))
    ax.axis("off")
    if strict:
        title = "Predicted median response latency for groups of exactly eight penguins"
    fig.text(.055,.946,title,fontsize=18,weight="bold",color="#18353d")
    fig.text(.055,.894,"Group size = 8 penguins   |   Standardized over behavioral contexts   |   95% bootstrap confidence intervals",fontsize=10)
    grid=[];grey=[]
    for typ,dist in product(TYPES,DISTANCES):
        g=data[(data.CallType==typ)&(data.Distance==dist)].set_index("endpoint")
        n=g.iloc[0].exact_size_playbacks if strict else g.iloc[0].playbacks
        row=[{"control":"Control sound","flat":"Flat call","mod":"Modulated call","tv":"Two-voice call"}[typ],str(dist),str(n)]
        flags=[False]*3
        for name in endpoints:
            r=g.loc[name]
            count=r.exact_size_events if strict else r.events
            cell=("No trials" if not n else "No response") if not count else f"{r.estimate:.1f}   [{r.lo95:.1f}, {r.hi95:.1f}]"
            row.append(cell);flags.append(not count)
        grid.append(row);grey.append(flags)
    widths=[.12,.095,.09]+[.695/len(endpoints)]*len(endpoints)
    table=ax.table(cellText=grid,colLabels=["Playback sound","Distance\n(meters)","Number of\nplaybacks",*headers],colWidths=widths,
                   cellLoc="center",bbox=[0,.11,1,.80])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row==0:
            cell.set_height(cell.get_height()*1.5)
            cell.set_facecolor("#205159");cell.set_text_props(color="white",weight="bold")
        else:
            cell.set_facecolor(["#e5eeed","#f4f7f7"][((row-1)//3)%2])
            if grey[row-1][col]:
                cell.set_text_props(color="#62727a",style="italic")
    support="exactly 8 penguins" if strict else "all observed group sizes"
    fig.text(.055,.08,f"Trials and observed-response checks use playback type and exact distance, with {support}.",fontsize=10)
    fig.text(.055,.045,"No response = trials available, but no corresponding behavior recorded. No trials = no matching playback. Neither establishes lack of perception.",fontsize=9)
    fig.subplots_adjust(left=.055,right=.95,top=.91,bottom=.04)
    fig.savefig(FIG/(("03_exact_size_support" if strict else ("01_latency_table" if measure=="Latency" else "02_intensity_table"))+".png"),dpi=180)
    plt.close(fig)


def validation_figure(metrics,t):
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.5))
    colors=["#287e72","#b7732c","#59689b"]
    behavior_labels=["Head raising","Vocalization","Approach"]
    model_labels=["Separate accelerated failure time models",
                  "Separate ridge-regularized accelerated failure time models",
                  "Accelerated failure time model with partially shared slopes"]
    for k in range(3):
        vals=np.sort(t[np.isfinite(t[:,k]),k])
        axes[0].step(np.r_[0,vals],np.r_[0,np.arange(1,len(vals)+1)/len(vals)],where="post",
                       label=f"{behavior_labels[k]} ({len(vals)} responses)",color=colors[k])
    axes[0].set(xlabel="Seconds after playback",ylabel="Fraction of observed responses",title="Observed onset among responders",xlim=(0,50))
    axes[0].legend(frameon=False,loc="upper left",bbox_to_anchor=(0,-.28))
    for i,mode in enumerate(MODELS):
        g=metrics[metrics.model==mode].set_index("endpoint").loc[TASKS]
        axes[1].bar(np.arange(4)+.24*(i-1),g.MAE_s,width=.23,label=model_labels[i],color=colors[i])
    axes[1].set(xticks=range(4),xticklabels=[*behavior_labels,"First response\nof any kind"],
                ylabel="Mean absolute error (seconds)",title="Prediction error in held-out experiments")
    axes[1].legend(frameon=False,loc="upper left",bbox_to_anchor=(0,-.28),fontsize=10)
    fig.subplots_adjust(left=.065,right=.98,bottom=.33,top=.90,wspace=.24)
    fig.savefig(FIG/"04_timing_validation.png",dpi=180);plt.close(fig)


def run(reps):
    FIG.mkdir(parents=True,exist_ok=True)
    df,t,y=prepare()
    gate_x=occurrence.design(df);x=gate_x[:,:8]
    groups=df.expID.to_numpy()
    pre=np.log1p(df.VocalNcallsBefore.to_numpy(float))
    results,mark_results=validate(x,t,y,pre,groups)
    lam=choose_penalty(x,t,groups,MODELS[1],SEED+999)
    grid=make_grid(df); gx=occurrence.design(grid)
    ctx=df.Context.value_counts(normalize=True).loc[CONTEXTS].to_numpy()
    # Fixed reference pre-playback vocal activity for the vocal intensity model.
    pre_grid=np.full(len(grid),np.median(pre))
    states=df.state.to_numpy()
    def refit(idx):
        gate=occurrence.fit(gate_x[idx],states[idx],occurrence.MODELS[-1])
        timing=fit_time(x[idx],t[idx],MODELS[1],lam)
        strength=fit_marks(x[idx],t[idx],y[idx],pre[idx],False)
        return scenarios(gate,timing,strength,gx,ctx,pre_grid)
    point=refit(np.arange(len(df)))
    rng=np.random.default_rng(SEED)
    pairs=np.array(list(df.groupby("expID").indices.values()))
    boot=[refit(pairs[rng.integers(0,len(pairs),len(pairs))].ravel()) for _ in range(reps)]
    draws=[np.stack([v[j] for v in boot]) for j in range(2)]
    tab=summarize_grid(df,t,point,draws)
    save(tab,"predictions")
    # Paired draws give uncertainty for comparisons, not just separate intervals.
    contrasts=[]
    for k,name in enumerate(TASKS):
        for i,(typ,dist) in enumerate(product(TYPES,DISTANCES)):
            if typ=="control": continue
            j=DISTANCES.index(dist)
            supported=all(((df.CallType==a)&(df.Distance==dist)&np.isfinite(t[:,k])).any() for a in [typ,"control"])
            if supported:
                lo,hi=np.quantile(draws[0][:,i,k]-draws[0][:,j,k],[.025,.975])
                contrasts.append([typ,dist,name,point[0][i,k]-point[0][j,k],lo,hi])
    save(pd.DataFrame(contrasts,columns=["versus_control","Distance","endpoint","difference_s","lo95","hi95"]),"latency_contrasts")
    distance_effects=[]
    for i,typ in enumerate(TYPES):
        near,far=3*i,3*i+2
        for k,name in enumerate(TASKS):
            g=tab[(tab.measure=="Latency") & (tab.CallType==typ) & (tab.endpoint==name)]
            if g[g.Distance.isin([50,150])].status.eq("Estimated").all():
                lo,hi=np.quantile(draws[0][:,far,k]-draws[0][:,near,k],[.025,.975])
                distance_effects.append([typ,name,point[0][far,k]-point[0][near,k],lo,hi])
    save(pd.DataFrame(distance_effects,columns=["CallType","endpoint","difference_150m_minus_50m_s","lo95","hi95"]),
         "distance_contrasts")
    table_figure(tab,"Latency");table_figure(tab,"Intensity");table_figure(tab,"Latency",strict=True)
    validation_figure(results,t)
    save(pd.DataFrame([{"penalty":lam,"bootstrap_replicates":reps,"pairs_per_replicate":len(pairs),"seed":SEED,"group_size":df.Npenguins.median(),
                        "before_calls":df.VocalNcallsBefore.median(),"intensity_coupling":False,"main_timing_model":MODELS[1],**df.Context.value_counts(normalize=True).to_dict()}]),"settings")
    print(results.round(3).to_string(index=False))
    print(mark_results.round(3).to_string(index=False))
    print("Final penalty:",lam,"; bootstrap replicates:",reps,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",type=int,default=1000)
    args=parser.parse_args()
    if args.bootstrap<2: parser.error("--bootstrap must be at least 2")
    run(args.bootstrap)
