"""Run python P4_part2.py in the same environment as P4_part1.py.

Full factorial stress test of the original correlation thresholds and fidelity.

The recognition layer is described by the trial-level pass fraction
q(d) = P(F_trial(d) * (1 - delta) >= tau).  The mean-fidelity radius is kept
for continuity with Problem 2.  Because q(d) can remain positive beyond that
mean radius, the behavioural communication radius is an *expected-benefit*
radius and may legitimately exceed the mean acoustic radius; q at every
reported radius is saved, and a minimum-recognition floor is tested as a
separate sensitivity.  The response layer remains a scenario assumption, not
an identified noise effect.
"""
import argparse
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression
import matplotlib.pyplot as plt

import P4_part1 as base

OUT = base.ROOT / "outputs_p4_part2"
FIG = OUT / "figures"
TAUS = (.50, .60, .70)
LOSSES = (0., .05, .10, .15, .20)
SCENARIOS = list(product(TAUS, LOSSES))
MAIN_ETA = .5
ETA_GRID = (0., .5, 1.)
Q_FLOORS = (0., .25, .5)
SEED = 42621
KNOTS = np.array(base.acoustic.DISTANCES)
AT_IDX = [np.flatnonzero(base.DIST == d)[0] for d in base.AT]
KEYS = [(0.5, 0), (.6, 0), (.7, 0), (.5, .1), (.5, .2), (.7, .2)]
PAIRS = [("mod", "flat"), ("tv", "flat"), ("tv", "mod")]


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def majority_radius(q):
    """Distance where the trial-level recognition probability crosses 0.5."""
    below = np.flatnonzero(q < .5)
    if len(below) == 0:
        return float(base.DIST[-1]), "at_least_upper_limit"
    j = int(below[0])
    if j == 0:
        return float(base.DIST[0]), "below_lower_limit"
    fraction = (q[j - 1] - .5) / (q[j - 1] - q[j])
    return float(base.DIST[j - 1] + fraction * (base.DIST[j] - base.DIST[j - 1])), "crossing"


def acoustics(cubes):
    """Mean curve, trial-level pass rates, retention ratios and both radii."""
    all_distances = np.r_[base.DIST, base.acoustic.GRID]
    all_f, weights = base.fidelity(cubes, all_distances)
    mean_f = all_f[:, :len(base.DIST)]
    pass_rates = []
    counts = []
    for block in cubes:
        available = np.isfinite(block) @ weights
        scores = np.divide(np.nan_to_num(block) @ weights, available,
                           out=np.full(available.shape, np.nan), where=available > 0)
        complete = scores[np.isfinite(scores).all(axis=1)]
        if len(complete) == 0:
            raise ValueError("No common sound trajectories")
        counts.append(len(complete))
        by_scenario = []
        for tau, loss in SCENARIOS:
            empirical = (complete >= tau / (1 - loss)).mean(axis=0)
            monotone = isotonic_regression(empirical, increasing=False).x
            by_scenario.append(np.interp(base.DIST, KNOTS, monotone))
        pass_rates.append(by_scenario)
    gates = np.swapaxes(np.array(pass_rates), 0, 1)   # scenario, call type, distance
    if np.any(gates > gates[:1] + 1e-12):
        raise ValueError("Nested recognition thresholds inconsistent")
    # Ratio only on supported baseline pass rates. In resamples with zero
    # baseline support in the far tail, carry forward the last estimable ratio.
    retain = np.ones_like(gates)
    for i in range(len(cubes)):
        estimable = gates[0, i] > 1e-12
        if not estimable.any():
            raise ValueError("No baseline threshold passes")
        for k in range(1, len(SCENARIOS)):
            retain[k, i] = np.interp(base.DIST, base.DIST[estimable],
                                     gates[k, i, estimable] / gates[0, i, estimable])
    ar = np.empty((len(SCENARIOS), len(cubes)))
    astatus = np.empty(ar.shape, dtype=object)
    for k, (tau, loss) in enumerate(SCENARIOS):
        for i, curve in enumerate(all_f[:, len(base.DIST):]):
            val, flag = base.acoustic.solve_range(curve * (1 - loss), tau)
            # A below-support radius is stored at the observed boundary for
            # clipped summaries and explicitly labeled when reported.
            ar[k, i] = np.clip(val, KNOTS.min(), KNOTS.max())
            astatus[k, i] = flag
    maj = np.empty(ar.shape)
    mstatus = np.empty(ar.shape, dtype=object)
    for k in range(len(SCENARIOS)):
        for i in range(len(cubes)):
            maj[k, i], mstatus[k, i] = majority_radius(gates[k, i])
    return mean_f, gates, np.clip(retain, 0, 1), ar, astatus, counts, maj, mstatus


def calculate(f, retain, prob, weights, domains, eta, gates, q_floor=0.):
    """Scenario index and response distribution.

    keep = 1 - eta * (1 - retain) is the share of the behavioural excess that
    survives the loss of recognition; gate = 1{q(d) >= q_floor} is a separate,
    optional minimum-recognition requirement.  With q_floor = 0 the gate is
    inactive and the communication radius is a pure expected-benefit radius.
    """
    original, _, p_any = base.index_components(f, prob, weights)
    p_control = base.include_average(1 - prob[:1, :, :, 0], weights)
    keep = (1 - eta * (1 - retain))[:, :, None, :]
    gate = (gates >= q_floor)[:, :, None, :]
    effective = keep * gate
    changed_p = p_control[None, ...] + effective * (p_any - p_control)[None, ...]
    physical = np.array([1 - loss for _, loss in SCENARIOS])[:, None, None, None]
    index = physical * effective * original[None, ...]
    radii = np.empty(index.shape[:-1])
    flags = np.empty(radii.shape, dtype=object)
    for k in range(len(SCENARIOS)):
        for i, t in enumerate(base.TYPES):
            for j, ctx in enumerate(base.LAYERS):
                radii[k, i, j], flags[k, i, j] = base.crossing(
                    index[k, i, j], domains[t, ctx], base.THRESHOLD)
    if q_floor == 0:
        ok = (np.allclose(changed_p[0], p_any) and np.allclose(index[0], original))
    else:
        ok = np.all(index[0] <= original + 1e-12)
    if not (ok and np.all((changed_p >= 0) & (changed_p <= 1))):
        raise ValueError("Scenario calibration failed")
    return radii, flags, changed_p, index


def paired_intervals(draws, scale=1):
    """Pointwise intervals for levels and scenario minus baseline, from paired draws."""
    q = np.quantile(draws, [.025, .975], axis=0)
    changes = np.quantile(scale * (draws - draws[:, :1]), [.025, .975], axis=0)
    return q, changes


def area_loss(draw_radius, base_radius, lower, upper):
    """Paired bootstrap area loss, undefined when either radius is censored."""
    valid = ((draw_radius > lower + 1e-6) & (draw_radius < upper - 1e-6)
             & (base_radius > lower + 1e-6) & (base_radius < upper - 1e-6))
    values = 100 * (1 - (draw_radius / base_radius) ** 2)
    if valid.sum() < 20:
        return np.nan, np.nan, np.nan, float(valid.mean())
    lo, hi = np.quantile(values[valid], [.025, .975])
    return float(values[valid].mean()), float(lo), float(hi), float(valid.mean())


def outputs(center, ac, boot, domains, alternative, floors, reference_df, nboot, size):
    f, gates, retain, ar, aflags, ncommon, maj, mstatus = ac
    radii, flags, p, index = center
    ac_ci, ac_diff = paired_intervals(boot[0])
    r_ci, r_diff = paired_intervals(boot[1])
    p_ci, p_diff = paired_intervals(boot[2], 100)
    rr = []
    aa = []
    pp = []
    ss = []
    lines = []
    contrasts = []
    floor_rows = []
    eta_rows = []
    qrows = []
    status_rows = []
    for k, (tau, loss) in enumerate(SCENARIOS):
        for i, t in enumerate(base.TYPES):
            q_at_mean = float(np.interp(ar[k, i], base.DIST, gates[k, i]))
            q_at_maj = float(np.interp(maj[k, i], base.DIST, gates[k, i]))
            aa.append([tau, loss, t, ar[k, i], aflags[k, i], maj[k, i], mstatus[k, i],
                       *ac_ci[:, k, i], ar[k, i] - ar[0, i], *ac_diff[:, k, i],
                       ncommon[i], q_at_mean, q_at_maj])
            for j, ctx in enumerate(base.LAYERS):
                a, b = domains[t, ctx]
                q_here = float(np.interp(radii[k, i, j], base.DIST, gates[k, i]))
                r0 = radii[0, i, j]
                point_ok = (flags[0, i, j] == "crossing" and flags[k, i, j] == "crossing"
                            and r0 > a + 1e-6 and radii[k, i, j] > a + 1e-6)
                point_area = 100 * (1 - (radii[k, i, j] / r0) ** 2) if point_ok else np.nan
                _, area_lo, area_hi, area_frac = area_loss(
                    boot[1][:, k, i, j], boot[1][:, 0, i, j], a, b)
                if not point_ok:
                    area_lo = area_hi = np.nan
                rr.append([tau, loss, t, ctx, a, b, radii[k, i, j], flags[k, i, j],
                           *r_ci[:, k, i, j], radii[k, i, j] - radii[0, i, j],
                           *r_diff[:, k, i, j],
                           np.mean(boot[1][:, k, i, j] == a),
                           np.mean(boot[1][:, k, i, j] == b),
                           np.mean((boot[1][:, k, i, j] > a + 1e-6)
                                   & (boot[1][:, k, i, j] < b - 1e-6)),
                           q_here, point_area, area_lo, area_hi, area_frac,
                           bool(radii[k, i, j] > ar[k, i] + 1e-9),
                           "≥" if flags[k, i, j] == "at_least_upper_limit" else
                           "<" if flags[k, i, j] == "below_lower_limit" else ""])
                for n, (dist, idx) in enumerate(zip(base.AT, AT_IDX)):
                    if not a <= dist <= b:
                        continue
                    pp.append([tau, loss, t, ctx, dist, p[0, i, j, idx], p[k, i, j, idx],
                               *p_ci[:, k, i, j, n],
                               100 * (p[k, i, j, idx] - p[0, i, j, idx]),
                               *p_diff[:, k, i, j, n],
                               (1 - loss) * f[i, idx], gates[k, i, idx],
                               retain[k, i, idx], index[k, i, j, idx],
                               radius_ok := bool(radii[k, i, j] >= dist),
                               q_here])
                if ctx == "standardized":
                    for eta, results in alternative.items():
                        rad, stat, prob, _ = results
                        for n, idx in enumerate(AT_IDX):
                            ss.append([tau, loss, t, eta, rad[k, i, j], stat[k, i, j],
                                       base.AT[n], prob[k, i, j, idx]])
                    for qf, results in floors.items():
                        rad, stat, prob, _ = results
                        floor_rows.append([qf, tau, loss, t, ctx, rad[k, i, j],
                                           stat[k, i, j],
                                           float(np.interp(rad[k, i, j], base.DIST, gates[k, i]))])
                    if (tau, loss) in KEYS:
                        for idx, d in enumerate(base.DIST):
                            if a <= d <= b:
                                lines.append([tau, loss, t, d, p[k, i, j, idx],
                                              index[k, i, j, idx]])
    for kk in range(len(SCENARIOS)):
        counts = {name: int(np.sum(flags[kk] == name)) for name in
                  ("crossing", "below_lower_limit", "at_least_upper_limit")}
        status_rows.append([SCENARIOS[kk][0], SCENARIOS[kk][1],
                            counts["crossing"], counts["below_lower_limit"],
                            counts["at_least_upper_limit"],
                            counts["crossing"] / flags[kk].size])
    for k, (tau, loss) in enumerate(SCENARIOS):
        for j, ctx in enumerate(base.LAYERS):
            for hi, lo in PAIRS:
                hi_i, lo_i = base.TYPES.index(hi), base.TYPES.index(lo)
                point_hi = radii[k, hi_i, j] - radii[0, hi_i, j]
                point_lo = radii[k, lo_i, j] - radii[0, lo_i, j]
                draw = ((boot[1][:, k, hi_i, j] - boot[1][:, 0, hi_i, j])
                        - (boot[1][:, k, lo_i, j] - boot[1][:, 0, lo_i, j]))
                dlo, dhi = np.quantile(draw, [.025, .975])
                ah, bh = domains[hi, ctx]
                al, bl = domains[lo, ctx]
                area_hi = 100 * (1 - (radii[k, hi_i, j] / radii[0, hi_i, j]) ** 2) \
                    if flags[0, hi_i, j] == flags[k, hi_i, j] == "crossing" else np.nan
                area_lo = 100 * (1 - (radii[k, lo_i, j] / radii[0, lo_i, j]) ** 2) \
                    if flags[0, lo_i, j] == flags[k, lo_i, j] == "crossing" else np.nan
                draw_hi = 100 * (1 - (boot[1][:, k, hi_i, j] / boot[1][:, 0, hi_i, j]) ** 2)
                draw_lo = 100 * (1 - (boot[1][:, k, lo_i, j] / boot[1][:, 0, lo_i, j]) ** 2)
                valid = ((boot[1][:, 0, hi_i, j] > ah + 1e-6) & (boot[1][:, 0, hi_i, j] < bh - 1e-6)
                         & (boot[1][:, k, hi_i, j] > ah + 1e-6) & (boot[1][:, k, hi_i, j] < bh - 1e-6)
                         & (boot[1][:, 0, lo_i, j] > al + 1e-6) & (boot[1][:, 0, lo_i, j] < bl - 1e-6)
                         & (boot[1][:, k, lo_i, j] > al + 1e-6) & (boot[1][:, k, lo_i, j] < bl - 1e-6))
                if valid.sum() >= 20:
                    alo, ahi = np.quantile((draw_hi - draw_lo)[valid], [.025, .975])
                else:
                    alo = ahi = np.nan
                contrasts.append([tau, loss, ctx, hi, lo, point_hi, point_lo,
                                  point_hi - point_lo, dlo, dhi,
                                  area_hi, area_lo,
                                  area_hi - area_lo if np.isfinite(area_hi) and np.isfinite(area_lo) else np.nan,
                                  alo, ahi,
                                  bool(flags[k, hi_i, j] == flags[k, lo_i, j] == "crossing"
                                       and flags[0, hi_i, j] == flags[0, lo_i, j] == "crossing")])
    for k, (tau, loss) in enumerate(SCENARIOS):
        for i, t in enumerate(base.TYPES):
            for j, ctx in enumerate(base.LAYERS):
                vals = [alternative[e][0][k, i, j] for e in ETA_GRID]
                idx = AT_IDX[1]
                probs = [alternative[e][2][k, i, j, idx] for e in ETA_GRID]
                eta_rows.append([tau, loss, t, ctx, *vals, min(vals), max(vals),
                                 *probs, min(probs), max(probs)])
    for k, (tau, loss) in enumerate(SCENARIOS):
        for i, t in enumerate(base.TYPES):
            for idx, (d, q) in enumerate(zip(base.DIST, gates[k, i])):
                qrows.append([tau, loss, t, d, q, gates[0, i, idx], retain[k, i, idx]])
    acoust = pd.DataFrame(aa, columns=["threshold", "fidelity_loss", "call", "radius_m",
                                       "status", "majority_radius_m", "majority_status",
                                       "low95_clipped", "high95_clipped",
                                       "change_m_clipped", "change_low95", "change_high95",
                                       "common_trials", "pass_fraction_at_mean_radius",
                                       "pass_fraction_at_majority_radius"])
    ranges = pd.DataFrame(rr, columns=["threshold", "fidelity_loss", "call", "context",
                                       "lower_support", "upper_support", "radius_m",
                                       "status", "low95_clipped", "high95_clipped",
                                       "change_m_clipped", "change_low95", "change_high95",
                                       "fraction_at_lower", "fraction_at_upper",
                                       "fraction_identified", "pass_fraction_at_radius",
                                       "area_loss_pct", "area_loss_low95", "area_loss_high95",
                                       "area_fraction_identified", "beyond_mean_radius",
                                       "report_as"])
    probs = pd.DataFrame(pp, columns=["threshold", "fidelity_loss", "call", "context",
                                      "distance", "baseline_probability",
                                      "response_probability", "low95", "high95",
                                      "change_percentage_points", "change_low95",
                                      "change_high95", "mean_fidelity",
                                      "threshold_pass_fraction", "relative_pass_retention",
                                      "effectiveness_index", "within_communication_radius",
                                      "pass_fraction_at_radius"])
    sensitivity = pd.DataFrame(ss, columns=["threshold", "fidelity_loss", "call",
                                            "behavior_sensitivity", "radius_m", "status",
                                            "distance", "response_probability"])
    curves = pd.DataFrame(lines, columns=["threshold", "fidelity_loss", "call", "distance",
                                          "response_probability", "index"])
    contrast = pd.DataFrame(contrasts, columns=["threshold", "fidelity_loss", "context",
                                                "call", "reference", "radius_change_call",
                                                "radius_change_reference", "difference_m",
                                                "low95", "high95", "area_loss_call",
                                                "area_loss_reference", "area_difference_pp",
                                                "area_low95", "area_high95", "identified"])
    floor_frame = pd.DataFrame(floor_rows, columns=["q_floor", "threshold", "fidelity_loss",
                                                    "call", "context", "radius_m", "status",
                                                    "pass_fraction_at_radius"])
    eta_frame = pd.DataFrame(eta_rows, columns=["threshold", "fidelity_loss", "call", "context",
                                                "radius_eta0", "radius_eta05", "radius_eta1",
                                                "radius_min", "radius_max", "p100_eta0",
                                                "p100_eta05", "p100_eta1", "p100_min", "p100_max"])
    recognition = pd.DataFrame(qrows, columns=["threshold", "fidelity_loss", "call", "distance",
                                               "pass_fraction", "baseline_pass_fraction",
                                               "relative_retention"])
    status = pd.DataFrame(status_rows, columns=["threshold", "fidelity_loss", "n_crossing",
                                                "n_below", "n_above", "fraction_crossing"])
    iso = []
    for i, t in enumerate(base.TYPES):
        target = float(radii[SCENARIOS.index((.7, 0.)), i, -1])
        xs = np.array(LOSSES, float)
        ys = np.array([radii[k, i, -1] for k in range(len(SCENARIOS))
                       if SCENARIOS[k][0] == .5])
        if target >= ys[0]:
            equivalent, note = 0., "no loss"
        elif target <= ys[-1]:
            equivalent, note = np.nan, ">20%"
        else:
            equivalent = float(np.interp(target, ys[::-1], xs[::-1]))
            note = "interpolated"
        iso.append([t, .7, 0., target,
                    equivalent * 100 if np.isfinite(equivalent) else np.nan, note])
    iso = pd.DataFrame(iso, columns=["call", "reference_threshold", "reference_loss",
                                     "target_radius_m", "equivalent_loss_pct", "note"])
    save(acoust, "acoustic_radius")
    save(ranges, "communication_radius")
    save(probs, "response_probability")
    save(sensitivity, "behavior_sensitivity")
    save(curves, "selected_curves")
    save(contrast, "call_type_contrasts")
    save(floor_frame, "recognition_floor_sensitivity")
    save(eta_frame, "eta_band")
    save(recognition, "recognition_probability")
    save(status, "scenario_status")
    save(iso, "iso_effect")
    # Reference rows match P2 and P4 part 1 before imposing any deterioration.
    old = pd.read_csv(base.ROOT / "outputs_p2/P2_effective_range.csv")
    matching = acoust[acoust.fidelity_loss == 0].merge(
        old, left_on=["call", "threshold"], right_on=["CallType", "tau"])
    radius_error = np.max(abs(matching.radius_m - matching.Range_m))
    r0 = ranges[(ranges.threshold == .5) & (ranges.fidelity_loss == 0)].merge(
        reference_df, left_on=["call", "context"],
        right_on=["call_type", "context"])
    err = np.max(abs(r0.radius_m - r0.radius))
    if len(matching) != 9 or len(r0) != 12 or max(radius_error, err) > .025:
        raise ValueError("Earlier radius results could not be reproduced")
    save(pd.DataFrame([{"bootstrap": nboot, "seed": SEED, "group_size": size,
                        "behavior_sensitivity": MAIN_ETA, "q_floor": 0.,
                        "maximum_P2_error_m": radius_error,
                        "maximum_P4_part1_error_m": err,
                        "n_scenarios": len(SCENARIOS),
                        "n_below_support": int((ranges.status == "below_lower_limit").sum()),
                        "n_rows_beyond_mean": int(ranges.beyond_mean_radius.sum())}]),
         "settings_and_checks")
    return acoust, ranges, probs, contrast, floor_frame, eta_frame, recognition, iso, status


def _label(ax, value, maximum, percent):
    text = f"{value:.1f}"
    if not percent:
        text = f"{value:.0f}"
    ax.text(0, 0, text, ha="center", va="center")


def plots(acoustic, radii, probs, recognition, eta_frame):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "grid.alpha": .32,
                         "grid.linewidth": .6, "font.size": 10})

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(FIG / (name + ".png"), dpi=185, bbox_inches="tight")
        plt.close(fig)

    for frame, col, name, title in [
            (acoustic, "radius_m", "01_acoustic_radius",
             "Mean-fidelity acoustic radius (meters)"),
            (radii[radii.context == "standardized"], "radius_m", "02_effective_radius",
             "Communication radius at 15 index points (meters)"),
            (probs[(probs.context == "standardized") & (probs.distance == 100)],
             "response_probability", "03_response_probability",
             "Group response probability at 100 meters (%)")]:
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), sharey=True)
        for ax, t in zip(axes, base.TYPES):
            part = frame[frame.call == t]
            val = part.pivot(index="threshold", columns="fidelity_loss", values=col) \
                .loc[list(TAUS), list(LOSSES)].to_numpy()
            percent = col == "response_probability"
            if percent:
                val = val * 100
            maximum = 100 if percent else 190
            ax.imshow(val, aspect="auto", vmin=0, vmax=maximum, cmap="YlGnBu")
            for y, x in np.ndindex(val.shape):
                label = f"{val[y, x]:.1f}"
                if not percent:
                    state = part[(part.threshold == TAUS[y])
                                 & (part.fidelity_loss == LOSSES[x])].iloc[0].status
                    if state in ("below_4m", "below_lower_limit"):
                        label = "<" + f"{val[y, x]:.0f}"
                    if state in ("above_256m", "at_least_upper_limit"):
                        label = "≥" + f"{val[y, x]:.0f}"
                ax.text(x, y, label, ha="center", va="center",
                        color="white" if val[y, x] / maximum > .53 else "#142e36")
            ax.set(title=base.LABELS[t], xticks=range(5),
                   xticklabels=[f"{int(l * 100)}%" for l in LOSSES],
                   yticks=range(3), yticklabels=["0.50", "0.60", "0.70"],
                   xlabel="Relative fidelity loss")
            ax.grid(False)
        axes[0].set_ylabel("Recognition threshold")
        fig.suptitle(title, fontsize=13)
        finish(fig, name)

    # Trial-level recognition curves: q(d) remains positive beyond the mean radius.
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), sharey=True)
    show = [(.5, 0.), (.7, 0.), (.5, .2), (.7, .2), (.7, .05)]
    colors = {(.5, 0.): "#2c7fb8", (.7, 0.): "#d95f02",
              (.5, .2): "#41ab5d", (.7, .2): "#e7298a", (.7, .05): "#756bb1"}
    for ax, t in zip(axes, base.TYPES):
        for key in show:
            part = recognition[(recognition.call == t)
                               & (recognition.threshold == key[0])
                               & (recognition.fidelity_loss == key[1])]
            ax.plot(part.distance, part.pass_fraction, color=colors[key], lw=1.8,
                    label=f"τ={key[0]:.2f}, δ={int(key[1] * 100)}%")
        mean_r = float(acoustic[(acoustic.threshold == .7) & (acoustic.fidelity_loss == .05)
                                & (acoustic.call == t)].iloc[0].radius_m)
        comm_r = float(radii[(radii.context == "standardized") & (radii.threshold == .7)
                             & (radii.fidelity_loss == .05) & (radii.call == t)].iloc[0].radius_m)
        ax.axvline(mean_r, color="black", lw=1, ls="--")
        ax.axvline(comm_r, color="black", lw=1, ls=":")
        ax.annotate("mean", xy=(mean_r, .93), xytext=(2, 0), textcoords="offset points",
                    fontsize=7, rotation=90, va="top")
        ax.annotate("comm", xy=(comm_r, .83), xytext=(2, 0), textcoords="offset points",
                    fontsize=7, rotation=90, va="top")
        ax.axhline(.5, color="#555555", lw=.8, ls="--")
        ax.axhline(.25, color="#999999", lw=.8, ls=":")
        ax.set(title=base.LABELS[t], xlabel="Distance (m)", ylim=(0, 1.02))
        ax.grid(True)
    axes[0].set_ylabel("Recognition probability q(d)")
    axes[0].legend(fontsize=7.2, loc="upper right")
    fig.suptitle("Trial-level recognition probability (vertical lines: τ=0.70, δ=5%)",
                 fontsize=13)
    finish(fig, "04_recognition_probability")

    # Eta band: the behavioural share is not identified, so show 0/0.5/1.
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.2), sharex=True, sharey=True)
    for row, tau in enumerate([.5, .7]):
        for col, t in enumerate(base.TYPES):
            ax = axes[row, col]
            part = eta_frame[(eta_frame.threshold == tau) & (eta_frame.call == t)
                             & (eta_frame.context == "standardized")]
            x = part.fidelity_loss.to_numpy() * 100
            ax.fill_between(x, part.radius_min, part.radius_max, color="#9ecae1", alpha=.55,
                            label="η ∈ [0,1]")
            ax.plot(x, part.radius_eta05, color="#08519c", lw=2, marker="o", ms=3,
                    label="η=0.5")
            ax.plot(x, part.radius_eta0, color="#636363", lw=1, ls="--", label="η=0")
            ax.plot(x, part.radius_eta1, color="#636363", lw=1, ls=":", label="η=1")
            ax.set(title=f"{base.LABELS[t]} (τ={tau:.2f})", xlabel="Fidelity loss (%)")
            ax.grid(True)
            if col == 0:
                ax.set_ylabel("Communication radius (m)")
            if row == 0 and col == 2:
                ax.legend(fontsize=7.5)
    fig.suptitle("Communication radius under an unidentified behavioural share η",
                 fontsize=13)
    finish(fig, "05_eta_band")


def write_docx(tables):
    """Build the Chinese report directly from the generated tables."""
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    acoust, ranges, probs, contrast, floors, eta_frame, recognition, iso, status = tables
    settings = pd.read_csv(OUT / "settings_and_checks.csv").iloc[0]
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(.8)
    sec.top_margin, sec.bottom_margin = Inches(.72), Inches(.71)
    for name, size in [("Normal", 10.5), ("Title", 18), ("Heading 1", 14), ("Caption", 9)]:
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
    doc.styles["Heading 1"].font.bold = True
    doc.styles["Title"].font.bold = True
    doc.styles["Caption"].font.bold = False
    title_pPr = doc.styles["Title"].element.get_or_add_pPr()
    for border in title_pPr.findall(qn("w:pBdr")):
        title_pPr.remove(border)

    def body(text, indent=True):
        par = doc.add_paragraph(text)
        par.paragraph_format.line_spacing = 1.5
        par.paragraph_format.space_after = Pt(4)
        if indent:
            par.paragraph_format.first_line_indent = Pt(21)
        return par

    def caption(text, keep=False):
        par = doc.add_paragraph(text, style="Caption")
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(10)
        par.paragraph_format.space_after = Pt(4)
        par.paragraph_format.keep_with_next = keep
        return par

    def picture(name, width=6.4):
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.add_run().add_picture(str(FIG / name), width=Inches(width))

    def table(header, rows, widths=None, font=8.5):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, text in enumerate(header):
            cell = t.rows[0].cells[j]
            cell.text = str(text)
            for par in cell.paragraphs:
                for run in par.runs:
                    run.bold = True
                    run.font.size = Pt(font)
                    run.font.name = "Times New Roman"
                    run._element.get_or_add_rPr().get_or_add_rFonts() \
                        .set(qn("w:eastAsia"), "宋体")
        for row in rows:
            cells = t.add_row().cells
            for j, text in enumerate(row):
                cells[j].text = str(text)
                for par in cells[j].paragraphs:
                    for run in par.runs:
                        run.font.size = Pt(font)
                        run.font.name = "Times New Roman"
                        run._element.get_or_add_rPr().get_or_add_rFonts() \
                            .set(qn("w:eastAsia"), "宋体")
        for ri, row in enumerate(t.rows):
            trPr = row._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
            if ri < len(t.rows) - 1:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        par.paragraph_format.keep_with_next = True
        if widths:
            for j, width in enumerate(widths):
                for row in t.rows:
                    row.cells[j].width = Inches(width)
        return t

    def get(frame, **query):
        return frame.query(" and ".join(f"{k}=={v!r}" for k, v in query.items())).iloc[0]

    base_row = get(ranges, threshold=.5, fidelity_loss=0., context="standardized")
    loss20 = {t: get(ranges, threshold=.5, fidelity_loss=.2, call=t, context="standardized")
              for t in base.TYPES}
    loss20_eta = {t: get(eta_frame, threshold=.5, fidelity_loss=.2, call=t,
                         context="standardized") for t in base.TYPES}
    severe = {t: get(ranges, threshold=.7, fidelity_loss=.2, call=t, context="standardized")
              for t in base.TYPES}
    q20 = {t: get(probs, threshold=.5, fidelity_loss=.2, call=t, context="standardized",
                  distance=100) for t in base.TYPES}
    q70 = {t: get(probs, threshold=.7, fidelity_loss=0., call=t, context="standardized",
                  distance=100) for t in base.TYPES}
    basep = {t: get(probs, threshold=.5, fidelity_loss=0., call=t, context="standardized",
                    distance=100) for t in base.TYPES}
    eta1 = {t: get(eta_frame, threshold=.7, fidelity_loss=.2, call=t, context="standardized")
            for t in base.TYPES}
    iso_d = iso.set_index("call").equivalent_loss_pct
    status20 = status[(status.threshold == .5) & (status.fidelity_loss == .2)].iloc[0]
    status70 = status[(status.threshold == .7) & (status.fidelity_loss == .2)].iloc[0]

    doc.add_paragraph("海上环境恶化对企鹅声学通信的影响", style="Title").alignment = \
        WD_ALIGN_PARAGRAPH.CENTER
    body(
        f"在8只企鹅的典型群体中，若各距离的信号相关性相对降低20%、识别阈值保持0.50，"
        f"平直型、调制型和双声型叫声的15分通信半径由{base_row.radius_m:.1f}、"
        f"{get(ranges, threshold=.5, fidelity_loss=0., call='mod', context='standardized').radius_m:.1f}、"
        f"{get(ranges, threshold=.5, fidelity_loss=0., call='tv', context='standardized').radius_m:.1f}米"
        f"分别变为{loss20['flat'].radius_m:.1f}、{loss20['mod'].radius_m:.1f}、"
        f"{loss20['tv'].radius_m:.1f}米。"
        f"由于行为对识别损失的敏感份额η无法由现有数据识别，η在[0,1]内变化时三类半径的区间分别为"
        f"{loss20_eta['flat'].radius_min:.1f}–{loss20_eta['flat'].radius_max:.1f}、"
        f"{loss20_eta['mod'].radius_min:.1f}–{loss20_eta['mod'].radius_max:.1f}、"
        f"{loss20_eta['tv'].radius_min:.1f}–{loss20_eta['tv'].radius_max:.1f}米；"
        f"正文表内的单点取中性设定η=0.5。以下结果为指定压力与行为假设下的预测。")

    doc.add_heading("1　识别阈值、信号衰减与识别概率", level=1)
    body(
        "构造3×5个情景。识别相关性阈值τ取0.50、0.60、0.70，各距离的相对相关性损失δ取0%、5%、"
        "10%、15%、20%。海浪或背景噪声对接收信号的影响用δ表征，识别所需相似度提高用τ表征。"
        "数据未提供这两个量与噪声分贝、风速或海况等级之间的转换关系。")
    body(
        "沿用第二问的三指标融合与单调平均保真度曲线。平均声学达标半径定义为平均保真度仍达到τ的"
        "最远距离。该半径是平均口径，不是硬检测边界：平均曲线会被低尾轨迹拖低，因此其外仍可能有"
        "部分轨迹达标。为刻画这一点，本文同时计算逐轨迹识别概率"
        "q(d)=P(F_i(d)(1−δ)≥τ)，并给出q(d)=0.50的多数轨迹半径。")
    picture("01_acoustic_radius.png")
    caption("图 1　15个情景下的平均保真度声学达标半径（米）。沿列向右增加衰减，沿行向下提高阈值。")
    body(
        f"阈值单独由0.50升至0.70时，平直、调制、双声型的平均声学半径依次由"
        f"{get(acoust, threshold=.5, fidelity_loss=0., call='flat').radius_m:.1f}降至"
        f"{get(acoust, threshold=.7, fidelity_loss=0., call='flat').radius_m:.1f}米、"
        f"{get(acoust, threshold=.5, fidelity_loss=0., call='mod').radius_m:.1f}降至"
        f"{get(acoust, threshold=.7, fidelity_loss=0., call='mod').radius_m:.1f}米、"
        f"{get(acoust, threshold=.5, fidelity_loss=0., call='tv').radius_m:.1f}降至"
        f"{get(acoust, threshold=.7, fidelity_loss=0., call='tv').radius_m:.1f}米。"
        f"双声型的平均曲线在中距离较平，阈值越过该段后半径随很小的附加损失快速收缩；"
        f"τ=0.70、δ=5%时其平均半径仅"
        f"{get(acoust, threshold=.7, fidelity_loss=.05, call='tv').radius_m:.1f}米，"
        f"但100米处仍有"
        f"{get(recognition, threshold=.7, fidelity_loss=.05, call='tv', distance=100.).pass_fraction*100:.1f}%"
        "的轨迹达标。这一差异说明平均半径与逐轨迹识别概率必须分开报告。")
    picture("04_recognition_probability.png")
    caption("图 2　逐轨迹识别概率q(d)。水平虚线为0.50，点线为0.25；竖直虚线为τ=0.70、δ=5%的"
            "平均声学半径，竖直点线为同一情景的通信半径。")

    doc.add_heading("2　从声学达标变化到有效通信半径", level=1)
    body(
        "在7个实测距离均有可用综合保真度的完整轨迹中，计算超过各个有效阈值的比例。平直、调制、"
        "双声型的共同轨迹数依次为"
        f"{int(get(acoust, threshold=.5, fidelity_loss=0., call='flat').common_trials)}、"
        f"{int(get(acoust, threshold=.5, fidelity_loss=0., call='mod').common_trials)}和"
        f"{int(get(acoust, threshold=.5, fidelity_loss=0., call='tv').common_trials)}。"
        "对各阈值的比例沿距离做单调回归，再在相邻距离间线性插值。")
    body(
        "用g表示新情景达标比例与同类型、同距离基准达标比例之比。令η为自然叫声相对于对照所多出的"
        "行为反应中，对识别损失敏感的份额。保留系数为w=1−η(1−g)。主设定η=0.5，另检验0和1。"
        "每个情境下，对第三问的自然叫声和对照八状态概率按w与1−w加权，假设对照背景行为保持基准值。"
        "通信半径是期望收益口径：保留系数按q(d)的比值对轨迹取期望，指数中的保真度按"
        "F(d)(1−δ)计入，因此可以大于平均声学半径；每个报告半径处的q值均在输出表中给出，"
        "以便区分“平均达标”与“仍有部分轨迹识别”。")
    picture("02_effective_radius.png")
    caption("图 3　主行为假设下的15分综合通信半径（米）。小于号表示在最近支持距离已经未达标，"
            "是支持边界而非精确交点。")
    body(
        f"只提高阈值至0.70时，三类通信半径的点估计依次为"
        f"{get(ranges, threshold=.7, fidelity_loss=0., call='flat', context='standardized').radius_m:.1f}、"
        f"{get(ranges, threshold=.7, fidelity_loss=0., call='mod', context='standardized').radius_m:.1f}和"
        f"{get(ranges, threshold=.7, fidelity_loss=0., call='tv', context='standardized').radius_m:.1f}米；"
        f"其中调制型在点估计上超过双声型，但该差异的配对区间见表2，点估计排序不作为结论。"
        f"若阈值维持0.50而相关性降低20%，半径分别损失"
        f"{-loss20['flat'].change_m_clipped:.1f}、{-loss20['mod'].change_m_clipped:.1f}和"
        f"{-loss20['tv'].change_m_clipped:.1f}米。"
        f"以τ=0.70、δ=5%的双声型为例，通信半径为"
        f"{get(ranges, threshold=.7, fidelity_loss=.05, call='tv', context='standardized').radius_m:.1f}米，"
        f"该处仍有"
        f"{get(ranges, threshold=.7, fidelity_loss=.05, call='tv', context='standardized').pass_fraction_at_radius * 100:.1f}%"
        "的轨迹可识别，因此该半径不是对平均曲线的外推，而是期望收益在部分识别下的延续。"
        f"它略大于双声型的多数轨迹半径"
        f"{get(acoust, threshold=.7, fidelity_loss=.05, call='tv').majority_radius_m:.1f}米"
        "（q=0.50），说明该结果对应约半数轨迹仍识别的距离。")
    body(
        "按半径计算的损失会低估生态影响，因为通信覆盖是面积。以基准半径为R0，面积损失为"
        "1−(R/R0)²。相关性降低20%、阈值0.50时，平直、调制、双声型的面积损失分别为"
        f"{loss20['flat'].area_loss_pct:.0f}%、{loss20['mod'].area_loss_pct:.0f}%和"
        f"{loss20['tv'].area_loss_pct:.0f}%；按米数排序时双声型绝对损失最大，按面积排序时平直型"
        "最脆弱，两个口径必须同时报告。")
    caption("表 1　相对基准的通信面积损失（%）。破折号表示半径已触底，面积不可识别。", keep=True)
    table(["阈值", "相关性损失", "平直型面积损失", "调制型面积损失", "双声型面积损失"],
          [[".50", "20%",
            f"{loss20['flat'].area_loss_pct:.0f}%", f"{loss20['mod'].area_loss_pct:.0f}%",
            f"{loss20['tv'].area_loss_pct:.0f}%"],
           [".70", "0%",
            f"{get(ranges, threshold=.7, fidelity_loss=0., call='flat', context='standardized').area_loss_pct:.0f}%",
            f"{get(ranges, threshold=.7, fidelity_loss=0., call='mod', context='standardized').area_loss_pct:.0f}%",
            f"{get(ranges, threshold=.7, fidelity_loss=0., call='tv', context='standardized').area_loss_pct:.0f}%"],
           [".70", "20%",
            f"{severe['flat'].area_loss_pct:.0f}%" if np.isfinite(severe['flat'].area_loss_pct) else "—",
            f"{severe['mod'].area_loss_pct:.0f}%" if np.isfinite(severe['mod'].area_loss_pct) else "—",
            f"{severe['tv'].area_loss_pct:.0f}%" if np.isfinite(severe['tv'].area_loss_pct) else "—"]],
          widths=[.7, 1.0, 1.5, 1.5, 1.5])

    def ci(value, lo, hi):
        if not (np.isfinite(value) and np.isfinite(lo) and np.isfinite(hi)):
            return "—"
        return f"{value:+.1f} [{lo:+.1f}, {hi:+.1f}]"

    contrast_rows = []
    cn = {"flat": "平直型", "mod": "调制型", "tv": "双声型"}
    for key in [(.5, .2), (.7, 0.)]:
        for hi, lo in [("mod", "flat"), ("tv", "flat"), ("tv", "mod")]:
            row = get(contrast, threshold=key[0], fidelity_loss=key[1],
                      context="standardized", call=hi, reference=lo)
            contrast_rows.append([f"{key[0]:.2f}", f"{int(key[1] * 100)}%",
                                  f"{cn[hi]}−{cn[lo]}",
                                  ci(row.difference_m, row.low95, row.high95),
                                  ci(row.area_difference_pp, row.area_low95, row.area_high95)])
    caption("表 2　叫声类型之间的配对差值（标准化情境）。差值为前者减后者；"
            "半径差为负表示前者损失更多，面积差为正表示前者损失更多。", keep=True)
    table(["阈值", "相关性损失", "对比", "半径差（米）[95%CI]",
           "面积差（百分点）[95%CI]"], contrast_rows,
          widths=[.55, .8, 1.35, 1.9, 1.9], font=8)
    r = get(contrast, threshold=.7, fidelity_loss=0., context="standardized",
            call="tv", reference="mod")
    sig_radius = (r.low95 > 0) or (r.high95 < 0)
    sig_area = (r.area_low95 > 0) or (r.area_high95 < 0)
    body(
        f"τ=0.70、δ=0时，双声型相对调制型的半径损失差为{r.difference_m:+.1f}米"
        f"（95%区间[{r.low95:+.1f}, {r.high95:+.1f}]），区间{'不含' if sig_radius else '含'}0；"
        f"面积损失差为{r.area_difference_pp:+.1f}个百分点"
        f"（[{r.area_low95:+.1f}, {r.area_high95:+.1f}]），区间{'不含' if sig_area else '含'}0。"
        + ("因此“调制型超过双声型”只在半径点估计上成立；按面积计，双声型的损失更大且区间不含0。"
           if sig_area and r.area_difference_pp > 0 else
           "因此“调制型超过双声型”只在半径点估计上成立；按面积计的差异未达显著。"))
    iso_flat, iso_mod = float(iso_d["flat"]), float(iso_d["mod"])
    iso_tv = "超过20%" if not np.isfinite(iso_d["tv"]) else f"约{float(iso_d['tv']):.1f}%"
    iso_low, iso_high = min(iso_flat, iso_mod), max(iso_flat, iso_mod)
    body(
        "提高τ与增加δ可以相互换算。以τ=0.70、δ=0的通信半径为等效应目标，在τ=0.50的曲线"
        f"上插值得到所需的相关性损失：平直型约{iso_flat:.1f}%，调制型约{iso_mod:.1f}%，"
        f"双声型{iso_tv}。因此在本数据中，把识别阈值从0.50提到0.70，大致相当于损失"
        f"{iso_low:.0f}%–{iso_high:.0f}%的信号相关性；双声型超过20%，阈值压力不低于中等噪声情景。")

    doc.add_heading("3　响应概率的变化与解释", level=1)
    body(
        "概率表示一次回放中，8只企鹅组成的群体至少出现一次抬头、发声或靠近。以下固定距离为100米，"
        "并统一三类情境构成。表内“下降”为相对基准的百分点变化，按未舍入的原始概率计算。"
        "概率与半径使用同一套八状态加权，因此当100米超出通信半径时，概率会回落到由w决定的"
        "行为超额水平。")
    picture("03_response_probability.png")
    caption("图 4　100米处的全部15个情景。每格数值为经情境标准化的群体响应概率（%）。")
    rows = []
    for key in [(.5, 0.), (.6, 0.), (.7, 0.), (.5, .1), (.5, .2), (.7, .2)]:
        cells = []
        for t in base.TYPES:
            row = get(probs, threshold=key[0], fidelity_loss=key[1], call=t,
                      context="standardized", distance=100)
            text = f"{row.response_probability * 100:.1f}%"
            if key != (.5, 0.):
                text += f"（↓{-row.change_percentage_points:.1f}）"
            cells.append(text)
        rows.append([f"{key[0]:.2f}", f"{int(key[1] * 100)}%", *cells])
    caption("表 3　100米处的群体响应概率（主设定η=0.5；括号内为相对基准的下降百分点）。", keep=True)
    table(["阈值", "相关性损失", "平直型", "调制型", "双声型"], rows,
          widths=[.8, 1.1, 1.4, 1.4, 1.4])
    body(
        f"阈值单独提高至0.70时，平直型的达标比例保留率约"
        f"{q70['flat'].relative_pass_retention * 100:.1f}%，调制型约"
        f"{q70['mod'].relative_pass_retention * 100:.1f}%，双声型约"
        f"{q70['tv'].relative_pass_retention * 100:.1f}%。平直型预测响应概率下降约"
        f"{-q70['flat'].change_percentage_points:.1f}个百分点。最严情景下，三类声学轨迹在100米处的"
        "估计达标比例均为零；主假设η=0.5仍保留一半行为超额，故可见概率为"
        f"{get(probs, threshold=.7, fidelity_loss=.2, call='flat', context='standardized', distance=100).response_probability * 100:.1f}%、"
        f"{get(probs, threshold=.7, fidelity_loss=.2, call='mod', context='standardized', distance=100).response_probability * 100:.1f}%及"
        f"{get(probs, threshold=.7, fidelity_loss=.2, call='tv', context='standardized', distance=100).response_probability * 100:.1f}%。"
        f"若η=1（行为完全依赖识别），三类均回落到对照水平约"
        f"{eta1['flat'].p100_eta1 * 100:.1f}%。")

    doc.add_heading("4　不确定性与稳健性", level=1)
    body(
        "传播实验按整条试验轨迹抽样，回放实验按139个实验编号整对抽样；每次重估保真度权重、"
        "达标比例及行为模型。同一批抽样同时计算基准和所有压力情景，使差值保留两端的相关性。"
        "阈值、损失和η视为给定。基准情景（τ=0.50、δ=0）与第二问、第四问第一部分的半径最大"
        f"复现误差分别为{settings.maximum_P2_error_m:.1e}米和{settings.maximum_P4_part1_error_m:.1e}米，"
        "说明情景层没有改动前序结果。")
    body(
        "η是本文最关键的未识别参数。图5给出η∈[0,1]的半径带：在δ=20%、τ=0.50时，平直型为"
        f"{loss20_eta['flat'].radius_min:.1f}–{loss20_eta['flat'].radius_max:.1f}米，调制型为"
        f"{loss20_eta['mod'].radius_min:.1f}–{loss20_eta['mod'].radius_max:.1f}米，双声型为"
        f"{loss20_eta['tv'].radius_min:.1f}–{loss20_eta['tv'].radius_max:.1f}米。"
        "前序回归中，保真度每提高0.1的响应优势比为1.18，95%区间0.79–1.76（p=0.415）；"
        "现有数据无法标定η，因此主文以区间而非单点作为结论。")
    picture("05_eta_band.png")
    caption("图 5　η∈[0,1]下的通信半径带。实线为η=0.5，虚线与点线为两个边界。")
    q_std = ranges[(ranges.context == "standardized")
                   & (ranges.status == "crossing")].pass_fraction_at_radius
    body(
        "若要求通信半径处仍有最低比例的轨迹可识别，可施加q(d)≥q_min的门槛。q_min=0为期望收益"
        "口径（主设定）；q_min=0.25和0.50分别要求至少四分之一和多数轨迹达标。门槛越高，"
        "强情景下的半径越短，但基本不改变弱情景的排序。"
        f"例如τ=0.70、δ=5%时，q_min=0.50把调制型由"
        f"{get(floors, q_floor=0., threshold=.7, fidelity_loss=.05, call='mod', context='standardized').radius_m:.1f}"
        f"米缩至{get(floors, q_floor=.5, threshold=.7, fidelity_loss=.05, call='mod', context='standardized').radius_m:.1f}"
        f"米、双声型由{get(floors, q_floor=0., threshold=.7, fidelity_loss=.05, call='tv', context='standardized').radius_m:.1f}"
        f"米缩至{get(floors, q_floor=.5, threshold=.7, fidelity_loss=.05, call='tv', context='standardized').radius_m:.1f}"
        f"米，平直型基本不变。标准化情境下，可识别半径处的q值约为{q_std.min():.2f}–{q_std.max():.2f}；"
        "最低值出现在强情景，q_min=0.50会对这些低q半径产生实质约束。该敏感性说明"
        "“行为扩大通信半径”依赖于仍有部分轨迹识别，而不是任意外推。")
    body(
        f"最后，{int(status20.n_crossing)}/12个（τ=0.50、δ=20%）和"
        f"{int(status70.n_crossing)}/12个（τ=0.70、δ=20%）类型—情境组合的通信半径可在支持范围内"
        "识别，其余在最近支持距离已未达标，只能报告上界。表中带“<”的数值是支持边界而非交点，"
        "不能当作精确半径；相应差值也只应解释为“至少损失”或“无法识别”。")

    doc.add_heading("5　小结", level=1)
    t70 = {t: get(ranges, threshold=.7, fidelity_loss=0., call=t, context="standardized")
           for t in base.TYPES}
    body(
        "本文将噪声与识别阈值统一为3×5个情景。结果有两条渠道：δ直接降低保真度并缩短平均声学"
        "半径；τ降低逐轨迹识别概率并削弱行为收益。"
        f"δ=20%、τ=0.50时，平直型的通信面积损失约{loss20['flat'].area_loss_pct:.0f}%，"
        f"高于调制型的{loss20['mod'].area_loss_pct:.0f}%和双声型的{loss20['tv'].area_loss_pct:.0f}%；"
        f"τ=0.70、δ=0时，三类半径降至{t70['flat'].radius_m:.1f}、{t70['mod'].radius_m:.1f}和"
        f"{t70['tv'].radius_m:.1f}米，面积损失为{t70['flat'].area_loss_pct:.0f}%、"
        f"{t70['mod'].area_loss_pct:.0f}%和{t70['tv'].area_loss_pct:.0f}%。"
        "η无法由数据识别，因此上述单点应视为中性情景；η∈[0,1]的"
        "区间和q_min门槛给出了结论的适用范围。对管理而言，降低噪声对平直型近源通信的边际收益"
        "最大，依赖单一叫声类型的保护策略不足以覆盖群体通信。")

    doc.save(OUT / "第四问第二部分_噪声与不利海况情景分析.docx")


def run(nboot):
    FIG.mkdir(parents=True, exist_ok=True)
    cubes, df = base.input_data()
    size, gx = base.probability_grid(df)
    bx = base.behavior.design(df)
    states = df.state.to_numpy()
    domains = base.get_domains(df)
    ctx = df.Context.value_counts(normalize=True).loc[base.CONTEXTS].to_numpy()

    def behavior_prob(idx):
        m = base.behavior.fit(bx[idx], states[idx], base.behavior.MODELS[-1])
        return base.probabilities(m, gx)

    pr = behavior_prob(np.arange(len(df)))
    ac = acoustics(cubes)
    alternative = {eta: calculate(ac[0], ac[2], pr, ctx, domains, eta, ac[1], 0.)
                   for eta in ETA_GRID}
    floors = {qf: calculate(ac[0], ac[2], pr, ctx, domains, MAIN_ETA, ac[1], qf)
              for qf in Q_FLOORS}
    point = alternative[MAIN_ETA]
    rng = np.random.default_rng(SEED)
    pairs = np.stack(list(df.groupby("expID").indices.values()))
    draw_a = []
    draw_r = []
    draw_p = []
    for _ in range(nboot):
        sampled = [c[rng.integers(len(c), size=len(c))] for c in cubes]
        f, gates, retain, rad, _, _, _, _ = acoustics(sampled)
        pb = behavior_prob(pairs[rng.integers(len(pairs), size=len(pairs))].ravel())
        radii, _, response, _ = calculate(f, retain, pb, ctx, domains, MAIN_ETA, gates, 0.)
        draw_a.append(rad)
        draw_r.append(radii)
        draw_p.append(response[..., AT_IDX])
    boot = tuple(np.stack(v) for v in (draw_a, draw_r, draw_p))
    tables = outputs(point, ac, boot, domains, alternative, floors,
                     pd.read_csv(base.OUT / "effective_radii.csv"), nboot, size)
    plots(*tables[:3], tables[6], tables[5])
    write_docx(tables)
    print(f"Completed {len(SCENARIOS)} scenarios and {nboot} paired scenario bootstrap comparisons.")
    print(tables[2].query("context=='standardized' and distance==100 and fidelity_loss==0.2 "
                          "and threshold==0.7").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("At least two bootstrap draws are required.")
    run(args.bootstrap)
