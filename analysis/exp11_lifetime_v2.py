"""E11: calibration lifetime with tolerance sensitivity and bootstrap CIs.

Same frozen-calibration design as E4, but every prediction after each weekly cutoff is stored, so the
event definition can vary without refitting:
  exceedance = |error| > tol x training mean (tol in 10/15/20%), event = first of two consecutive.
RMST over 28 d from Kaplan-Meier; CI of RMST differences vs Spec by cluster bootstrap over cutoffs
(cutoffs overlap in their follow-up, so the CI is approximate and stated as such).
"""
import warnings

import numpy as np
import pandas as pd

from common import OUT, WL, SpecPrep, audit_digestate, audit_hsw, load_stream
from exp03_dmlcal import DMLCal, fit_pls
from exp04_lifetime import MIN_TRAIN, N_ANCHOR, rmst

warnings.filterwarnings("ignore")
TOLS = [0.10, 0.15, 0.20]
TAU = 28
N_BOOT = 2000
TARGETS = [("dig", ["TS", "VS", "VA", "Alk", "TAC"]), ("hsw", ["TS", "VS"])]


def frozen_predictions(df, tname, stream):
    s = df[df[tname].notna() & ~df[f"bad_{tname}"]].sort_values("day").reset_index(drop=True)
    X = SpecPrep("sg1").transform(s[WL].values)
    y = s[tname].values.astype(float)
    t = s["day"].values.astype(float)
    out = []
    for c in np.arange(t[MIN_TRAIN - 1], t.max() - 7, 7):
        tr, fu = t <= c, t > c
        if tr.sum() < MIN_TRAIN or fu.sum() < 3:
            continue
        a = np.where(tr)[0][-N_ANCHOR:]
        spec = fit_pls(X[tr], y[tr])
        bias = np.mean(y[a] - spec.predict(X[a]).ravel())
        dml = DMLCal().fit(X[tr], y[tr], t[tr])
        lvl_y = dml.g.predict(np.array([[t[a].mean()]]))[0]
        preds = {"Persist": np.full(fu.sum(), y[a[-1]]),
                 "Spec": spec.predict(X[fu]).ravel(),
                 "SpecBias": spec.predict(X[fu]).ravel() + bias,
                 "AnchorDML": lvl_y + dml.theta.predict(X[fu] - X[a].mean(0)).ravel()}
        for k, p in preds.items():
            out += [dict(stream=stream, target=tname, cutoff=c, ybar=y[tr].mean(), model=k, dt=d - c, y=v, p=q)
                    for d, v, q in zip(t[fu], y[fu], p)]
    return out


def lifetimes(P, tol):
    rows = []
    for (s, tn, c, m), g in P.groupby(["stream", "target", "cutoff", "model"]):
        g = g.sort_values("dt")
        exc = (np.abs(g.y - g.p) > tol * g.ybar.iloc[0]).values
        two = exc[:-1] & exc[1:]
        T, E = (g.dt.values[np.argmax(two)], 1) if two.any() else (g.dt.values[-1], 0)
        rows.append(dict(stream=s, target=tn, cutoff=c, model=m, time=T, event=E, tol=tol))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allp = []
    for stream, targets in TARGETS:
        df = load_stream(stream)
        df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
        for tname in targets:
            allp += frozen_predictions(df, tname, stream)
            print("done", stream, tname, flush=True)
    P = pd.DataFrame(allp)
    P.to_csv(OUT / "E11_frozen_predictions.csv", index=False)
    L = pd.concat([lifetimes(P, tol) for tol in TOLS])
    L.to_csv(OUT / "E11_lifetimes.csv", index=False)
    rng = np.random.default_rng(0)
    summ = []
    for (s, tn, tol), g in L.groupby(["stream", "target", "tol"]):
        cuts = g.cutoff.unique()
        W = g.pivot(index="cutoff", columns="model", values="time"), g.pivot(index="cutoff", columns="model", values="event")
        base = {m: rmst(W[0][m], W[1][m], TAU) for m in W[0].columns}
        boot = {m: [] for m in W[0].columns}
        for _ in range(N_BOOT):
            b = rng.choice(cuts, len(cuts))
            r_spec = rmst(W[0].loc[b, "Spec"], W[1].loc[b, "Spec"], TAU)
            for m in W[0].columns:
                boot[m].append(rmst(W[0].loc[b, m], W[1].loc[b, m], TAU) - r_spec)
        for m in W[0].columns:
            summ.append(dict(stream=s, target=tn, tol=tol, model=m, RMST=base[m], dRMST_vs_Spec=base[m] - base["Spec"],
                             lo=np.percentile(boot[m], 2.5), hi=np.percentile(boot[m], 97.5), n_cut=len(cuts)))
    S = pd.DataFrame(summ).round(2)
    S.to_csv(OUT / "E11_rmst_summary.csv", index=False)
    print(S[S.tol == 0.15].to_string())
