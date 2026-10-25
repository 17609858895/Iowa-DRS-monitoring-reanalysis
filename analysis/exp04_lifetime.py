"""E4: calibration lifetime as a time-to-event outcome.

At each weekly cutoff c, every model is calibrated on labels <= c and then frozen (no further labs).
For later labelled days, an exceedance is |error| > tol (tol = TOL_FRAC x training mean of y).
Event time = days from c to the first of two consecutive exceedances; censored at the last label.

Models (all frozen at c):
  Persist     last lab value before c
  Spec        absolute PLS
  SpecBias    absolute PLS + mean residual of the 5 most recent labels (bias update at c)
  AnchorDML   y(c-level) + theta(x_t - x_c-level): partially linear calibration anchored at c
"""
import warnings

import numpy as np
import pandas as pd

from common import (DIG_TARGETS, HSW_TARGETS, OUT, WL, SpecPrep, audit_digestate, audit_hsw,
                    load_stream)
from exp03_dmlcal import DMLCal, fit_pls

warnings.filterwarnings("ignore")
TOL_FRAC = 0.15
MIN_TRAIN = 20
N_ANCHOR = 5


def km(times, events, grid):
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    s, out, at_risk = 1.0, [], len(t)
    surv = {}
    for u in np.unique(t):
        d = ((t == u) & (e == 1)).sum()
        n = (t >= u).sum()
        s *= 1 - d / n if n else 1
        surv[u] = s
    keys = np.array(sorted(surv))
    return np.array([surv[keys[keys <= g][-1]] if (keys <= g).any() else 1.0 for g in grid])


def rmst(times, events, tau):
    grid = np.arange(0, tau)
    return km(times, events, grid).sum()


def run_target(df, tname, stream):
    m = df[tname].notna() & ~df[f"bad_{tname}"]
    s = df[m].sort_values("day").reset_index(drop=True)
    X = SpecPrep("sg1").transform(s[WL].values)
    y = s[tname].values.astype(float)
    t = s["day"].values.astype(float)
    rows = []
    cutoffs = np.arange(t[MIN_TRAIN - 1], t.max() - 7, 7)
    for c in cutoffs:
        tr, fu = t <= c, t > c
        if tr.sum() < MIN_TRAIN or fu.sum() < 3:
            continue
        tol = TOL_FRAC * y[tr].mean()
        a = np.where(tr)[0][-N_ANCHOR:]
        spec = fit_pls(X[tr], y[tr])
        bias = np.mean(y[a] - spec.predict(X[a]).ravel())
        dml = DMLCal().fit(X[tr], y[tr], t[tr])
        lvl_y = dml.g.predict(np.array([[t[a].mean()]]))[0]
        lvl_x = X[a].mean(0)
        preds = {
            "Persist": np.full(fu.sum(), y[a[-1]]),
            "Spec": spec.predict(X[fu]).ravel(),
            "SpecBias": spec.predict(X[fu]).ravel() + bias,
            "AnchorDML": lvl_y + dml.theta.predict(X[fu] - lvl_x).ravel(),
        }
        tf, yf = t[fu] - c, y[fu]
        for k, p in preds.items():
            exc = np.abs(yf - p) > tol
            two = exc[:-1] & exc[1:]
            if two.any():
                T, E = tf[np.argmax(two)], 1
            else:
                T, E = tf[-1], 0
            rows.append(dict(stream=stream, target=tname, cutoff=c, model=k, time=T, event=E,
                             mae_14d=np.mean(np.abs(yf - p)[tf <= 14]) / y[tr].mean() * 100))
    return rows


if __name__ == "__main__":
    allrows = []
    for stream, targets in [("dig", ["TS", "VS", "VA", "Alk", "TAC"]), ("hsw", ["TS", "VS"])]:
        df = load_stream(stream)
        df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
        for tname in targets:
            r = run_target(df, tname, stream)
            allrows += r
            R = pd.DataFrame(r)
            line = " ".join(f"{k}: RMST28={rmst(g.time, g.event, 28):.1f}d ev={g.event.sum()}/{len(g)} MAE14={g.mae_14d.median():.1f}%"
                            for k, g in R.groupby("model"))
            print(f"{stream} {tname:4s} {line}", flush=True)
    pd.DataFrame(allrows).to_csv(OUT / "E4_lifetime.csv", index=False)
