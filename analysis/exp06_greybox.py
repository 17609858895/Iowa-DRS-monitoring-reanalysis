"""E6: grey-box (feed mass-balance) propagation of a stale digestate lab anchor.

CSTR volatile-solids balance (both digesters lumped, V = 3671.8 m3):
  VS_{d+1} = VS_d + (Q_d / V) * ((1 - f) * VSin_d - VS_d)
Q_d: daily TWAS+PS+HSW volume; VSin_d: flow-weighted feed VS (TWAS/PS lab, HSW lab or spectral);
f: fraction of feed VS destroyed, fitted on training anchor pairs only (labels <= cutoff).
Compared at staleness k with persistence of the same anchor (identical anchors, identical days).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import OUT, ROOT, audit_digestate, load_stream, metrics
from exp05_hsw_load_dml import daily_hsw_vs

warnings.filterwarnings("ignore")
V = 3671.849e3  # L
KS = [1, 3, 7, 14, 28]


def feed_table(hsw_col):
    L = pd.read_csv(ROOT / "20_Loading.csv", encoding="utf-8-sig")
    L.columns = [c.strip() for c in L.columns]
    L["day"] = (pd.to_datetime(L["Date"]) - pd.Timestamp("2023-12-20")).dt.days
    vs = daily_hsw_vs()
    L = L.merge(vs, on="day", how="left").set_index("day")
    L = L.reindex(range(L.index.min(), L.index.max() + 1))
    for c in ["TWAS Observed VS mg", "PS Observed VS mg", hsw_col]:
        L[c] = L[c].interpolate(limit_direction="both")
    q = L[["TWAS Fed liters", "PS Fed liters", "HSW Fed liters"]].interpolate(limit_direction="both")
    mass = (q["TWAS Fed liters"] * L["TWAS Observed VS mg"] + q["PS Fed liters"] * L["PS Observed VS mg"]
            + q["HSW Fed liters"] * L[hsw_col])
    Q = q.sum(1)
    return pd.DataFrame({"Q": Q, "VSin": mass / Q})


def propagate(v0, d0, d1, feed, f):
    v = v0
    for d in range(int(d0), int(d1)):
        if d in feed.index and not np.isnan(feed.at[d, "Q"]):
            r = feed.at[d, "Q"] / V
            v = v + r * ((1 - f) * feed.at[d, "VSin"] - v)
    return v


if __name__ == "__main__":
    dig = audit_digestate(load_stream("dig"))
    s = dig[dig["VS"].notna()].groupby("day")["VS"].mean()
    days, y = s.index.values, s.values
    rows = []
    for hsw_col, tag in [("VS_lab", "GB_labHSW"), ("VS_fill", "GB_specHSW")]:
        feed = feed_table(hsw_col)
        for k in KS:
            for i, t in enumerate(days):
                known = days <= t - k
                if known.sum() < 20 or t < feed.index.min() + 1:
                    continue
                a = np.where(known)[0][-1]
                tr_idx = np.where(known)[0]
                pairs = [(p, q) for p in tr_idx for q in tr_idx if 1 <= days[q] - days[p] <= max(k, 3) + 3]
                if len(pairs) < 10:
                    continue

                def loss(f):
                    return np.mean([(propagate(y[p], days[p], days[q], feed, f) - y[q]) ** 2 for p, q in pairs[-60:]])

                f = minimize_scalar(loss, bounds=(0.0, 0.9), method="bounded").x
                rows.append(dict(model=tag, k=k, day=t, y=y[i], persist=y[a],
                                 pred=propagate(y[a], days[a], t, feed, f), f=f))
        R = pd.DataFrame([r for r in rows if r["model"] == tag])
        for k, g in R.groupby("k"):
            print(f"{tag} k={k:2d} n={len(g)} Persist R2={metrics(g.y, g.persist)['R2']:.2f} "
                  f"GreyBox R2={metrics(g.y, g.pred)['R2']:.2f}  MAE {metrics(g.y, g.persist)['MAE']:.0f} -> "
                  f"{metrics(g.y, g.pred)['MAE']:.0f}  f={g.f.median():.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "E6_greybox.csv", index=False)
