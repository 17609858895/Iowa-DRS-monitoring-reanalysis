"""E2: online soft-sensor protocol with stale lab anchors.

For a labelled day t and staleness k, only labels with day <= t-k are known (training + anchors).
Models are retrained on a weekly cutoff grid (labels <= cutoff), anchors use every label <= t-k.

Methods (same spectra, same labels, same cutoff):
  Persist   y of most recent anchor
  Abs       absolute PLS f(x_t)
  AbsBias   f(x_t) + mean_a [y_a - f(x_a)]           (classical slope/bias update, nearest competitor)
  Delta     mean_a [y_a + g(x_t - x_a)]              (g trained on within-window pairs: dx -> dy)
  Fuse      inverse-variance fusion of Delta and Persist with age-dependent variance
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (DIG_TARGETS, HSW_TARGETS, OUT, WL, SpecPrep, audit_digestate, audit_hsw,
                    load_stream, metrics)

warnings.filterwarnings("ignore")
PREP = "sg1"
H_PAIR = 21          # max lag (days) of training pairs for g
N_ANCHOR = 5         # most recent anchors used at prediction
KS = [1, 3, 7, 14, 28]
MIN_TRAIN = 20


def fit_abs(X, y):
    gs = GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)),
                      {"plsregression__n_components": list(range(1, min(10, len(y) // 3) + 1))},
                      cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error")
    gs.fit(X, y)
    return gs.best_estimator_, -gs.best_score_


def make_pairs(X, y, day, h):
    i, j = np.where((np.abs(day[:, None] - day[None, :]) <= h) & (day[:, None] != day[None, :]))
    return X[i] - X[j], y[i] - y[j], (day[i] // 14)


def fit_delta(X, y, day):
    dX, dy, g = make_pairs(X, y, day, H_PAIR)
    nlv = list(range(1, 11))
    cv = GroupKFold(min(5, len(np.unique(g))))
    gs = GridSearchCV(PLSRegression(scale=True), {"n_components": nlv}, cv=cv, scoring="neg_mean_squared_error")
    gs.fit(dX, dy, groups=g)
    return gs.best_estimator_, -gs.best_score_


def lag_variance(y, day, max_k=60):
    """Empirical E[(y_t - y_{t-k})^2] by lag (structure function), smoothed & monotone."""
    lag = np.abs(day[:, None] - day[None, :])
    d2 = (y[:, None] - y[None, :]) ** 2
    out = np.full(max_k + 1, np.nan)
    for k in range(1, max_k + 1):
        m = (lag >= max(1, k - 1)) & (lag <= k + 1)
        if m.sum() >= 5:
            out[k] = d2[m].mean()
    s = pd.Series(out).interpolate(limit_direction="both").values
    return np.maximum.accumulate(np.nan_to_num(s, nan=np.nanmean(s)))


def run_target(df, t, stream):
    X = SpecPrep(PREP).transform(df[WL].values)
    y = df[t].values.astype(float)
    day = df["day"].values
    lab = ~np.isnan(y) & ~df[f"bad_{t}"].values
    Xl, yl, dl = X[lab], y[lab], day[lab]
    order = np.argsort(dl)
    Xl, yl, dl = Xl[order], yl[order], dl[order]
    cache = {}
    rows = []
    for k in KS:
        for idx in range(len(yl)):
            cut_anchor = dl[idx] - k
            known = dl <= cut_anchor
            if known.sum() < MIN_TRAIN:
                continue
            cut_model = (cut_anchor // 7) * 7
            tr = dl <= cut_model
            if tr.sum() < MIN_TRAIN:
                tr = known
                cut_model = cut_anchor
            if cut_model not in cache:
                fa, mse_a = fit_abs(Xl[tr], yl[tr])
                fd, mse_d = fit_delta(Xl[tr], yl[tr], dl[tr])
                v = lag_variance(yl[tr], dl[tr])
                cache[cut_model] = (fa, mse_a, fd, mse_d, v)
            fa, mse_a, fd, mse_d, v = cache[cut_model]
            a_idx = np.where(known)[0][-N_ANCHOR:]
            xt = Xl[idx:idx + 1]
            age = dl[idx] - dl[a_idx[-1]]
            p_persist = yl[a_idx[-1]]
            p_abs = float(fa.predict(xt).ravel()[0])
            p_bias = p_abs + float(np.mean(yl[a_idx] - fa.predict(Xl[a_idx]).ravel()))
            p_delta = float(np.mean(yl[a_idx] + fd.predict(xt - Xl[a_idx]).ravel()))
            vp = v[min(age, len(v) - 1)]
            vd = mse_d  # pair-level CV error of g (single-anchor delta prediction)
            p_fuse = (p_delta / vd + p_persist / vp) / (1 / vd + 1 / vp)
            rows.append(dict(stream=stream, target=t, k=k, day=dl[idx], age=age, y=yl[idx],
                             Persist=p_persist, Abs=p_abs, AbsBias=p_bias, Delta=p_delta, Fuse=p_fuse))
    return rows


if __name__ == "__main__":
    allrows = []
    for stream in ["dig", "hsw"]:
        df = load_stream(stream)
        df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
        for t in (DIG_TARGETS if stream == "dig" else HSW_TARGETS):
            r = run_target(df, t, stream)
            allrows += r
            R = pd.DataFrame(r)
            for k, g in R.groupby("k"):
                s = " ".join(f"{m}={metrics(g.y, g[m])['R2']:.2f}" for m in ["Persist", "Abs", "AbsBias", "Delta", "Fuse"])
                print(f"{stream} {t:7s} k={k:2d} n={len(g):3d} {s}", flush=True)
    P = pd.DataFrame(allrows)
    P.to_csv(OUT / "E2_online_predictions.csv", index=False)
    summ = []
    for (s, t, k), g in P.groupby(["stream", "target", "k"]):
        for m in ["Persist", "Abs", "AbsBias", "Delta", "Fuse"]:
            r = metrics(g.y, g[m]); r.update(stream=s, target=t, k=k, model=m); summ.append(r)
    pd.DataFrame(summ).to_csv(OUT / "E2_online_metrics.csv", index=False)
