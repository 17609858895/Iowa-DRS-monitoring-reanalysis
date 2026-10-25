"""E3: temporal-shortcut decomposition and double-ML (partially linear) spectral calibration.

y = g(t) + theta(x - m(t)) + e      g, m = smooth functions of sampling day (nuisances)
Cross-fitted on the training set; prediction = g_hat(t) + theta(x - m_hat(t)).

Compared, on identical splits:
  Time      GP on day only (calendar shortcut, no spectra)
  Spec      PLS on spectra only (authors' model class)
  Spec+Time GP on spectral PCs + day (naive concatenation)
  DMLCal    partially linear calibration above
Splits: KS 70/30 (authors), 20x repeated random 70/30, leave-one-block-out (5 blocks, 3 d buffer).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (DIG_TARGETS, HSW_TARGETS, OUT, WL, SpecPrep, audit_digestate, audit_hsw,
                    kennard_stone, load_stream, metrics)
from exp01_benchmark import split_blocks

warnings.filterwarnings("ignore")
PREP = "sg1"


def gp_time():
    return GaussianProcessRegressor(C() * RBF(10.0, (1.0, 200.0)) + WhiteKernel(), normalize_y=True,
                                    n_restarts_optimizer=2, random_state=0)


def fit_pls(X, y, max_lv=10):
    gs = GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)),
                      {"plsregression__n_components": list(range(1, min(max_lv, len(y) // 4) + 1))},
                      cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error")
    return gs.fit(X, y).best_estimator_


class TimeSmoother:
    """m(t): kernel (Nadaraya-Watson) smoother of spectra over day; bandwidth tied to g's GP length."""

    def __init__(self, bw):
        self.bw = bw

    def fit(self, t, X):
        self.t, self.X = t, X
        return self

    def predict(self, t):
        w = np.exp(-0.5 * ((t[:, None] - self.t[None, :]) / self.bw) ** 2)
        return (w @ self.X) / w.sum(1, keepdims=True)


class DMLCal:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def fit(self, X, y, t):
        self.g = gp_time().fit(t[:, None], y)
        bw = float(np.clip(self.g.kernel_.k1.k2.length_scale, 2.0, 60.0))
        ry, rX = np.zeros_like(y), np.zeros_like(X)
        for tr, te in KFold(self.n_folds, shuffle=True, random_state=1).split(X):
            g = gp_time().fit(t[tr, None], y[tr])
            ry[te] = y[te] - g.predict(t[te, None])
            rX[te] = X[te] - TimeSmoother(bw).fit(t[tr], X[tr]).predict(t[te])
        self.m = TimeSmoother(bw).fit(t, X)
        self.theta = fit_pls(rX, ry)
        return self

    def predict(self, X, t):
        return self.g.predict(t[:, None]) + self.theta.predict(X - self.m.predict(t)).ravel()


def evaluate(X, y, t, tr, te):
    out = {}
    out["Time"] = gp_time().fit(t[tr, None], y[tr]).predict(t[te, None])
    out["Spec"] = fit_pls(X[tr], y[tr]).predict(X[te]).ravel()
    pca = PCA(10, random_state=0).fit(X[tr])
    Z = np.column_stack([StandardScaler().fit(pca.transform(X[tr])).transform(pca.transform(X)), t / 30.0])
    gp = GaussianProcessRegressor(C() * RBF(np.ones(Z.shape[1])) + WhiteKernel(), normalize_y=True, random_state=0)
    out["Spec+Time"] = gp.fit(Z[tr], y[tr]).predict(Z[te])
    out["DMLCal"] = DMLCal().fit(X[tr], y[tr], t[tr]).predict(X[te], t[te])
    return out


def main():
    rows, preds = [], []
    for stream, targets in [("dig", DIG_TARGETS), ("hsw", HSW_TARGETS)]:
        df = load_stream(stream)
        df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
        for tname in targets:
            m = df[tname].notna() & ~df[f"bad_{tname}"]
            s = df[m].reset_index(drop=True)
            X = SpecPrep(PREP).transform(s[WL].values)
            y = s[tname].values.astype(float)
            t = s["day"].values.astype(float)
            n = len(y)
            schemes = {"KS": [(kennard_stone(SpecPrep("sg2").transform(s[WL].values), int(round(0.7 * n))), None)]}
            schemes["KS"] = [(schemes["KS"][0][0], np.setdiff1d(np.arange(n), schemes["KS"][0][0]))]
            rng = np.random.default_rng(0)
            rnd = []
            for _ in range(20):
                p = rng.permutation(n)
                rnd.append((p[: int(0.7 * n)], p[int(0.7 * n):]))
            schemes["Random20"] = rnd
            schemes["Block"] = split_blocks(t.astype(int))
            for sch, splits in schemes.items():
                acc = {}
                for rep, (tr, te) in enumerate(splits):
                    out = evaluate(X, y, t, tr, te)
                    for k, p in out.items():
                        acc.setdefault(k, []).append((rep, y[te], p))
                        preds += [dict(stream=stream, target=tname, scheme=sch, rep=rep, model=k, day=d_, y=a, p=b)
                                  for d_, a, b in zip(t[te], y[te], p)]
                for k, lst in acc.items():
                    if sch == "Random20":
                        r2s = [metrics(a, b)["R2"] for _, a, b in lst]
                        rr = dict(R2=np.median(r2s), R2_q25=np.percentile(r2s, 25), R2_q75=np.percentile(r2s, 75),
                                  NRMSE=np.median([metrics(a, b)["NRMSE"] for _, a, b in lst]))
                    else:
                        a = np.concatenate([x[1] for x in lst]); b = np.concatenate([x[2] for x in lst])
                        rr = metrics(a, b)
                    rr.update(stream=stream, target=tname, scheme=sch, model=k, n_lab=n)
                    rows.append(rr)
                line = " ".join(f"{r['model']}={r['R2']:.2f}" for r in rows if r["stream"] == stream and
                                r["target"] == tname and r["scheme"] == sch)
                print(f"{stream} {tname:7s} {sch:8s} {line}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "E3_dmlcal_metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "E3_dmlcal_predictions.csv", index=False)


if __name__ == "__main__":
    main()
