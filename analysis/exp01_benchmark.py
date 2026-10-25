"""E1: conventional absolute-spectrum benchmark under two splits.

Split A: Kennard-Stone 70/30 (authors' protocol, for comparability).
Split B: leave-one-time-block-out (5 contiguous blocks, 3-day buffer) -> evaluation across omitted time segments.
Labels: audited (flagged artefacts dropped) and raw.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import nnls
import lightgbm as lgb

from common import (DIG_TARGETS, HSW_TARGETS, OUT, WL, SpecPrep, audit_digestate, audit_hsw,
                    kennard_stone, load_stream, metrics)

warnings.filterwarnings("ignore")
RNG = 0


class SafePCA(PCA):
    """PCA whose component count is clipped to the fold size (small HSW targets, n = 25)."""

    def fit(self, X, y=None):
        self.n_components = min(self._k, X.shape[0] - 1, X.shape[1])
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        self.n_components = min(self._k, X.shape[0] - 1, X.shape[1])
        return super().fit_transform(X, y)

    def __init__(self, n_components=15, random_state=None):
        super().__init__(n_components=n_components, random_state=random_state)
        self._k = n_components


def pls(prep, max_lv=10):
    pipe = make_pipeline(SpecPrep(prep), StandardScaler(), PLSRegression(scale=False))
    return GridSearchCV(pipe, {"plsregression__n_components": list(range(1, max_lv + 1))},
                        cv=KFold(5, shuffle=True, random_state=RNG), scoring="neg_root_mean_squared_error")


def model_zoo():
    pca = lambda: SafePCA(15, random_state=RNG)
    return {
        "PLS_sg2": lambda: pls("sg2"),
        "PLS_sg1": lambda: pls("sg1"),
        "PLS_snv_sg1": lambda: pls("snv_sg1"),
        "PLS_abs_sg1": lambda: pls("abs_sg1"),
        "PLS_snv": lambda: pls("snv"),
        "Ridge_sg1": lambda: make_pipeline(SpecPrep("sg1"), StandardScaler(), RidgeCV(np.logspace(-3, 4, 30))),
        "SVR_sg1": lambda: GridSearchCV(make_pipeline(SpecPrep("sg1"), StandardScaler(), pca(), SVR()),
                                        {"svr__C": [1, 10, 100], "svr__epsilon": [0.05, 0.2]},
                                        cv=KFold(5, shuffle=True, random_state=RNG)),
        "GPR_sg1": lambda: make_pipeline(SpecPrep("sg1"), StandardScaler(), SafePCA(10, random_state=RNG),
                                         GaussianProcessRegressor(ConstantKernel() * RBF(5.0) + WhiteKernel(),
                                                                  normalize_y=True, random_state=RNG)),
        "RF_sg1": lambda: make_pipeline(SpecPrep("sg1"), pca(), RandomForestRegressor(300, min_samples_leaf=3, random_state=RNG)),
        "ET_sg1": lambda: make_pipeline(SpecPrep("sg1"), pca(), ExtraTreesRegressor(300, min_samples_leaf=3, random_state=RNG)),
        "LGBM_sg1": lambda: make_pipeline(SpecPrep("sg1"), pca(), lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03,
                                                                                    num_leaves=7, min_child_samples=8,
                                                                                    subsample=0.8, subsample_freq=1,
                                                                                    colsample_bytree=0.8, verbose=-1,
                                                                                    random_state=RNG)),
    }


BASE_FOR_STACK = ["PLS_sg2", "PLS_sg1", "PLS_snv_sg1", "PLS_abs_sg1", "GPR_sg1", "SVR_sg1", "ET_sg1"]


class NNLSStack:
    """Non-negative stacked generalisation over inner-CV out-of-fold predictions."""

    def __init__(self, zoo):
        self.zoo = zoo

    def fit(self, X, y):
        cv = KFold(5, shuffle=True, random_state=RNG)
        Z = np.column_stack([cross_val_predict(self.zoo[k](), X, y, cv=cv).ravel() for k in BASE_FOR_STACK])
        A = np.column_stack([Z, np.ones(len(y))])
        w, _ = nnls(A, y)
        self.w_ = w
        self.models_ = [self.zoo[k]().fit(X, y) for k in BASE_FOR_STACK]
        return self

    def predict(self, X):
        Z = np.column_stack([m.predict(X).ravel() for m in self.models_])
        return Z @ self.w_[:-1] + self.w_[-1]


def split_ks(X, y):
    tr = kennard_stone(SpecPrep("sg2").transform(X), int(round(0.7 * len(y))))
    te = np.setdiff1d(np.arange(len(y)), tr)
    return [(tr, te)]


def split_blocks(day, n_blocks=5, buffer=3):
    order = np.argsort(day)
    blocks = np.array_split(order, n_blocks)
    out = []
    for b in blocks:
        lo, hi = day[b].min(), day[b].max()
        tr = np.where((day < lo - buffer) | (day > hi + buffer))[0]
        out.append((tr, b))
    return out


def run(stream, labels="audited"):
    df = load_stream(stream)
    df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
    targets = DIG_TARGETS if stream == "dig" else HSW_TARGETS
    zoo = model_zoo()
    rows, preds = [], []
    for t in targets:
        m = df[t].notna()
        if labels == "audited":
            m &= ~df[f"bad_{t}"]
        sub = df[m].reset_index(drop=True)
        X, y, day = sub[WL].values, sub[t].values.astype(float), sub["day"].values
        for split_name, splits in [("KS", split_ks(X, y)), ("Block", split_blocks(day))]:
            names = list(zoo) + ["Stack_NNLS"]
            for name in names:
                yt, yp = [], []
                for tr, te in splits:
                    mdl = NNLSStack(zoo) if name == "Stack_NNLS" else zoo[name]()
                    mdl.fit(X[tr], y[tr])
                    p = np.asarray(mdl.predict(X[te])).ravel()
                    yt.append(y[te]); yp.append(p)
                    preds += [dict(stream=stream, target=t, split=split_name, model=name, labels=labels,
                                   day=d_, y=a, p=b) for d_, a, b in zip(day[te], y[te], p)]
                yt, yp = np.concatenate(yt), np.concatenate(yp)
                r = metrics(yt, yp)
                r.update(stream=stream, target=t, split=split_name, model=name, labels=labels, n_lab=len(y))
                rows.append(r)
                print(f"{stream} {t:7s} {split_name:5s} {name:12s} R2={r['R2']:.3f} NRMSE={r['NRMSE']:.1f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(preds)


if __name__ == "__main__":
    labels = sys.argv[1] if len(sys.argv) > 1 else "audited"
    streams = sys.argv[2:] or ["dig", "hsw"]
    for s in streams:
        a, b = run(s, labels)
        a.to_csv(OUT / f"E1_benchmark_{labels}_{s}.csv", index=False)
        b.to_csv(OUT / f"E1_predictions_{labels}_{s}.csv", index=False)
