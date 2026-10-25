"""E13-E16: supplementary analyses (one target per call so they can run in parallel).

  preproc  E13  PLS under 6 preprocessings vs the date-only GP, 20 random 70/30 splits (paired)
  ablate   E14  DML-Cal ablation on the same 20 splits:
                full | no cross-fitting | no spectral time-centring m(t) | concat GP(PCs + day) | PLS | date
  labels   E15  raw vs audited labels (VA, Alk, FOS, TAC), 20 random splits: PLS, GPR, date
  repro    E16  recovered authors' test days (from 10_/14_ prediction files): published R2, R2 recomputed from
                the released predictions, refitted PLS (sg2) and date-only GP on recovered days (raw labels); FOS recovery is partial
Usage: python exp13_si.py MODE stream:target   |   python exp13_si.py collect
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import (OUT, ROOT, WL, SpecPrep, audit_digestate, audit_hsw, load_stream, metrics)
from exp01_benchmark import model_zoo
from exp03_dmlcal import DMLCal, TimeSmoother, fit_pls, gp_time

warnings.filterwarnings("ignore")
PARTS = OUT / "E13_16_parts"
PREPS = ["raw", "snv", "sg1", "sg2", "snv_sg1", "abs_sg1"]


def splits(n, k=20):
    out = []
    for r in range(k):
        p = np.random.default_rng(r).permutation(n)
        out.append((p[: int(0.7 * n)], p[int(0.7 * n):]))
    return out


def r2(y, p):
    return metrics(y, p)["R2"]


def data(stream, t, raw=False):
    df = load_stream(stream)
    df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
    m = df[t].notna() if raw else df[t].notna() & ~df[f"bad_{t}"]
    s = df[m].reset_index(drop=True)
    return s, s[WL].values, s[t].values.astype(float), s["day"].values.astype(float)


def run_preproc(stream, t):
    s, Xr, y, d = data(stream, t)
    rows = []
    for rep, (tr, te) in enumerate(splits(len(y))):
        pdate = gp_time().fit(d[tr, None], y[tr]).predict(d[te, None])
        rows.append(dict(model="Date", prep="-", rep=rep, R2=r2(y[te], pdate)))
        for pp in PREPS:
            X = SpecPrep(pp).transform(Xr)
            rows.append(dict(model="PLS", prep=pp, rep=rep, R2=r2(y[te], fit_pls(X[tr], y[tr]).predict(X[te]).ravel())))
    return rows


class DMLNoCross(DMLCal):
    def fit(self, X, y, t):
        self.g = gp_time().fit(t[:, None], y)
        bw = float(np.clip(self.g.kernel_.k1.k2.length_scale, 2.0, 60.0))
        self.m = TimeSmoother(bw).fit(t, X)
        self.theta = fit_pls(X - self.m.predict(t), y - self.g.predict(t[:, None]))
        return self


class DMLNoM(DMLCal):
    def fit(self, X, y, t):
        self.g = gp_time().fit(t[:, None], y)
        ry = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=1).split(X):
            ry[te] = y[te] - gp_time().fit(t[tr, None], y[tr]).predict(t[te, None])
        self.theta = fit_pls(X, ry)
        return self

    def predict(self, X, t):
        return self.g.predict(t[:, None]) + self.theta.predict(X).ravel()


def run_ablate(stream, t):
    s, Xr, y, d = data(stream, t)
    X = SpecPrep("sg1").transform(Xr)
    rows = []
    for rep, (tr, te) in enumerate(splits(len(y))):
        P = {"Full": DMLCal().fit(X[tr], y[tr], d[tr]).predict(X[te], d[te]),
             "NoCrossFit": DMLNoCross().fit(X[tr], y[tr], d[tr]).predict(X[te], d[te]),
             "NoTimeCentring": DMLNoM().fit(X[tr], y[tr], d[tr]).predict(X[te], d[te]),
             "PLS": fit_pls(X[tr], y[tr]).predict(X[te]).ravel(),
             "Date": gp_time().fit(d[tr, None], y[tr]).predict(d[te, None])}
        pca = PCA(10, random_state=0).fit(X[tr])
        Z = np.column_stack([StandardScaler().fit(pca.transform(X[tr])).transform(pca.transform(X)), d / 30.0])
        gp = GaussianProcessRegressor(C() * RBF(np.ones(Z.shape[1])) + WhiteKernel(), normalize_y=True, random_state=0)
        P["Concat"] = gp.fit(Z[tr], y[tr]).predict(Z[te])
        rows += [dict(model=k, rep=rep, R2=r2(y[te], v)) for k, v in P.items()]
    return rows


def run_labels(stream, t):
    rows = []
    zoo = model_zoo()
    for lab in ["raw", "audited"]:
        s, Xr, y, d = data(stream, t, raw=(lab == "raw"))
        X = SpecPrep("sg1").transform(Xr)
        for rep, (tr, te) in enumerate(splits(len(y))):
            rows += [dict(labels=lab, n=len(y), model="PLS", rep=rep, R2=r2(y[te], fit_pls(X[tr], y[tr]).predict(X[te]).ravel())),
                     dict(labels=lab, n=len(y), model="GPR", rep=rep, R2=r2(y[te], zoo["GPR_sg1"]().fit(Xr[tr], y[tr]).predict(Xr[te]))),
                     dict(labels=lab, n=len(y), model="Date", rep=rep, R2=r2(y[te], gp_time().fit(d[tr, None], y[tr]).predict(d[te, None])))]
    return rows


def run_repro(stream, t):
    fn = "10_Digestate_predictions_output.csv" if stream == "dig" else "14_HSW_predictions_output.csv"
    mf = "12_Digestate_model_metrics_summary.csv" if stream == "dig" else "16_HSW_model_metrics_summary.csv"
    P = pd.read_csv(ROOT / fn, encoding="utf-8-sig")
    P.columns = [c.replace("- Observed", " - Observed").replace("  ", " ") for c in P.columns]
    te_col = [c for c in P.columns if c.split(" ")[0] == t and "Testing" in c][0]
    pr_col = [c for c in P.columns if c.split(" ")[0] == t and "Predicted" in c][0]
    P["day"] = (pd.to_datetime(P["Date"]).dt.normalize() - pd.Timestamp("2023-12-20")).dt.days
    test = P[P[te_col].notna()]
    pub = pd.read_csv(ROOT / mf, encoding="utf-8-sig").set_index("Parameter").loc[t, "R2 (Test)"]
    rec = r2(test[te_col].values, test[pr_col].values)
    s, Xr, y, d = data(stream, t, raw=True)
    te = np.where(np.isin(d.astype(int), test["day"].values))[0]
    tr = np.setdiff1d(np.arange(len(y)), te)
    X = SpecPrep("sg2").transform(Xr)
    ours = r2(y[te], fit_pls(X[tr], y[tr]).predict(X[te]).ravel())
    date = r2(y[te], gp_time().fit(d[tr, None], y[tr]).predict(d[te, None]))
    return [dict(n_test_authors=len(test), n_test_matched=len(te), n_train=len(tr), published=pub,
                 recomputed_from_author_predictions=rec, our_PLS_sg2=ours, date_only=date)]


MODES = {"preproc": run_preproc, "ablate": run_ablate, "labels": run_labels, "repro": run_repro}

if __name__ == "__main__":
    PARTS.mkdir(exist_ok=True)
    if sys.argv[1] == "collect":
        for mode in MODES:
            fs = sorted(PARTS.glob(f"{mode}_*.csv"))
            if fs:
                pd.concat([pd.read_csv(f) for f in fs]).to_csv(OUT / f"E_si_{mode}.csv", index=False)
        sys.exit()
    mode, key = sys.argv[1], sys.argv[2]
    stream, t = key.split(":")
    rows = MODES[mode](stream, t)
    pd.DataFrame([dict(stream=stream, target=t, **r) for r in rows]).to_csv(PARTS / f"{mode}_{stream}_{t}.csv", index=False)
    print(mode, key, "done", flush=True)
