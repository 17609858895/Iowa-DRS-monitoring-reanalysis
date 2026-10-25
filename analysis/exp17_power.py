"""E17: semi-synthetic calibration of the diagnostics (known truth).

Real spectra X and real sampling days t of one target are kept. The synthetic label is
    y* = g(t) + beta * sd(g) * z(x) + eps,
  g     date-only GP fitted to the real label (realistic calendar structure),
  z(x)  a standardised spectral score made orthogonal to time: first sg1 principal component minus its
        kernel-smoothed date trend (so z carries sample-specific spectral variation, not the date),
  eps   Gaussian noise with the SD of the real residual y - g(t).
beta = 0 omits the injected spectral-score term; beta > 0 introduces known simulated spectral signal.

Per replicate: Kennard-Stone R2 of PLS; 5-fold x 2 out-of-fold predictions for date GP, spectral GPR and
DML-Cal; calendar-orthogonal skill CO = 1 - MSE/MSE_date with a 7-day moving-block bootstrap (500); a
detection is the lower 95% bound of the incremental R2 over the date model,
dR2_cal = (MSE_date - MSE_model) / Var(y), above 0 (bootstrap of the MSE difference, full-sample Var(y)). Wrong-day gap: R2(true spectra) - R2(spectra from a day +-3 d away).
Usage: python exp17_power.py stream:target   |   python exp17_power.py collect
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold

from common import OUT, WL, SpecPrep, audit_digestate, audit_hsw, kennard_stone, load_daily, load_stream, metrics
from exp01_benchmark import model_zoo
from exp03_dmlcal import DMLCal, TimeSmoother, fit_pls, gp_time
from exp07_wrongday import swapped
from exp10_skill_ci import block_boot

warnings.filterwarnings("ignore")
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
REPS = 30
N_BOOT = 500
PARTS = OUT / "E17_parts"


def one_rep(Xr, X, t, g, z, sd_eps, beta, rng, daily, ks_tr, ks_te):
    y = g + beta * g.std() * z + rng.normal(0, sd_eps, len(g))
    zoo = model_zoo()
    out = dict(R2_KS_PLS=metrics(y[ks_te], fit_pls(X[ks_tr], y[ks_tr]).predict(X[ks_te]).ravel())["R2"])
    P = {k: np.zeros(len(y)) for k in ["Date", "GPR", "DMLCal"]}
    cnt = np.zeros(len(y))
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=2, random_state=int(rng.integers(1e6))).split(X):
        P["Date"][te] += gp_time().fit(t[tr, None], y[tr]).predict(t[te, None])
        P["GPR"][te] += zoo["GPR_sg1"]().fit(Xr[tr], y[tr]).predict(Xr[te])
        P["DMLCal"][te] += DMLCal().fit(X[tr], y[tr], t[tr]).predict(X[te], t[te])
        cnt[te] += 1
    P = {k: v / cnt for k, v in P.items()}
    se = {k: (y - p) ** 2 for k, p in P.items()}
    boots = [block_boot(t, rng) for _ in range(N_BOOT)]
    for k in ["GPR", "DMLCal"]:
        bs = [(se["Date"][b].mean() - se[k][b].mean()) / np.var(y) for b in boots]
        out[f"dR2_{k}"] = (se["Date"].mean() - se[k].mean()) / np.var(y)
        out[f"dR2_{k}_lo"] = np.percentile(bs, 2.5)
        out[f"detect_{k}"] = float(np.percentile(bs, 2.5) > 0)
    out["R2_Date"] = 1 - se["Date"].mean() / np.var(y)
    out["R2_GPR"] = 1 - se["GPR"].mean() / np.var(y)
    # wrong-day gap on one random 70/30 split
    p = rng.permutation(len(y)); tr, te = p[: int(.7 * len(y))], p[int(.7 * len(y)):]
    r_true = metrics(y[te], zoo["GPR_sg1"]().fit(Xr[tr], y[tr]).predict(Xr[te]))["R2"]
    Xs = swapped(Xr, t.astype(int), daily, 3, rng)
    r_swap = metrics(y[te], zoo["GPR_sg1"]().fit(Xs[tr], y[tr]).predict(Xs[te]))["R2"]
    out["wrongday_gap"] = r_true - r_swap
    # date-shift reference: the same +-3 d displacement applied to the dates of a date-only model
    tj = t + rng.choice([d for d in range(-3, 4) if d != 0], len(t))
    d_true = metrics(y[te], gp_time().fit(t[tr, None], y[tr]).predict(t[te, None]))["R2"]
    d_shift = metrics(y[te], gp_time().fit(tj[tr, None], y[tr]).predict(tj[te, None]))["R2"]
    out["date_gap"] = d_true - d_shift
    out["excess_gap"] = out["wrongday_gap"] - out["date_gap"]
    return out


def run(stream, tname):
    df = load_stream(stream)
    df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
    s = df[df[tname].notna() & ~df[f"bad_{tname}"]].reset_index(drop=True)
    Xr = s[WL].values
    X = SpecPrep("sg1").transform(Xr)
    t = s["day"].values.astype(float)
    yr = s[tname].values.astype(float)
    gp = gp_time().fit(t[:, None], yr)
    g = gp.predict(t[:, None])
    sd_eps = np.std(yr - g)
    pc = PCA(1).fit_transform(X)[:, 0]
    bw = float(np.clip(gp.kernel_.k1.k2.length_scale, 2.0, 60.0))
    resid = pc - TimeSmoother(bw).fit(t, pc[:, None]).predict(t)[:, 0]
    z = (resid - resid.mean()) / resid.std()
    daily = load_daily(stream).groupby("day", as_index=False)[WL].mean()
    ks_tr = kennard_stone(SpecPrep("sg2").transform(Xr), int(round(.7 * len(t))))
    ks_te = np.setdiff1d(np.arange(len(t)), ks_tr)
    rows = []
    for beta in BETAS:
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * rep + int(beta * 100))
            r = one_rep(Xr, X, t, g, z, sd_eps, beta, rng, daily, ks_tr, ks_te)
            rows.append(dict(stream=stream, target=tname, beta=beta, rep=rep, corr_z_t=np.corrcoef(z, t)[0, 1], **r))
    return rows


if __name__ == "__main__":
    PARTS.mkdir(exist_ok=True)
    if sys.argv[1] == "collect":
        pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob("*.csv"))]).to_csv(OUT / "E17_power.csv", index=False)
        sys.exit()
    stream, tname = sys.argv[1].split(":")
    beta_sel = [float(b) for b in sys.argv[2:]] or BETAS
    BETAS = beta_sel
    rows = run(stream, tname)
    tag = "_".join(f"{b:g}" for b in beta_sel)
    pd.DataFrame(rows).to_csv(PARTS / f"{stream}_{tname}_{tag}.csv", index=False)
    print("done", stream, tname, tag, flush=True)
