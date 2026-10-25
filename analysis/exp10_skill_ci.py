"""E10: fixed out-of-fold predictions, loss intervals and date displacement.

Part 1 averages ten repeated five-fold out-of-fold predictions per observation.
The legacy CO = 1 - MSE_model/MSE_date is retained as an auxiliary output.
The manuscript's primary increment and intervals are computed by exp10b_dR2.py:
(MSE_date - MSE_model)/Var(y), with full target-series variance held fixed.
Seven-calendar-day circular block resampling acts on fixed prediction losses;
model-refitting and model-selection uncertainty are not included.

Part 2 independently displaces calendar inputs within +/-w days to provide the
reference for the spectral-displacement curves. Different replacement mechanics
prevent treating the two perturbations as an exact conditional test.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from common import (DIG_TARGETS, HSW_TARGETS, OUT, WL, SpecPrep, audit_digestate, audit_hsw,
                    load_stream)
from exp01_benchmark import model_zoo
from exp03_dmlcal import DMLCal, fit_pls, gp_time

warnings.filterwarnings("ignore")
N_BOOT, BLOCK = 2000, 7
WS = [1, 3, 7, 14, 28]


def oof(X, Xr, y, t):
    zoo = model_zoo()
    acc = {k: np.zeros(len(y)) for k in ["Date", "PLS", "GPR", "DMLCal"]}
    cnt = np.zeros(len(y))
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=10, random_state=0).split(X):
        acc["Date"][te] += gp_time().fit(t[tr, None], y[tr]).predict(t[te, None])
        acc["PLS"][te] += fit_pls(X[tr], y[tr]).predict(X[te]).ravel()
        acc["GPR"][te] += zoo["GPR_sg1"]().fit(Xr[tr], y[tr]).predict(Xr[te])
        acc["DMLCal"][te] += DMLCal().fit(X[tr], y[tr], t[tr]).predict(X[te], t[te])
        cnt[te] += 1
    return {k: v / cnt for k, v in acc.items()}


def block_boot(t, rng):
    """Circular block resample of sample indices, blocks of BLOCK calendar days.

    Blocks wrap around the end of the record, so every day has the same inclusion probability
    (the non-circular moving-block scheme under-samples the first and last days)."""
    t0 = t.min()
    span = t.max() - t0 + 1
    rel = t - t0
    idx = []
    while len(idx) < len(t):
        s0 = rng.integers(0, span)
        idx += list(np.where(((rel - s0) % span) < BLOCK)[0])
    return np.array(idx[: len(t)])


def main(only=None):
    """only: list of 'stream:target'; each target writes its own part files (parallel-safe)."""
    for stream, targets in [("dig", DIG_TARGETS), ("hsw", HSW_TARGETS)]:
        df = load_stream(stream)
        df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
        for tname in targets:
            if only and f"{stream}:{tname}" not in only:
                continue
            part = OUT / "E10_parts" / f"{stream}_{tname}"
            if (part.parent / f"{part.name}_skill.csv").exists():
                continue
            part.parent.mkdir(exist_ok=True)
            rows, jit = [], []
            s = df[df[tname].notna() & ~df[f"bad_{tname}"]].reset_index(drop=True)
            Xr = s[WL].values
            X = SpecPrep("sg1").transform(Xr)
            y = s[tname].values.astype(float)
            t = s["day"].values.astype(float)
            P = oof(X, Xr, y, t)
            pd.DataFrame({"day": t, "y": y, **P}).to_csv(OUT / f"E10_oof_{stream}_{tname}.csv", index=False)
            rng = np.random.default_rng(0)
            se = {k: (y - p) ** 2 for k, p in P.items()}
            boots = {k: [] for k in ["PLS", "GPR", "DMLCal"]}
            r2b = {k: [] for k in P}
            for _ in range(N_BOOT):
                b = block_boot(t, rng)
                md = se["Date"][b].mean()
                for k in boots:
                    boots[k].append(1 - se[k][b].mean() / md)
                v = np.var(y[b])
                for k in P:
                    r2b[k].append(1 - se[k][b].mean() / v)
            for k in P:
                r2 = 1 - se[k].mean() / np.var(y)
                row = dict(stream=stream, target=tname, model=k, n=len(y), R2=r2,
                           R2_lo=np.percentile(r2b[k], 2.5), R2_hi=np.percentile(r2b[k], 97.5))
                if k != "Date":
                    co = 1 - se[k].mean() / se["Date"].mean()
                    row.update(CO=co, CO_lo=np.percentile(boots[k], 2.5), CO_hi=np.percentile(boots[k], 97.5),
                               P_CO_le0=np.mean(np.array(boots[k]) <= 0))
                rows.append(row)
            co = {r["model"]: r for r in rows if r["stream"] == stream and r["target"] == tname and r["model"] != "Date"}
            print(f"{stream} {tname:7s} " + " ".join(f"{k}: CO={v['CO']:.2f} [{v['CO_lo']:.2f},{v['CO_hi']:.2f}]"
                                                    for k, v in co.items()), flush=True)
            # Part 2: date-jitter null (random 70/30 x 20, 5 draws per w)
            rng2 = np.random.default_rng(1)
            for w in [0] + WS:
                for dr in range(1 if w == 0 else 5):
                    shift = 0 if w == 0 else rng2.choice([d for d in range(-w, w + 1) if d != 0], len(t))
                    tj = t + shift
                    for r in range(20):
                        p = np.random.default_rng(r).permutation(len(y))
                        tr, te = p[: int(0.7 * len(y))], p[int(0.7 * len(y)):]
                        pr = gp_time().fit(tj[tr, None], y[tr]).predict(tj[te, None])
                        jit.append(dict(stream=stream, target=tname, w=w, draw=dr, rep=r,
                                        R2=1 - np.mean((y[te] - pr) ** 2) / np.var(y[te])))
            pd.DataFrame(rows).to_csv(part.parent / f"{part.name}_skill.csv", index=False)
            pd.DataFrame(jit).to_csv(part.parent / f"{part.name}_jitter.csv", index=False)


def collect():
    parts = OUT / "E10_parts"
    pd.concat([pd.read_csv(f) for f in sorted(parts.glob("*_skill.csv"))]).to_csv(OUT / "E10_calendar_orthogonal_skill.csv", index=False)
    pd.concat([pd.read_csv(f) for f in sorted(parts.glob("*_jitter.csv"))]).to_csv(OUT / "E10_date_jitter_null.csv", index=False)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["collect"]:
        collect()
    else:
        main(sys.argv[1:] or None)
