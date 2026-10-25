"""E7: local wrong-day spectrum diagnostic.

For each labelled observation, replace its spectrum with an available daily
spectrum from a different day within +/-w days, for both training and testing.
The GPR model is evaluated under Kennard-Stone and repeated random allocation.
This perturbation measures local sample-matching sensitivity; it is not an exact
conditional-randomisation test or proof of chemical selectivity. Pair it with the
date-displacement reference computed by exp10_skill_ci.py.
"""
import warnings

import numpy as np
import pandas as pd

from common import (DIG_TARGETS, HSW_TARGETS, OUT, WL, SpecPrep, audit_digestate, audit_hsw,
                    kennard_stone, load_daily, load_stream, metrics)
from exp01_benchmark import model_zoo

warnings.filterwarnings("ignore")
WS = [0, 1, 3, 7, 14, 28]
N_DRAW = 10


def swapped(X_true, days, daily, w, rng):
    if w == 0:
        return X_true
    lut = dict(zip(daily["day"].values, daily[WL].values))
    out = X_true.copy()
    for i, d in enumerate(days):
        cands = [d + k for k in range(-w, w + 1) if k != 0 and (d + k) in lut]
        if cands:
            out[i] = lut[cands[rng.integers(len(cands))]]
    return out


def main():
    zoo = model_zoo()
    rows = []
    for stream, targets in [("dig", DIG_TARGETS), ("hsw", HSW_TARGETS)]:
        df = load_stream(stream)
        df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
        daily = load_daily(stream).groupby("day", as_index=False)[WL].mean()
        for t in targets:
            s = df[df[t].notna() & ~df[f"bad_{t}"]].reset_index(drop=True)
            Xt, y, days = s[WL].values, s[t].values.astype(float), s["day"].values
            n = len(y)
            ks_tr = kennard_stone(SpecPrep("sg2").transform(Xt), int(round(0.7 * n)))
            rng0 = np.random.default_rng(0)
            splits = [("KS", ks_tr, np.setdiff1d(np.arange(n), ks_tr))]
            for r in range(20):
                p = rng0.permutation(n)
                splits.append((f"R{r}", p[: int(0.7 * n)], p[int(0.7 * n):]))
            for w in WS:
                draws = 1 if w == 0 else N_DRAW
                for dr in range(draws):
                    rng = np.random.default_rng(100 + dr)
                    X = swapped(Xt, days, daily, w, rng)
                    for name, tr, te in (splits if dr < 3 or w == 0 else splits[:1]):
                        m = zoo["GPR_sg1"]().fit(X[tr], y[tr])
                        rows.append(dict(stream=stream, target=t, w=w, draw=dr, split=name,
                                         R2=metrics(y[te], m.predict(X[te]))["R2"]))
            R = pd.DataFrame([r for r in rows if r["stream"] == stream and r["target"] == t])
            ks = R[R.split == "KS"].groupby("w").R2.median().round(2).to_dict()
            rnd = R[R.split != "KS"].groupby("w").R2.median().round(2).to_dict()
            print(f"{stream} {t:7s} KS {ks} | Random {rnd}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "E7_wrongday.csv", index=False)


if __name__ == "__main__":
    main()
