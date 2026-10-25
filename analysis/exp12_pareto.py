"""E12: laboratory frequency vs monitoring error (operational Pareto).

A 30-day initial calibration campaign uses every available lab value. Afterwards labs are kept only
every m calendar days (m in 1..14, up to three phases). On every labelled day that is NOT a kept lab
day, each strategy predicts using kept labs strictly before that day:
  Persist    last kept lab value
  DateGP     GP on day fitted to kept labs (calendar extrapolation)
  SpecPLS    absolute PLS refitted on kept labs every 14 d
  AnchorDML  partially linear calibration refitted every 14 d, anchored on the 5 latest kept labs
Reported: lab fraction (kept labs / evaluated days) and NMAE on non-lab days (% of mean).
"""
import warnings

import numpy as np
import pandas as pd

from common import OUT, WL, SpecPrep, audit_digestate, audit_hsw, load_stream
from exp03_dmlcal import DMLCal, fit_pls, gp_time

warnings.filterwarnings("ignore")
MS = [1, 2, 3, 5, 7, 10, 14]
WARM = 30
REFIT = 14
N_ANCHOR = 5
TARGETS = [("dig", ["TS", "VS", "Alk", "TAC"]), ("hsw", ["TS", "VS"])]


def simulate(X, y, t, m, phase):
    t0 = t.min()
    warm = t < t0 + WARM
    kept = warm | (((t - t0 - WARM - phase) % m) == 0)
    rows, cache = [], {}
    for i in np.where(~warm)[0]:
        if kept[i]:
            continue
        avail = kept & (t < t[i])
        if avail.sum() < 15:
            continue
        cut = t0 + WARM + ((t[i] - t0 - WARM) // REFIT) * REFIT
        tr = kept & (t < cut) if (kept & (t < cut)).sum() >= 15 else avail
        key = (cut, int(tr.sum()))
        if key not in cache:
            cache[key] = (fit_pls(X[tr], y[tr]), DMLCal().fit(X[tr], y[tr], t[tr]), gp_time().fit(t[tr, None], y[tr]))
        pls, dml, gpd = cache[key]
        a = np.where(avail)[0][-N_ANCHOR:]
        lvl = y[a].mean()  # anchors may post-date the refit, so the level comes from the labs themselves
        rows.append(dict(day=t[i], y=y[i], Persist=y[a[-1]],
                         DateGP=gp_time().fit(t[avail, None], y[avail]).predict(t[i:i + 1, None])[0],
                         SpecPLS=pls.predict(X[i:i + 1]).ravel()[0],
                         AnchorDML=lvl + dml.theta.predict(X[i:i + 1] - X[a].mean(0, keepdims=True)).ravel()[0]))
    n_eval = (~warm).sum()
    return pd.DataFrame(rows), (kept & ~warm).sum() / n_eval


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["collect"]:
        pd.concat([pd.read_csv(f) for f in sorted((OUT / "E12_parts").glob("*.csv"))]).to_csv(OUT / "E12_pareto.csv", index=False)
        sys.exit()
    only = sys.argv[1:]
    out = []
    for stream, targets in TARGETS:
        df = load_stream(stream)
        df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
        for tname in targets:
            if only and f"{stream}:{tname}" not in only:
                continue
            s = df[df[tname].notna() & ~df[f"bad_{tname}"]].sort_values("day").reset_index(drop=True)
            X = SpecPrep("sg1").transform(s[WL].values)
            y = s[tname].values.astype(float)
            t = s["day"].values.astype(int)
            for m in MS:
                for ph in sorted({0, m // 3, (2 * m) // 3}):
                    R, frac = simulate(X, y, t, m, ph)
                    if R.empty:
                        continue
                    for k in ["Persist", "DateGP", "SpecPLS", "AnchorDML"]:
                        out.append(dict(stream=stream, target=tname, m=m, phase=ph, lab_frac=frac, model=k,
                                        n_eval=len(R), NMAE=100 * np.mean(np.abs(R.y - R[k])) / y.mean()))
                g = pd.DataFrame([o for o in out if o["stream"] == stream and o["target"] == tname and o["m"] == m])
                if not g.empty:
                    print(stream, tname, m, g.groupby("model").NMAE.mean().round(1).to_dict(), flush=True)
    (OUT / "E12_parts").mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(OUT / "E12_parts" / f"{'_'.join(x.replace(':', '-') for x in only) or 'all'}.csv", index=False)
