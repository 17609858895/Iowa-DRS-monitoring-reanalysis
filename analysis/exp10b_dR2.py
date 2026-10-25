"""E10b: incremental R2 over the calendar, dR2_cal = (MSE_date - MSE_model) / Var(y), recomputed from the
saved E10 out-of-fold predictions (no refit). 7-day moving-block bootstrap, 2000 resamples.
Bootstrap of the MSE difference scaled by the full-sample Var(y) (resample-wise Var(y[b]) is unstable
when a resample comes mostly from one sampling period). Replaces the ratio form 1 - MSE/MSE_date, whose bootstrap is unstable when the date model is near-perfect."""
import numpy as np
import pandas as pd

from common import DIG_TARGETS, HSW_TARGETS, OUT
from exp10_skill_ci import block_boot

rows = []
for stream, targets in [("dig", DIG_TARGETS), ("hsw", HSW_TARGETS)]:
    for t in targets:
        d = pd.read_csv(OUT / f"E10_oof_{stream}_{t}.csv")
        y, day = d.y.values, d.day.values
        se = {k: (y - d[k].values) ** 2 for k in ["Date", "PLS", "GPR", "DMLCal"]}
        rng = np.random.default_rng(0)
        boots = [block_boot(day, rng) for _ in range(2000)]
        for k in ["PLS", "GPR", "DMLCal"]:
            bs = np.array([(se["Date"][b].mean() - se[k][b].mean()) / np.var(y) for b in boots])
            rows.append(dict(stream=stream, target=t, model=k, n=len(y),
                             R2_date=1 - se["Date"].mean() / np.var(y), R2_model=1 - se[k].mean() / np.var(y),
                             dR2=(se["Date"].mean() - se[k].mean()) / np.var(y),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p_le0=np.mean(bs <= 0)))
R = pd.DataFrame(rows)
R.to_csv(OUT / "E10b_incremental_R2.csv", index=False)
print(R.round(3).to_string())
