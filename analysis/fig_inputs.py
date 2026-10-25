"""Frozen inputs for figures that are not produced by E-scripts: drift PCs and DML-Cal coefficients."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import OUT, WAVELENGTH, WL, SpecPrep, audit_digestate, audit_hsw, load_daily, load_stream
from exp03_dmlcal import DMLCal, fit_pls

FI = OUT / "fig_inputs"
FI.mkdir(exist_ok=True)
rows = []
for stream in ["dig", "hsw"]:
    d = load_daily(stream).groupby("day", as_index=False)[WL].mean()
    X = SpecPrep("sg1").transform(d[WL].values)
    pca = PCA(3).fit(X)
    sc = pca.transform(X)
    rows.append(pd.DataFrame({"stream": stream, "day": d.day, "PC1": sc[:, 0], "PC2": sc[:, 1],
                              "mean_reflectance": d[WL].values.mean(1)}))
    pd.DataFrame({"wavelength": WAVELENGTH, "mean_raw": d[WL].values.mean(0),
                  "sd_raw": d[WL].values.std(0)}).to_csv(FI / f"mean_spectrum_{stream}.csv", index=False)
pd.concat(rows).to_csv(FI / "daily_spectral_pcs.csv", index=False)

coef = []
for stream, t in [("hsw", "TS"), ("hsw", "VS"), ("dig", "Alk"), ("dig", "TS")]:
    df = load_stream(stream)
    df = audit_digestate(df) if stream == "dig" else audit_hsw(df)
    s = df[df[t].notna() & ~df[f"bad_{t}"]]
    X = SpecPrep("sg1").transform(s[WL].values)
    m = DMLCal().fit(X, s[t].values.astype(float), s["day"].values.astype(float))
    pls = m.theta.named_steps["plsregression"]; sc = m.theta.named_steps["standardscaler"]
    b = pls.coef_.ravel() / sc.scale_ if pls.coef_.shape[0] == 1 else pls.coef_[:, 0] / sc.scale_
    pl = fit_pls(X, s[t].values.astype(float)).named_steps
    b0 = pl["plsregression"].coef_.ravel() / pl["standardscaler"].scale_
    coef.append(pd.DataFrame({"stream": stream, "target": t, "wavelength": WAVELENGTH,
                              "theta_dml": b / np.abs(b).max(), "beta_pls": b0 / np.abs(b0).max()}))
pd.concat(coef).to_csv(FI / "dmlcal_coefficients.csv", index=False)
print("ok")
