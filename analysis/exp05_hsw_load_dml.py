"""E5 (exploratory): does HSW organic load drive later digestate VA / Alk?

Daily HSW VS: lab value when measured, otherwise spectral prediction (PLS trained on labelled HSW,
out-of-fold for labelled days). Treatment D = mean HSW OLR over days t-L..t-1 (kg VS m-3 d-1).
Outcome Y = digestate lab value on day t. Controls W = TWAS/PS OLR over the same window, smooth day trend.
Partially linear DML (5-fold cross-fitting, RF nuisances); treatment built from (i) lab-only VS,
(ii) spectra-filled VS. Window L in {3, 7, 14}.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

from common import OUT, ROOT, WL, SpecPrep, audit_digestate, load_daily, load_stream
from exp03_dmlcal import fit_pls

warnings.filterwarnings("ignore")
V_DIG = 3671.849  # m3, two digesters


def daily_hsw_vs():
    lab = load_stream("hsw")
    lab = lab[lab["VS"].notna()]
    X = SpecPrep("sg1").transform(lab[WL].values)
    y = lab["VS"].values
    oof = cross_val_predict(fit_pls(X, y), X, y, cv=KFold(5, shuffle=True, random_state=0)).ravel()
    model = fit_pls(X, y)
    dly = load_daily("hsw")
    dly["VS_spec"] = model.predict(SpecPrep("sg1").transform(dly[WL].values)).ravel()
    dly = dly.groupby("day", as_index=False)["VS_spec"].mean()
    labd = pd.DataFrame({"day": lab["day"].values, "VS_lab": y, "VS_oof": oof}).groupby("day", as_index=False).mean()
    out = dly.merge(labd, on="day", how="outer").sort_values("day")
    out["VS_fill"] = out["VS_lab"].fillna(out["VS_spec"])
    return out


def build(L):
    load = pd.read_csv(ROOT / "20_Loading.csv", encoding="utf-8-sig")
    load.columns = [c.strip() for c in load.columns]
    load["day"] = (pd.to_datetime(load["Date"]) - pd.Timestamp("2023-12-20")).dt.days
    vs = daily_hsw_vs()
    f = load.merge(vs, on="day", how="left")
    f["D_lab"] = f["HSW Fed liters"] * f["VS_lab"] * 1e-6 / V_DIG
    f["D_spec"] = f["HSW Fed liters"] * f["VS_fill"] * 1e-6 / V_DIG
    f = f.set_index("day").reindex(range(f["day"].min(), f["day"].max() + 1))
    roll = lambda s: s.shift(1).rolling(L, min_periods=max(2, L // 2)).mean()
    feats = pd.DataFrame({"D_lab": roll(f["D_lab"]), "D_spec": roll(f["D_spec"]),
                          "TWAS": roll(f["TWAS Observed Load"]), "PS": roll(f["PS Observed Load"])})
    feats["day"] = feats.index
    return feats


def dml(Y, D, W, seed=0):
    rf = lambda: RandomForestRegressor(300, min_samples_leaf=5, random_state=seed)
    cv = KFold(5, shuffle=True, random_state=seed)
    ry = Y - cross_val_predict(rf(), W, Y, cv=cv)
    rd = D - cross_val_predict(rf(), W, D, cv=cv)
    theta = np.sum(rd * ry) / np.sum(rd * rd)
    psi = (ry - theta * rd) * rd
    se = np.sqrt(np.mean(psi**2) / np.mean(rd * rd) ** 2 / len(Y))
    return theta, se


if __name__ == "__main__":
    dig = audit_digestate(load_stream("dig"))
    rows = []
    for L in [3, 7, 14]:
        F = build(L)
        for tname in ["VA", "Alk", "TS", "VS"]:
            m = dig[tname].notna() & ~dig[f"bad_{tname}"]
            y = dig.loc[m, ["day", tname]].groupby("day").mean()
            J = F.join(y, how="inner")
            for dcol in ["D_lab", "D_spec"]:
                Jc = J.dropna(subset=[dcol, "TWAS", "PS", tname])
                if len(Jc) < 25:
                    continue
                W = Jc[["TWAS", "PS", "day"]].values
                ests = [dml(Jc[tname].values, Jc[dcol].values, W, s) for s in range(5)]
                th = np.median([e[0] for e in ests]); se = np.median([e[1] for e in ests])
                sd_d = Jc[dcol].std()
                rows.append(dict(L=L, outcome=tname, treatment=dcol, n=len(Jc), theta=th, se=se,
                                 effect_per_sd=th * sd_d, z=th / se))
                print(f"L={L:2d} {tname:4s} {dcol:6s} n={len(Jc):3d} theta*sdD={th*sd_d:8.1f} z={th/se:5.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "E5_hsw_load_dml.csv", index=False)
