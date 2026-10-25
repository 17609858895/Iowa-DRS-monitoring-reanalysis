"""Shared data loading, label audit, preprocessing and metrics for the Iowa DRS co-digestion study.

Public source files in data/raw (or IOWA_DRS_DATA_DIR) are read only.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("IOWA_DRS_DATA_DIR", REPO_ROOT / "data" / "raw")).expanduser().resolve()
OUT = Path(os.environ.get("IOWA_DRS_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)

WL = [f"wl{i}" for i in range(1, 262)]
WAVELENGTH = np.arange(400, 1705, 5)
DIG_TARGETS = ["TS", "VS", "VA", "Alk", "FOS", "TAC"]
HSW_TARGETS = ["TS", "VS", "COD", "VFA", "Fat", "Protein", "Carbs"]


def load_stream(stream: str) -> pd.DataFrame:
    """Return labelled spectra sorted by time. stream in {'dig', 'hsw'}."""
    fn = "02_AllTrainDigestate.csv" if stream == "dig" else "04_AllTrainHSW.csv"
    df = pd.read_csv(ROOT / fn, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["Date"] = pd.to_datetime(df["Date"])
    # 12/21/2021 lies outside the campaign and sits between 12/20/2023 and 12/22/2023 in file order;
    # provisionally aligned to 2023; no complete drop-row sensitivity was retained.
    bad = df["Date"].dt.year == 2021
    df["flag_date_typo"] = bad
    df.loc[bad, "Date"] = df.loc[bad, "Date"] + pd.DateOffset(years=2)
    df = df.sort_values("Date").reset_index(drop=True)
    df["day"] = (df["Date"].dt.normalize() - pd.Timestamp("2023-12-20")).dt.days
    return df


def audit_digestate(df: pd.DataFrame) -> pd.DataFrame:
    """Flag numerical patterns without treating them as confirmed data errors."""
    df = df.copy()
    for c in DIG_TARGETS:
        df[f"bad_{c}"] = False
    # (1) FOS/TAC identical to VA/Alk on the same day; fraction identities remain unresolved
    same = (df["FOS"] == df["VA"]) & df["FOS"].notna()
    df.loc[same, "bad_FOS"] = True
    same = (df["TAC"] == df["Alk"]) & df["TAC"].notna()
    df.loc[same, "bad_TAC"] = True
    # (2) TAC sequence equal to an earlier non-overlapping sequence (run of >=3 values)
    lab = df["TAC"].dropna().round(4)
    vals, idx = lab.values, lab.index.values
    for a in range(len(vals)):
        for b in range(a):
            run = 0
            while a + run < len(vals) and b + run < a and vals[a + run] == vals[b + run]:
                run += 1
            if run >= 3:
                df.loc[idx[a:a + run], "bad_TAC"] = True
    # (3) non-integer VA/Alk values within otherwise predominantly integer records
    for c in ["VA", "Alk"]:
        frac = (df[c] % 1).abs() > 1e-6
        df.loc[frac & df[c].notna(), f"bad_{c}"] = True
    return df


def audit_hsw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in HSW_TARGETS:
        df[f"bad_{c}"] = False
    return df


def load_daily(stream: str) -> pd.DataFrame:
    fn = "06_DECtoMAY_Muscatine_Digestate.csv" if stream == "dig" else "08_DECtoMAY_Muscatine_HSW.csv"
    df = pd.read_csv(ROOT / fn, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = (df["Date"].dt.normalize() - pd.Timestamp("2023-12-20")).dt.days
    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------- preprocessing
class SpecPrep(BaseEstimator, TransformerMixin):
    """Row-wise spectral preprocessing with no learned cross-sample state."""

    def __init__(self, kind="sg2", window=15, poly=2):
        self.kind, self.window, self.poly = kind, window, poly

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if self.kind == "raw":
            return X
        if self.kind == "abs":  # pseudo-absorbance
            return np.log10(1.0 / np.clip(X, 1e-6, None))
        if self.kind == "snv":
            return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        if self.kind in ("sg1", "sg2"):
            d = 1 if self.kind == "sg1" else 2
            return savgol_filter(X, self.window, self.poly, deriv=d, axis=1)
        if self.kind in ("snv_sg1", "snv_sg2"):
            Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
            d = 1 if self.kind.endswith("1") else 2
            return savgol_filter(Z, self.window, self.poly, deriv=d, axis=1)
        if self.kind == "abs_sg1":
            Z = np.log10(1.0 / np.clip(X, 1e-6, None))
            return savgol_filter(Z, self.window, self.poly, deriv=1, axis=1)
        raise ValueError(self.kind)


# ---------------------------------------------------------------- splits & metrics
def kennard_stone(X, n_train):
    X = np.asarray(X, float)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    sel = [i, j]
    rest = set(range(len(X))) - set(sel)
    while len(sel) < n_train:
        r = np.array(sorted(rest))
        mind = d[np.ix_(r, sel)].min(1)
        k = r[np.argmax(mind)]
        sel.append(k)
        rest.remove(k)
    return np.array(sel)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    res = y - p
    rmse = np.sqrt(np.mean(res**2))
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(res**2) / ss if ss > 0 else np.nan
    return dict(R2=r2, RMSE=rmse, NRMSE=100 * rmse / np.mean(y), MAE=np.mean(np.abs(res)),
                RPD=np.std(y, ddof=1) / rmse if rmse > 0 else np.nan, n=len(y))
