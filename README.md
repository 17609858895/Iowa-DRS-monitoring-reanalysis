# Temporal and spectral information in anaerobic codigestion monitoring

Core numerical analysis code accompanying **Spectral information and calibration lifetime in full-scale anaerobic codigestion monitoring**. Code contact: Chong Liu.

The study distinguishes predictive information beyond sampling date, temporal calibration stability, and estimation between laboratory measurements. The repository contains the essential numerical analysis, not the manuscript, raw data, precomputed results, publication artwork, or model weights. It does not provide an externally validated monitoring service.

## Public source data

Kramin, Z. A. M. and Just, C. L. (2025), *Dataset and Code for Rapid In-line Monitoring of Full-Scale Anaerobic Co-Digestion Using Diffuse Reflectance Spectroscopy and Machine Learning*.

- Dataset DOI: https://doi.org/10.25820/data.008042
- Official record: https://iro.uiowa.edu/esploro/outputs/dataset/9984948241202771
- Associated source paper: https://doi.org/10.1021/acs.energyfuels.5c04529

The repository manifest fixes the 21 public attachments used for this analysis by filename, stable official download URL, size and locally recorded SHA-256. These hashes are verification records for the downloaded snapshot, not publisher-supplied checksums. The raw source files are not redistributed here. Read their official README and cite the source dataset and paper. See [NOTICE.md](NOTICE.md) for attribution and licensing scope.

## Installation and input check

Use Python 3.11 or 3.12 and the versions in `requirements.txt`. The numerical environment was checked with NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1 and scikit-learn 1.5.1. LightGBM is used by the benchmark and imported by the shared model factory.

```text
python -m pip install -r requirements.txt
python download_public_data.py
python verify_code.py --data-dir data/raw
```

`download_public_data.py` stores official attachments in `data/raw/`, verifies the recorded sizes and hashes, and refuses an existing mismatched file. To verify an already downloaded source directory without copying or downloading:

```text
python download_public_data.py --check-existing PATH_TO_PUBLIC_SOURCE_FILES
```

Inputs default to `data/raw/` and outputs to `results/`; both directories are ignored by Git. Set `IOWA_DRS_DATA_DIR` and `IOWA_DRS_RESULTS_DIR`, or use the run-plan options below, to use other directories. The code does not edit raw input files.

## Reproduce the numerical analyses

First inspect the plan. Commands are printed but no models are fitted unless `--execute` is present:

```text
python run_analysis.py --stage core
python run_analysis.py --stage core --execute
```

The complete run includes repeated cross-validation and bootstrap/simulation experiments and can take a long time. Each stage can be run separately, in this order:

| Stage | Scientific output | Main source |
|---|---|---|
| `benchmark` | Learner sensitivity under Kennard–Stone and temporal-block evaluation | `exp01_benchmark.py` |
| `calendar` | Date-only, spectral and calendar-adjusted prediction across partitions | `exp03_dmlcal.py` |
| `displacement` | Local wrong-day spectral replacement | `exp07_wrongday.py` |
| `skill` | Averaged out-of-fold predictions, date displacement and incremental R² intervals | `exp10_skill_ci.py`, `exp10b_dR2.py` |
| `lifetime` | Frozen predictions, tolerance sensitivity, 28-day RMST and paired cutoff intervals | `exp11_lifetime_v2.py`, `exp04_lifetime.py` |
| `schedule` | Forward-only retained-laboratory schedules and normalized errors | `exp12_pareto.py` |
| `supplementary` | Preprocessing, component ablation, label sensitivity and recovered published test days | `exp13_si.py` |
| `synthetic` | Known added spectral signal, detection frequencies and perturbation diagnostics | `exp17_power.py` |

For example:

```text
python run_analysis.py --stage lifetime --data-dir PATH_TO_PUBLIC_SOURCE_FILES --results-dir results --execute
```

The `skill` stage must finish before `exp10b_dR2.py`, which reads the saved out-of-fold predictions. Part-file collectors are included by the runner after their fitting jobs. The E10 script skips a target whose completed skill part exists; use an empty output directory for a fresh rerun. For target-specific execution, consult the CLI in `exp10_skill_ci.py`, `exp12_pareto.py`, `exp13_si.py` and `exp17_power.py`.

Optional stages are `raw-benchmark`, `exploratory` (retrospective load associations and greybox propagation), `figure-inputs` (numerical spectral profiles) and `legacy-online` (an earlier anchored-calibration analysis). They preserve related analyses and negative results but are not all prerequisites for the primary endpoints. Publication figure layouts, manually coded literature context, formatted supplementary tables and frozen paper outputs remain in the manuscript's local delivery package; this is a code-only numerical release.

## Definitions and analysis boundaries

- Primary incremental R² is `(MSE_date - MSE_model) / Var(y)`, using complete target-specific evaluation-set variance. The older CO ratio in the E10 output is auxiliary. The reported incremental intervals come from `exp10b_dR2.py`.
- Circular seven-calendar-day block resampling holds the averaged out-of-fold predictions fixed. Its pointwise intervals do not include model refitting or selection uncertainty and are not corrected for simultaneous comparisons.
- The RMST helper sums survival at integer days 0–27 for the interval [0, 28]. A completely surviving calibration therefore contributes 28 days. Its cutoff-bootstrap differences are approximate because cutoffs share follow-up observations.
- The released digestate timestamp in 2021 is provisionally aligned to 2023, with the original date flag retained in memory. A saved full drop-row sensitivity is not supplied. Numerical pattern flags are not confirmed transcription or analytical errors. Literal VA/Alk/FOS/TAC labels are retained because source sample-fraction descriptions conflict.
- FOS has 11 recovered published test days versus 14 reported test samples; it is a partial reconstruction. The refitted PLS is not an exact replication of every original preprocessing choice.
- Interior temporal blocks can have training observations on both sides. They assess transport across omitted time segments; the separate laboratory-schedule analysis uses earlier retained observations only. Its retained-reference fraction is computed over observed reference days after warm-up. Normalized schedule error uses the full target-series mean.
- Spectral GPR supplies the feed-solids incremental-skill result but is not included in the laboratory-schedule experiment. That experiment compares persistence, date-only GPR, spectral PLS and anchored calendar-adjusted calibration; it does not determine the schedule-specific benefit of the spectral GPR increment.
- Load/greybox analyses use retrospective filling and interpolation. No causal intervention, methane-production improvement, independent-facility validation or deployment claim follows from these computations.

## Verification and provenance

`verify_code.py` checks every Python file's syntax, four boundary cases that catch a 29-day RMST off-by-one error, circular-block index validity and optional screened target counts. It does not refit models. The released numerical routines retain the original analysis implementation; input/output configuration and explanatory comments were adapted for a standalone repository. File-level origin and release hashes are in `provenance/code_origin.json`.

No raw data, stored predictions, frozen scientific results, manuscripts, supplementary documents, private credentials or generated figures are committed. Numerical outputs are regenerated locally from the public source files by the explicit execution commands above. Exact floating-point output can depend on the stated dependency versions and computational environment.
