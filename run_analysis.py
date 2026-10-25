"""Print the analysis plan; execute it only when --execute is explicitly supplied."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DIG = ['TS', 'VS', 'VA', 'Alk', 'FOS', 'TAC']
HSW = ['TS', 'VS', 'COD', 'VFA', 'Fat', 'Protein', 'Carbs']
KEYS = ['dig:' + t for t in DIG] + ['hsw:' + t for t in HSW]


def plan(stage):
    groups = {
        'benchmark': [('exp01_benchmark.py', 'audited')],
        'raw-benchmark': [('exp01_benchmark.py', 'raw')],
        'calendar': [('exp03_dmlcal.py',)],
        'displacement': [('exp07_wrongday.py',)],
        'skill': [('exp10_skill_ci.py',), ('exp10_skill_ci.py', 'collect'), ('exp10b_dR2.py',)],
        'lifetime': [('exp11_lifetime_v2.py',)],
        'schedule': [('exp12_pareto.py',), ('exp12_pareto.py', 'collect')],
        'supplementary': [( 'exp13_si.py', mode, key)
                          for mode in ['preproc', 'ablate', 'repro'] for key in KEYS]
                         + [('exp13_si.py', 'labels', 'dig:' + t) for t in ['VA', 'Alk', 'FOS', 'TAC']]
                         + [('exp13_si.py', 'collect')],
        'synthetic': [('exp17_power.py', key) for key in ['dig:Alk', 'dig:TS', 'hsw:TS']]
                     + [('exp17_power.py', 'collect')],
        'exploratory': [('exp05_hsw_load_dml.py',), ('exp06_greybox.py',)],
        'figure-inputs': [('fig_inputs.py',)],
        'legacy-online': [('exp02_anchored.py',)],
    }
    if stage == 'core':
        return [job for name in ['benchmark', 'calendar', 'displacement', 'skill', 'lifetime',
                                 'schedule', 'supplementary', 'synthetic'] for job in groups[name]]
    return groups[stage]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='core', choices=['core', 'benchmark', 'raw-benchmark', 'calendar',
                        'displacement', 'skill', 'lifetime', 'schedule', 'supplementary', 'synthetic',
                        'exploratory', 'figure-inputs', 'legacy-online'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--execute', action='store_true', help='Refit models and write numerical outputs.')
    args = parser.parse_args()
    env = os.environ.copy()
    if args.data_dir is not None:
        env['IOWA_DRS_DATA_DIR'] = str(args.data_dir.expanduser().resolve())
    if args.results_dir is not None:
        env['IOWA_DRS_RESULTS_DIR'] = str(args.results_dir.expanduser().resolve())
    for job in plan(args.stage):
        command = [sys.executable, str(ROOT / 'analysis' / job[0]), *job[1:]]
        print(subprocess.list2cmdline(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.execute:
        print('Plan only. Supply --execute to fit models and write results.')


if __name__ == '__main__':
    main()
