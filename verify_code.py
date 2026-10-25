"""Check syntax, RMST horizon and optional source-label coverage without model fitting."""
from pathlib import Path
import argparse
import ast
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def extracted_functions(file, names, extras=None):
    tree = ast.parse(file.read_text(encoding='utf8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    namespace = {'np': np, **(extras or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Also read source files and check screened label counts.')
    args = parser.parse_args()
    files = list((ROOT / 'analysis').glob('*.py')) + list(ROOT.glob('*.py'))
    for file in files:
        ast.parse(file.read_text(encoding='utf8'), filename=str(file))
    namespace = extracted_functions(ROOT / 'analysis/exp04_lifetime.py', ['km', 'rmst'])
    rmst = namespace['rmst']
    # A calibration still surviving beyond the horizon must contribute exactly
    # 28 days, not 29. An event at day 4 contributes exactly four days.
    assert rmst([40], [0], 28) == 28
    assert rmst([4], [1], 28) == 4
    assert rmst([28], [1], 28) == 28
    assert rmst([0], [1], 28) == 0
    sampler = extracted_functions(ROOT / 'analysis/exp10_skill_ci.py', ['block_boot'], {'BLOCK': 7})['block_boot']
    draw = sampler(np.array([0, 1, 8, 13, 21]), np.random.default_rng(0))
    assert len(draw) == 5 and np.all((draw >= 0) & (draw < 5))
    out = {'syntax_files': len(files), 'rmst_horizon_cases': 4,
           'circular_block_index_check': 'passed', 'model_refitting': False}
    if args.data_dir is not None:
        os.environ['IOWA_DRS_DATA_DIR'] = str(args.data_dir.expanduser().resolve())
        sys.path.insert(0, str(ROOT / 'analysis'))
        import common
        expected = {'dig': {'TS': 140, 'VS': 140, 'VA': 73, 'Alk': 75, 'FOS': 40, 'TAC': 48},
                    'hsw': {'TS': 121, 'VS': 120, 'COD': 42, 'VFA': 42, 'Fat': 41, 'Protein': 25, 'Carbs': 25}}
        actual = {}
        for stream, targets in expected.items():
            frame = common.load_stream(stream)
            frame = common.audit_digestate(frame) if stream == 'dig' else common.audit_hsw(frame)
            actual[stream] = {target: int((frame[target].notna() & ~frame['bad_' + target]).sum()) for target in targets}
        assert actual == expected, (actual, expected)
        out['screened_label_counts'] = actual
        out['portable_data_path'] = common.ROOT == args.data_dir.expanduser().resolve()
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
