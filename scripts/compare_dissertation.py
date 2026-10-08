"""
Phase 1 acceptance check on the real 2015 data: compare this pipeline's
dissertation-multiseed runs against the CSVs the dissertation scripts wrote.

    python scripts/compare_dissertation.py --legacy-dir <dissertation output folder>

Reads multiseed_full_metrics.csv (multiseed_full_metrics.py: AE x5, CNN x3,
LSTM x3, range-F1-tuned) and, if present, baseline_results.csv
(baselines.py: k-means and IsolationForest, seed 42). Prints every
(model, seed, metric) pair and exits non-zero if any differ by more than
--tol. Expect exact agreement only in the same software environment the
dissertation used; a different torch or sklearn version can move results,
which is itself worth knowing.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from icsad.store import ResultsStore

MULTISEED_METRICS = ['point_f1', 'range_f1', 'range_fbeta_3', 'range_fbeta_1_3', 'numenta', 'na_early']
BASELINE_METRICS = ['point_precision', 'point_recall', 'point_f1', 'range_precision',
                    'range_recall', 'range_f1', 'numenta', 'na_early']
MODEL_NAMES = {'AE': 'ae', 'CNN': 'cnn', 'LSTM': 'lstm', 'kmeans': 'kmeans',
               'isolation_forest': 'iforest'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy-dir', required=True)
    ap.add_argument('--store', default='results/results.sqlite')
    ap.add_argument('--experiment', default='dissertation-multiseed')
    ap.add_argument('--tol', type=float, default=1e-9)
    args = ap.parse_args()

    new = ResultsStore(args.store).results(args.experiment, 'mean', 'range_f1')
    if new.empty:
        sys.exit(f'no results for {args.experiment!r} in {args.store}')
    new = new.set_index(['model', 'seed'])

    legacy_dir = Path(args.legacy_dir)
    rows = []
    ms = pd.read_csv(legacy_dir / 'multiseed_full_metrics.csv')
    for _, r in ms.iterrows():
        key = (MODEL_NAMES[r['model']], int(r['seed']))
        for m in MULTISEED_METRICS + ['w']:
            rows.append((*key, m, r[m], new.loc[key, m] if key in new.index else float('nan')))
    bl_path = legacy_dir / 'baseline_results.csv'
    if bl_path.exists():
        bl = pd.read_csv(bl_path, index_col=0)
        for name, r in bl.iterrows():
            key = (MODEL_NAMES[name], 42)
            for m in BASELINE_METRICS:
                rows.append((*key, m, r[m], new.loc[key, m] if key in new.index else float('nan')))

    df = pd.DataFrame(rows, columns=['model', 'seed', 'metric', 'dissertation', 'new'])
    df['abs_diff'] = (df['new'] - df['dissertation']).abs()
    df['match'] = df['abs_diff'] <= args.tol
    with pd.option_context('display.max_rows', None, 'display.width', 160):
        print(df.to_string(index=False, float_format='{:.6f}'.format))
    n_bad = int((~df['match']).sum())
    print(f'\n{len(df) - n_bad}/{len(df)} values match within {args.tol}')
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
