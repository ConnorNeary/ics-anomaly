"""
Command line. Run from the project root (paths in configs are relative to it).

  python -m icsad prepare-swat2015 --raw-dir <folder with the 2015 .xlsx files>
  python -m icsad run configs/dissertation_multiseed.yaml
  python -m icsad retune --experiment dissertation-multiseed --tuning-metric point_f1
  python -m icsad summary --experiment dissertation-multiseed
"""

import argparse
import logging
import os
import sys

DEFAULT_STORE = 'results/results.sqlite'
SUMMARY_METRICS = ('range_f1', 'point_f1', 'false_alarms', 'attacks_detected')


def _prepare_swat2015(args):
    import pandas as pd

    from .data import read_feature_list
    from .data import swat2015
    from .data.features import KS_THRESHOLD, PAPER_REPORTED_DROPS_SWAT, ks_feature_table

    out = swat2015.build(args.raw_dir, args.out_dir)
    normal = pd.read_parquet(out / 'normal.parquet')
    attack = pd.read_parquet(out / 'attack.parquet')
    table = ks_feature_table(normal, attack, read_feature_list(out / 'feature_list.txt'), args.ks_threshold)
    table.to_csv(out / 'ks_feature_selection.csv', index=False)
    kept = table.loc[~table['drop'], 'feature'].tolist()
    (out / 'feature_list_ks_filtered.txt').write_text('\n'.join(kept), encoding='utf-8')
    n_drop = int(table['drop'].sum())
    print(f'K-S selection: dropped {n_drop} of {len(table)} features (ks_star > {args.ks_threshold}): '
          f'{table.loc[table["drop"], "feature"].tolist()}')
    if args.ks_threshold == KS_THRESHOLD and n_drop != PAPER_REPORTED_DROPS_SWAT:
        print(f'note: the paper reports dropping {PAPER_REPORTED_DROPS_SWAT}; threshold not adjusted.')
    print(f'wrote {out}')


def _run(args):
    from .config import load_config
    from .data import load_dataset
    from .pipeline import run_experiment
    from .store import ResultsStore

    cfg = load_config(args.config)
    if cfg.runtime.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before CUDA initialises
    store = ResultsStore(args.store or cfg.runtime.store)
    run_experiment(cfg, load_dataset(cfg.data), store, only_models=args.models)
    _print_summary(store, cfg.experiment, cfg.detection.aggregation, cfg.detection.tuning_metric,
                   SUMMARY_METRICS)


def _retune(args):
    from .evaluation import evaluate_run
    from .store import ResultsStore

    store = ResultsStore(args.store)
    runs = store.runs(args.experiment)
    if not runs:
        sys.exit(f'no finished runs for experiment {args.experiment!r} in {args.store}')
    for run in runs:
        if args.models and run['model'] not in args.models:
            continue
        try:
            evaluate_run(store, run['run_id'], args.aggregation, args.tuning_metric)
        except ValueError as exc:  # e.g. IsolationForest has no per-feature errors
            logging.warning('%s: %s', run['run_id'], exc)
    _print_summary(store, args.experiment, args.aggregation, args.tuning_metric, SUMMARY_METRICS)


def _summary(args):
    from .store import ResultsStore

    _print_summary(ResultsStore(args.store), args.experiment, args.aggregation,
                   args.tuning_metric, args.metrics)


def _print_summary(store, experiment, aggregation, tuning_metric, metric_names):
    import pandas as pd

    table = store.summary(metric_names, experiment, aggregation, tuning_metric)
    if table.empty:
        print('no results')
        return
    print(f'\nTest-partition results: score aggregation={aggregation}, tau/w tuned for {tuning_metric} '
          f'on attack-val. Distribution across seeds; count = number of seeds.')
    with pd.option_context('display.width', 200, 'display.max_columns', None,
                           'display.float_format', '{:.4f}'.format):
        for name in metric_names:
            print(f'\n{name}')
            print(table[name].to_string())


def main(argv=None):
    from .evaluation import TUNING_METRICS

    parser = argparse.ArgumentParser(prog='icsad')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('prepare-swat2015', help='build parquet tables + feature lists from the 2015 workbooks')
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--out-dir', default='data/swat2015')
    p.add_argument('--ks-threshold', type=float, default=50)
    p.set_defaults(func=_prepare_swat2015)

    p = sub.add_parser('run', help='run every (model, seed) in a config; finished runs are skipped')
    p.add_argument('config')
    p.add_argument('--store', help='override runtime.store')
    p.add_argument('--models', nargs='+', help='only these model names')
    p.set_defaults(func=_run)

    for name, func, helptext in [('retune', _retune, 're-evaluate cached runs, no retraining'),
                                 ('summary', _summary, 'print distributions across seeds')]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument('--store', default=DEFAULT_STORE)
        p.add_argument('--experiment', required=(name == 'retune'))
        p.add_argument('--tuning-metric', default='range_f1', choices=sorted(TUNING_METRICS))
        p.add_argument('--aggregation', default='mean', help="'mean', 'max' or 'top<k>'")
        if name == 'retune':
            p.add_argument('--models', nargs='+')
        else:
            p.add_argument('--metrics', nargs='+', default=list(SUMMARY_METRICS))
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    args.func(args)


if __name__ == '__main__':
    main()
