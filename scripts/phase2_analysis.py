"""
Run the pre-registered Phase 2 analysis (prereg/phase2.md) over the
phase2-seeds runs in the results store and write a report.

    python scripts/phase2_analysis.py [--store results/results.sqlite] [--out results/phase2]

All constants live in src/icsad/phase2.py and are fixed by the
pre-registration. This script only loads runs, calls it, and writes:
  report.md      verdicts and tables
  *.csv          every per-run number behind them
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from icsad import phase2
from icsad.data.swat2015 import ATTACKS
from icsad.store import ResultsStore


def fmt_ci(ci):
    return f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', default='results/results.sqlite')
    ap.add_argument('--experiment', default='phase2-seeds')
    ap.add_argument('--out', default='results/phase2')
    args = ap.parse_args(argv)

    store = ResultsStore(args.store)
    runs_meta = [r for r in store.runs(args.experiment) if r['model'] in phase2.MODELS]
    if not runs_meta:
        raise SystemExit(f'no finished {args.experiment!r} runs in {args.store}')
    aggs = (phase2.BASELINE, phase2.PRIMARY) + phase2.EXPLORATORY
    runs = [phase2.load_run(r, aggs) for r in runs_meta]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h1 = phase2.h1(runs)
    curve = phase2.tradeoff_curve(runs)
    h2 = phase2.h2(runs)
    test_rows = phase2.describe_test_partition(runs, ATTACKS)

    pd.DataFrame(h1['rows']).to_csv(out / 'h1_per_run.csv', index=False)
    pd.DataFrame(curve).to_csv(out / 'tradeoff_curve_per_run.csv', index=False)
    pd.DataFrame(h2['rows']).to_csv(out / 'h2_per_run.csv', index=False)
    pd.DataFrame(h2['summary']).to_csv(out / 'h2_summary.csv', index=False)
    pd.DataFrame(test_rows).to_csv(out / 'test_partition_descriptive.csv', index=False)
    (out / 'h1_summary.json').write_text(json.dumps({k: v for k, v in h1.items() if k != 'rows'},
                                                    indent=2), encoding='utf-8')

    p = h1['pooled']
    lines = [
        '# Phase 2 results (pre-registered: prereg/phase2.md)', '',
        f'Runs: {args.experiment}, seeds per model {p["n_seeds"]}. Dataset: SWaT Dec 2015.',
        f'Thresholds from benign-val only: q = {phase2.Q_PRIMARY}, w = {phase2.W}.', '',
        '## H1: top3 vs mean, 2015 attack-val', '',
        f'**Verdict: {p["verdict"]}.** Pooled mean paired difference '
        f'{p["mean_diff_attacks"]:+.2f} attacks of {p["n_attacks"]} '
        f'(95% bootstrap interval {fmt_ci(p["ci95"])}; needed >= +{phase2.MIN_IMPROVEMENT} with the '
        f'interval above 0). False alarms per run: mean {p["fa_mean"]:.1f} -> top3 {p["fa_top3"]:.1f} '
        f'(guard {"passed" if p["fa_guard_passed"] else "FAILED"}).', '',
        '| model | seeds | mean diff (attacks) | 95% interval | FA mean | FA top3 | guard | exploratory verdict |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for m, s in h1['per_model'].items():
        lines.append(f'| {m} | {s["n_seeds"]} | {s["mean_diff_attacks"]:+.2f} | {fmt_ci(s["ci95"])} | '
                     f'{s["fa_mean"]:.1f} | {s["fa_top3"]:.1f} | '
                     f'{"pass" if s["fa_guard_passed"] else "fail"} | {s["verdict"]} |')

    per = pd.DataFrame(h1['rows'])
    table = per.groupby(['model', 'aggregation'])[['detected', 'false_alarms']].agg(
        ['mean', 'std', 'min', 'max']).round(2)
    lines += ['', 'Attacks detected and false alarms per run, all aggregations '
              '(max, top2, top5 exploratory):', '', '```', table.to_string(), '```', '',
              '## H2: calibration from normal data, mean aggregation', '',
              f'Holds if realised exceedance is within x{phase2.H2_FACTOR:g} of nominal for at least '
              f'{phase2.H2_MIN_FRACTION:.0%} of seeds.', '', '```',
              pd.DataFrame(h2['summary']).to_string(index=False), '```', '',
              '## Secondary, descriptive: 2015 test partition (generated the hypothesis)', '',
              'Fraction of seeds detecting each test attack, per aggregation:', '', '```']
    desc = pd.DataFrame(test_rows)
    lines.append(desc.pivot_table(index=['attack_id', 'targets', 'model'], columns='aggregation',
                                  values='detected', aggfunc='mean').round(2).to_string())
    lines += ['```', '', 'Trade-off curve data: tradeoff_curve_per_run.csv.']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:12]))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
