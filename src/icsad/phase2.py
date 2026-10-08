"""
Pre-registered Phase 2 analysis (prereg/phase2.md). Every constant below is
fixed by that document; changing one is a deviation and must be reported as
such.

H1: top3 vs mean aggregation, attacks detected on 2015 attack-val at a
    threshold set from benign-val only, paired per (model, seed), with a
    seed-and-attack bootstrap and a false-alarm guard.
H2: realised exceedance rate on the attack workbook's normal rows versus the
    nominal 1 - q, with and without a post-attack recovery window.

Attack-val labels are used only to count outcomes, never to choose anything.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import aggregation as agg
from .detection import apply_threshold_rule
from .metrics import false_positive_segments, to_segments

MODELS = ('ae', 'cnn', 'lstm', 'kmeans')
BASELINE, PRIMARY = 'mean', 'top3'
EXPLORATORY = ('max', 'top2', 'top5')
Q_PRIMARY, W = 0.9995, 10
Q_CURVE = (0.99, 0.995, 0.999, 0.9995, 0.9999)
MIN_IMPROVEMENT = 1          # attacks, i.e. 1/14 range recall
FA_RATIO, FA_ABS = 1.25, 2   # guard: top3 FA <= max(1.25 x mean FA, mean FA + 2)
H2_FACTOR, H2_MIN_FRACTION = 2.0, 0.8
RECOVERY_ROWS = 600          # 10 minutes at 1 Hz
N_BOOT, BOOT_SEED = 10_000, 0
PARTITIONS = ('benign_val', 'attack_val', 'test')


@dataclass
class RunArrays:
    run_id: str
    model: str
    seed: int
    scores: dict      # aggregation -> partition -> (n,) scores
    labels: dict      # 'attack_val' / 'test' -> (n,) 0/1
    raw_idx: dict     # 'attack_val' / 'test' -> (n,) attack-workbook row


def load_run(run, aggregations):
    d = Path(run['artifact_dir'])
    per = {p: np.load(d / f'perfeat_{p}.npy') for p in PARTITIONS}
    norm = agg.fit_normaliser(per['benign_val'])
    scores = {'mean': {p: np.load(d / f'scores_{p}.npy') for p in PARTITIONS}}
    for a in aggregations:
        if a != 'mean':
            scores[a] = {p: agg.aggregate(per[p], a, norm) for p in PARTITIONS}
    return RunArrays(run['run_id'], run['model'], int(run['seed']), scores,
                     {p: np.load(d / f'labels_{p}.npy').astype(int) for p in ('attack_val', 'test')},
                     {p: np.load(d / f'rawidx_{p}.npy') for p in ('attack_val', 'test')})


def detect_by_attack(scores_benign, scores_eval, y_eval, q=Q_PRIMARY, w=W):
    """Threshold from benign-val only. Returns (per-attack 0/1 detection
    vector in segment order, false-alarm event count)."""
    tau = float(np.quantile(scores_benign, q))
    y_pred = apply_threshold_rule(scores_eval, tau, w)
    starts, ends = to_segments(y_eval)
    detected = np.array([int(y_pred[s:e + 1].any()) for s, e in zip(starts, ends)], dtype=int)
    return detected, int(false_positive_segments(y_eval, y_pred))


def bootstrap_paired(diff_by_model, n_boot=N_BOOT, seed=BOOT_SEED):
    """diff_by_model: model -> (n_seeds, n_attacks) array of per-attack
    detection differences (top3 - mean). Resamples seeds within each model
    and attacks (shared across models, since they are the same attacks).
    Returns (pooled_stats, per_model_stats), each an array of n_boot
    bootstrap means of attacks-detected difference per run."""
    rng = np.random.default_rng(seed)
    models = sorted(diff_by_model)
    n_attacks = next(iter(diff_by_model.values())).shape[1]
    per_model = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        attacks = rng.integers(0, n_attacks, n_attacks)
        for m in models:
            D = diff_by_model[m]
            seeds = rng.integers(0, D.shape[0], D.shape[0])
            per_model[m][b] = D[seeds][:, attacks].sum(axis=1).mean()
    pooled = np.mean([per_model[m] for m in models], axis=0)
    return pooled, per_model


def h1_verdict(mean_diff, ci, fa_mean, fa_top3):
    """Decision rule from prereg section 5 (H1)."""
    guard = fa_top3 <= max(FA_RATIO * fa_mean, fa_mean + FA_ABS)
    lo, hi = ci
    if mean_diff >= MIN_IMPROVEMENT and lo > 0 and guard:
        return 'supported', guard
    if hi < MIN_IMPROVEMENT or not guard:
        return 'not supported', guard
    return 'unresolved', guard


def h1(runs, q=Q_PRIMARY, w=W):
    """runs: list of RunArrays (all models, all seeds)."""
    rows = []
    diffs = {}
    for r in runs:
        y = r.labels['attack_val']
        out = {}
        for a in (BASELINE, PRIMARY) + EXPLORATORY:
            det, fa = detect_by_attack(r.scores[a]['benign_val'], r.scores[a]['attack_val'], y, q, w)
            out[a] = (det, fa)
            rows.append({'model': r.model, 'seed': r.seed, 'aggregation': a,
                         'detected': int(det.sum()), 'n_attacks': len(det), 'false_alarms': fa})
        diffs.setdefault(r.model, []).append(out[PRIMARY][0] - out[BASELINE][0])
    diff_by_model = {m: np.vstack(v) for m, v in diffs.items()}
    models = sorted(diff_by_model)
    pooled_boot, model_boot = bootstrap_paired(diff_by_model)

    def mean_fa(model, a):
        return float(np.mean([x['false_alarms'] for x in rows
                              if x['model'] == model and x['aggregation'] == a]))

    def result(mean_diff, boot, fa_mean, fa_top3, n_seeds):
        ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
        verdict, guard = h1_verdict(mean_diff, ci, fa_mean, fa_top3)
        return {'mean_diff_attacks': mean_diff, 'ci95': ci, 'fa_mean': fa_mean, 'fa_top3': fa_top3,
                'fa_guard_passed': guard, 'verdict': verdict, 'n_seeds': n_seeds,
                'n_attacks': int(next(iter(diff_by_model.values())).shape[1])}

    # per model: mean over seeds of (attacks detected under top3 - under mean)
    model_diff = {m: float(diff_by_model[m].sum(axis=1).mean()) for m in models}
    per_model = {m: result(model_diff[m], model_boot[m], mean_fa(m, BASELINE), mean_fa(m, PRIMARY),
                           int(diff_by_model[m].shape[0]))
                 for m in models}
    # pooled: equal weight per model (model as stratum)
    pooled = result(float(np.mean([model_diff[m] for m in models])), pooled_boot,
                    float(np.mean([mean_fa(m, BASELINE) for m in models])),
                    float(np.mean([mean_fa(m, PRIMARY) for m in models])),
                    {m: int(diff_by_model[m].shape[0]) for m in models})
    return {'pooled': pooled, 'per_model': per_model, 'rows': rows}


def describe_test_partition(runs, attacks):
    """Secondary, descriptive only: which 2015 test attacks each aggregation
    detects at the primary operating point. These are the attacks that
    generated H1. attacks: list of swat2015.Attack (start/end are workbook rows)."""
    rows = []
    for r in runs:
        y, idx = r.labels['test'], r.raw_idx['test']
        starts, _ = to_segments(y)
        meta = []
        for s in starts:
            row = idx[s]
            match = next((a for a in attacks if a.start <= row < a.end), None)
            meta.append((match.attack_id, '/'.join(match.targets)) if match else (None, None))
        for a in (BASELINE, PRIMARY) + EXPLORATORY:
            det, _ = detect_by_attack(r.scores[a]['benign_val'], r.scores[a]['test'], y)
            for (attack_id, targets), d in zip(meta, det):
                rows.append({'model': r.model, 'seed': r.seed, 'aggregation': a,
                             'attack_id': attack_id, 'targets': targets, 'detected': int(d)})
    return rows


def tradeoff_curve(runs, w=W):
    """Secondary: mean attacks detected and false alarms per (model,
    aggregation, q), benign-val thresholds only."""
    rows = []
    for r in runs:
        for a in (BASELINE, PRIMARY) + EXPLORATORY:
            for q in Q_CURVE:
                det, fa = detect_by_attack(r.scores[a]['benign_val'], r.scores[a]['attack_val'],
                                           r.labels['attack_val'], q, w)
                rows.append({'model': r.model, 'seed': r.seed, 'aggregation': a, 'q': q,
                             'detected': int(det.sum()), 'false_alarms': fa})
    return rows


def _attack_ends(run):
    """Workbook rows at which each labelled attack ends, from both partitions."""
    idx = np.concatenate([run.raw_idx['attack_val'], run.raw_idx['test']])
    lab = np.concatenate([run.labels['attack_val'], run.labels['test']])
    order = np.argsort(idx)
    idx, lab = idx[order], lab[order]
    is_end = (lab == 1) & np.append((lab[1:] == 0) | (np.diff(idx) > 1), True)
    return idx[is_end]


def h2(runs):
    """Realised exceedance on normal rows vs nominal, mean aggregation."""
    rows = []
    for r in runs:
        ends = _attack_ends(r)
        idx = np.concatenate([r.raw_idx['attack_val'], r.raw_idx['test']])
        lab = np.concatenate([r.labels['attack_val'], r.labels['test']])
        s = np.concatenate([r.scores['mean']['attack_val'], r.scores['mean']['test']])
        in_recovery = np.zeros(len(idx), dtype=bool)
        for e in ends:
            in_recovery |= (idx > e) & (idx <= e + RECOVERY_ROWS)
        for q in Q_CURVE:
            tau = float(np.quantile(r.scores['mean']['benign_val'], q))
            nominal = 1 - q
            for excluded, mask in (('recovery excluded', (lab == 0) & ~in_recovery),
                                   ('no exclusion', lab == 0)):
                realised = float((s[mask] > tau).mean())
                rows.append({'model': r.model, 'seed': r.seed, 'q': q, 'rows': excluded,
                             'nominal': nominal, 'realised': realised,
                             'within_factor': nominal / H2_FACTOR <= realised <= nominal * H2_FACTOR})
    summary = []
    for model in sorted({x['model'] for x in rows}):
        for q in Q_CURVE:
            for excluded in ('recovery excluded', 'no exclusion'):
                sel = [x for x in rows if x['model'] == model and x['q'] == q and x['rows'] == excluded]
                frac = float(np.mean([x['within_factor'] for x in sel]))
                summary.append({'model': model, 'q': q, 'rows': excluded, 'nominal': 1 - q,
                                'realised_median': float(np.median([x['realised'] for x in sel])),
                                'realised_min': float(min(x['realised'] for x in sel)),
                                'realised_max': float(max(x['realised'] for x in sel)),
                                'seeds_within': f"{sum(x['within_factor'] for x in sel)}/{len(sel)}",
                                'holds': frac >= H2_MIN_FRACTION})
    return {'rows': rows, 'summary': summary}
