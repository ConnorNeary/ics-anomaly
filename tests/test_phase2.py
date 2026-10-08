"""The pre-registered analysis, tested before it ever sees real results."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from icsad import phase2
from icsad.pipeline import run_one
from icsad.store import ResultsStore


@pytest.mark.parametrize('mean_diff,ci,fa_mean,fa_top3,expected', [
    (1.5, (0.5, 2.5), 10, 11, 'supported'),
    (1.5, (0.5, 2.5), 10, 20, 'not supported'),     # guard fails: 20 > max(12.5, 12)
    (0.2, (-0.3, 0.8), 10, 10, 'not supported'),    # interval excludes +1
    (0.8, (0.1, 1.6), 10, 10, 'unresolved'),        # below +1 but +1 not excluded
    (1.2, (-0.2, 2.6), 10, 10, 'unresolved'),       # interval includes 0
    (1.0, (0.2, 1.8), 0, 2, 'supported'),           # small counts: +2 allowance
])
def test_h1_decision_rule(mean_diff, ci, fa_mean, fa_top3, expected):
    assert phase2.h1_verdict(mean_diff, ci, fa_mean, fa_top3)[0] == expected


def test_bootstrap_is_reproducible_and_exact_when_no_variation():
    diff = {'ae': np.ones((10, 14), dtype=int), 'cnn': np.ones((10, 14), dtype=int)}
    pooled, per_model = phase2.bootstrap_paired(diff, n_boot=200)
    assert np.all(pooled == 14) and np.all(per_model['ae'] == 14)
    noisy = {'ae': np.random.default_rng(1).integers(-1, 2, (10, 14))}
    a, _ = phase2.bootstrap_paired(noisy, n_boot=200)
    b, _ = phase2.bootstrap_paired(noisy, n_boot=200)
    np.testing.assert_array_equal(a, b)


def test_threshold_comes_from_benign_scores_only():
    benign = np.linspace(0, 1, 10_001)                 # q=0.9995 -> tau ~0.9995
    y = np.zeros(200, dtype=int)
    y[50:80] = 1
    y[120:150] = 1
    scores = np.zeros(200)
    scores[60:75] = 2.0                                # first attack crosses tau for 15 >= w steps
    scores[125:130] = 2.0                              # second only 5 < w steps
    scores[170:185] = 2.0                              # 15 steps outside any attack -> 1 false alarm
    detected, fa = phase2.detect_by_attack(benign, scores, y)
    assert detected.tolist() == [1, 0]
    assert fa == 1


def _run(labels_av, labels_te, raw_av, raw_te, s_av, s_te, benign):
    sc = {'benign_val': benign, 'attack_val': s_av, 'test': s_te}
    return phase2.RunArrays('r', 'ae', 1, {'mean': sc}, {'attack_val': labels_av, 'test': labels_te},
                            {'attack_val': raw_av, 'test': raw_te})


def test_h2_excludes_ten_minutes_after_each_attack():
    n = 3_000
    lab = np.zeros(n, dtype=int)
    lab[100:200] = 1
    s = np.zeros(n)
    s[200:800] = 5.0                       # high scores only in the 600 rows after the attack
    run = _run(lab[:1_000], lab[1_000:], np.arange(1_000), np.arange(1_000, n),
               s[:1_000], s[1_000:], np.linspace(0, 1, 10_000))
    np.testing.assert_array_equal(phase2._attack_ends(run), [199])
    out = {(x['q'], x['rows']): x for x in phase2.h2([run])['rows']}
    assert out[(0.9995, 'recovery excluded')]['realised'] == 0.0
    assert out[(0.9995, 'no exclusion')]['realised'] == pytest.approx(600 / (n - 100))


@pytest.mark.slow
def test_analysis_script_end_to_end_on_synthetic(make_cfg, synthetic, tmp_path):
    cfg = make_cfg([{'name': 'ae', 'seeds': [1, 2]}, {'name': 'cnn', 'params': {'history': 30}, 'seeds': [1, 2]}],
                   experiment='phase2-seeds')
    store = ResultsStore(tmp_path / 'r.sqlite')
    for m in cfg.models:
        for seed in m.seeds:
            run_one(cfg, m, seed, synthetic, store)
    store.close()

    path = Path(__file__).resolve().parents[1] / 'scripts' / 'phase2_analysis.py'
    spec = importlib.util.spec_from_file_location('phase2_analysis', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / 'phase2'
    script.main(['--store', str(tmp_path / 'r.sqlite'), '--out', str(out)])

    report = (out / 'report.md').read_text(encoding='utf-8')
    assert any(v in report for v in ('Verdict: supported', 'Verdict: not supported', 'Verdict: unresolved'))
    for name in ('h1_per_run.csv', 'h2_summary.csv', 'tradeoff_curve_per_run.csv', 'h1_summary.json'):
        assert (out / name).exists()
