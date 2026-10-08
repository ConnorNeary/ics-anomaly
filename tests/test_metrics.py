"""
Unit tests for metrics.py.

Run with:  python -m pytest tests/test_metrics.py
"""

import unittest

import numpy as np
from sklearn.metrics import f1_score

from icsad import metrics


def segment(y, start, length, value=1):
    y[start:start + length] = value


class TestSegments(unittest.TestCase):

    def test_to_segments_basic(self):
        y = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
        starts, ends = metrics.to_segments(y)
        self.assertEqual(starts, [1, 5, 7])
        self.assertEqual(ends, [2, 5, 9])

    def test_to_segments_edges(self):
        y = np.array([1, 1, 0, 1])
        starts, ends = metrics.to_segments(y)
        self.assertEqual(starts, [0, 3])
        self.assertEqual(ends, [1, 3])


class TestRangePrecisionRecall(unittest.TestCase):
    """All calls use (y_true, y_pred) order, matching the module's public API."""

    def test_perfect_detection(self):
        y_true = np.zeros(50, dtype=int)
        segment(y_true, 5, 10)
        segment(y_true, 30, 5)
        y_pred = y_true.copy()
        self.assertEqual(metrics.range_recall(y_true, y_pred), 1.0)
        self.assertEqual(metrics.range_precision(y_true, y_pred), 1.0)
        self.assertEqual(metrics.range_f1(y_true, y_pred), 1.0)

    def test_no_predictions(self):
        y_true = np.zeros(50, dtype=int)
        segment(y_true, 5, 10)
        y_pred = np.zeros(50, dtype=int)
        self.assertEqual(metrics.range_recall(y_true, y_pred), 0.0)
        self.assertEqual(metrics.range_precision(y_true, y_pred), 0.0)
        self.assertEqual(metrics.range_f1(y_true, y_pred), 0.0)

    def test_all_false_positives(self):
        y_true = np.zeros(50, dtype=int)
        segment(y_true, 5, 10)
        y_pred = np.zeros(50, dtype=int)
        segment(y_pred, 20, 3)  # nowhere near the true segment
        self.assertEqual(metrics.range_recall(y_true, y_pred), 0.0)
        self.assertEqual(metrics.range_precision(y_true, y_pred), 0.0)

    def test_partial_overlap_counts_as_full_existence_credit(self):
        # 1 timestep of overlap = full coverage under existence-reward (Sec. 5.2).
        y_true = np.zeros(50, dtype=int)
        segment(y_true, 5, 10)
        y_pred = np.zeros(50, dtype=int)
        segment(y_pred, 14, 1)  # touches only the last point of the segment
        self.assertEqual(metrics.range_recall(y_true, y_pred), 1.0)

    def test_asymmetric_precision_numerator(self):
        # 2 true attacks (1 detected, 1 missed) + 1 stray FP segment.
        # TP=1, FP=1 -> range_precision = 1/2. (Next test shows where this
        # diverges from "correct predicted / total predicted".)
        y_true = np.zeros(60, dtype=int)
        segment(y_true, 5, 5)
        segment(y_true, 30, 5)
        y_pred = np.zeros(60, dtype=int)
        segment(y_pred, 5, 5)     # detects first true attack
        segment(y_pred, 45, 3)    # false alarm, no true overlap
        self.assertEqual(metrics.range_precision(y_true, y_pred), 0.5)
        self.assertEqual(metrics.range_recall(y_true, y_pred), 0.5)

    def test_precision_numerator_uses_true_segment_count_not_predicted_count(self):
        # 1 true attack detected by 3 fragmented predicted segments, no FPs.
        # range_precision = 1/(1+0) = 1.0 despite 3 predicted segments --
        # confirms numerator is true-segment count, not predicted-segment count.
        y_true = np.zeros(60, dtype=int)
        segment(y_true, 10, 20)
        y_pred = np.zeros(60, dtype=int)
        segment(y_pred, 10, 1)
        segment(y_pred, 15, 1)
        segment(y_pred, 29, 1)
        self.assertEqual(metrics.range_precision(y_true, y_pred), 1.0)
        self.assertEqual(metrics.range_recall(y_true, y_pred), 1.0)


class TestFig4Reproduction(unittest.TestCase):
    """
    Reproduces Fig. 4 of Fung et al.: two cases that both score point-F1 =
    0.75 but range-F1 = 1.0 and ~0.17 respectively.

    Case 1 ("all attacks detected, no false positives"): 6 true attacks
    [1,1,1,1,1,20]; first five fully detected, sixth half detected, 0 FPs.
    -> point-F1=0.75, range-recall=range-precision=6/6=1.0, range-F1=1.0

    Case 2 ("one attack detected, five false positives"): 6 true attacks
    [3,3,3,3,3,60]; first five missed, sixth fully detected, 5 stray FP
    segments elsewhere.
    -> point-F1=0.75, range-recall=range-precision=1/6, range-F1~=0.17
    """

    def _case_1(self):
        n = 200
        y_true = np.zeros(n, dtype=int)
        y_pred = np.zeros(n, dtype=int)
        pos = 5
        for _ in range(5):
            segment(y_true, pos, 1)
            segment(y_pred, pos, 1)  # fully detected
            pos += 6
        segment(y_true, pos, 20)
        segment(y_pred, pos, 10)     # half detected
        return y_true, y_pred

    def _case_2(self):
        n = 200
        y_true = np.zeros(n, dtype=int)
        y_pred = np.zeros(n, dtype=int)
        pos = 5
        for _ in range(5):
            segment(y_true, pos, 3)  # completely missed
            pos += 8
        segment(y_true, pos, 60)
        segment(y_pred, pos, 60)     # fully detected
        fp_pos = pos + 70
        for _ in range(5):
            segment(y_pred, fp_pos, 5)  # false alarm
            fp_pos += 10
        return y_true, y_pred

    def test_case_1_point_f1(self):
        y_true, y_pred = self._case_1()
        self.assertAlmostEqual(f1_score(y_true, y_pred), 0.75, places=6)

    def test_case_1_range_f1(self):
        y_true, y_pred = self._case_1()
        self.assertAlmostEqual(metrics.range_f1(y_true, y_pred), 1.0, places=6)

    def test_case_2_point_f1(self):
        y_true, y_pred = self._case_2()
        self.assertAlmostEqual(f1_score(y_true, y_pred), 0.75, places=6)

    def test_case_2_range_f1(self):
        y_true, y_pred = self._case_2()
        self.assertAlmostEqual(metrics.range_f1(y_true, y_pred), 1 / 6, places=6)

    def test_both_cases_share_point_f1_but_diverge_on_range_f1(self):
        t1, p1 = self._case_1()
        t2, p2 = self._case_2()
        self.assertAlmostEqual(f1_score(t1, p1), f1_score(t2, p2), places=6)
        self.assertGreater(metrics.range_f1(t1, p1) - metrics.range_f1(t2, p2), 0.8)


class TestArgumentSwapDetection(unittest.TestCase):
    """
    Fig. 4 tests can't catch a y_true/y_pred swap: range-F1 is symmetric in
    P/R, and both Fig. 4 cases have precision==recall anyway. Uses an
    asymmetric scenario instead.

    3 true attacks, 1 prediction overlapping only the first, no false alarms:
        range_recall == 1/3, range_precision == 1.0

    beta=3 -> range_fbeta=5/6, beta=1/3 -> range_fbeta=5/14 (verified
    against reference fb13_score/fb31_score); swapping arguments exchanges
    the two values exactly.
    """

    def _three_attacks_one_detected(self):
        y_true = np.zeros(60, dtype=int)
        segment(y_true, 5, 5)    # detected
        segment(y_true, 20, 5)   # missed
        segment(y_true, 35, 5)   # missed
        y_pred = np.zeros(60, dtype=int)
        segment(y_pred, 5, 5)    # overlaps only the first true range, no false alarms
        return y_true, y_pred

    def test_precision_and_recall_are_not_equal_and_not_symmetric(self):
        y_true, y_pred = self._three_attacks_one_detected()
        self.assertEqual(metrics.range_recall(y_true, y_pred), 1 / 3)
        self.assertEqual(metrics.range_precision(y_true, y_pred), 1.0)
        # swap must change both values -- the actual bug this guards against
        self.assertEqual(metrics.range_recall(y_pred, y_true), 1.0)
        self.assertEqual(metrics.range_precision(y_pred, y_true), 1 / 3)

    def test_fbeta_values_are_sensitive_to_argument_order(self):
        y_true, y_pred = self._three_attacks_one_detected()
        f3 = metrics.range_fbeta(y_true, y_pred, beta=3)
        f13 = metrics.range_fbeta(y_true, y_pred, beta=1 / 3)
        f3_swapped = metrics.range_fbeta(y_pred, y_true, beta=3)
        f13_swapped = metrics.range_fbeta(y_pred, y_true, beta=1 / 3)

        self.assertAlmostEqual(f3, 5 / 6, places=6)
        self.assertAlmostEqual(f13, 5 / 14, places=6)
        # swap must move the score materially
        self.assertNotAlmostEqual(f3, f3_swapped, places=2)
        self.assertNotAlmostEqual(f13, f13_swapped, places=2)
        # for this scenario, swapped beta=3 lands exactly on un-swapped beta=1/3
        # (internal-consistency check, not a general claim)
        self.assertAlmostEqual(f3_swapped, f13, places=6)
        self.assertAlmostEqual(f13_swapped, f3, places=6)


class TestRangeFbetaDirection(unittest.TestCase):
    """Guards range_fbeta's beta-inversion against being "fixed" back to
    the naive (paper-reference-wrong) direction."""

    def _high_precision_low_recall(self):
        # 1 detected (TP), 4 missed (FN), 0 FPs -> precision=1.0, recall=0.2
        y_true = np.zeros(80, dtype=int)
        y_pred = np.zeros(80, dtype=int)
        pos = 5
        segment(y_true, pos, 5)
        segment(y_pred, pos, 5)  # only this one detected
        pos += 10
        for _ in range(4):
            segment(y_true, pos, 5)  # missed
            pos += 10
        return y_true, y_pred

    def test_recall_weighted_penalises_low_recall_harder_than_precision_weighted(self):
        y_true, y_pred = self._high_precision_low_recall()
        prec = metrics.range_precision(y_true, y_pred)
        rec = metrics.range_recall(y_true, y_pred)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 0.2)

        recall_weighted = metrics.range_fbeta_recall_weighted(y_true, y_pred)
        precision_weighted = metrics.range_fbeta_precision_weighted(y_true, y_pred)

        # recall-weighted should sit closer to the low recall value,
        # precision-weighted closer to the high precision value.
        self.assertLess(recall_weighted, precision_weighted)
        self.assertLess(recall_weighted, rec + 0.15)
        self.assertGreater(precision_weighted, prec - 0.5)


class TestNumenta(unittest.TestCase):

    def test_early_detection_scores_higher_than_late_detection(self):
        y_true = np.zeros(100, dtype=int)
        segment(y_true, 10, 40)  # attack from 10..49

        y_pred_early = np.zeros(100, dtype=int)
        segment(y_pred_early, 11, 1)  # detect almost immediately

        y_pred_late = np.zeros(100, dtype=int)
        segment(y_pred_late, 48, 1)  # detect right before it ends

        early_score = metrics.numenta_score(y_true, y_pred_early)
        late_score = metrics.numenta_score(y_true, y_pred_late)
        self.assertGreater(early_score, late_score)

    def test_false_positive_reduces_score(self):
        y_true = np.zeros(100, dtype=int)
        segment(y_true, 10, 40)

        y_pred = np.zeros(100, dtype=int)
        segment(y_pred, 15, 1)

        y_pred_with_fp = y_pred.copy()
        segment(y_pred_with_fp, 70, 3)

        score = metrics.numenta_score(y_true, y_pred)
        score_with_fp = metrics.numenta_score(y_true, y_pred_with_fp)
        self.assertGreater(score, score_with_fp)

    def test_na_early_defaults_to_the_reference_repos_20_percent(self):
        # Reference repo is authoritative (produced the paper's published
        # numbers) even though the prose says "25% point" -- must default to 0.2.
        y_true = np.zeros(100, dtype=int)
        segment(y_true, 10, 40)
        y_pred = np.zeros(100, dtype=int)
        segment(y_pred, 20, 1)

        expected_repo_value = metrics.numenta_score(
            y_true, y_pred, kappa=10, position_bias=0.2, fp_weight=-0.5)
        paper_prose_value = metrics.numenta_score(
            y_true, y_pred, kappa=10, position_bias=0.25, fp_weight=-0.5)

        self.assertEqual(metrics.na_early(y_true, y_pred), expected_repo_value)
        self.assertNotEqual(expected_repo_value, paper_prose_value)
        # 0.25 must still be reachable for the later sensitivity comparison
        self.assertEqual(metrics.na_early(y_true, y_pred, position_bias=0.25),
                          paper_prose_value)

    def test_no_true_anomalies_and_no_predictions_is_zero(self):
        y_true = np.zeros(50, dtype=int)
        y_pred = np.zeros(50, dtype=int)
        self.assertEqual(metrics.numenta_score(y_true, y_pred), 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
