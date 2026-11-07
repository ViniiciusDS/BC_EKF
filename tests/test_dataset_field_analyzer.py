"""Regressão: métricas reais são distâncias, não ground truth fictício."""
import json
import unittest
from pathlib import Path

import numpy as np

from src.analysis.algo_metrics import (
    compute_matched_track_difference_stats,
    compute_single_tag_field_stats,
)


class TestSingleTagFieldAnalyzer(unittest.TestCase):
    def test_distances_to_nominal_polyline_not_time_synced(self):
        route = np.array([[0, 0], [2, 0], [2, 2]], dtype=float)
        estimates = {"lms": {"posicoes": np.array([
            [0.5, 0.0], [1.5, 1.0], [2.0, 1.0],
        ])}}
        stats = compute_single_tag_field_stats(
            estimates, "reference_route", reference_route=route,
        )
        self.assertEqual(stats["lms"]["n"], 3)
        self.assertAlmostEqual(stats["lms"]["mae"], 1 / 3)
        self.assertAlmostEqual(stats["lms"]["rmse"], np.sqrt(1/3))

    def test_encoder_comparison_aligns_at_original_sample_indices(self):
        odom = np.column_stack((
            np.arange(7, dtype=float),
            np.zeros(7, dtype=float),
            np.zeros(7, dtype=float),
        ))
        indices = np.array([1, 3, 6], dtype=int)
        estimates = {"gauss_newton": {"posicoes": np.array([
            [1, 0.5], [3, 0.0], [6, -1.0],
        ])}}
        stats = compute_single_tag_field_stats(
            estimates, "encoder_route",
            odometry_poses=odom, sample_indices=indices,
        )
        self.assertEqual(stats["gauss_newton"]["n"], 3)
        self.assertAlmostEqual(stats["gauss_newton"]["mae"], 0.5)
        self.assertAlmostEqual(stats["gauss_newton"]["rmse"], np.sqrt(1.25 / 3))
        self.assertEqual(
            stats["gauss_newton"]["comparison_type"],
            "difference_to_odometry_not_ground_truth",
        )

    def test_nonfinite_samples_keep_temporal_index_alignment(self):
        ref = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
        res = {"lms": {"posicoes": np.array([
            [1, 0], [np.nan, np.nan], [22, 0],
        ])}}
        stats = compute_matched_track_difference_stats(res, ref)
        self.assertEqual(stats["lms"]["n"], 2)
        self.assertAlmostEqual(stats["lms"]["mae"], 1.5)

    def test_mismatched_sample_counts_never_truncate_silently(self):
        res = {"lms": {"posicoes": np.array([[0, 0], [1, 0]])}}
        ref = np.array([[0, 0], [1, 0], [2, 0]])
        self.assertIsNone(compute_matched_track_difference_stats(res, ref))
        self.assertIsNone(compute_single_tag_field_stats(
            res, "encoder_route", odometry_poses=ref,
            sample_indices=[0, 1, 2],
        ))

    def test_unsynced_nominal_route_not_used_as_time_truth(self):
        self.assertIsNone(compute_single_tag_field_stats(
            {"lms": {"posicoes": np.array([[0, 0]])}},
            "reference_synced", reference_route=[[0, 0], [1, 1]],
        ))

    def test_cluster_mode_is_explicit_dispersion_not_accuracy(self):
        results = {"lms": {"posicoes": np.array([[0, 0], [2, 0]])}}
        stats = compute_single_tag_field_stats(results, "cluster")
        self.assertAlmostEqual(stats["lms"]["rmse"], 1.0)

    def test_invalid_encoder_index_does_not_use_unsynchronized_rows(self):
        estimates = {"lms": {"posicoes": np.array([[0, 0], [1, 1]])}}
        odom = np.array([[0, 0], [1, 1]])
        for ids in [[0, 2], [-1, 0], [0.0, 1.0]]:
            self.assertIsNone(compute_single_tag_field_stats(
                estimates, "encoder_route", odometry_poses=odom,
                sample_indices=np.asarray(ids),
            ))

    def test_comparison_does_not_change_input_trajectories(self):
        odom = np.array([[0., 0.], [3., 4.]], dtype=float)
        est = np.array([[1., 0.], [3., 5.]], dtype=float)
        b1, b2 = odom.copy(), est.copy()
        compute_matched_track_difference_stats({"lms": {"posicoes": est}}, odom)
        np.testing.assert_array_equal(odom, b1)
        np.testing.assert_array_equal(est, b2)

    def test_estimated_initial_pose_is_a_separate_profile(self):
        root = Path(__file__).resolve().parents[1] / "anchor_sets"
        previous = json.loads((root / "passeio_01_4_ancoras.json").read_text())
        estimated = json.loads((root / "passeio_01_4_ancoras_pose_estimada.json").read_text())
        self.assertEqual(previous["robot_2026"]["initial_theta_deg"], 180)
        self.assertEqual(estimated["robot_2026"]["initial_theta_deg"], 7)
        self.assertAlmostEqual(estimated["robot_2026"]["initial_x_m"], -1.41)
        self.assertEqual(estimated["anchors_xy"], previous["anchors_xy"])
        self.assertEqual(
            estimated["pcnt_slot_to_anchor_index"],
            previous["pcnt_slot_to_anchor_index"],
        )
        self.assertIn("HIPOTESE", estimated["meta"]["initial_pose_status"])


if __name__ == "__main__":
    unittest.main()
