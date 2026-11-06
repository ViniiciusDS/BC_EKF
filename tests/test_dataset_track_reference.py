"""Testes: geometria aproximada nunca substitui a referência independente."""
import unittest
import numpy as np

from src.ui.algo_modes.dataset_lap_view import split_laps
from src.ui.algo_modes.dataset_track_reference import build_track_from_first_complete_lap


class TestDatasetApproxTrack(unittest.TestCase):
    def make_lap(self):
        # Uma volta fechada (aproximação elíptica), e uma volta deslocada.
        theta = np.linspace(0, 2*np.pi, 101)
        first = np.column_stack([
            1.2*np.cos(theta), 0.7*np.sin(theta), np.zeros_like(theta)
        ])
        second = first + np.array([2.0, 0.0, 0.0])
        poses = np.vstack((first, second))
        cnt = np.r_[np.zeros(len(first)), np.ones(len(second))]
        return poses, split_laps(cnt)

    def test_first_lap_only_and_closed(self):
        poses, laps = self.make_lap()
        baseline = poses.copy()
        result = build_track_from_first_complete_lap(poses, laps)
        self.assertIsNotNone(result)
        self.assertEqual(result.source_lap, 1)
        self.assertEqual(result.xy.shape, (257, 2))
        np.testing.assert_allclose(result.xy[0], result.xy[-1])
        self.assertTrue(2.1 < result.span_x_m < 2.5)
        self.assertTrue(1.2 < result.span_y_m < 1.5)
        self.assertIn("NOT ground truth", result.provenance)
        np.testing.assert_array_equal(poses, baseline)

    def test_no_completed_laps(self):
        xy = np.zeros((30, 3))
        self.assertIsNone(build_track_from_first_complete_lap(
            xy, split_laps(np.zeros(30))
        ))

    def test_rejects_large_open_gap(self):
        # Trajetória retilínea não deve ser fechada artificialmente em uma pista.
        x = np.linspace(0, 4, 60)
        one = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
        poses = np.vstack((one, one))
        laps = split_laps(np.r_[np.zeros(60), np.ones(60)])
        self.assertIsNone(build_track_from_first_complete_lap(poses, laps))

    def test_uneven_sampling_and_repeated_points(self):
        poses, laps = self.make_lap()
        poses[4] = poses[3]
        result = build_track_from_first_complete_lap(poses, laps)
        self.assertIsNotNone(result)
        self.assertGreater(result.perimeter_m, 4)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            build_track_from_first_complete_lap(np.array([1, 2]), ())
        with self.assertRaises(ValueError):
            build_track_from_first_complete_lap(
                np.ones((20, 3)), (), sample_count=4
            )


if __name__ == "__main__":
    unittest.main()
