"""Regressão do importador de odometria 2026 (não depende de Pygame)."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from src.ui.algo_modes.dataset_odometry_pipeline import (
    ROBOT2026_PROFILE,
    list_robot2026_sources,
    load_robot2026_odometry,
    resolve_robot2026_source,
)


class TestDatasetOdometry2026(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "raw").mkdir()
        (self.root / "processed").mkdir()

    def write(self, subdir, filename, data):
        path = self.root / subdir / filename
        path.write_text(data, encoding="utf-8")
        return path

    def test_4560_cpr_preserves_every_sample(self):
        file = self.write(
            "raw", "pcntLog.csv",
            "100,0,0,0,0\n350,2280,-2280,0,0\n600,4560,-4560,0,0\n",
        )
        data = load_robot2026_odometry(file)
        self.assertEqual(ROBOT2026_PROFILE.ticks_per_wheel_rev, 4560)
        self.assertEqual(data.sample_count, 3)
        self.assertAlmostEqual(data.duration_s, 0.5)
        self.assertAlmostEqual(data.poses_xytheta[-1, 0], 2 * math.pi * .03425)
        self.assertAlmostEqual(data.poses_xytheta[-1, 1], 0)
        self.assertAlmostEqual(data.poses_xytheta[-1, 2], 0)

    def test_session_reset_not_sorted(self):
        file = self.write(
            "raw", "pcntLog.csv",
            "100,0,0,0,0\n350,4560,-4560,1,0\n100,0,0,0,0\n350,2280,-2280,1,0\n",
        )
        data = load_robot2026_odometry(file, session_id=2)
        self.assertEqual(data.total_sessions, 2)
        self.assertEqual(data.sample_count, 2)
        self.assertAlmostEqual(data.poses_xytheta[-1, 0], math.pi * .03425)
        with self.assertRaisesRegex(ValueError, "Sessão 3 inexistente"):
            load_robot2026_odometry(file, session_id=3)

    def test_converted_encoder_sign_correction(self):
        file = self.write(
            "processed", "robot2026_session_02_encoder.csv",
            "timestamp,left_ticks,right_ticks\n0,0,0\n250,4560,4560\n",
        )
        data = load_robot2026_odometry(file)
        self.assertEqual(data.session_id, 2)
        self.assertAlmostEqual(data.poses_xytheta[-1, 0], 2 * math.pi * .03425)
        self.assertAlmostEqual(data.poses_xytheta[-1, 2], 0)

    def test_converted_odometry_does_not_reuse_2280_pose(self):
        file = self.write(
            "processed", "robot2026_session_02_odometry.csv",
            "timestamp_ms,left_ticks,right_ticks,x_m,y_m,theta_rad\n0,0,0,0,0,0\n250,4560,4560,999,999,999\n",
        )
        data = load_robot2026_odometry(file)
        self.assertAlmostEqual(data.poses_xytheta[-1, 0], 2 * math.pi * .03425)
        self.assertLess(data.poses_xytheta[-1, 0], 1)

    def test_list_and_resolve_safe_source_paths(self):
        self.write("raw", "pcntLog.csv", "0,0,0,0,0\n250,1,-1,0,0\n")
        self.write(
            "processed", "robot2026_session_01_encoder.csv",
            "timestamp,left_ticks,right_ticks\n0,0,0\n250,1,1\n",
        )
        self.write("processed", "other.txt", "not odometry")
        options = list_robot2026_sources(self.root / "raw", self.root / "processed")
        self.assertEqual(len(options), 2)
        self.assertIn("Bruto: pcntLog.csv", options)
        self.assertEqual(resolve_robot2026_source(
            options[0], self.root / "raw", self.root / "processed"
        ).suffix, ".csv")
        with self.assertRaises(ValueError):
            resolve_robot2026_source(
                "Bruto: ../escape.csv", self.root / "raw", self.root / "processed"
            )

    def test_invalid_timestamp_is_rejected(self):
        file = self.write(
            "processed", "robot2026_session_02_encoder.csv",
            "timestamp,left_ticks,right_ticks\n0,0,0\n0,5,5\n",
        )
        with self.assertRaisesRegex(ValueError, "Timestamp não crescente"):
            load_robot2026_odometry(file)


if __name__ == "__main__":
    unittest.main()
