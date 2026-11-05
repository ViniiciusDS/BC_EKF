"""Testes de regressão do adaptador ESP32 2026."""
import math
import tempfile
import unittest
from pathlib import Path

from src.odometry.pcntlog_adapter import (
    RawPcntRow, Robot2026Profile, convert_session, convert_uwb_session,
    export_pcnt_sessions, load_pcnt_sessions,
)


class TestPcntLogAdapter(unittest.TestCase):
    def setUp(self):
        self.cfg = Robot2026Profile()

    def test_one_revolution_straight(self):
        rows = [RawPcntRow(1000, 50, -50, 0, 0), RawPcntRow(1250, 2330, -2330, 0, 0)]
        od = convert_session(rows, self.cfg)
        self.assertAlmostEqual(od[1]["x_m"], 2 * math.pi * self.cfg.wheel_radius_m, places=10)
        self.assertAlmostEqual(od[1]["y_m"], 0.0, places=10)
        self.assertAlmostEqual(od[1]["theta_rad"], 0.0, places=10)
        self.assertEqual(od[1]["right_ticks"], 2280)

    def test_rotation_has_correct_wheel_sign(self):
        rows = [RawPcntRow(100, 0, 0, 0, 0), RawPcntRow(350, -500, -500, 0, 0)]
        od = convert_session(rows, self.cfg)
        self.assertAlmostEqual(od[1]["ds_m"], 0, places=10)
        self.assertGreater(od[1]["dtheta_rad"], 0)

    def test_sessions_not_time_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pcntLog.csv"
            p.write_text("1000,0,0,0,0\n1250,100,-100,0,0\n200,0,0,0,0\n450,200,-200,0,0\n", encoding="utf-8")
            sessions = load_pcnt_sessions(p)
            self.assertEqual([len(s) for s in sessions], [2,2])
            manifest = export_pcnt_sessions(p, Path(tmp)/"out", self.cfg)
            self.assertEqual([m["source_rows"] for m in manifest], [2,2])
            self.assertTrue((Path(tmp)/"out"/"robot2026_session_02_encoder.csv").exists())

    def test_uwb_units_and_invalid(self):
        triple = ((120.0, 4.0, 2.0), (-1.0, 5.0, -255.0), (100.0, 1.0, 1.0),
                  (200.0, 9.0, 3.0), (300.0, 25.0, 5.0))
        rows = [RawPcntRow(1000, 0, 0, 0, 0, triple), RawPcntRow(1250, 0, 0, 0, 0, triple)]
        uwb = convert_uwb_session(rows)
        self.assertEqual(len(uwb), 8)
        self.assertAlmostEqual(uwb[0]["range"], 1.2)
        self.assertAlmostEqual(uwb[0]["sigma"], 0.02)
        self.assertAlmostEqual(uwb[0]["variance_m2"], 0.0004)
        self.assertEqual(sum(r["possibly_reused"] for r in uwb), 4)


if __name__ == "__main__":
    unittest.main()
