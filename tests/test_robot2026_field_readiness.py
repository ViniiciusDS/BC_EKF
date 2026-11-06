"""Regressão dos arquivos e perfis de ensaio do robô 2026."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ui.algo_modes.dataset_odometry_pipeline import (
    list_robot2026_sources, resolve_robot2026_source,
)
from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    inspect_robot2026_capture, read_robot2026_profile_from_anchor_json,
    validate_robot2026_anchor_geometry, load_robot2026_single_tag_uwb,
)


class TestRobot2026FieldReadiness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "raw").mkdir()
        (self.root / "processed").mkdir()
        self.anchors = self.root / "setup.json"
        self.anchors.write_text(json.dumps({
            "anchors_xy": [
                [0, 0, 1], [2, 0, 1], [0, 2, 1], [2, 2, 1], [1, 1, 1]
            ],
            "pcnt_slot_to_anchor_index": [0, 1, 2, 3, 4],
            "robot_2026": {
                "ticks_per_wheel_rev": 4560,
                "wheel_radius_m": 0.03425,
                "wheel_base_m": 0.229,
                "initial_x_m": 1.5,
                "initial_y_m": 3.1,
                "initial_theta_deg": 90,
            }
        }), encoding="utf-8")

    def line(self, t, left, right, offset=100):
        fields = [str(t), str(left), str(right), "0", "0"]
        for i in range(5):
            fields.extend([str(offset+i*10), "4", "2"])
        return ",".join(fields)

    def write(self, name, lines):
        file = self.root / "raw" / name
        file.write_text("\n".join(lines)+"\n", encoding="utf-8")
        return file

    def test_arbitrary_firmware_filename_is_selectable(self):
        path = self.write("BATERIA_7_C2.CSV", ["10,0,0,0,0", "260,5,-5,0,0"])
        options = list_robot2026_sources(
            self.root / "raw", self.root / "processed"
        )
        self.assertIn("Bruto: BATERIA_7_C2.CSV", options)
        self.assertEqual(
            resolve_robot2026_source(options[0], self.root / "raw", self.root / "processed"),
            path,
        )

    def test_profile_from_layout_sets_origin_and_angles(self):
        p = read_robot2026_profile_from_anchor_json(self.anchors)
        self.assertAlmostEqual(p.initial_x_m, 1.5)
        self.assertAlmostEqual(p.initial_y_m, 3.1)
        self.assertAlmostEqual(p.initial_theta_rad, math.pi / 2)
        path = self.write("capture.csv", [
            self.line(100, 0, 0), self.line(350, 4560, -4560, offset=110),
        ])
        res = load_robot2026_single_tag_uwb(
            path, session_id=1, anchor_count=5,
            slot_to_anchor_index=(0, 1, 2, 3, 4), profile=p,
        )
        self.assertAlmostEqual(res.odometry.poses_xytheta[0, 0], 1.5)
        self.assertAlmostEqual(res.odometry.poses_xytheta[0, 1], 3.1)
        self.assertAlmostEqual(
            res.odometry.poses_xytheta[-1, 1] - 3.1, 2*math.pi*.03425
        )

    def test_default_profile_is_still_4560(self):
        self.anchors.write_text(json.dumps({"robot_2026": {}}))
        p = read_robot2026_profile_from_anchor_json(self.anchors)
        self.assertEqual(p.ticks_per_wheel_rev, 4560)
        self.assertAlmostEqual(p.initial_theta_rad, 0)

    def test_rejects_typo_or_conflicting_angles(self):
        self.anchors.write_text(json.dumps({
            "robot_2026": {"initial_theta_deg": 90, "initial_theta_rad": 1.57}
        }))
        with self.assertRaisesRegex(ValueError, "OU"):
            read_robot2026_profile_from_anchor_json(self.anchors)
        self.anchors.write_text(json.dumps({
            "robot_2026": {"ticks_per_wheel_revv": 4560}
        }))
        with self.assertRaisesRegex(ValueError, "desconhecidos"):
            read_robot2026_profile_from_anchor_json(self.anchors)

    def test_rejects_collinear_anchor_geometry(self):
        with self.assertRaisesRegex(ValueError, "colineares"):
            validate_robot2026_anchor_geometry(
                np.array([[0, 0], [1, 1], [2, 2]])
            )
        validate_robot2026_anchor_geometry(
            np.array([[0, 0], [1, 0], [0, 1]])
        )

    def test_example_anchor_layout_is_blocked(self):
        self.anchors.write_text(json.dumps({
            "meta": {"example_only": True},
            "anchors_xy": [[0, 0], [1, 0], [0, 1]],
            "pcnt_slot_to_anchor_index": [0, 1, 2, None, None],
        }))
        path = self.write("capture.csv", [
            self.line(100, 0, 0), self.line(350, 5, -5, 110),
        ])
        report = inspect_robot2026_capture(
            path, session_id=1, anchor_path=self.anchors
        )
        self.assertEqual(report["status"], "blocked")
        self.assertIn("EXEMPLO", report["warnings"][0])

    def test_preflight_reports_only_odometry_for_five_columns(self):
        path = self.write("old_log.csv", ["100,0,0,0,0", "350,100,-100,0,0"])
        report = inspect_robot2026_capture(path)
        self.assertEqual(report["status"], "odometry_only")
        self.assertEqual(report["columns"], 5)
        self.assertEqual(report["pcnt_samples"], 2)

    def test_preflight_flags_missing_mapping(self):
        path = self.write("new_log.csv", [self.line(100, 0, 0), self.line(350, 1, -1, 110)])
        report = inspect_robot2026_capture(path)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("JSON", report["warnings"][0])
        self.assertEqual(report["columns"], 20)

    def test_preflight_ready_and_time_reset(self):
        path = self.write("run2026.csv", [
            self.line(100, 0, 0), self.line(350, 1, -1, 110),
            self.line(100, 0, 0, 120), self.line(350, 2, -2, 130),
        ])
        report = inspect_robot2026_capture(
            path, session_id=2, anchor_path=self.anchors
        )
        self.assertEqual(report["status"], "ready_for_static_batch")
        self.assertEqual(report["sessions_detected"], 2)
        self.assertEqual(report["uwb_snapshots"], 2)
        self.assertEqual(report["start_pose_xytheta"][0], 1.5)

    def test_preflight_blocks_zero_fresh_snapshots(self):
        path = self.write("run.csv", [
            self.line(100, 0, 0), self.line(350, 1, -1),
        ])
        # Primeiro registro possui ranges frescos: para bloquear, todas
        # as cinco médias devem ser inválidas ou sentinelas.
        path.write_text(
            "100,0,0,0,0," + ",".join(["-1,0,0"]*5) + "\n"
            "350,1,-1,0,0," + ",".join(["-1,0,0"]*5) + "\n"
        )
        report = inspect_robot2026_capture(
            path, anchor_path=self.anchors
        )
        self.assertEqual(report["status"], "blocked")
        self.assertIn("Nenhuma linha", report["warnings"][0])


if __name__ == "__main__":
    unittest.main()
