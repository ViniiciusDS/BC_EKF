"""Regressões da auditoria odométrica PR14, sem pygame ou PCNT real."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.odometry.field_calibration_audit import audit_field_odometry
from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    read_robot2026_profile_from_anchor_json,
)
from tools.audit_field_odometry_2026 import main as cli_main


class TestFieldCalibrationAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "pcnt.csv"
        self.anchors = self.root / "lab_correct.json"
        self.f7_path = self.root / "f7.json"
        # Formato real: 5 colunas PCNT, sem cabeçalho; sem UWB
        # porque os limites das voltas são fornecidos pelo F7.
        rows = [f"{i*250},{i*100},{-i*100},0,0" for i in range(25)]
        self.log.write_text("\n".join(rows)+"\n", encoding="utf-8")
        self.anchors.write_text(json.dumps({
            "robot_2026": {
                "wheel_radius_m": .03425,
                "wheel_base_m": .229,
                "ticks_per_wheel_rev": 4560,
                "invert_left": False, "invert_right": True,
                "swap_encoder_columns": False,
                "left_scale": 1., "right_scale": 1.,
                "initial_x_m": .15, "initial_y_m": 1.38,
                "initial_theta_deg": 0,
            }
        }), encoding="utf-8")
        stops = (4, 10, 16, 22, 25)
        starts = (0, 4, 10, 16, 22)
        segments = []
        for i, (start, stop) in enumerate(zip(starts, stops)):
            complete = 1 <= i <= 3
            seg = {
                "segment_number": i+1,
                "complete": complete,
                "pcnt_start_inclusive": start,
                "pcnt_stop_exclusive": stop,
                "pcnt_samples": stop-start,
                "algorithms": {
                    "lms": {
                        "odometry_disagreement": {
                            "rms_m": (.3*i if complete else 2.5)
                        }
                    },
                },
            }
            segments.append(seg)
        self.f7 = {
            "schema": "bc_ekf.field_lap_metrics.v1",
            "segmentation_type": "physical_start_gate",
            "pcnt_samples": 25,
            "uwb_snapshots": 10,
            "source_file": self.log.name,
            "source_sha256": hashlib.sha256(self.log.read_bytes()).hexdigest(),
            "anchor_config_file": self.anchors.name,
            "anchor_config_sha256": hashlib.sha256(self.anchors.read_bytes()).hexdigest(),
            "session_id": 1,
            "complete_laps": 3,
            "partial_segments": 2,
            "gate_returns": 4,
            "gate_crossing_times_s": [1.,2.5,4.,5.5],
            "segments": segments,
        }
        self.f7_path.write_text(json.dumps(self.f7), encoding="utf-8")
        self.profile = read_robot2026_profile_from_anchor_json(self.anchors)

    def audit(self, **kwargs):
        return audit_field_odometry(
            self.f7, self.log, self.anchors, self.profile, **kwargs
        )

    def test_full_lap_boundaries_and_descriptive_slope(self):
        before = self.log.read_bytes()
        result = self.audit(algorithm="lms", expected_lap_turn_deg=-360.)
        self.assertEqual(result["complete_lap_count"], 3)
        self.assertEqual(result["session_id"], 1)
        self.assertIsNone(result["selected_calibration"])
        self.assertFalse(result["applied_to_original_profile"])
        self.assertEqual(len(result["scenarios"]), 1)
        for observed, expected in zip(
            result["uwb_odom_disagreement_from_f7"]["per_complete_lap_rms_m"],
            (.3, .6, .9),
        ):
            self.assertAlmostEqual(observed, expected)
        trend = result["uwb_odom_disagreement_from_f7"]["linear_trend"]
        self.assertAlmostEqual(trend["slope_m_per_lap"], .3)
        self.assertAlmostEqual(trend["r_squared_descriptive"], 1.)
        scenarios = result["scenarios"][0]["complete_laps"]
        self.assertEqual(
            [(item["gate_start_index_pcnt"], item["gate_end_index_pcnt"]) for item in scenarios],
            [(4,10),(10,16),(16,22)]
        )
        for lap in scenarios:
            self.assertAlmostEqual(lap["gate_to_gate_closure_m"], lap["odom_path_m"])
            self.assertAlmostEqual(lap["odom_net_heading_deg_unwrapped"], 0.)
            self.assertAlmostEqual(lap["heading_residual_deg_assumed"], 360.)
        self.assertEqual(before, self.log.read_bytes())

    def test_scale_variant_is_not_selected_and_changes_yaw_as_expected(self):
        result = self.audit(scale_candidate=(1.20, .80))
        self.assertEqual(len(result["scenarios"]), 2)
        a, b = result["scenarios"]
        self.assertEqual(a["name"], "perfil_de_campo")
        self.assertEqual(b["name"], "escalas_experimentais")
        self.assertEqual(a["profile_used"]["left_scale"], 1.)
        self.assertEqual(b["profile_used"]["right_scale"], .8)
        self.assertLess(b["complete_laps"][0]["odom_net_heading_deg_unwrapped"], 0.)
        self.assertIsNone(result["expected_lap_turn_deg_assumption"])
        self.assertIsNone(b["complete_laps"][0]["heading_residual_deg_assumed"])
        self.assertIsNone(result["selected_calibration"])

    def test_rejects_mismatched_provenance_hash_or_filename(self):
        self.f7["source_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "SHA-256 do CSV"):
            self.audit()
        self.f7["source_sha256"] = hashlib.sha256(self.log.read_bytes()).hexdigest()
        self.f7["anchor_config_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "SHA-256 do perfil"):
            self.audit()
        self.f7["anchor_config_sha256"] = hashlib.sha256(self.anchors.read_bytes()).hexdigest()
        self.f7["anchor_config_file"] = "different.json"
        with self.assertRaisesRegex(ValueError, "nome do JSON"):
            self.audit()

    def test_rejects_invalid_lap_segmentation_instead_of_reinterpreting(self):
        self.f7["segments"][2]["pcnt_start_inclusive"] = 9
        with self.assertRaisesRegex(ValueError, "sem sobreposição"):
            self.audit()
        self.f7["segments"][2]["pcnt_start_inclusive"] = 10
        self.f7["segmentation_type"] = "time_windows"
        with self.assertRaisesRegex(ValueError, "portão físico"):
            self.audit()

    def test_rejects_missing_algorithm_or_bad_candidate(self):
        with self.assertRaisesRegex(ValueError, "Algoritmo"):
            self.audit(algorithm="lmsp")
        with self.assertRaisesRegex(ValueError, "positivas"):
            self.audit(scale_candidate=(0.,1.))
        with self.assertRaisesRegex(ValueError, "Giro esperado"):
            self.audit(expected_lap_turn_deg=0)

    def test_report_detects_session_reset_and_wrong_counts(self):
        self.f7["pcnt_samples"] = 24
        with self.assertRaises(ValueError):
            self.audit()
        self.f7["pcnt_samples"] = 25
        self.f7["gate_returns"] = 2
        with self.assertRaisesRegex(ValueError, "retorno"):
            self.audit()

    def test_cli_exports_json_and_csv_without_overwriting(self):
        output = self.root / "audit.json"
        csv_out = self.root / "audit.csv"
        result = cli_main([
            "--log", str(self.log), "--anchors", str(self.anchors),
            "--laps", str(self.f7_path), "--expected-lap-turn-deg", "-360",
            "--out", str(output), "--csv", str(csv_out)
        ])
        self.assertEqual(result["complete_lap_count"], 3)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema"],
                         "bc_ekf.field_odometry_audit.v1")
        csv_lines = csv_out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(csv_lines), 4)
        prior = output.read_bytes()
        with self.assertRaises(SystemExit):
            cli_main([
                "--log",str(self.log), "--anchors",str(self.anchors),
                "--laps",str(self.f7_path), "--out",str(output)
            ])
        self.assertEqual(prior, output.read_bytes())

    def test_cli_reports_bad_f7_profile_without_creating_partial_output(self):
        f7 = json.loads(self.f7_path.read_text(encoding="utf-8"))
        f7["anchor_config_sha256"] = "bad"
        self.f7_path.write_text(json.dumps(f7), encoding="utf-8")
        output = self.root / "should_not_exist.json"
        with self.assertRaises(SystemExit):
            cli_main([
                "--log",str(self.log), "--anchors",str(self.anchors),
                "--laps",str(self.f7_path), "--out",str(output)
            ])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
