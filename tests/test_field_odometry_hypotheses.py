"""Regressões do PR #11: alinhamento, sinais, escalas, sessões e integridade."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.odometry.pcntlog_adapter import Robot2026Profile
from src.odometry.field_trajectory_diagnostic import (
    compare_odometry_hypotheses, load_nominal_route,
    load_straight_calibration_candidates, route_distance_rms,
)


class TestFieldOdometryHypotheses(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "pcnt.csv"
        self.anchors = self.root / "anchors.json"
        self.route = self.root / "route.json"
        self.profile = Robot2026Profile(
            wheel_radius_m=0.03425, wheel_base_m=0.229,
            ticks_per_wheel_rev=4560, swap_encoder_columns=True,
            invert_left=True, invert_right=False,
            initial_x_m=0.15, initial_y_m=1.38,
        )
        self.anchors.write_text(json.dumps({
            "robot_2026": {
                "wheel_radius_m": 0.03425, "wheel_base_m": 0.229,
                "ticks_per_wheel_rev": 4560,
                "swap_encoder_columns": True, "invert_left": True,
                "invert_right": False, "initial_x_m": 0.15, "initial_y_m": 1.38,
            },
        }), encoding="utf-8")
        self.route.write_text(json.dumps({
            "waypoints": [[0.15, 1.38], [2.15, 1.38]]
        }), encoding="utf-8")

    def write_log(self, movements, reset=False):
        left = right = 0
        rows = ["0,0,0,0,0"]
        for i, (dl, dr) in enumerate(movements, 1):
            left += dl
            right += dr
            rows.append(f"{i*250},{right},{-left},0,0")
        if reset:
            rows += ["0,0,0,0,0", "250,200,-200,0,0"]
        self.log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_pose_changes_alignment_not_local_shape(self):
        self.write_log([(1200, 1200), (1200, 1200)])
        report, trace = compare_odometry_hypotheses(
            self.log, self.profile, measured_pose=(2.0, 3.0, 120)
        )
        cases = {c["hypothesis"]: c for c in report["cases"]}
        original, pose = cases["original"], cases["pose_medida"]
        self.assertEqual(len(trace), len(report["cases"]) * 3)
        self.assertAlmostEqual(original["start_xy_m"][0], 0.15)
        self.assertAlmostEqual(pose["start_xy_m"][0], 2.0)
        for i in (0, 1):
            self.assertAlmostEqual(
                original["relative_end_robot_frame_m"][i],
                pose["relative_end_robot_frame_m"][i], places=10
            )
        self.assertAlmostEqual(
            original["heading_change_deg"], pose["heading_change_deg"]
        )
        self.assertIsNone(report["best_case"])
        self.assertFalse(report["calibration_applied"])

    def test_calibration_scales_distance_without_modifying_csv(self):
        self.write_log([(1000, 1000), (1000, 1000)])
        before = hashlib.sha256(self.log.read_bytes()).hexdigest()
        report, _ = compare_odometry_hypotheses(
            self.log, self.profile, scale_candidates=(1.2, 1.2),
        )
        cases = {c["hypothesis"]: c for c in report["cases"]}
        self.assertAlmostEqual(
            cases["escalas_reta"]["integrated_abs_forward_m"],
            cases["original"]["integrated_abs_forward_m"] * 1.2,
        )
        mirrored = cases["trocar_colunas_e_sinais_com_escalas_reta"]
        calibrated = cases["escalas_reta"]
        self.assertAlmostEqual(
            mirrored["integrated_abs_forward_m"],
            calibrated["integrated_abs_forward_m"],
        )
        self.assertAlmostEqual(
            mirrored["heading_change_deg"],
            -calibrated["heading_change_deg"],
        )
        self.assertAlmostEqual(
            mirrored["relative_end_robot_frame_m"][0],
            calibrated["relative_end_robot_frame_m"][0],
        )
        self.assertAlmostEqual(
            mirrored["relative_end_robot_frame_m"][1],
            -calibrated["relative_end_robot_frame_m"][1],
        )
        self.assertAlmostEqual(
            mirrored["profile"]["left_scale"], 1.2
        )
        self.assertEqual(before, report["raw_log_sha256"])
        self.assertEqual(before, hashlib.sha256(self.log.read_bytes()).hexdigest())

    def test_combined_remap_swaps_unequal_scale_candidates_by_channel(self):
        """Escalas do PR10 acompanham os canais brutos quando trocados."""
        self.write_log([(1000, 1000), (1000, 1400)])
        report, _ = compare_odometry_hypotheses(
            self.log, self.profile, scale_candidates=(1.22, 1.20)
        )
        cases = {case["hypothesis"]: case for case in report["cases"]}
        calibrated = cases["escalas_reta"]
        mirrored = cases["trocar_colunas_e_sinais_com_escalas_reta"]
        self.assertAlmostEqual(mirrored["profile"]["left_scale"], 1.20)
        self.assertAlmostEqual(mirrored["profile"]["right_scale"], 1.22)
        self.assertAlmostEqual(
            calibrated["integrated_abs_forward_m"],
            mirrored["integrated_abs_forward_m"],
        )
        self.assertAlmostEqual(
            calibrated["heading_change_deg"],
            -mirrored["heading_change_deg"],
        )

    def test_sign_and_swap_hypotheses_are_always_explicit(self):
        self.write_log([(600, 600)])
        report, trace = compare_odometry_hypotheses(self.log, self.profile)
        names = {c["hypothesis"] for c in report["cases"]}
        self.assertTrue({
            "original", "trocar_colunas", "trocar_colunas_e_sinais",
            "inverter_esquerda", "inverter_direita", "inverter_ambas",
            "heading_mais_90", "heading_menos_90",
        } <= names)
        self.assertEqual(len(trace), 16)
        self.assertEqual(self.profile.invert_left, True)

    def test_swap_and_signs_preserve_forward_direction_but_reverse_turn(self):
        """Mapeamento alternativo preserva +X e espelha curva esquerda/direita."""
        # O primeiro trecho avança em linha reta no sentido físico +X.
        # Na sequência, o encoder direito avança mais: curva CCW original.
        self.write_log([(600, 600), (300, 700), (300, 700)])
        report, traces = compare_odometry_hypotheses(self.log, self.profile)
        by_name = {case["hypothesis"]: case for case in report["cases"]}
        original = by_name["original"]
        mirror = by_name["trocar_colunas_e_sinais"]
        self.assertAlmostEqual(
            original["integrated_abs_forward_m"],
            mirror["integrated_abs_forward_m"],
        )
        self.assertAlmostEqual(
            original["left_wheel_total_m"], mirror["right_wheel_total_m"]
        )
        self.assertAlmostEqual(
            original["right_wheel_total_m"], mirror["left_wheel_total_m"]
        )
        self.assertAlmostEqual(
            original["heading_change_deg"],
            -mirror["heading_change_deg"]
        )
        self.assertAlmostEqual(
            original["relative_end_robot_frame_m"][0],
            mirror["relative_end_robot_frame_m"][0]
        )
        self.assertAlmostEqual(
            original["relative_end_robot_frame_m"][1],
            -mirror["relative_end_robot_frame_m"][1]
        )
        p = {name: sorted(
            (t for t in traces if t["hypothesis"] == name),
            key=lambda t: t["timestamp_s"]
        ) for name in ("original", "trocar_colunas_e_sinais")}
        for name in p:
            self.assertGreater(p[name][1]["x_m"], .15)
            self.assertAlmostEqual(p[name][1]["y_m"], 1.38, places=9)
        self.assertIsNone(report["best_case"])
        self.assertFalse(report["calibration_applied"])

    def test_session_reset_is_not_concatenated(self):
        self.write_log([(100, 100), (100, 100)], reset=True)
        report, trace = compare_odometry_hypotheses(
            self.log, self.profile, session_id=2
        )
        self.assertEqual(report["session_count"], 2)
        self.assertEqual(report["cases"][0]["samples"], 2)
        self.assertAlmostEqual(report["cases"][0]["duration_s"], .25)
        self.assertEqual(len(trace), 2 * len(report["cases"]))

    def test_route_is_geometric_only_not_position_rmse(self):
        self.write_log([(500, 500), (500, 500)])
        route = load_nominal_route(self.route)
        report, _ = compare_odometry_hypotheses(
            self.log, self.profile, nominal_route=route
        )
        self.assertAlmostEqual(
            report["cases"][0]["nominal_route_rms_distance_m"], 0.0
        )
        self.assertIsNone(report["best_case"])
        self.assertTrue(report["nominal_route_supplied"])
        self.assertAlmostEqual(
            route_distance_rms([(1., 1.)], [(0., 0.), (2., 0.)]), 1.0
        )

    def test_rejects_invalid_scales_and_malformed_route(self):
        self.write_log([(100, 100)])
        with self.assertRaises(ValueError):
            compare_odometry_hypotheses(
                self.log, self.profile, scale_candidates=(-1, 1)
            )
        with self.assertRaises(ValueError):
            compare_odometry_hypotheses(
                self.log, self.profile, measured_pose=(0, float("nan"), 0)
            )
        self.route.write_text('{"waypoints":[[1,1],[1,1]]}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_nominal_route(self.route)

    def calibration_fixture(self, path, left=1.2, right=1.21):
        profile = {
            "wheel_radius_m": .03425, "wheel_base_m": .229,
            "ticks_per_wheel_rev": 4560, "left_scale": 1.,
            "right_scale": 1., "swap_encoder_columns": True,
            "invert_left": True, "invert_right": False,
        }
        payload = {
            "schema": "bc_ekf.assisted_encoder_calibration.v1",
            "reference": {"kind": "straight", "value": 3.0},
            "profile_used": profile,
            "warnings": [],
            "suggestions_not_applied": {
                "left_scale_candidate": left,
                "right_scale_candidate": right,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_calibration_reports_averaged_only_when_profiles_match(self):
        p1 = self.root / "run1.json"
        p2 = self.root / "run2.json"
        self.calibration_fixture(p1, 1.2, 1.21)
        self.calibration_fixture(p2, 1.24, 1.19)
        result = load_straight_calibration_candidates([p1, p2], self.profile)
        self.assertAlmostEqual(result["left_scale"], 1.22)
        self.assertAlmostEqual(result["right_scale"], 1.2)
        self.assertEqual(result["count"], 2)
        incompatible = Robot2026Profile(ticks_per_wheel_rev=2280)
        with self.assertRaisesRegex(ValueError, "incompatível"):
            load_straight_calibration_candidates([p1], incompatible)

    def test_cli_export_and_existing_output_protected(self):
        from tools.compare_field_odometry_2026 import main
        self.write_log([(200, 200)])
        output = self.root / "summary.json"
        traces = self.root / "traces.csv"
        first = main([
            "--log", str(self.log), "--anchors", str(self.anchors),
            "--route", str(self.route), "--out", str(output),
            "--traces-csv", str(traces),
        ])
        self.assertEqual(first["schema"], "bc_ekf.field_odometry_hypotheses.v1")
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["best_case"], None)
        self.assertIn("hypothesis,timestamp_s", traces.read_text(encoding="utf-8"))
        before = output.read_bytes()
        with self.assertRaises(SystemExit):
            main([
                "--log", str(self.log), "--anchors", str(self.anchors),
                "--out", str(output),
            ])
        self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
