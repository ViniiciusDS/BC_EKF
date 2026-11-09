"""Regressões da calibração assistida: sinal, escala, reset, entrada intacta."""
import math
import tempfile
import unittest
from pathlib import Path

from src.odometry.pcntlog_adapter import Robot2026Profile
from src.odometry.assisted_calibration import (
    CalibrationReference, analyze_encoder_calibration,
)


class TestAssistedEncoderCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "pcntLog.csv"
        self.profile = Robot2026Profile(
            wheel_radius_m=.03425, wheel_base_m=.229,
            ticks_per_wheel_rev=4560,
            swap_encoder_columns=True, invert_left=True, invert_right=False,
        )
        self.mpt = 2 * math.pi * .03425 / 4560

    def rows(self, increments, reset=False):
        # CSV físico: coluna 2 WR, coluna 3 WL negativo.
        right = left = 0
        lines = ["0,0,0,0,0"]
        for i, (dl, dr) in enumerate(increments, 1):
            left += dl
            right += dr
            lines.append(f"{250*i},{right},{-left},0,0")
        if reset:
            lines.extend(["0,0,0,0,0", "250,100,-100,0,0"])
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_straight_recovers_individual_scale_candidates(self):
        self.rows([(1000, 1000), (1000, 1000)])
        before = self.path.read_bytes()
        target = 2000 * self.mpt * 1.10
        res = analyze_encoder_calibration(
            self.path, self.profile, CalibrationReference("straight", target)
        )
        self.assertAlmostEqual(res["suggestions_not_applied"]["left_scale_candidate"], 1.1)
        self.assertAlmostEqual(res["suggestions_not_applied"]["right_scale_candidate"], 1.1)
        self.assertAlmostEqual(res["observed"]["signed_relative_error"], -0.1 / 1.1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_straight_with_large_wheel_mismatch_only_warns(self):
        """Uma curva não deve calibrar escalas como se fosse reta perfeita."""
        self.rows([(1000, 1800)])
        result = analyze_encoder_calibration(
            self.path, self.profile,
            CalibrationReference("straight", 0.07),
        )
        self.assertEqual(result["suggestions_not_applied"], {})
        self.assertTrue(any("Diferença" in w for w in result["warnings"]))
        self.assertNotEqual(
            result["observed"]["left_wheel_m"],
            result["observed"]["right_wheel_m"],
        )

    def test_turn_recovers_base_candidate_from_opposite_wheels(self):
        self.rows([(-1000, 1000)])
        target_deg = math.degrees((2000 * self.mpt) / .25)
        res = analyze_encoder_calibration(
            self.path, self.profile, CalibrationReference("turn", target_deg)
        )
        self.assertAlmostEqual(res["suggestions_not_applied"]["wheel_base_m_candidate"], .25)
        self.assertEqual(res["warnings"], [])

    def test_turn_with_translation_refuses_wheelbase(self):
        """Uma curva fechada com centro deslocando não é giro estacionário."""
        self.rows([(-1000, 1600)])
        result = analyze_encoder_calibration(
            self.path, self.profile,
            CalibrationReference("turn", 30),
        )
        self.assertNotIn("wheel_base_m_candidate", result["suggestions_not_applied"])
        self.assertTrue(any("translação" in w for w in result["warnings"]))

    def test_turn_wrong_direction_does_not_suggest_positive_base(self):
        self.rows([(1000, -1000)])
        res = analyze_encoder_calibration(
            self.path, self.profile, CalibrationReference("turn", 40)
        )
        self.assertNotIn("wheel_base_m_candidate", res["suggestions_not_applied"])
        self.assertTrue(res["warnings"])

    def test_bad_straight_sign_does_not_propose_negative_scale(self):
        self.rows([(-400, 400)])
        res = analyze_encoder_calibration(
            self.path, self.profile, CalibrationReference("straight", 1.0)
        )
        self.assertNotIn("left_scale_candidate", res["suggestions_not_applied"])
        self.assertTrue(res["warnings"])

    def test_can_select_second_session_after_timestamp_reset(self):
        self.rows([(100, 100)], reset=True)
        res = analyze_encoder_calibration(
            self.path, self.profile, CalibrationReference("straight", 0.1),
            session_id=2,
        )
        self.assertEqual(res["session_id"], 2)
        self.assertEqual(res["samples"], 2)

    def test_window_is_relative_to_session_start(self):
        self.rows([(100, 100), (200, 200), (300, 300)])
        res = analyze_encoder_calibration(
            self.path, self.profile, CalibrationReference(
                "straight", .1, start_s=.25, end_s=.75
            )
        )
        # Inclusive t=.25/.50/.75 -> 3 samples, 500 encoder ticks total.
        self.assertEqual(res["samples"], 3)
        self.assertAlmostEqual(res["observed"]["left_wheel_m"], 500 * self.mpt)

    def test_rejects_empty_window_and_invalid_reference(self):
        self.rows([(100, 100)])
        with self.assertRaisesRegex(ValueError, "ao menos duas"):
            analyze_encoder_calibration(
                self.path, self.profile,
                CalibrationReference("straight", .5, start_s=9)
            )
        for ref in (CalibrationReference("straight", 0),
                    CalibrationReference("turn", float("nan")),
                    CalibrationReference("straight", -1)):
            with self.assertRaises(ValueError):
                analyze_encoder_calibration(self.path, self.profile, ref)


if __name__ == "__main__":
    unittest.main()
