"""Regressões do pré-diagnóstico de campo: nenhuma amostra é reescrita."""
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.field_diagnostic import (
    build_field_diagnostic, save_field_diagnostic,
)


class TestFieldDiagnostic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anchor = self.root / "anchors.json"
        self.anchor.write_text(json.dumps({
            "anchors_xy": [
                [0.77, 1.77, 1.85], [0, 0, 1.63],
                [-2.66, 0.13, 1.28], [-3.04, 2.1, 0.86],
            ],
            "pcnt_slot_to_anchor_index": [None, 0, 1, 2, 3],
            "robot_2026": {
                "wheel_radius_m": 0.03425,
                "wheel_base_m": 0.229,
                "ticks_per_wheel_rev": 4560,
                "swap_encoder_columns": True,
                "invert_left": True,
                "invert_right": False,
            },
            "uwb_2026": {"min_valid_anchors": 4},
        }), encoding="utf-8")

    @staticmethod
    def _row(t, step, base):
        # Cinco colunas PCNT, slot 0 inativo, quatro triplas em cm/cm2/cm.
        cols = [t, step * 50, -step * 50, 0, 0, 0, 0, -255]
        for slot in range(4):
            cols.extend([base + slot * 10, 4, 2])
        return ",".join(str(x) for x in cols)

    def test_counts_reused_and_new_without_modifying_log(self):
        file = self.root / "pcnt_test.csv"
        file.write_text("\n".join([
            self._row(0, 0, 180), self._row(250, 1, 180),
            self._row(500, 2, 181),
        ]) + "\n", encoding="utf-8")
        original = file.read_bytes()
        report = build_field_diagnostic(file, self.anchor, session_id=1)
        self.assertEqual(file.read_bytes(), original)
        self.assertEqual(report["summary"]["status"], "ready_for_static_batch")
        self.assertEqual(report["summary"]["uwb_snapshots"], 2)
        self.assertEqual(report["summary"]["pcnt_samples"], 3)
        self.assertEqual(report["slot_quality"][0]["invalid"], 3)
        self.assertEqual(report["slot_quality"][1]["new_valid"], 2)
        self.assertEqual(report["slot_quality"][1]["repeated_valid"], 1)
        out = save_field_diagnostic(report, self.root / "reports" / "check.json")
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["session_id"], 1)

    def test_separates_reset_sessions(self):
        file = self.root / "multi.csv"
        file.write_text("\n".join([
            self._row(0, 0, 180), self._row(250, 1, 181),
            self._row(0, 0, 200), self._row(250, 1, 201),
        ]) + "\n", encoding="utf-8")
        report = build_field_diagnostic(file, self.anchor, session_id=2)
        self.assertEqual(report["summary"]["sessions_detected"], 2)
        self.assertEqual(report["summary"]["pcnt_samples"], 2)

    def test_odometry_only_is_reported_not_fabricated(self):
        file = self.root / "encoder.csv"
        file.write_text("0,0,0,0,0\n250,100,-100,0,0\n")
        report = build_field_diagnostic(file, self.anchor)
        self.assertEqual(report["summary"]["status"], "odometry_only")
        self.assertEqual(report["slot_quality"], [])
        self.assertFalse(report["ground_truth_available"])


if __name__ == "__main__":
    unittest.main()
