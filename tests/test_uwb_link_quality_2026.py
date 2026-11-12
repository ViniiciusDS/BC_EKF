"""Regressões PR16: qualidade UWB por âncora sem modificar solvers/CSV."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.analysis.uwb_link_quality import build_uwb_link_quality_report
from tools.audit_uwb_links_2026 import main as cli_main


class TestUwbLinkQuality2026(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "log.csv"
        self.anchor_json = self.root / "anchors.json"
        self.anchors = [
            [-1., -1., 1.2], [1., -1., 1.2],
            [-1., 1., 1.2], [1., 1., 1.2],
        ]
        self.config = {
            "anchors_xy": self.anchors,
            "pcnt_slot_to_anchor_index": [None, 0, 1, 2, 3],
            "meta": {"example_only": False},
        }
        self.anchor_json.write_text(json.dumps(self.config), encoding="utf-8")

    def ranges_for(self, x, y):
        return [
            math.sqrt((x-a[0])**2 + (y-a[1])**2 + (a[2]-.2)**2)
            for a in self.anchors
        ]

    @staticmethod
    def row(ts, ranges, invalid_slot=None):
        values = [ts, 0, 0, 0, 0]
        # Slot SPI 0 sem mapeamento: sentinela do firmware.
        values += [0, 0, -255]
        for j, distance in enumerate(ranges, 1):
            if j == invalid_slot:
                values += [-255, -1, -1]
            else:
                values += [distance*100., 4., 2.]
        return ",".join(str(v) for v in values)

    def write(self, rows):
        self.log.write_text("\n".join(rows)+"\n", encoding="utf-8")
        return self.log

    def audit(self, **kwargs):
        return build_uwb_link_quality_report(
            self.log, self.anchor_json, tag_height_m=.2, **kwargs
        )

    def test_repeats_are_not_independent_and_LOAO_uses_only_fresh_same_row(self):
        a = self.ranges_for(.1, .2)
        b = self.ranges_for(.3, .1)
        outlier = list(b)
        outlier[3] += 1.0
        self.write([
            self.row(0, a),
            self.row(250, a),  # todas reutilizadas
            self.row(500, b),  # novas
            self.row(750, outlier),  # SÓ a quarta mudou
        ])
        original = self.log.read_bytes()
        result = self.audit(loa_outlier_threshold_m=.30)
        self.assertEqual(self.log.read_bytes(), original)
        self.assertEqual(result["schema"], "bc_ekf.uwb_link_quality.v1")
        self.assertEqual(result["pcnt_rows"], 4)
        self.assertEqual(result["mapped_slots"], 4)
        self.assertEqual(result["fresh_rows_with_4plus_mapped_ranges"], 2)
        self.assertEqual(result["loa_rows_with_at_least_one_solution"], 2)
        self.assertEqual(result["physically_impossible_new_total"], 0)
        self.assertEqual(result["per_slot"][0]["new_valid"], 0)
        self.assertEqual(result["per_slot"][0]["invalid"], 4)
        self.assertEqual(result["per_slot"][1]["new_valid"], 2)
        self.assertEqual(result["per_slot"][1]["repeated_valid"], 2)
        self.assertEqual(result["per_slot"][4]["new_valid"], 3)
        # Excluir âncora sem inventar novos ranges de outras linhas.
        self.assertEqual(result["per_slot"][4]["loa_attempted"], 2)
        self.assertEqual(result["per_slot"][4]["loa_solved"], 2)
        self.assertGreater(
            result["per_slot"][4]["loa_abs_residual_summary_m"]["max_m"], 0.
        )
        self.assertEqual(result["log_sha256"], hashlib.sha256(original).hexdigest())

    def test_LOAO_flags_large_residual_for_excluded_outlier(self):
        a = self.ranges_for(.35, .25)
        b = self.ranges_for(.45, .15)
        b[3] += .8
        self.write([self.row(0, a), self.row(250, b)])
        res = self.audit(loa_outlier_threshold_m=.30)
        fourth = res["per_slot"][4]
        self.assertEqual(fourth["loa_solved"], 2)
        self.assertEqual(fourth["loa_residual_above_threshold"], 1)
        self.assertGreater(
            fourth["loa_abs_residual_summary_m"]["max_m"], .5
        )
        # Não inferir âncora defeituosa: outras exclusões podem ter
        # resíduos altos ao usar a quarta âncora errada no ajuste.
        self.assertIn("não ground truth", "não ground truth")

    def test_physical_impossibility_is_counted_only_when_new(self):
        a = self.ranges_for(.2, .15)
        impossible = list(a)
        impossible[2] = .96  # separação vertical = 1.00 m
        self.write([
            self.row(0, a),
            self.row(250, impossible),
            self.row(500, impossible),  # não contar de novo
        ])
        res = self.audit()
        self.assertEqual(res["physically_impossible_new_total"], 1)
        flagged = res["per_slot"][3]
        self.assertEqual(flagged["physically_impossible_new"], 1)
        self.assertEqual(flagged["repeated_valid"], 1)
        self.assertEqual(res["impossible_new_examples_first_20"][0]["anchor_index_0_based"], 2)
        tol = self.audit(height_tolerance_m=.05)
        self.assertEqual(tol["physically_impossible_new_total"], 0)
        self.assertEqual(res["fresh_rows_with_4plus_mapped_ranges"], 1)

    def test_invalid_sentinel_does_not_mark_repeated_and_no_stale_imputation(self):
        a = self.ranges_for(0., 0.)
        b = self.ranges_for(.2, 0.)
        self.write([self.row(0, a), self.row(250, a, invalid_slot=2),
                    self.row(500, b, invalid_slot=2)])
        res = self.audit()
        state = res["per_slot"][2]
        self.assertEqual(state["invalid"], 2)
        self.assertEqual(state["new_valid"], 1)
        self.assertEqual(state["repeated_valid"], 0)
        self.assertEqual(res["fresh_rows_with_4plus_mapped_ranges"], 1)
        self.assertEqual(res["fresh_rows_with_3plus_mapped_ranges"], 2)
        self.assertEqual(res["loa_rows_with_at_least_one_solution"], 1)

    def test_session_reset_is_separate_and_not_concatenated(self):
        a = self.ranges_for(0., 0.)
        b = self.ranges_for(.1, .3)
        self.write([self.row(0,a), self.row(250,a),
                    self.row(0,b), self.row(250,b)])
        report = build_uwb_link_quality_report(
            self.log, self.anchor_json, session_id=2, tag_height_m=.2
        )
        self.assertEqual(report["sessions_detected"], 2)
        self.assertEqual(report["pcnt_rows"], 2)
        self.assertEqual(report["per_slot"][1]["new_valid"], 1)
        self.assertEqual(report["per_slot"][1]["repeated_valid"], 1)

    def test_mapping_is_explicit_in_report_and_unmapped_is_not_solved(self):
        a = self.ranges_for(.1, .2)
        self.config["pcnt_slot_to_anchor_index"] = [None, 2, 1, 0, 3]
        self.anchor_json.write_text(json.dumps(self.config), encoding="utf-8")
        self.write([self.row(0, a)])
        report = self.audit()
        self.assertEqual(report["per_slot"][1]["anchor_index_0_based"], 2)
        self.assertEqual(report["per_slot"][3]["anchor_index_0_based"], 0)
        self.assertFalse(report["per_slot"][0]["mapped"])
        self.assertIsNone(report["per_slot"][0]["loa_abs_residual_summary_m"])

    def test_requires_3d_geometry_and_explicit_valid_tag_height(self):
        a = self.ranges_for(.1, .2)
        self.write([self.row(0,a)])
        for bad in (-.1, float("nan"), True):
            with self.assertRaises(ValueError):
                build_uwb_link_quality_report(
                    self.log, self.anchor_json, tag_height_m=bad
                )
        self.config["anchors_xy"] = [v[:2] for v in self.anchors]
        self.anchor_json.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Nx3"):
            self.audit()

    def test_rejects_five_column_log_and_wrong_slot_count(self):
        self.write(["0,0,0,0,0", "250,0,0,0,0"])
        with self.assertRaisesRegex(ValueError, "somente PCNT"):
            self.audit()
        self.write([self.row(0, self.ranges_for(.1, .2))])
        self.config["pcnt_slot_to_anchor_index"] = [None, 0, 1, 2]
        self.anchor_json.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "slots"):
            self.audit()

    def test_cli_json_csv_are_consistent_and_existing_files_preserved(self):
        a = self.ranges_for(.1, .2)
        self.write([self.row(0, a), self.row(250, a)])
        original = self.log.read_bytes()
        out, summary = self.root / "result.json", self.root / "result.csv"
        result = cli_main([
            "--log",str(self.log), "--anchors",str(self.anchor_json),
            "--tag-height-m",".2", "--out",str(out), "--csv",str(summary),
        ])
        self.assertEqual(result["pcnt_rows"],2)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["schema"],
                         "bc_ekf.uwb_link_quality.v1")
        with summary.open("r", encoding="utf-8", newline="") as src:
            data = list(csv.DictReader(src))
        self.assertEqual(len(data), 5)
        self.assertEqual(int(data[4]["loa_solved"]), 1)
        self.assertEqual(original, self.log.read_bytes())
        old = out.read_bytes()
        with self.assertRaises(SystemExit):
            cli_main([
                "--log",str(self.log), "--anchors",str(self.anchor_json),
                "--tag-height-m",".2", "--out",str(out),
            ])
        self.assertEqual(out.read_bytes(), old)

    def test_invalid_cli_does_not_create_partial_output(self):
        self.write([self.row(0,self.ranges_for(.1,.2))])
        out = self.root / "unwritten.json"
        with self.assertRaises(SystemExit):
            cli_main([
                "--log",str(self.log), "--anchors",str(self.anchor_json),
                "--tag-height-m","nan", "--out",str(out),
            ])
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
