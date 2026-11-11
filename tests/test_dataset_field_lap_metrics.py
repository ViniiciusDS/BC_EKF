"""Regressões PR13: métricas por volta física, sem ground truth nem duplicação."""
from __future__ import annotations

import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analysis.field_lap_metrics import (
    build_field_lap_metrics, save_field_lap_metrics,
)
from src.ui.algo_modes.dataset_lap_view import LapSlice


class TestFieldLapMetrics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.t = np.arange(8, dtype=float)
        self.odom = np.column_stack((
            np.arange(8, dtype=float), np.zeros(8), np.zeros(8)
        ))
        self.indices = np.asarray([0, 2, 3, 4, 6, 7], dtype=int)
        self.laps = (
            LapSlice(1, 0, 3, -1, False),
            LapSlice(2, 2, 6, -1, True),
            LapSlice(3, 5, 8, -1, False),
        )
        x = self.odom[self.indices, :2].copy()
        self.estimates = {"lms": {
            "posicoes": x + [0., .2],
            "algo_key_actual": "lms",
        }}
        self.nominal_route = [[-1., 0.], [10., 0.]]

    def analyze(self, **kwargs):
        return build_field_lap_metrics(
            self.laps, self.t, self.odom, self.indices,
            self.estimates,
            reference_route=kwargs.get("reference_route", self.nominal_route),
            gate_diagnostics={"crossing_times_s": [3., 6.]},
        )

    def test_unique_pcnt_and_uwb_samples_at_shared_drawing_boundaries(self):
        """LapSlice sobrepõe um ponto no desenho, não nas estatísticas."""
        before_t, before_odom = self.t.copy(), self.odom.copy()
        report = self.analyze()
        pieces = report["segments"]
        self.assertEqual([(s["pcnt_start_inclusive"], s["pcnt_stop_exclusive"])
                          for s in pieces], [(0, 3), (3, 6), (6, 8)])
        self.assertEqual([s["pcnt_samples"] for s in pieces], [3, 3, 2])
        self.assertEqual([s["uwb_snapshots"] for s in pieces], [2, 2, 2])
        self.assertEqual(sum(s["pcnt_samples"] for s in pieces), len(self.t))
        self.assertEqual(sum(s["uwb_snapshots"] for s in pieces), len(self.indices))
        self.assertEqual(report["complete_laps"], 1)
        self.assertEqual(report["partial_segments"], 2)
        self.assertEqual(report["gate_returns"], 2)
        np.testing.assert_array_equal(self.t, before_t)
        np.testing.assert_array_equal(self.odom, before_odom)

    def test_nominal_distance_and_disagreement_are_distinct_and_not_gt(self):
        report = self.analyze()
        data = report["segments"][1]["algorithms"]["lms"]
        self.assertAlmostEqual(data["nominal_route_distance"]["rms_m"], .2)
        self.assertAlmostEqual(data["nominal_route_distance"]["mean_m"], .2)
        self.assertAlmostEqual(data["odometry_disagreement"]["rms_m"], .2)
        self.assertAlmostEqual(data["valid_fraction"], 1.)
        self.assertEqual(data["implementation"], "lms")
        self.assertTrue(report["reference_type"].endswith("not_ground_truth"))
        self.assertEqual(report["segmentation_type"], "physical_start_gate")
        self.assertFalse(any("rmse_position" in str(x) for x in report["segments"]))

    def test_closure_gap_is_diagnostic_not_real_drift(self):
        report = self.analyze()
        odom = report["segments"][1]["odometry"]
        self.assertAlmostEqual(odom["path_length_m"], 2.)
        self.assertAlmostEqual(odom["closure_gap_m"], 2.)
        self.assertTrue(odom["closure_is_not_ground_truth_error"])
        self.assertAlmostEqual(
            report["segments"][1]["algorithms"]["lms"]["estimated_closure_gap_m"], 1.
        )

    def test_invalid_estimates_keep_time_alignment_and_valid_coverage(self):
        self.estimates["lms"]["posicoes"][3] = [float("nan"), float("nan")]
        report = self.analyze()
        values = report["segments"][1]["algorithms"]["lms"]
        self.assertEqual(values["valid_samples"], 1)
        self.assertEqual(values["invalid_samples"], 1)
        self.assertAlmostEqual(values["valid_fraction"], .5)
        self.assertEqual(values["nominal_route_distance"]["count"], 1)
        self.assertIsNone(values["estimated_closure_gap_m"])
        self.assertEqual(report["segments"][1]["uwb_snapshots"], 2)

    def test_missing_nominal_route_does_not_fabricate_accuracy(self):
        report = self.analyze(reference_route=None)
        for piece in report["segments"]:
            self.assertIsNone(piece["algorithms"]["lms"]["nominal_route_distance"])
            self.assertIsNotNone(piece["algorithms"]["lms"]["odometry_disagreement"])

    def test_no_snapshots_inside_one_segment_is_explicit(self):
        self.indices = np.asarray([0, 2, 6, 7], dtype=int)
        self.estimates = {"lms": {
            "posicoes": self.odom[self.indices, :2].copy()
        }}
        report = self.analyze()
        empty = report["segments"][1]
        self.assertEqual(empty["uwb_snapshots"], 0)
        res = empty["algorithms"]["lms"]
        self.assertEqual(res["valid_samples"], 0)
        self.assertIsNone(res["valid_fraction"])
        self.assertIsNone(res["nominal_route_distance"])
        self.assertIsNone(res["odometry_disagreement"])
        self.assertIsNone(res["estimated_closure_gap_m"])

    def test_rejects_unsynchronized_and_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "índices UWB"):
            build_field_lap_metrics(
                self.laps, self.t, self.odom,
                np.asarray([0, 3, 3, 4, 6, 7]),
                self.estimates,
            )
        with self.assertRaisesRegex(ValueError, "posições XY"):
            build_field_lap_metrics(
                self.laps, self.t, self.odom, self.indices,
                {"lms": {"posicoes": np.zeros((5, 2))}},
            )
        with self.assertRaisesRegex(ValueError, "volta física completa"):
            build_field_lap_metrics(
                [LapSlice(1, 0, 8, -1, False)], self.t, self.odom,
                self.indices, self.estimates,
            )
        with self.assertRaises(ValueError):
            build_field_lap_metrics(
                self.laps, self.t[::-1], self.odom,
                self.indices, self.estimates,
            )

    def test_multiple_estimators_keep_independent_valid_counts(self):
        second = self.odom[self.indices, :2] + [0, .5]
        second[1] = [np.nan, np.nan]
        self.estimates["gauss_newton"] = {
            "posicoes": second, "algo_key_actual": "gauss_newton"
        }
        report = self.analyze()
        self.assertEqual(report["algorithm_keys"], ["lms", "gauss_newton"])
        per = report["segments"][0]["algorithms"]
        self.assertEqual(per["lms"]["valid_samples"], 2)
        self.assertEqual(per["gauss_newton"]["valid_samples"], 1)
        self.assertAlmostEqual(
            per["gauss_newton"]["nominal_route_distance"]["rms_m"], .5
        )

    def test_json_and_csv_export_incremental_does_not_overwrite(self):
        report = self.analyze()
        first, csv_first = save_field_lap_metrics(report, self.folder, "LAB7A_s01")
        raw1 = first.read_bytes()
        self.assertEqual(json.loads(raw1)["complete_laps"], 1)
        with csv_first.open("r", encoding="utf-8") as stream:
            data = list(csv.DictReader(stream))
        self.assertEqual(len(data), 3)
        self.assertEqual([row["complete"] for row in data],
                         ["False", "True", "False"])
        self.assertEqual(data[1]["algorithm"], "lms")
        second, csv_second = save_field_lap_metrics(report, self.folder, "LAB7A_s01")
        self.assertNotEqual(first, second)
        self.assertNotEqual(csv_first, csv_second)
        self.assertEqual(first.read_bytes(), raw1)

    def test_export_cleans_only_its_own_partial_json_on_serialization_error(self):
        """JSON inválido não pode deixar artefato quebrado nem apagar antigos."""
        old, _ = save_field_lap_metrics(self.analyze(), self.folder, "LAB7A")
        original = old.read_bytes()
        broken = self.analyze()
        broken["bad_numeric_value"] = float("nan")
        with self.assertRaises(ValueError):
            save_field_lap_metrics(broken, self.folder, "LAB7A")
        self.assertEqual(old.read_bytes(), original)
        self.assertFalse((self.folder / "LAB7A_voltas_002.json").exists())
        self.assertFalse((self.folder / "LAB7A_voltas_002.csv").exists())

    def test_rejects_unsafe_export_stem(self):
        with self.assertRaises(ValueError):
            save_field_lap_metrics(self.analyze(), self.folder, "../bad")

    def test_real_gui_hotkey_is_wired_and_stale_laps_reset(self):
        """Teste AST da ligação real no Pygame sem importar dependências GUI."""
        root = Path(__file__).resolve().parents[1]
        source = (root / "src/ui/algo_modes/dataset_mode.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        klass = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "DatasetMode")
        methods = {node.name: node for node in klass.body
                   if isinstance(node, ast.FunctionDef)}
        self.assertIn("_export_field_lap_metrics", methods)
        calls = [
            node for node in ast.walk(methods["handle_events"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_export_field_lap_metrics"
        ]
        self.assertTrue(calls, "F7 precisa chamar o exportador")
        f7 = [node for node in ast.walk(methods["handle_events"])
              if isinstance(node, ast.Attribute) and node.attr == "K_F7"]
        self.assertTrue(f7)
        exports = {node.func.id for node in ast.walk(methods["_export_field_lap_metrics"])
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("build_field_lap_metrics", exports)
        self.assertIn("save_field_lap_metrics", exports)
        guard = ast.get_source_segment(source, methods["_export_field_lap_metrics"])
        self.assertIn("self._odom_is_time_window", guard)
        refresh = ast.get_source_segment(source, methods["_refresh_uwb_gate_laps_after_batch"])
        self.assertIn("self._odom_gate_diagnostics = None", refresh)
        self.assertIn("split_time_windows", refresh)


if __name__ == "__main__":
    unittest.main()
