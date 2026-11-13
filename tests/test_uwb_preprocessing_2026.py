"""PR17: contrato genérico UWB, políticas opt-in e regressão de batch/UI.

Testes sintéticos e independentes do laboratório: não exigem pygame.
O simulador deve continuar idêntico quando não há política configurada.
"""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.uwb.preprocessing import (
    BELOW_VERTICAL_MIN,
    EXCESSIVE_RANGE,
    EXCESSIVE_REPORTED_SIGMA,
    INVALID_REPORTED_SIGMA,
    MISSING_RANGE,
    NONPOSITIVE_RANGE,
    POSSIBLY_REPEATED,
    UwbPreprocessConfig,
    load_uwb_preprocess_config,
    preprocess_uwb_batch,
)
from src.uwb.algoritmos_estaticos import run_batch


class TestGenericUwbPreprocessing(unittest.TestCase):
    def setUp(self):
        self.anchors = np.array([
            [0., 0., 1.], [2., 0., 1.], [0., 2., 1.],
            [2., 2., 1.], [1., 1., 1.],
        ])
        p = np.array([.6, .4, .2])
        base = np.linalg.norm(self.anchors-p, axis=1)
        self.ranges = np.tile(base, (4, 1))
        self.sigmas = np.full_like(self.ranges, .02)

    def test_off_is_bitwise_legacy_passthrough_including_nonfinite(self):
        self.ranges[1, 2] = np.nan
        self.ranges[2, 0] = -255.
        self.sigmas[3, 4] = np.inf
        before_d, before_s = self.ranges.copy(), self.sigmas.copy()
        for config in (None, UwbPreprocessConfig(), {"mode": "off"}):
            result = preprocess_uwb_batch(
                self.anchors, self.ranges, self.sigmas, config=config
            )
            np.testing.assert_array_equal(result.distances, before_d)
            np.testing.assert_array_equal(result.deviations, before_s)
            self.assertEqual(result.summary["flagged_cells"], 0)
            self.assertEqual(result.summary["newly_masked_finite_positive_cells"], 0)
            self.assertEqual(np.count_nonzero(result.flags), 0)
        self.assertIsNot(before_d, result.distances)
        np.testing.assert_array_equal(self.ranges, before_d)
        np.testing.assert_array_equal(self.sigmas, before_s)

    def test_flag_reports_reasons_but_never_edits_cells(self):
        self.ranges[0, 0] = .5  # menor que |1.0-0.2|=.8
        self.ranges[0, 1] = 9.  # acima do máximo
        self.ranges[1, 2] = -1
        self.ranges[2, 3] = np.nan
        self.sigmas[3, 1] = .5
        self.sigmas[3, 2] = np.nan
        d, s = self.ranges.copy(), self.sigmas.copy()
        cfg = UwbPreprocessConfig(
            mode="flag", tag_heights_m=.2, max_range_m=5.,
            max_reported_sigma_m=.3,
        )
        result = preprocess_uwb_batch(self.anchors,d,s,config=cfg)
        for i,j,reason in [
            (0,0,BELOW_VERTICAL_MIN), (0,1,EXCESSIVE_RANGE),
            (1,2,NONPOSITIVE_RANGE), (2,3,MISSING_RANGE),
            (3,1,EXCESSIVE_REPORTED_SIGMA), (3,2,INVALID_REPORTED_SIGMA)
        ]:
            self.assertTrue(result.flags[i,j] & reason)
        self.assertEqual(result.summary["flagged_cells"], 6)
        self.assertEqual(result.summary["newly_masked_finite_positive_cells"], 0)
        np.testing.assert_array_equal(result.distances, d)
        np.testing.assert_array_equal(result.deviations, s)
        self.assertEqual(result.summary["per_anchor_flagged_cells"], [1,2,2,1,0])

    def test_mask_preserves_rows_and_pcnt_sample_alignment(self):
        self.ranges[1, 0] = .2
        self.ranges[2, 1] = 500.
        before = self.ranges.copy()
        idx = np.array([10, 40, 70, 100], dtype=np.int64)
        times = np.array([.10, .25, .75, 1.25])
        out = preprocess_uwb_batch(
            self.anchors, self.ranges, self.sigmas,
            config={"mode":"mask", "tag_heights_m":.2, "max_range_m":5.},
            sample_indices=idx, timestamps_s=times,
        )
        self.assertEqual(out.distances.shape, self.ranges.shape)
        self.assertTrue(np.isnan(out.distances[1,0]))
        self.assertTrue(np.isnan(out.deviations[1,0]))
        self.assertTrue(np.isnan(out.distances[2,1]))
        self.assertEqual(out.summary["newly_masked_finite_positive_cells"], 2)
        self.assertFalse(out.accepted_mask[1,0])
        np.testing.assert_array_equal(out.sample_indices, idx)
        np.testing.assert_array_equal(out.timestamps_s, times)
        np.testing.assert_array_equal(self.ranges, before)

    def test_two_tags_supported_with_explicit_heights(self):
        ranges = np.full((3, 2, 5), 2.)
        ranges[0, 0, 0] = .5   # centro tag baixo: impossível
        ranges[0, 1, 0] = .5   # tag alta: possível
        result = preprocess_uwb_batch(
            self.anchors,ranges,
            config=UwbPreprocessConfig(
                mode="mask", tag_heights_m=(.2, 1.1)
            ),
        )
        self.assertEqual(result.summary["rows"], 3)
        self.assertEqual(result.summary["tags"], 2)
        self.assertEqual(result.summary["total_cells"], 30)
        self.assertTrue(result.flags[0,0,0] & BELOW_VERTICAL_MIN)
        self.assertEqual(result.flags[0,1,0],0)
        self.assertTrue(np.isnan(result.distances[0,0,0]))
        self.assertAlmostEqual(result.distances[0,1,0], .5)
        with self.assertRaisesRegex(ValueError, "2 alturas"):
            preprocess_uwb_batch(
                self.anchors,ranges,
                config={"mode":"flag", "tag_heights_m":[.2]}
            )

    def test_repeat_detection_is_opt_in_and_only_consecutive(self):
        ranges = self.ranges.copy()
        sigmas = self.sigmas.copy()
        ranges[2] += .1
        no_repeat = preprocess_uwb_batch(
            self.anchors,ranges,sigmas,config={"mode":"flag"}
        )
        self.assertEqual(no_repeat.summary["flag_counts"]["possibly_repeated_consecutively"],0)
        flagged = preprocess_uwb_batch(
            self.anchors,ranges,sigmas,
            config={"mode":"flag","detect_consecutive_repeats":True}
        )
        self.assertEqual(
            flagged.summary["flag_counts"]["possibly_repeated_consecutively"],5
        )
        self.assertFalse(np.any(flagged.flags[0] & POSSIBLY_REPEATED))
        self.assertTrue(np.all(flagged.flags[1] & POSSIBLY_REPEATED))
        # Igualdade das distâncias NÃO basta se sigma muda na linha.
        sigmas[1,0] = .03
        modified = preprocess_uwb_batch(
            self.anchors,ranges,sigmas,
            config={"mode":"flag","detect_consecutive_repeats":True}
        )
        self.assertFalse(modified.flags[1,0] & POSSIBLY_REPEATED)

    def test_vertical_tolerance_is_explicit_not_hard_coded(self):
        self.ranges[0,0] = .78  # 2 cm abaixo do limite
        zero = preprocess_uwb_batch(
            self.anchors,self.ranges,
            config={"mode":"flag","tag_heights_m":.2}
        )
        tolerance = preprocess_uwb_batch(
            self.anchors,self.ranges,
            config={"mode":"flag","tag_heights_m":.2,
                    "vertical_tolerance_m":.03}
        )
        self.assertTrue(zero.flags[0,0] & BELOW_VERTICAL_MIN)
        self.assertFalse(tolerance.flags[0,0] & BELOW_VERTICAL_MIN)

    def test_shape_validation_rejects_wrong_data_without_truncation(self):
        with self.assertRaisesRegex(ValueError,"Quantidade de colunas"):
            preprocess_uwb_batch(self.anchors, self.ranges[:,:-1])
        with self.assertRaisesRegex(ValueError,"mesma forma"):
            preprocess_uwb_batch(self.anchors,self.ranges,self.sigmas[:-1])
        with self.assertRaisesRegex(ValueError,"Nx3"):
            preprocess_uwb_batch(
                self.anchors[:,:2],self.ranges,
                config={"mode":"flag","tag_heights_m":.2}
            )
        with self.assertRaisesRegex(ValueError,"requer deviations"):
            preprocess_uwb_batch(
                self.anchors,self.ranges,
                config={"mode":"flag","max_reported_sigma_m":.5}
            )
        with self.assertRaisesRegex(ValueError,"sample_indices"):
            preprocess_uwb_batch(self.anchors,self.ranges,sample_indices=[0,2,2,4])
        with self.assertRaisesRegex(ValueError,"timestamps_s"):
            preprocess_uwb_batch(self.anchors,self.ranges,timestamps_s=[0,.1,.1,.2])

    def test_policy_parameter_validation_fails_closed(self):
        for policy in (
            {"mode":"automatic_filter"},
            {"mode":"mask","max_range_m":-1},
            {"mode":"flag","tag_heights_m":float("nan")},
            {"mode":"mask","detect_consecutive_repeats":1},
            {"mode":"flag","typo":True},
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                UwbPreprocessConfig.from_dict(policy)
        with self.assertRaises(ValueError):
            preprocess_uwb_batch(self.anchors,self.ranges,config=123)

    def test_config_json_opt_in_and_no_implicit_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"anchors.json"
            p.write_text(json.dumps({"anchors_xy":self.anchors.tolist()}),
                         encoding="utf-8")
            self.assertIsNone(load_uwb_preprocess_config(p))
            p.write_text(json.dumps({
                "anchors_xy":self.anchors.tolist(),
                "uwb_preprocessing":{"mode":"flag","tag_heights_m":.4},
            }),encoding="utf-8")
            config = load_uwb_preprocess_config(p)
            self.assertEqual(config.mode,"flag")
            self.assertEqual(config.tag_heights_m,.4)
            p.write_text(json.dumps({"uwb_preprocessing":{"mode":"typo"}}),
                         encoding="utf-8")
            with self.assertRaises(ValueError):
                load_uwb_preprocess_config(p)

    def test_same_batch_algorithms_see_same_inputs_in_flag(self):
        args = dict(anchors_Nx3=self.anchors, distances=self.ranges,
                    deviations=self.sigmas, algoritmos=["lms","gauss_newton"])
        original = run_batch(**args)
        flagged = run_batch(**args,preprocess_config={"mode":"flag","max_range_m":5.})
        for algo in args["algoritmos"]:
            np.testing.assert_array_equal(
                original[algo]["posicoes"],flagged[algo]["posicoes"]
            )
            self.assertNotIn("preprocessing", original[algo])
            self.assertEqual(flagged[algo]["preprocessing"]["flagged_cells"],0)

    def test_mask_uses_same_excluded_cells_for_every_algorithm(self):
        anomalous = self.ranges.copy()
        anomalous[1,2] = 200.
        args = dict(anchors_Nx3=self.anchors,distances=anomalous,
                    deviations=self.sigmas,algoritmos=["lms","gauss_newton"])
        result = run_batch(
            **args,
            preprocess_config={"mode":"mask","max_range_m":20.}
        )
        self.assertEqual(result["lms"]["preprocessing"]["flagged_cells"],1)
        self.assertEqual(result["lms"]["preprocessing"],result["gauss_newton"]["preprocessing"])
        self.assertEqual(result["lms"]["posicoes"].shape,(4,3))
        self.assertAlmostEqual(anomalous[1,2],200.)
        self.assertTrue(np.any(np.isfinite(result["lms"]["posicoes"])))

    def test_bc_ekf_is_not_silently_preprocessed_as_single_tag(self):
        with self.assertRaisesRegex(ValueError,"BC-EKF"):
            run_batch(
                anchors_Nx3=self.anchors,distances=self.ranges,
                algoritmos=["bc_ekf"],
                preprocess_config={"mode":"mask","max_range_m":100.}
            )

    def test_dataset_gui_wires_generic_policy_to_static_batch(self):
        root = Path(__file__).resolve().parents[1]
        path = root/"src/ui/algo_modes/dataset_mode.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body
                   if isinstance(n,ast.ClassDef) and n.name=="DatasetMode")
        fn = next(n for n in cls.body
                  if isinstance(n,ast.FunctionDef) and n.name=="_run_batch")
        names = [
            node.func.id for node in ast.walk(fn)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
        ]
        self.assertIn("load_uwb_preprocess_config",names)
        run = next(n for n in ast.walk(fn)
                   if isinstance(n,ast.Call) and
                   isinstance(n.func,ast.Name) and n.func.id=="run_batch")
        self.assertIn("preprocess_config",[key.arg for key in run.keywords])


if __name__=="__main__":
    unittest.main()
