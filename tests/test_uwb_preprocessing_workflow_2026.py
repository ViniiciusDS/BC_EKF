"""PR17: seleção, prévia e exportação opcional de CSV UWB.

Os testes usam dados sintéticos e funcionam sem Pygame/Tk no CI.
Validação visual do modal e diálogo nativo permanece no Windows.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.uwb.preprocessing import UwbPreprocessConfig
from src.uwb.preprocessing_workflow import (
    _read_interleaved,
    execute_preprocessing,
    load_preprocessing_input,
    save_preprocessed_dataset,
)


class TestIndependentPreprocessingWorkflow(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        self.anchors=self.root/"anchors.json"
        self.anchors.write_text(json.dumps({
            "anchors_xy":[
                [0,0,1.], [2,0,1.], [0,2,1.],
                [2,2,1.], [1,1,1.],
            ],
            "pcnt_slot_to_anchor_index":[0,1,2,3,4],
            "robot_2026":{
                "wheel_radius_m":.03425,"wheel_base_m":.229,
                "ticks_per_wheel_rev":4560,
                "invert_left":False,"invert_right":True,
            },
            "uwb_2026":{"min_valid_anchors":4,"min_sigma_m":.02},
        }),encoding="utf-8")
        self.measure=self.root/"measure.csv"

    def write_pairs(self,dists,sigmas):
        a=np.asarray(dists,dtype=float)
        b=np.asarray(sigmas,dtype=float)
        if a.ndim==3:
            m,t,n=a.shape
            a=a.transpose(0,2,1).reshape(m,n*t)
            b=b.transpose(0,2,1).reshape(m,n*t)
        rows=np.empty((len(a),a.shape[1]*2))
        rows[:,0::2]=a
        rows[:,1::2]=b
        with self.measure.open("w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        return self.measure

    def test_independent_file_selection_does_not_touch_loaded_dataset(self):
        matrix=np.array([[1.2,1.5,1.7,1.8,2.0],
                         [1.2,1.5,.1,1.8,2.0],
                         [1.3,1.6,1.7,1.9,2.1]],dtype=float)
        sigmas=np.full_like(matrix,.02)
        self.write_pairs(matrix,sigmas)
        original_bytes=self.measure.read_bytes()
        source,xyz=load_preprocessing_input(
            self.measure,self.anchors,source_type="ranges"
        )
        result,report=execute_preprocessing(
            source,xyz,self.anchors,
            UwbPreprocessConfig(mode="mask",tag_heights_m=.2)
        )
        self.assertEqual(report["rows"],3)
        self.assertEqual(report["anchors"],5)
        self.assertEqual(report["flagged_cells"],1)
        self.assertEqual(report["newly_masked_finite_positive_cells"],1)
        self.assertEqual(report["first_30_flagged_cells"][0]["row"],1)
        self.assertEqual(report["first_30_flagged_cells"][0]["anchor_index"],2)
        self.assertTrue(np.isnan(result.distances[1,2]))
        np.testing.assert_allclose(source.distances,matrix)
        self.assertEqual(self.measure.read_bytes(),original_bytes)
        self.assertEqual(report["source_sha256"],hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(report["original_sample_indices"],[0,1,2])

        path=self.root/"exported.csv"
        csv_out,meta=save_preprocessed_dataset(path,result,report)
        self.assertEqual(csv_out,path)
        self.assertEqual(meta,path.with_suffix(".preproc.json"))
        loaded_d,loaded_s=_read_interleaved(path,5)
        np.testing.assert_allclose(loaded_d,result.distances,equal_nan=True)
        np.testing.assert_allclose(loaded_s,result.deviations,equal_nan=True)
        exported_report=json.loads(meta.read_text(encoding="utf-8"))
        self.assertEqual(exported_report["source_sha256"],report["source_sha256"])
        self.assertEqual(exported_report["first_30_flagged_cells"],report["first_30_flagged_cells"])
        self.assertEqual(self.measure.read_bytes(),original_bytes)

        old_csv=path.read_bytes()
        old_meta=meta.read_bytes()
        with self.assertRaises(FileExistsError):
            save_preprocessed_dataset(path,result,report)
        self.assertEqual(path.read_bytes(),old_csv)
        self.assertEqual(meta.read_bytes(),old_meta)

    def test_flag_leaves_ranges_identical_and_allows_opt_in_export(self):
        matrix=np.full((2,5),1.5)
        matrix[0,0]=.4
        self.write_pairs(matrix,np.full_like(matrix,.02))
        inp,xyz=load_preprocessing_input(self.measure,self.anchors,source_type="ranges")
        original=inp.distances.copy()
        res,report=execute_preprocessing(
            inp,xyz,self.anchors,
            {"mode":"flag","tag_heights_m":.2}
        )
        np.testing.assert_array_equal(res.distances,original)
        self.assertEqual(report["flagged_cells"],1)
        self.assertEqual(report["newly_masked_finite_positive_cells"],0)
        self.assertEqual(report["config"]["mode"],"flag")

    def test_pcnt_independent_input_retains_real_source_indices_and_times(self):
        pcnt=self.root/"pcntLog.csv"
        def row(ts,base):
            vals=[ts,ts//10,-ts//10,0,0]
            for j in range(5):
                vals.extend([base+10*j,4.,2.])
            return ",".join(map(str,vals))
        pcnt.write_text("\n".join([
            row(0,100),row(250,100),row(500,110)
        ])+"\n",encoding="utf-8")
        before=pcnt.read_bytes()
        inp,xyz=load_preprocessing_input(
            pcnt,self.anchors,source_type="pcnt",session_id=1
        )
        np.testing.assert_array_equal(inp.sample_indices,[0,2])
        np.testing.assert_allclose(inp.timestamps_s,[0,.5])
        self.assertEqual(inp.source_type,"pcnt")
        r,report=execute_preprocessing(
            inp,xyz,self.anchors,UwbPreprocessConfig(mode="flag")
        )
        self.assertEqual(report["original_sample_indices"],[0,2])
        self.assertEqual(report["original_timestamps_s"],[0.,.5])
        self.assertTrue(any("sem odometria" in n for n in report["source_notes"]))
        out,_=save_preprocessed_dataset(self.root/"uwb_only.csv",r,report)
        self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()),2)
        self.assertEqual(pcnt.read_bytes(),before)

    def test_two_tag_data_exports_interleaved_in_correct_order(self):
        d=np.array([[
            [1.,2.,3.,4.,5.],
            [11.,12.,13.,14.,15.],
        ]])
        s=np.full_like(d,.03)
        self.write_pairs(d,s)
        inp,anchors=load_preprocessing_input(
            self.measure,self.anchors,source_type="ranges"
        )
        self.assertEqual(inp.distances.shape,(1,2,5))
        np.testing.assert_array_equal(inp.distances,d)
        res,report=execute_preprocessing(
            inp,anchors,self.anchors,
            {"mode":"flag","tag_heights_m":[.2,1.1]}
        )
        self.assertEqual(report["tags"],2)
        out,_=save_preprocessed_dataset(self.root/"two_tags.csv",res,report)
        actual,actual_sigma=_read_interleaved(out,5)
        np.testing.assert_array_equal(actual,d)
        np.testing.assert_array_equal(actual_sigma,s)

    def test_invalid_or_mismatched_geometry_and_data_fail_without_output(self):
        self.measure.write_text("1,0.1,2,0.1,3\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Esperadas"):
            load_preprocessing_input(
                self.measure,self.anchors,source_type="ranges"
            )
        with self.assertRaises(ValueError):
            load_preprocessing_input(
                self.measure,self.anchors,source_type="wrong"
            )
        self.assertFalse((self.root/"result.csv").exists())

    def test_jsonl_legacy_front_tag_is_explicitly_documented(self):
        path=self.root/"data.jsonl"
        path.write_text(json.dumps({
            "z_k":[1.,11.,2.,12.,3.,13.,4.,14.,5.,15.]
        })+"\n",encoding="utf-8")
        inp,xyz=load_preprocessing_input(
            path,self.anchors,source_type="ranges"
        )
        self.assertEqual(inp.distances.shape,(1,5))
        np.testing.assert_array_equal(inp.distances[0],[1,2,3,4,5])
        self.assertIsNone(inp.deviations)
        self.assertIn("somente frontal",inp.notes[0])
        res,report=execute_preprocessing(
            inp,xyz,self.anchors,{"mode":"off"}
        )
        exported,_=save_preprocessed_dataset(
            self.root/"jsonl_front.csv",res,report
        )
        _,sigmas=_read_interleaved(exported,5)
        self.assertTrue(np.isnan(sigmas).all())

    def test_gui_button_modal_and_event_priority_wired_without_pygame_ci(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/"src/ui/algo_modes/dataset_mode.py").read_text(encoding="utf-8"))
        cls=next(n for n in tree.body
                 if isinstance(n,ast.ClassDef) and n.name=="DatasetMode")
        methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        joined=ast.unparse(methods["handle_events"])
        self.assertIn("self.preprocessing_modal.handle_event(event)",joined)
        self.assertIn("self.btn_preprocessor.hit(pos)",joined)
        # Modal deve cobrir o Dataset, não abrir seleção ao clicar em
        # outros lugares da tela e propagar o evento até Run batch.
        self.assertLess(
            joined.index("self.preprocessing_modal.handle_event(event)"),
            joined.index("self.btn_run_batch.hit(pos)")
        )
        draw=ast.unparse(methods["draw"])
        self.assertIn("self.btn_export.draw(self.host.screen)",draw)
        self.assertIn("self.btn_preprocessor.draw(self.host.screen)",draw)
        self.assertIn("self.preprocessing_modal.draw()",draw)
        modal=(root/"src/ui/algo_modes/preprocessing_modal.py").read_text(encoding="utf-8")
        self.assertIn('self._choose_file()',modal)
        self.assertIn('self._analyze()',modal)
        self.assertIn('self._save()',modal)
        self.assertIn('self._choose_anchors()',modal)
        self.assertNotIn("transform.smoothscale",modal)


if __name__=="__main__":
    unittest.main()
