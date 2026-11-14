"""Regressões do novo log PCNT+duas tags de novembro/2026.

A suíte cria dados 39-colunas sintéticos com cabeçalho REAL:
time_ms,left,right,t1_a1_avg/std,...,t2_a9_avg/std.
Não inclui nenhum log de laboratório no repositório. A ingestão
usa timestamps reais, preserva todos os encoders e deduplica UWB.
"""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.odometry.pcntlog_adapter import Robot2026Profile
from src.fusion.tag_geometry import resolve_tag_geometry
from src.ui.algo_modes.dataset_two_tag2026_pipeline import (
    load_dual_tag_pcnt,load_dual_tag_settings,
)
from src.uwb.algoritmos_estaticos import run_batch


class TestDualTagPcnt2026(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.anchors=self.root/"anchors.json"
        self.anchors.write_text(json.dumps({
            "anchors_xy":[
                [1.38,2.28,.85],[2.98,1.98,1.70],
                [2.76,-.70,.98],[1.38,-.70,.96],
                [0.,-.70,1.28],[1.38,1.20,1.53],
                [0.,2.68,1.76],
            ],
            "uwb_dual_2026":{
                "anchor_ids_uwb":[3,4,5,6,7,8,9],
                "range_unit":"cm","std_unit":"cm",
                "min_valid_anchors_per_tag":4,
                "min_sigma_m":.02,"pair_gap_tolerance_m":.05,
            },
        }),encoding="utf8")
        self.csv=self.root/"two_tags.csv"
        self.profile=Robot2026Profile(
            wheel_radius_m=.03425,wheel_base_m=.229,
            ticks_per_wheel_rev=4560,invert_right=True,
        )
        self.g=resolve_tag_geometry({
            "preset":"left_right","baseline_m":.36,
            "height_m":.215,"channel_order":["right","left"],
        },default_baseline_m=.25,default_height_m=.20)

    def write_log(self,*,count=30,updates=(5,10,15,20,25),bad_header=False):
        columns=["time_ms","left","right"]
        for tag in (1,2):
            for anchor in range(1,10):
                columns.extend([f"t{tag}_a{anchor}_avg",
                                f"t{tag}_a{anchor}_std"])
        if bad_header:
            columns[-1]="t2_a9_missing"
        rows=[]
        epoch=0
        for i in range(count):
            if i in updates:
                epoch+=1
            line=[343+i*250,100*i,-98*i]
            for tag in (1,2):
                for anchor in range(1,10):
                    if anchor<3 or epoch==0:
                        line.extend([0.,0.])
                    else:
                        # Distâncias legítimas aproximadas (cm) com
                        # offsets distintos e sigma em centímetros.
                        line.extend([
                            200.+(anchor-3)*6+(tag-1)*12+epoch*2,
                            2.+.1*(tag-1),
                        ])
            rows.append(line)
        with self.csv.open("w",encoding="utf8",newline="") as file:
            writer=csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)
        return self.csv

    def load(self):
        return load_dual_tag_pcnt(
            self.csv,self.anchors,profile=self.profile,
            baseline_m=self.g.baseline_m,session_id=1,
        )

    def test_real_column_layout_preserves_pcnt_and_5_fresh_joint_updates(self):
        self.write_log()
        raw_before=self.csv.read_bytes()
        record=self.load()
        self.assertEqual(record.odometry.sample_count,30)
        self.assertEqual(record.sample_count,5)
        np.testing.assert_array_equal(
            record.sample_indices,[5,10,15,20,25]
        )
        self.assertEqual(record.repeated_rows,24)
        self.assertEqual(record.rejected_rows,1)
        self.assertEqual(record.partial_refresh_rows,0)
        self.assertEqual(record.anchor_ids,(3,4,5,6,7,8,9))
        self.assertEqual(record.header_anchor_ids,tuple(range(1,10)))
        self.assertEqual(record.distances_by_tag.shape,(5,2,7))
        self.assertEqual(record.deviations_by_tag.shape,(5,2,7))
        self.assertEqual(record.z_hist.shape,(14,30))
        np.testing.assert_allclose(record.timestamps_s,
                                   record.odometry.timestamps_s[record.sample_indices])
        self.assertTrue(np.isnan(record.z_hist[:,6]).all())
        np.testing.assert_allclose(record.z_hist[:,5],
                                   record.interleaved_distances[0])
        self.assertAlmostEqual(record.distances_by_tag[0,0,0],2.02)
        self.assertAlmostEqual(record.distances_by_tag[0,1,0],2.14)
        self.assertAlmostEqual(record.deviations_by_tag[0,0,0],.02)
        self.assertAlmostEqual(record.deviations_by_tag[0,1,0],.021)
        self.assertEqual(self.csv.read_bytes(),raw_before)
        # Encoder continua progredindo nos 30 instantes,
        # inclusive quando não houve um UWB novo.
        self.assertGreater(record.odometry.poses_xytheta[-1,0],0.)
        self.assertGreater(record.odometry.timestamps_s[-1],7.)

    def test_two_tag_orientation_and_center_offset(self):
        self.assertEqual(self.g.channels,("right","left"))
        np.testing.assert_allclose(self.g.offsets,[
            [0.,-.18,.215],[0.,.18,.215]
        ])
        self.assertAlmostEqual(self.g.baseline_m,.36)
        self.write_log()
        record=self.load()
        self.assertTrue(np.all(np.isfinite(record.interleaved_distances)))
        self.assertTrue(np.all(np.isfinite(record.interleaved_deviations)))

    def test_pcnt_provenance_and_bc_batch_compare_same_fresh_uwb_indices(self):
        self.write_log()
        rec=self.load()
        pcnt=rec.odometry
        Ts=np.diff(pcnt.timestamps_s)
        kwargs={
            "T":float(np.median(Ts)),
            "timestamps_s":pcnt.timestamps_s,
            "anchors":np.asarray(json.loads(self.anchors.read_text())["anchors_xy"]).T,
            "odometry_noisy":np.vstack([pcnt.v_m_s,pcnt.omega_rad_s]),
            "z_hist":rec.z_hist,
            "output_indices":rec.sample_indices,
            "l":.18,"z_c":.215,"sigma_uwb":.15,
            "tag_offsets_xyz":self.g.offsets,
            "x0":pcnt.poses_xytheta[0],
        }
        results=run_batch(
            anchors_Nx3=kwargs["anchors"].T,
            distances=rec.distances_by_tag[:,0,:],
            deviations=rec.deviations_by_tag[:,0,:],
            algoritmos=["bc_ekf"],
            bc_ekf_data=kwargs,
        )
        r=results["bc_ekf"]
        self.assertEqual(r["posicoes"].shape,(5,3))
        self.assertEqual(r["full_pcnt_poses"].shape,(30,3))
        np.testing.assert_allclose(
            r["posicoes"],r["full_pcnt_poses"][rec.sample_indices]
        )
        # Sem UWB em k<5, o EKF fica em predição nos instantes iniciais.
        np.testing.assert_allclose(
            r["full_pcnt_poses"][:5],
            pcnt.poses_xytheta[:5],atol=1e-10
        )

    def test_bad_header_anchor_mapping_and_units_are_rejected(self):
        self.write_log(bad_header=True)
        with self.assertRaisesRegex(ValueError,"Campo não reconhecido"):
            self.load()
        self.write_log()
        root=json.loads(self.anchors.read_text())
        root["uwb_dual_2026"]["range_unit"]="m"
        self.anchors.write_text(json.dumps(root))
        with self.assertRaisesRegex(ValueError,"range_unit=cm"):
            self.load()
        root["uwb_dual_2026"]["range_unit"]="cm"
        root["uwb_dual_2026"]["anchor_ids_uwb"]=[2,3,4,5,6,7,9]
        self.anchors.write_text(json.dumps(root))
        with self.assertRaisesRegex(ValueError,"Nenhum snapshot"):
            self.load()

    def test_bad_time_and_session_boundary_never_joined(self):
        self.write_log(count=30,updates=(5,10,15,20,25))
        with self.csv.open("a",encoding="utf8",newline="") as out:
            writer=csv.writer(out)
            line=list(csv.reader(self.csv.open()))[5]
            line[0]=100
            writer.writerow(line)
        record=self.load()
        self.assertEqual(record.odometry.total_sessions,2)
        with self.assertRaisesRegex(ValueError,"Sessão"):
            load_dual_tag_pcnt(
                self.csv,self.anchors,profile=self.profile,
                baseline_m=.36,session_id=3
            )

    def test_ui_distinguishes_dual_from_one_tag_and_preserves_old_loader(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"src/ui/algo_modes/dataset_mode.py").read_text(encoding="utf8")
        modal=(root/"src/ui/algo_modes/dataset_modal.py").read_text(encoding="utf8")
        self.assertIn("real_odom_uwb_dual_2026",modal)
        self.assertIn("Real (PCNT + 2 tags 2026)",modal)
        self.assertIn("def _apply_robot2026_dual_tag_config(",source)
        self.assertIn("load_dual_tag_pcnt(",source)
        self.assertIn("def _apply_robot2026_single_tag_config(",source)
        self.assertIn('run_one_tag_ekf = (',source)


if __name__=="__main__":
    unittest.main()
