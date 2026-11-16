"""PR20: regressões da auditoria angular independente de duas tags.

GT conhecido APENAS nesta trajetória sintética. Não usa Pygame,
SciPy, CSV real, pseudo-GT nem altera a parametrização de encoders.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analysis.dual_tag_heading_audit import (
    HeadingAuditConfig,audit_dual_tag_heading,
)
from tools.audit_dual_tag_heading_2026 import save_heading_audit,main


def _trajectory(n=401,dt=.25,*,angular_start_deg=10,angular_speed_deg_s=1):
    t=np.arange(n)*dt
    th=np.deg2rad(angular_start_deg+angular_speed_deg_s*t)
    xy=np.column_stack((1.+.55*np.sin(.04*t),1.+.4*np.cos(.04*t)))
    # Deriva introduzida deliberadamente nos encoders (NÃO no UWB).
    odom_theta=th+np.deg2rad(.20*t)
    odo=np.column_stack((xy,odom_theta))
    return t,th,xy,odo


class TestDualTagHeadingAudit(unittest.TestCase):
    def setUp(self):
        self.anchors=np.asarray([
            [-.5,-.5,.9],[3.,-.5,1.4],[-.5,3.,1.1],
            [3.,3.,1.7],[1.3,-1.,1.55],[1.3,3.4,1.8],
            [.1,1.3,1.2],
        ])
        # T1=direita, T2=esquerda em metros (frame robô).
        self.offsets=np.asarray([[0.,-.18,.215],[0.,+.18,.215]])
        self.cfg=HeadingAuditConfig(
            sigma_floor_m=.08,max_heading_sigma_deg=55,
            max_fit_rms_m=.22,max_baseline_error_m=.12,
            window_s=10,focus_start_s=80,focus_end_s=120,
        )

    def make_input(self,*,n=401,start=10,speed=1,noise=0.):
        t,th,xy,odom=_trajectory(
            n,angular_start_deg=start,angular_speed_deg_s=speed
        )
        idx=np.arange(0,n,8,dtype=int)
        uwb=np.empty((len(idx),2,len(self.anchors)))
        for j,k in enumerate(idx):
            ct,st=np.cos(th[k]),np.sin(th[k])
            for tag,local in enumerate(self.offsets):
                px=xy[k,0]+ct*local[0]-st*local[1]
                py=xy[k,1]+st*local[0]+ct*local[1]
                delta=self.anchors-np.array([px,py,local[2]])
                uwb[j,tag]=np.linalg.norm(delta,axis=1)
        if noise:
            rng=np.random.default_rng(412)
            uwb+=rng.normal(0,noise,uwb.shape)
        sigma=np.full_like(uwb,.02)
        dl=np.full(len(t),34.,dtype=float)
        dr=np.full(len(t),33.,dtype=float)
        dl[0]=dr[0]=0.
        return dict(
            timestamps_s=t,odometry_xytheta=odom,
            pcnt_delta_left_ticks=dl,pcnt_delta_right_ticks=dr,
            sample_indices=idx,distances_by_tag=uwb,
            deviations_by_tag=sigma,anchors_xyz=self.anchors,
            tag_offsets_xyz=self.offsets,config=self.cfg,
        )

    def test_known_synthetic_angular_drift_is_recovered_without_gt_in_report(self):
        p=self.make_input()
        report=audit_dual_tag_heading(**p)
        s=report["summary"]
        self.assertEqual(s["pcnt_samples"],401)
        self.assertEqual(s["uwb_epochs"],51)
        self.assertEqual(s["accepted_heading_epochs"],51)
        rows=report["uwb_snapshots"]
        self.assertEqual(rows[0]["quality_reason"],"ok")
        self.assertAlmostEqual(rows[0]["relative_drift_deg"],0.,places=5)
        self.assertAlmostEqual(rows[-1]["relative_drift_deg"],-20.,delta=1.)
        self.assertEqual(len(report["pcnt_steps"]),401)
        self.assertEqual(len(report["windows"]),11)
        # A série sintética termina em 100s; só 80–90, 90–100,
        # e 100–110 (última janela vazia) podem intersectar o foco.
        self.assertEqual(len(s["focus_window"]),3)
        self.assertEqual(report["geometry"]["baseline_m"],.36)
        self.assertIn("não é GT",report["warnings"][0])

    def test_heading_does_not_falsely_jump_across_plus_minus_180(self):
        p=self.make_input(start=165,speed=3.3)
        # Sem deriva de encoder: heading verdadeiro permanece contínuo
        # apesar de atan2(UWB) atravessar ±180 várias vezes.
        t=p["timestamps_s"]
        p["odometry_xytheta"][:,2]=np.deg2rad(165+3.3*t)
        r=audit_dual_tag_heading(**p)
        ok=[x for x in r["uwb_snapshots"] if x["valid"]]
        self.assertGreater(len(ok),35)
        self.assertLess(max(abs(x["relative_drift_deg"]) for x in ok),2)
        self.assertLess(max(abs(x["difference_unwrapped_deg"]) for x in ok),2)

    def test_missing_and_inconsistent_ranges_are_flagged_not_interpolated(self):
        p=self.make_input(noise=.005)
        p["distances_by_tag"][4,0,:]=np.nan
        p["distances_by_tag"][7,1,:]+=2.
        output=audit_dual_tag_heading(**p)
        rows=output["uwb_snapshots"]
        self.assertFalse(rows[4]["valid"])
        self.assertEqual(rows[4]["quality_reason"],"insufficient_or_singular_ranges")
        self.assertFalse(rows[7]["valid"])
        self.assertIn(rows[7]["quality_reason"],
                      ("range_fit_residual","tag_baseline_inconsistent"))
        self.assertIsNone(rows[4]["theta_uwb_wrapped_deg"])
        self.assertEqual(len(rows),51)
        self.assertEqual(output["summary"]["pcnt_samples"],401)

    def test_no_good_heading_outputs_nulls_but_retains_pcnt(self):
        p=self.make_input()
        p["distances_by_tag"][:]=np.nan
        result=audit_dual_tag_heading(**p)
        self.assertEqual(result["summary"]["accepted_heading_epochs"],0)
        self.assertEqual(len(result["pcnt_steps"]),401)
        self.assertTrue(all(r["difference_unwrapped_deg"] is None
                            for r in result["uwb_snapshots"]))
        self.assertTrue(all(window["quality"]=="insufficient_uwb"
                            for window in result["windows"]))

    def test_robust_to_a_single_anchor_outlier(self):
        p=self.make_input(noise=.005)
        p["distances_by_tag"][10,0,2]+=1.1
        r=audit_dual_tag_heading(**p)
        self.assertEqual(r["summary"]["uwb_epochs"],51)
        self.assertEqual(len(r["uwb_snapshots"]),51)
        # Se o robusto aceitar, o baseline deve continuar plausível;
        # se rejeitar, há motivo explícito, sem inventar heading.
        point=r["uwb_snapshots"][10]
        self.assertTrue(
            (point["valid"] and
             abs(point["measured_baseline_m"]-.36)<.12)
            or not point["valid"]
        )

    def test_custom_geometry_orientation_is_not_assumed_left_right(self):
        p=self.make_input(noise=0.)
        # Rotação de offsets 90°: agora canal0 traseiro, canal1 dianteiro.
        # Atualizar ranges gerados em vez de apenas alterar perfil.
        transformed=self.offsets.copy()
        transformed[:,:2]=np.column_stack(
            (-self.offsets[:,1],self.offsets[:,0])
        )
        t=p["timestamps_s"]
        th=np.deg2rad(10+t)
        xy=np.column_stack((1.+.55*np.sin(.04*t),1.+.4*np.cos(.04*t)))
        indices=p["sample_indices"]
        p["tag_offsets_xyz"]=transformed
        for j,k in enumerate(indices):
            ct,st=np.cos(th[k]),np.sin(th[k])
            for tag,local in enumerate(transformed):
                pos=np.array([
                    xy[k,0]+ct*local[0]-st*local[1],
                    xy[k,1]+st*local[0]+ct*local[1],local[2]
                ])
                p["distances_by_tag"][j,tag]=np.linalg.norm(self.anchors-pos,axis=1)
        r=audit_dual_tag_heading(**p)
        self.assertEqual(r["summary"]["accepted_heading_epochs"],51)
        self.assertAlmostEqual(
            r["uwb_snapshots"][-1]["relative_drift_deg"],-20,delta=1.
        )

    def test_reject_non_monotonic_indices_bad_shapes_and_geometry(self):
        base=self.make_input()
        cases=[
            ({"sample_indices":base["sample_indices"][::-1]},"índices"),
            ({"timestamps_s":np.zeros(401)},"timestamps"),
            ({"tag_offsets_xyz":np.zeros((2,3))},"mesmo XY"),
            ({"distances_by_tag":np.ones((51,7))},"ranges"),
            ({"pcnt_delta_right_ticks":np.array([1.])},"odometria"),
        ]
        for update,pattern in cases:
            with self.subTest(update=list(update)),self.assertRaisesRegex(ValueError,pattern):
                audit_dual_tag_heading(**{**base,**update})
        with self.assertRaises(ValueError):
            HeadingAuditConfig(max_fit_rms_m=-.1)
        with self.assertRaises(ValueError):
            HeadingAuditConfig(focus_start_s=130,focus_end_s=120)

    def test_output_csv_json_are_finite_serializable_and_non_destructive(self):
        p=self.make_input(noise=.01)
        source=p["distances_by_tag"].copy()
        r=audit_dual_tag_heading(**p)
        with tempfile.TemporaryDirectory() as folder:
            outputs=save_heading_audit(r,folder,plots=False)
            self.assertEqual(len(outputs),4)
            serialized=json.loads(Path(outputs["json"]).read_text(encoding="utf8"))
            self.assertEqual(serialized["schema"],"bc_ekf.dual_tag_heading_audit.v1")
            self.assertEqual(len(Path(outputs["pcnt"]).read_text().splitlines()),402)
            self.assertEqual(len(Path(outputs["snapshots"]).read_text().splitlines()),52)
            self.assertGreater(len(Path(outputs["windows"]).read_text().splitlines()),1)
            np.testing.assert_array_equal(source,p["distances_by_tag"])

    def test_cli_complete_39_column_pipeline_with_original_hashes(self):
        """Teste e2e PCNT+T1/T2 sintético, sem tocar em dados do LAB."""
        root=Path(__file__).resolve().parents[1]
        anchors=root/"anchor_sets/lab_duas_tags_A3_A9_14_11_2026.json"
        tags=root/"tag_profiles/lab_14_11_2026_t1_direita_t2_esquerda_xyz_custom.json"
        xyz=np.asarray(json.loads(anchors.read_text())["anchors_xy"])
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            raw=folder/"pcnt_new_dual.csv"
            out=folder/"export"
            fields=["time_ms","left","right"]
            for tag in (1,2):
                for anchor in range(1,10):
                    fields.extend((f"t{tag}_a{anchor}_avg",f"t{tag}_a{anchor}_std"))
            rows=[]
            epoch=-1
            for k in range(26):
                if k>=2 and (k-2)%4==0:
                    epoch+=1
                line=[343+k*250,80*k,-80*k]
                for tag in (1,2):
                    for anchor in range(1,10):
                        if anchor<3 or epoch<0:
                            line.extend([0.,0.])
                        else:
                            angle=.03*max(epoch,0)
                            local_y=-.18 if tag==1 else .18
                            pos=np.array([
                                .7-math.sin(angle)*local_y,
                                1.2+math.cos(angle)*local_y,
                                .215,
                            ])
                            cm=100*float(np.linalg.norm(pos-xyz[anchor-3]))
                            line.extend([cm,2.])
                rows.append(line)
            with raw.open("w",encoding="utf8",newline="") as handle:
                writer=csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            original_bytes=raw.read_bytes()
            status=main([
                "--pcnt",str(raw),"--anchors",str(anchors),
                "--tags",str(tags),"--session","1",
                "--output-dir",str(out),"--no-plots",
                "--focus-start","0","--focus-end","8",
            ])
            self.assertEqual(status,0)
            report=json.loads(
                (out/"dual_tag_heading_audit.json").read_text(encoding="utf8")
            )
            self.assertEqual(report["summary"]["pcnt_samples"],26)
            self.assertEqual(report["summary"]["uwb_epochs"],6)
            self.assertEqual(report["provenance"]["pcnt_sha256"],
                             hashlib.sha256(original_bytes).hexdigest())
            self.assertEqual(report["provenance"]["source_uwb_epochs"],6)
            self.assertEqual(report["geometry"]["baseline_m"],.36)
            self.assertEqual(raw.read_bytes(),original_bytes)
            self.assertTrue((out/"dual_tag_heading_audit_snapshots.csv").is_file())
            self.assertTrue((out/"dual_tag_heading_audit_pcnt.csv").is_file())
            self.assertTrue((out/"dual_tag_heading_audit_windows.csv").is_file())

    def test_cli_import_and_dual_parser_connection_are_stable(self):
        root=Path(__file__).resolve().parents[1]
        cli=(root/"tools/audit_dual_tag_heading_2026.py").read_text(encoding="utf8")
        self.assertIn("load_dual_tag_pcnt(",cli)
        self.assertIn("read_robot2026_profile_from_anchor_json",cli)
        self.assertIn("pcnt_sha256",cli)
        self.assertIn("tag_profile_sha256",cli)
        self.assertIn("import matplotlib",cli)
        pipeline=(root/"src/ui/algo_modes/dataset_two_tag2026_pipeline.py").read_text(
            encoding="utf8"
        )
        self.assertIn('pcnt_delta_left_ticks=',pipeline)
        self.assertIn('pcnt_delta_right_ticks=',pipeline)


if __name__=="__main__":
    unittest.main()
