"""PR18: EKF de uma tag e odometria com regressões sem Pygame/CSV LAB7A.

Verifica modelo cinemático, Jacobianas, covariância Joseph, ausência
de medição e comparação sintética com ground truth realmente conhecido.
Diferentemente do LAB7A, os dados sintéticos POSSUEM ground truth.
"""
from __future__ import annotations

import ast
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.fusion.single_tag_range_ekf import (
    SingleTagEkfConfig, load_single_tag_ekf_config,
    run_single_tag_range_ekf,
)
from src.ui.algo_modes.dataset_single_tag_fusion import run_real_single_tag_fusion


def _integrate(t,v,w,x0):
    """Integra o mesmo modelo midpoint usado pelo adaptador de PCNT."""
    x=np.array(x0,dtype=float).copy()
    out=np.empty((len(t),3),dtype=float)
    out[0]=x
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]
        ds=v[k]*dt
        dth=w[k]*dt
        x[0]+=ds*math.cos(x[2]+.5*dth)
        x[1]+=ds*math.sin(x[2]+.5*dth)
        x[2]+=dth
        out[k]=x
    return out


class TestOneTagRangeEkf(unittest.TestCase):
    def setUp(self):
        self.anchors=np.array([
            [-1.,-1.,1.5],[4.,-1.,1.6],[-1.,4.,1.3],
            [4.,4.,1.4],[1.5,-1.5,1.9],[1.5,4.5,1.2],
        ])
        self.t=np.arange(501,dtype=float)*.1
        self.v=np.full(len(self.t),.32)
        self.v[0]=0.
        self.w=np.zeros(len(self.t))
        self.w[105:155]=.54
        self.w[240:285]=-.43
        self.w[390:440]=.5
        self.true=_integrate(self.t,self.v,self.w,[.2,.2,0.])
        self.odom_v=self.v*1.08
        self.odom_w=self.w*1.11
        self.odom=_integrate(self.t,self.odom_v,self.odom_w,[.2,.2,0.])
        self.idx=np.arange(0,len(self.t),5,dtype=int)
        rng=np.random.default_rng(410)
        tag_height=.215
        delta=self.true[self.idx,:2][:,None,:] - self.anchors[None,:,:2]
        truth_ranges=np.sqrt(
            np.sum(delta**2,axis=2)
            +(tag_height-self.anchors[None,:,2])**2
        )
        self.d=truth_ranges+rng.normal(0,.025,truth_ranges.shape)
        self.s=np.full(self.d.shape,.04)
        self.cfg=SingleTagEkfConfig(
            tag_height_m=tag_height,
            range_sigma_floor_m=.12,
            initial_position_sigma_m=.7,
            initial_heading_sigma_deg=30.,
            motion_sigma_m_sqrt_m=.09,
            turn_sigma_rad_sqrt_rad=.15,
            stationary_sigma_m_sqrt_s=.003,
            heading_sigma_rad_sqrt_s=.03,
        )

    def run_filter(self,**kwargs):
        params=dict(
            timestamps_s=self.t,v_m_s=self.odom_v,omega_rad_s=self.odom_w,
            anchors_xyz=self.anchors,ranges_m=self.d,
            sample_indices=self.idx,sample_timestamps_s=self.t[self.idx],
            deviations_m=self.s,initial_pose=self.odom[0],
            config=self.cfg,
        )
        params.update(kwargs)
        return run_single_tag_range_ekf(**params)

    def test_synthetic_fusion_reduces_position_error_against_known_truth(self):
        out=self.run_filter()
        err_odom=np.linalg.norm(self.odom[self.idx,:2]-self.true[self.idx,:2],axis=1)
        err_fused=np.linalg.norm(out.sampled_xytheta[:,:2]-self.true[self.idx,:2],axis=1)
        rms_od=float(np.sqrt(np.mean(err_odom*err_odom)))
        rms_fus=float(np.sqrt(np.mean(err_fused*err_fused)))
        self.assertGreater(rms_od,.2)
        self.assertLess(rms_fus,rms_od*.7)
        self.assertEqual(out.summary["pcnt_samples"],501)
        self.assertEqual(out.summary["uwb_epochs"],len(self.idx))
        self.assertEqual(out.summary["corrections_applied"],len(self.idx))
        self.assertEqual(out.summary["total_ranges_used"],self.d.size)
        self.assertFalse(any("ground_truth" in key for key in out.summary))

    def test_masked_ranges_and_missing_epochs_never_create_samples(self):
        d=self.d.copy()
        d[4]=np.nan
        d[9,:4]=np.nan
        d[12,0]=np.nan
        d[15,2]=-3.
        out=self.run_filter(ranges_m=d)
        self.assertEqual(out.summary["uwb_epochs"],len(self.idx))
        self.assertEqual(out.summary["corrections_applied"],len(self.idx)-2)
        self.assertFalse(out.updates_applied[4])
        self.assertFalse(out.updates_applied[9])
        self.assertTrue(out.updates_applied[12])
        self.assertEqual(out.used_ranges_per_epoch[12],len(self.anchors)-1)
        self.assertEqual(len(out.poses_xytheta),len(self.t))
        np.testing.assert_array_equal(out.sample_indices,self.idx)
        np.testing.assert_array_equal(out.sampled_xytheta,out.poses_xytheta[self.idx])

    def test_all_invalid_ranges_is_exactly_odometry_predict_only(self):
        out=self.run_filter(ranges_m=np.full_like(self.d,np.nan))
        self.assertEqual(out.summary["corrections_applied"],0)
        self.assertEqual(out.summary["predict_only_uwb_epochs"],len(self.idx))
        np.testing.assert_allclose(out.poses_xytheta[:,:2],self.odom[:,:2],atol=1e-10)
        self.assertTrue(np.all(np.linalg.eigvalsh(out.covariances)>=-1e-12))
        self.assertTrue(np.isnan(out.innovation_rms_per_epoch_m).all())

    def test_heading_is_not_instantly_observed_with_central_tag(self):
        t=np.array([0.,.1])
        v=np.zeros(2)
        w=np.zeros(2)
        anchors=self.anchors
        true_ranges=np.linalg.norm(anchors-np.array([.5,.6,.215]),axis=1)
        out=run_single_tag_range_ekf(
            timestamps_s=t,v_m_s=v,omega_rad_s=w,
            anchors_xyz=anchors,
            ranges_m=true_ranges[None,:],
            sample_indices=np.array([0]),
            initial_pose=np.array([.2,.1,1.1]),
            config=self.cfg,
        )
        self.assertAlmostEqual(out.poses_xytheta[0,2],1.1,places=12)
        # Com alavanca conhecida (tag não central), a observação
        # contém derivada direta em theta e P permite correção angular.
        config=SingleTagEkfConfig(
            **{**self.cfg.__dict__,"tag_offset_x_m":.3}
        )
        shifted=run_single_tag_range_ekf(
            timestamps_s=t,v_m_s=v,omega_rad_s=w,anchors_xyz=anchors,
            ranges_m=true_ranges[None,:],sample_indices=np.array([0]),
            initial_pose=np.array([.2,.1,1.1]),config=config,
        )
        self.assertGreater(abs(shifted.poses_xytheta[0,2]-1.1),.001)

    def test_covariance_is_symmetric_psd_and_heading_is_wrapped(self):
        out=self.run_filter()
        self.assertTrue(np.isfinite(out.poses_xytheta).all())
        self.assertTrue(np.isfinite(out.covariances).all())
        np.testing.assert_allclose(
            out.covariances,np.transpose(out.covariances,(0,2,1)),atol=1e-10
        )
        eigenvalues=np.linalg.eigvalsh(out.covariances)
        self.assertGreater(np.min(eigenvalues),-1e-9)
        self.assertTrue(np.all(np.abs(out.poses_xytheta[:,2])<=math.pi+1e-12))
        self.assertLess(np.trace(out.covariances[-1]),np.trace(out.covariances[0]))

    def test_no_reported_sigma_uses_configured_floor(self):
        out=self.run_filter(deviations_m=None)
        self.assertEqual(out.summary["corrections_applied"],len(self.idx))
        invalid=self.s.copy()
        invalid[3,0]=np.nan
        invalid[4,1]=-1.
        out2=self.run_filter(deviations_m=invalid)
        self.assertEqual(out2.summary["sigma_fallback_ranges"],2)
        self.assertTrue(np.isfinite(out2.sampled_xytheta).all())
        fixed=SingleTagEkfConfig(**{**self.cfg.__dict__,"use_reported_sigma":False})
        a=self.run_filter(deviations_m=invalid,config=fixed)
        b=self.run_filter(deviations_m=None,config=fixed)
        np.testing.assert_allclose(a.sampled_xytheta,b.sampled_xytheta)

    def test_minimum_one_range_update_is_supported_but_explicit(self):
        cfg=SingleTagEkfConfig(**{**self.cfg.__dict__,"min_ranges_for_update":1})
        one=np.full_like(self.d,np.nan)
        one[:,0]=self.d[:,0]
        out=self.run_filter(ranges_m=one,config=cfg)
        self.assertEqual(out.summary["corrections_applied"],len(self.idx))
        self.assertEqual(out.summary["total_ranges_used"],len(self.idx))
        with self.assertRaisesRegex(ValueError,"excede"):
            self.run_filter(config=SingleTagEkfConfig(
                **{**self.cfg.__dict__,"min_ranges_for_update":len(self.anchors)+1}
            ))

    def test_invalid_indices_time_shapes_and_config_rejected(self):
        cases=[
            ({"sample_indices":self.idx[::-1]}, "índices"),
            ({"sample_indices":np.repeat(0,len(self.idx))}, "índices"),
            ({"sample_timestamps_s":self.t[self.idx]+.01}, "timestamps UWB"),
            ({"ranges_m":self.d[:,:-1]}, "ranges"),
            ({"deviations_m":self.s[:-1]}, "sigmas"),
            ({"v_m_s":np.full_like(self.t,np.nan)}, "v/omega"),
        ]
        for parameters,message in cases:
            with self.subTest(case=message),self.assertRaisesRegex(ValueError,message):
                self.run_filter(**parameters)
        for data in (
            {"range_sigma_floor_m":0.},
            {"initial_position_sigma_m":-1.},
            {"use_reported_sigma":1},
            {"min_ranges_for_update":0},
            {"tag_offset_x_m":float("nan")},
            {"unknown":1},
        ):
            with self.subTest(config=data),self.assertRaises(ValueError):
                SingleTagEkfConfig.from_dict(data)

    def test_config_loading_requires_real_height_or_explicit_preprocessing_height(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"layout.json"
            path.write_text(json.dumps({"anchors_xy":self.anchors.tolist()}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"tag_height_m"):
                load_single_tag_ekf_config(path)
            path.write_text(json.dumps({
                "uwb_preprocessing":{"mode":"flag","tag_heights_m":.215},
            }),encoding="utf-8")
            self.assertAlmostEqual(load_single_tag_ekf_config(path).tag_height_m,.215)
            path.write_text(json.dumps({
                "uwb_preprocessing":{"tag_heights_m":.215},
                "single_tag_ekf":{"tag_height_m":.3,"range_sigma_floor_m":.2},
            }),encoding="utf-8")
            cfg=load_single_tag_ekf_config(path)
            self.assertAlmostEqual(cfg.tag_height_m,.3)
            self.assertAlmostEqual(cfg.range_sigma_floor_m,.2)

    def test_adapter_preserves_all_pcnt_samples_and_uwb_alignment(self):
        odom=SimpleNamespace(
            timestamps_s=self.t,
            v_m_s=self.odom_v,omega_rad_s=self.odom_w,
            poses_xytheta=self.odom,
        )
        uwb=SimpleNamespace(sample_indices=self.idx,
                            timestamps_s=self.t[self.idx])
        original=self.d.copy()
        result=run_real_single_tag_fusion(
            odometry=odom,uwb=uwb,anchors_xyz=self.anchors,
            distances=self.d,deviations=self.s,config=self.cfg,
        )
        self.assertEqual(result["posicoes"].shape,(len(self.idx),3))
        self.assertEqual(result["full_pcnt_poses"].shape,(len(self.t),3))
        self.assertEqual(result["full_pcnt_covariances"].shape,(len(self.t),3,3))
        np.testing.assert_array_equal(
            result["posicoes"],result["full_pcnt_poses"][self.idx]
        )
        self.assertEqual(result["algo_key_actual"],"single_tag_range_ekf")
        self.assertNotIn("preprocessing",result)
        np.testing.assert_array_equal(self.d,original)

    def test_adapter_uses_mask_policy_but_preserves_input_and_indices(self):
        odom=SimpleNamespace(
            timestamps_s=self.t,v_m_s=self.odom_v,
            omega_rad_s=self.odom_w,poses_xytheta=self.odom,
        )
        uwb=SimpleNamespace(sample_indices=self.idx,timestamps_s=self.t[self.idx])
        ranges=self.d.copy()
        ranges[3,2]=.5  # impossível pela separação vertical da âncora 2
        baseline=run_real_single_tag_fusion(
            odometry=odom,uwb=uwb,anchors_xyz=self.anchors,
            distances=ranges,deviations=self.s,config=self.cfg,
            preprocess_config={"mode":"flag","tag_heights_m":.215},
        )
        masked=run_real_single_tag_fusion(
            odometry=odom,uwb=uwb,anchors_xyz=self.anchors,
            distances=ranges,deviations=self.s,config=self.cfg,
            preprocess_config={"mode":"mask","tag_heights_m":.215},
        )
        self.assertEqual(baseline["preprocessing"]["flagged_cells"],1)
        self.assertEqual(masked["preprocessing"]["newly_masked_finite_positive_cells"],1)
        self.assertEqual(masked["fusion_diagnostics"]["uwb_epochs"],len(self.idx))
        self.assertEqual(masked["fusion_used_ranges"][3],len(self.anchors)-1)
        self.assertAlmostEqual(ranges[3,2],.5)
        self.assertTrue(np.isfinite(masked["posicoes"]).all())

    def test_ui_batch_and_analyzer_bind_ekf_without_changing_bc_two_tag(self):
        root=Path(__file__).resolve().parents[1]
        mode=(root/"src/ui/algo_modes/dataset_mode.py").read_text(encoding="utf-8")
        render=(root/"src/ui/algo_modes/dataset_render.py").read_text(encoding="utf-8")
        shared=(root/"src/ui/algo_modes/shared.py").read_text(encoding="utf-8")
        tree=ast.parse(mode)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DatasetMode")
        methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        fn=ast.unparse(methods["_run_batch"])
        self.assertIn("run_one_tag_ekf",fn)
        self.assertIn("run_real_single_tag_fusion(",fn)
        self.assertIn('raw_results["bc_ekf"] = fusion',fn)
        self.assertIn("preprocess_config=pre_cfg",fn)
        self.assertIn('self._real_preproc_report is not None',fn)
        self.assertIn("run_batch(",fn)  # outros algoritmos ainda presentes
        self.assertIn('◉ EKF 1 tag',mode)
        self.assertIn('display_names={"bc_ekf": "EKF 1 tag"}',render)
        self.assertIn("display_names: Mapping[str, str] | None",shared)
        # O código BC-EKF original deve continuar existindo, sem refatoração destrutiva.
        source=(root/"src/bc_ekf.py").read_text(encoding="utf-8")
        self.assertIn("def run_bc_ekf_from_data(",source)


if __name__=="__main__":
    unittest.main()
