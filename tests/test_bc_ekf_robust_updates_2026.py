"""PR21: regressões do BC-EKF com timestamps, UWB parcial e NIS.

Todas as medições são sintéticas. Nunca inferir ground truth dos
ranges reais. Testa API batch/step com numpy somente e preserva
geometria das tags no referencial do robô.
"""
import json
import math
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.bc_ekf import (
    _measurement_model,run_bc_ekf,run_bc_ekf_from_data,run_bc_ekf_step,
)
from src.fusion.uwb_ekf_update import (
    compact_uwb_diagnostic,correct_uwb_ranges,
)
from src.uwb.algoritmos_estaticos import run_batch


class TestRobustBcEkf(unittest.TestCase):
    def setUp(self):
        self.anchors=np.asarray([
            [-.5,3.,-.5,3.,1.25],
            [-.5,-.5,3.,3.,1.5],
            [1.1,1.3,1.5,1.7,.95],
        ])
        self.tags=np.array([[0,-.18,.215],[0,.18,.215]],dtype=float)
        self.x0=np.array([.9,1.,0.],dtype=float)
        self.true=np.array([1.2,1.3,.18],dtype=float)
        self.h,_=_measurement_model(
            self.true,self.anchors,.18,.215,tag_offsets_xyz=self.tags
        )
        self.m=len(self.h)

    def batch(self,*,T=.2,time=None,z=None,odo=None,**kwargs):
        n=5
        if z is None:
            z=np.full((self.m,n),np.nan)
        if odo is None:
            odo=np.zeros((2,n))
        return run_bc_ekf_from_data(
            T=T,anchors=self.anchors,odometry_noisy=odo,z_hist=z,
            l=.18,z_c=.215,sigma_uwb=.15,x0=self.x0,
            tag_offsets_xyz=self.tags,timestamps_s=time,**kwargs,
        )

    def test_real_dt_exact_prediction_with_no_uwb(self):
        t=np.array([0,.1,.4,1.2,2.])
        odo=np.zeros((2,5))
        odo[0,1:]=.5
        state,diag=self.batch(
            T=.1,time=t,odo=odo,return_diagnostics=True
        )
        np.testing.assert_allclose(state[0],[.9,.95,1.1,1.5,1.9],atol=1e-12)
        np.testing.assert_allclose(state[1],np.ones(5),atol=1e-12)
        self.assertEqual(diag["summary"]["sampling_mode"],"real_dt")
        self.assertAlmostEqual(diag["summary"]["min_dt_s"],.1)
        self.assertAlmostEqual(diag["summary"]["max_dt_s"],.8)
        self.assertEqual(diag["summary"]["prediction_only_steps"],4)
        self.assertEqual(len(diag["steps"]),4)
        self.assertEqual(diag["summary"]["updates_accepted"],0)
        self.assertTrue(all(e["status"]=="no_valid_ranges" for e in diag["steps"]))
        fixed=self.batch(T=.1,odo=odo)
        self.assertAlmostEqual(fixed[0,-1],1.1)
        self.assertGreater(state[0,-1]-fixed[0,-1],.7)

    def test_real_dt_updates_on_new_measurement_not_legacy_periodicity(self):
        z=np.full((self.m,5),np.nan)
        z[:,1]=self.h
        z[:,3]=self.h
        time=np.array([0,.11,.62,.83,1.5])
        odo=np.zeros((2,5))
        state,diag=self.batch(
            time=time,z=z,odo=odo,return_diagnostics=True
        )
        self.assertEqual(diag["summary"]["updates_accepted"],2)
        self.assertEqual(
            [e["pcnt_index"] for e in diag["steps"] if e["status"]=="accepted"],
            [1,3],
        )
        self.assertGreater(np.linalg.norm(state[:2,-1]-self.x0[:2]),.02)
        self.assertTrue(all(np.isfinite(state.reshape(-1))))

    def test_legacy_schedule_and_return_type_unchanged(self):
        z=np.tile(self.h[:,None],(1,5))
        odo=np.zeros((2,5))
        output=self.batch(T=.1,z=z,odo=odo,update_hz=5.)
        self.assertEqual(output.shape,(3,5))
        hist,diag=self.batch(
            T=.1,z=z,odo=odo,update_hz=5.,return_diagnostics=True
        )
        np.testing.assert_array_equal(output,hist)
        self.assertEqual(diag["summary"]["scheduled_update_ratio"],2)
        self.assertEqual(diag["summary"]["updates_accepted"],2)
        self.assertEqual(diag["steps"][0]["status"],"not_scheduled")
        self.assertEqual(diag["steps"][1]["status"],"accepted")

    def test_partial_ranges_are_used_and_missing_ranges_do_not_poison_update(self):
        z=np.full((self.m,5),np.nan)
        z[:,1]=self.h
        z[1,1]=np.nan
        z[3,1]=np.inf
        z[6,1]=0.
        z[7,1]=-5.
        output,diag=self.batch(
            time=np.array([0,.2,.4,.6,.8]),
            z=z,return_diagnostics=True,
        )
        step=diag["steps"][0]
        self.assertEqual(step["status"],"accepted")
        self.assertEqual(step["used_ranges"],self.m-4)
        self.assertEqual(step["dropped_ranges"],4)
        self.assertEqual(diag["summary"]["partial_updates_accepted"],1)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreater(np.linalg.norm(output[:2,1]-self.x0[:2]),.02)
        json.dumps(diag,allow_nan=False)

    def test_optional_minimum_range_count_skips_sparse_observation(self):
        z=np.full((self.m,5),np.nan)
        z[0,1]=self.h[0]
        z[1,1]=self.h[1]
        states,report=self.batch(
            z=z,time=np.arange(5)*.2,
            min_valid_ranges=3,return_diagnostics=True,
        )
        self.assertEqual(report["steps"][0]["status"],"insufficient_ranges")
        np.testing.assert_array_equal(states[:,1],self.x0)

    def test_optional_nis_gate_diagnoses_and_rejects_large_innovation(self):
        z=np.full((self.m,5),np.nan)
        z[:,1]=self.h+5.
        state,diag=self.batch(
            z=z,time=np.arange(5)*.2,
            nis_gate_per_dof=3.,return_diagnostics=True,
        )
        first=diag["steps"][0]
        self.assertEqual(first["status"],"nis_gate_rejected")
        self.assertGreater(first["nis_per_dof"],3.)
        self.assertEqual(diag["summary"]["nis_gate_rejected"],1)
        np.testing.assert_array_equal(state[:,1],self.x0)

    def test_individual_range_gate_rejects_one_outlier_but_accepts_others(self):
        h_pred,H=_measurement_model(self.x0,self.anchors,.18,.215,self.tags)
        z=h_pred.copy()
        z[5]+=10.
        p=np.eye(3)*.05
        x,P,diag=correct_uwb_ranges(
            self.x0,p,z,h_pred,H,np.eye(self.m)*.04,
            range_gate_sigma=4.,
        )
        self.assertEqual(diag["status"],"accepted")
        self.assertNotIn(5,diag["used_indices"])
        self.assertEqual(diag["used_ranges"],self.m-1)
        self.assertAlmostEqual(diag["nis"],0.,places=8)
        self.assertTrue(np.all(np.linalg.eigvalsh(P)>=-1e-12))
        self.assertTrue(np.isfinite(x).all())

    def test_step_debug_contains_nis_and_original_fields_for_partial_update(self):
        z=self.h.copy()
        z[2]=np.nan
        args=dict(
            x_est=self.x0,P=np.eye(3)*.1,
            u_k=np.array([0.,0.]),z_k=z,anchors=self.anchors,
            l=.18,z_c=.215,Q=np.eye(3)*.0001,
            R=np.eye(self.m)*.0225,dt=.3,
            tag_offsets_xyz=self.tags,debug=True,
        )
        x,P,diag=run_bc_ekf_step(**args)
        self.assertEqual(diag["status"],"accepted")
        self.assertEqual(diag["used_ranges"],self.m-1)
        self.assertEqual(diag["S"].shape,(self.m-1,self.m-1))
        self.assertEqual(diag["K"].shape,(3,self.m-1))
        self.assertTrue(all(k in diag for k in (
            "innov","S","K","h","H","x_pred","P_pred","dt","nis"
        )))
        self.assertTrue(np.all(np.linalg.eigvalsh(P)>-1e-10))
        missing=run_bc_ekf_step(
            **{**args,"z_k":np.full(self.m,np.nan)}
        )
        self.assertEqual(missing[2]["status"],"no_valid_ranges")
        self.assertTrue(np.isfinite(x).all())

    def test_batch_and_step_partial_corrections_agree_at_single_timestep(self):
        odo=np.zeros((2,5))
        z=np.full((self.m,5),np.nan)
        z[:,1]=self.h
        z[0,1]=np.nan
        states,diag=self.batch(
            z=z,odo=odo,time=np.arange(5)*.2,return_diagnostics=True,
        )
        step=run_bc_ekf_step(
            x_est=self.x0,P=np.diag([.1]*3),
            u_k=np.zeros(2),z_k=z[:,1],anchors=self.anchors,
            l=.18,z_c=.215,Q=np.diag([1e-4]*3),
            R=np.eye(self.m)*.0225,dt=.2,tag_offsets_xyz=self.tags,
            debug=True,
        )
        np.testing.assert_allclose(states[:,1],step[0],atol=1e-10)
        self.assertEqual(diag["steps"][0]["status"],step[2]["status"])

    def test_bad_timestamps_and_covariance_fail_without_silent_fallback(self):
        for t in ([0,.1,.1,.2,.3],[0,.3,.2,.4,.5],[0,.1,np.nan,.3,.4]):
            with self.subTest(times=t),self.assertRaisesRegex(ValueError,"timestamps"):
                self.batch(time=np.array(t))
        h,H=_measurement_model(self.x0,self.anchors,.18,.215,self.tags)
        with self.assertRaisesRegex(ValueError,"variâncias positivas"):
            correct_uwb_ranges(
                self.x0,np.eye(3),h,h,H,np.zeros((self.m,self.m)),
            )
        with self.assertRaisesRegex(ValueError,"nis_gate_per_dof"):
            self.batch(nis_gate_per_dof=-1.)
        with self.assertRaisesRegex(ValueError,"min_valid_ranges"):
            self.batch(min_valid_ranges=0)

    def test_numpy_only_batch_exposes_diagnostics_aligned_with_uwb_indices(self):
        n=8
        odo=np.zeros((2,n))
        z=np.full((self.m,n),np.nan)
        indices=np.array([2,5,7])
        for k in indices:
            z[:,k]=self.h
        z[4,5]=np.nan
        bc=dict(
            T=.1,timestamps_s=np.arange(n)*.11,
            odometry_noisy=odo,z_hist=z,
            l=.18,z_c=.215,sigma_uwb=.15,
            x0=self.x0,tag_offsets_xyz=self.tags,
            output_indices=indices,
        )
        results=run_batch(
            anchors_Nx3=self.anchors.T,
            distances=np.tile(self.h[0::2],(len(indices),1)),
            deviations=np.full((len(indices),self.anchors.shape[1]),.15),
            algoritmos=["bc_ekf"],bc_ekf_data=bc,
        )
        data=results["bc_ekf"]
        self.assertEqual(data["full_pcnt_poses"].shape,(n,3))
        np.testing.assert_allclose(
            data["posicoes"],data["full_pcnt_poses"][indices]
        )
        diag=data["update_diagnostics"]
        self.assertEqual(diag["summary"]["updates_accepted"],3)
        self.assertEqual(diag["summary"]["partial_updates_accepted"],1)
        self.assertEqual(len(diag["steps"]),n-1)

    def test_synthetic_wrapper_no_longer_uses_undefined_time_axis(self):
        mocked=types.SimpleNamespace(
            apply_uwb_errors=lambda distance,sigma:distance
        )
        with patch.dict(sys.modules,{"src.utils":mocked}):
            # Segundo retorno é ground truth sintético; nunca usar
            # esse procedimento para julgar logs reais.
            outputs=run_bc_ekf(
                T=.1,t_final=.4,anchors=self.anchors,
                sigma_v=0.,sigma_w=0.,sigma_uwb=.05,
                l=.18,z_c=.215,tag_offsets_xyz=self.tags,
            )
        self.assertEqual(len(outputs),5)
        self.assertEqual(outputs[4].shape,(3,5))
        self.assertTrue(np.isfinite(outputs[4]).all())


if __name__=="__main__":
    unittest.main()
