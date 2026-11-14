"""PR19: presets/canais XYZ para BC-EKF, legado e regressão do Dataset.

Sem Pygame; testa numericamente modelo, Jacobiana, step/batch, geração
sintética e adaptação de dados reais/simulados. Usa quatro âncoras
sintéticas, não dados ou medidas particulares de um laboratório.
"""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.bc_ekf import (
    _generate_uwb_measurements,
    _measurement_model,
    run_bc_ekf_from_data,
    run_bc_ekf_step,
)
from src.fusion.tag_geometry import (
    TagGeometry, load_tag_geometry, resolve_tag_geometry,
)
from src.ui.algo_modes.dataset_bc_prep import (
    prepare_real_bc_ekf_data, prepare_simulated_bc_ekf_data,
)


class TestTwoTagGeometry(unittest.TestCase):
    def setUp(self):
        self.a=np.array([
            [-1., 2., 0., 4.],
            [-2.,-1., 4., 3.],
            [1.0, 1.2, 1.7, 1.4],
        ])
        self.x=np.array([.4,1.2,.47])
        self.baseline=.38
        self.height=.22

    def geom(self,settings=None,selection="auto"):
        return resolve_tag_geometry(
            settings,default_baseline_m=self.baseline,
            default_height_m=self.height,selection=selection
        )

    def test_legacy_geometry_without_json_is_strictly_longitudinal(self):
        g=self.geom()
        self.assertEqual(g.preset,"front_rear")
        self.assertEqual(g.source,"legacy")
        self.assertEqual(g.channels,("front","rear"))
        np.testing.assert_allclose(
            g.offsets,[[.19,0,.22],[-.19,0,.22]],rtol=0,atol=1e-14
        )
        legacy_h,legacy_H=_measurement_model(
            self.x,self.a,self.baseline/2,self.height
        )
        explicit_h,explicit_H=_measurement_model(
            self.x,self.a,self.baseline/2,self.height,
            tag_offsets_xyz=g.offsets
        )
        np.testing.assert_array_equal(legacy_h,explicit_h)
        np.testing.assert_array_equal(legacy_H,explicit_H)

    def test_longitudinal_matches_previous_equations(self):
        x,y,th=self.x
        l=self.baseline/2
        expected=[]
        jac=[]
        for i in range(self.a.shape[1]):
            for sign in (+1,-1):
                p=np.array([x+sign*l*math.cos(th),
                            y+sign*l*math.sin(th),self.height])
                delta=p-self.a[:,i]
                r=np.linalg.norm(delta)
                expected.append(r)
                jac.append((
                    delta[0]/r,delta[1]/r,
                    sign*l*(-delta[0]*math.sin(th)
                            +delta[1]*math.cos(th))/r
                ))
        h,H=_measurement_model(self.x,self.a,l,self.height)
        np.testing.assert_allclose(h,expected,atol=1e-13)
        np.testing.assert_allclose(H,jac,atol=1e-13)

    def test_left_right_presets_at_zero_heading(self):
        geometry=self.geom({"preset":"left_right"})
        self.assertEqual(geometry.channels,("left","right"))
        np.testing.assert_allclose(
            geometry.offsets,[[0,.19,.22],[0,-.19,.22]]
        )
        pose=np.array([.3,.6,0.])
        h,H=_measurement_model(
            pose,self.a,.19,.22,tag_offsets_xyz=geometry.offsets
        )
        left=np.array([.3,.6+.19,.22])
        right=np.array([.3,.6-.19,.22])
        for i in range(self.a.shape[1]):
            self.assertAlmostEqual(h[2*i],np.linalg.norm(left-self.a[:,i]))
            self.assertAlmostEqual(h[2*i+1],np.linalg.norm(right-self.a[:,i]))
        self.assertTrue(np.all(np.isfinite(H)))

    def test_channel_order_can_be_reversed_without_mutating_measurements(self):
        g=self.geom({
            "preset":"left_right",
            "channel_order":["right","left"],
        })
        self.assertEqual(g.channels,("right","left"))
        np.testing.assert_allclose(
            g.offsets,[[0,-.19,.22],[0,.19,.22]]
        )
        h,_=_measurement_model(self.x,self.a,.19,.22,g.offsets)
        normal,_=_measurement_model(
            self.x,self.a,.19,.22,
            self.geom({"preset":"left_right"}).offsets
        )
        np.testing.assert_array_equal(h[0::2],normal[1::2])
        np.testing.assert_array_equal(h[1::2],normal[0::2])

    def test_custom_offsets_have_independent_heights_and_xy(self):
        g=self.geom({
            "preset":"custom",
            "tags":[
                {"name":"unitA","x_m":.13,"y_m":.17,"z_m":.22},
                {"name":"unitB","x_m":-.02,"y_m":-.21,"z_m":.28},
            ],
        })
        self.assertEqual(g.channels,("unitA","unitB"))
        self.assertEqual(g.preset,"custom")
        self.assertGreater(g.baseline_m,.4)
        self.assertEqual(g.as_dict()["tags"][1]["z_m"],.28)
        g.offsets[0,0]=999
        self.assertAlmostEqual(g.offsets[0,0],.13)

    def test_arbitrary_jacobian_matches_finite_difference(self):
        candidates=[
            self.geom({"preset":"front_rear"}),
            self.geom({"preset":"left_right"}),
            self.geom({"preset":"custom","tags":[
                {"name":"a","x_m":.1,"y_m":.3,"z_m":.1},
                {"name":"b","x_m":-.17,"y_m":-.07,"z_m":.44},
            ]}),
        ]
        for geom in candidates:
            with self.subTest(preset=geom.preset):
                h,H=_measurement_model(self.x,self.a,.19,.22,geom.offsets)
                eps=1e-6
                numerical=np.empty_like(H)
                for dim in range(3):
                    delta=np.zeros(3)
                    delta[dim]=eps
                    plus,_=_measurement_model(
                        self.x+delta,self.a,.19,.22,geom.offsets
                    )
                    minus,_=_measurement_model(
                        self.x-delta,self.a,.19,.22,geom.offsets
                    )
                    numerical[:,dim]=(plus-minus)/(2*eps)
                np.testing.assert_allclose(H,numerical,rtol=1e-6,atol=1e-7)
                self.assertEqual(len(h),8)

    def test_json_file_presets_and_ui_override(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.json"
            p.write_text(json.dumps({"anchors_xy":self.a.T.tolist()}),encoding="utf8")
            g=load_tag_geometry(
                p,default_baseline_m=self.baseline,default_height_m=self.height
            )
            self.assertEqual(g.source,"legacy")
            forced=load_tag_geometry(
                p,default_baseline_m=self.baseline,
                default_height_m=self.height,selection="left_right"
            )
            self.assertEqual(forced.preset,"left_right")
            p.write_text(json.dumps({
                "anchors_xy":self.a.T.tolist(),
                "tag_geometry":{
                    "preset":"left_right","baseline_m":.51,"height_m":.3,
                    "channel_order":["right","left"],
                },
            }),encoding="utf8")
            loaded=load_tag_geometry(
                p,default_baseline_m=self.baseline,default_height_m=self.height
            )
            self.assertEqual(loaded.source,"json")
            np.testing.assert_allclose(loaded.offsets[0],[0,-.255,.3])
            override=load_tag_geometry(
                p,default_baseline_m=self.baseline,
                default_height_m=self.height,selection="front_rear"
            )
            np.testing.assert_allclose(override.offsets[0],[.255,0,.3])

    def test_independent_profile_overrides_anchor_inline_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            anchors=folder/"anchors.json"
            profile=folder/"tags.json"
            # O JSON das âncoras NÃO deve ser alterado ao trocar tags.
            anchors.write_text(json.dumps({
                "anchors_xy":self.a.T.tolist(),
                "tag_geometry":{"preset":"front_rear","baseline_m":.38}
            }),encoding="utf8")
            before=anchors.read_bytes()
            profile.write_text(json.dumps({
                "meta":{"example_only":False},
                "tag_geometry":{
                    "preset":"left_right","baseline_m":.44,
                    "height_m":.23,"channel_order":["right","left"],
                }
            }),encoding="utf8")
            g=load_tag_geometry(
                anchors,default_baseline_m=self.baseline,
                default_height_m=self.height,tag_profile_json=profile,
            )
            self.assertEqual(g.source,"profile")
            self.assertEqual(g.preset,"left_right")
            np.testing.assert_allclose(g.offsets,[[0,-.22,.23],[0,.22,.23]])
            self.assertEqual(before,anchors.read_bytes())
            forced=load_tag_geometry(
                anchors,default_baseline_m=self.baseline,
                default_height_m=self.height,tag_profile_json=profile,
                selection="front_rear"
            )
            self.assertEqual(forced.source,"profile+ui")
            np.testing.assert_allclose(
                forced.offsets,[[.22,0,.23],[-.22,0,.23]]
            )
            profile.write_text(json.dumps({
                "meta":{"example_only":True},
                "tag_geometry":{"preset":"left_right","baseline_m":.44}
            }),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"EXEMPLO"):
                load_tag_geometry(
                    anchors,default_baseline_m=self.baseline,
                    default_height_m=self.height,tag_profile_json=profile,
                )
            self.assertEqual(before,anchors.read_bytes())

    def test_custom_option_without_coordinates_and_bad_geometry_fail(self):
        cases=[
            {"preset":"custom"},
            {"preset":"left_right","baseline_m":0.},
            {"preset":"left_right","channel_order":["left","left"]},
            {"preset":"front_rear","unexpected":1},
            {"preset":"custom","tags":[
                {"name":"a","x_m":0.,"y_m":0.,"z_m":.2},
                {"name":"a","x_m":.1,"y_m":0.,"z_m":.2},
            ]},
            {"preset":"custom","tags":[
                {"name":"a","x_m":0.,"y_m":0.,"z_m":.2},
                {"name":"b","x_m":0.,"y_m":0.,"z_m":.2},
            ]},
            {"preset":"custom","tags":[
                {"name":"a","x_m":0.,"y_m":0.,"z_m":.2},
                {"name":"b","x_m":float("nan"),"y_m":0.,"z_m":.2},
            ]},
        ]
        for settings in cases:
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                self.geom(settings)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.json"
            p.write_text('{"anchors_xy":[]}',encoding="utf8")
            with self.assertRaisesRegex(ValueError,"Personalizado"):
                load_tag_geometry(
                    p,default_baseline_m=.3,default_height_m=.2,
                    selection="custom"
                )

    def test_legacy_batch_poses_identical_to_explicit_offsets(self):
        count=20
        T=.05
        odo=np.zeros((2,count))
        odo[0]=.2
        odo[1]=.12
        x0=np.array([1.,1.,.2])
        z=np.zeros((2*self.a.shape[1],count))
        for i in range(count):
            pose=np.array([1.+i*.005,1.+i*.005,.2+i*.006])
            z[:,i]=_measurement_model(pose,self.a,.19,.22)[0]
        kwargs=dict(T=T,anchors=self.a,odometry_noisy=odo,z_hist=z,
                    l=.19,z_c=.22,sigma_uwb=.05,x0=x0)
        baseline=run_bc_ekf_from_data(**kwargs)
        explicit=run_bc_ekf_from_data(
            **kwargs,tag_offsets_xyz=self.geom().offsets
        )
        np.testing.assert_array_equal(baseline,explicit)

    def test_legacy_step_matches_configured_step(self):
        x=np.array([1.,1.,.3])
        P=np.eye(3)*.1
        Q=np.eye(3)*.0001
        R=np.eye(8)*.01
        z=_measurement_model(x,self.a,.19,.22)[0]
        args=dict(x_est=x,P=P,u_k=np.array([.2,.01]),
                  z_k=z,anchors=self.a,l=.19,z_c=.22,
                  Q=Q,R=R,dt=.1)
        legacy=run_bc_ekf_step(**args)
        explicit=run_bc_ekf_step(
            **args,tag_offsets_xyz=self.geom().offsets
        )
        for v0,v1 in zip(legacy,explicit):
            np.testing.assert_array_equal(v0,v1)
        newer=run_bc_ekf_step(
            **args,tag_offsets_xyz=self.geom({"preset":"left_right"}).offsets
        )
        self.assertGreater(np.linalg.norm(legacy[0]-newer[0]),1e-4)

    def test_synthetic_generator_uses_same_model_and_channel_order(self):
        x=np.array([[.3,.6],[.2,.9],[.1,.3]])
        g=self.geom({"preset":"left_right"})
        fake=types.SimpleNamespace(apply_uwb_errors=lambda x,s:x)
        with patch.dict(sys.modules,{"src.utils":fake}):
            z=_generate_uwb_measurements(
                x,self.a,.19,.22,0.0,tag_offsets_xyz=g.offsets
            )
        self.assertEqual(z.shape,(8,2))
        for i in range(2):
            expected,_=_measurement_model(x[:,i],self.a,.19,.22,g.offsets)
            np.testing.assert_array_equal(z[:,i],expected)

    def test_simulated_bc_dataset_receives_tag_offsets_without_changing_columns(self):
        from types import SimpleNamespace
        cfg=SimpleNamespace(TAG_BASELINE=.38,TAG_HEIGHT=.22,TIME_STEP=.1,
                            UWB_NOISE_STD=.05)
        g=self.geom({"preset":"left_right"})
        route=np.array([[0.,0.],[.1,0.],[.2,.1],[.3,.1]])
        d=np.full((len(route),8),2.)
        sig=np.full_like(d,.05)
        r=prepare_simulated_bc_ekf_data(
            batch_dists=d,batch_devs=sig,dataset_anchors=self.a.T,
            dataset_path=None,dataset_route=route,route_waypoints=route,
            cfg=cfg,resample_polyline_fn=lambda a,n:a,tag_geometry=g,
        )
        self.assertTrue(r.ok,r.message)
        np.testing.assert_array_equal(r.bc_ekf_data["tag_offsets_xyz"],g.offsets)
        np.testing.assert_array_equal(r.bc_ekf_data["z_hist"],d.T)
        self.assertEqual(r.batch_dists.shape,(len(route),4))

    def test_real_bc_dataset_receives_tag_offsets_without_mixing_channels(self):
        from types import SimpleNamespace
        cfg=SimpleNamespace(TAG_BASELINE=.38,TAG_HEIGHT=.22,TIME_STEP=.1,
                            UWB_NOISE_STD=.05)
        g=self.geom({"preset":"custom","tags":[
            {"name":"left","x_m":0.,"y_m":.2,"z_m":.21},
            {"name":"right","x_m":0.,"y_m":-.2,"z_m":.23},
        ]})
        rows=[]
        for k in range(3):
            for i in range(self.a.shape[1]):
                rows.append(dict(
                    timestamp=float(k),anchor_id=i,
                    range_front=1.+i+.1*k,range_rear=2.+i+.1*k,
                    sigma_front=.05,sigma_rear=.05,
                ))
        poses=np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.]])
        r=prepare_real_bc_ekf_data(
            encoder_samples=[{}],uwb_rows=rows,dataset_anchors=self.a.T,
            cfg=cfg,build_pose_path_fn=lambda _:poses.copy(),
            resample_pose_path_fn=lambda a,n:a,
            apply_initial_pose_fn=lambda a:a,tag_geometry=g,
        )
        self.assertTrue(r.ok,r.message)
        self.assertEqual(r.bc_ekf_data["tag_geometry_info"]["channel_order"],
                         ["left","right"])
        self.assertAlmostEqual(r.bc_ekf_data["z_hist"][0,0],1.)
        self.assertAlmostEqual(r.bc_ekf_data["z_hist"][1,0],2.)
        np.testing.assert_array_equal(r.bc_ekf_data["tag_offsets_xyz"],g.offsets)

    def test_ui_preset_and_batch_are_wired_with_legacy_fallback(self):
        root=Path(__file__).resolve().parents[1]
        modal=(root/"src/ui/algo_modes/dataset_modal.py").read_text(encoding="utf8")
        mode=(root/"src/ui/algo_modes/dataset_mode.py").read_text(encoding="utf8")
        batch=(root/"src/uwb/algoritmos_estaticos.py").read_text(encoding="utf8")
        self.assertIn('"Auto (JSON / legado)"',modal)
        self.assertIn('"Esquerda / direita"',modal)
        self.assertIn('"Personalizado (JSON)"',modal)
        self.assertIn('self.rows_real_bc',modal)
        self.assertIn('FormDropdownRow("Perfil tags:", self.dd_tag_profile)',modal)
        self.assertIn('m._modal_tag_profile=',modal)
        self.assertIn('self.rows_sim_bc',modal)
        self.assertIn('selection=getattr(self, "_modal_tag_preset", "auto")',mode)
        self.assertIn('tag_offsets_xyz=bc_ekf_data.get("tag_offsets_xyz")',batch)
        self.assertIn('"tag_geometry_info"',mode)


if __name__=="__main__":
    unittest.main()
