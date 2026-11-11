"""Regressões PR12: portão físico, configuração e integração visual Dataset."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ui.algo_modes.dataset_gate_laps import split_start_gate_laps
from src.ui.algo_modes.dataset_uwb2026_pipeline import read_robot2026_view_settings
from tools.create_robot2026_field_variant import main as make_variant


def synthetic_clockwise_laps(repeats=2):
    """Trajetória densa: saída +X, giro -Y e retorno à origem por -X."""
    corners = [
        (0., 0.), (2., 0.), (2., -1.),
        (-.5, -1.), (-.5, 0.), (0., 0.),
    ]
    one = []
    for start, end in zip(corners, corners[1:]):
        segments = int(round(np.linalg.norm(np.array(end)-start) / .1))
        for j in range(segments):
            a = j / segments
            one.append(np.array(start)*(1-a)+np.array(end)*a)
    pts = [*one] * repeats + [
        np.array([0., 0.]), np.array([.1, 0.]), np.array([.2, 0.]),
    ]
    xy = np.array(pts, dtype=float)
    ts = np.arange(len(xy), dtype=float) * .4
    return xy, ts


def synthetic_corner_return_laps(repeats=3):
    """LAB7A simplificado: saída +X, retorno +Y, curva no canto."""
    corners = [(0., 0.), (2., 0.), (2., -1.5), (0., -1.5), (0., 0.)]
    single = []
    for a, b in zip(corners, corners[1:]):
        a, b = np.array(a), np.array(b)
        parts = int(round(np.linalg.norm(b-a) / .1))
        for j in range(parts):
            alpha = j / parts
            single.append(a*(1-alpha) + b*alpha)
    pts = [*single] * repeats + [
        np.array([0., 0.]), np.array([.1, 0.]),
    ]
    xy = np.asarray(pts, dtype=float)
    t = np.arange(len(xy), dtype=float) * .5
    return xy, t


class TestStartGateLapSegmentation(unittest.TestCase):
    def options(self):
        return dict(
            start_xy=[0., 0.], direction_xy=[1., 0.],
            half_width_m=.4, hysteresis_m=.12,
            min_lap_s=15., min_lap_distance_m=4.5,
            departure_m=.8, max_step_m=.8,
        )

    def corner_options(self):
        return dict(
            start_xy=[0., 0.],
            direction_xy=[1., 0.],
            gate_mode="corner_return",
            return_direction_xy=[0., 1.],
            return_radius_m=.5,
            smoothing_window=1,
            approach_lookback_s=2.5,
            approach_min_m=.06,
            max_gap_s=8.,
            half_width_m=.4,
            hysteresis_m=.12,
            min_lap_s=20.,
            min_lap_distance_m=4.5,
            departure_m=.8,
            max_step_m=.8,
        )

    def test_lab7a_corner_return_counts_laps_without_crossing_x_zero(self):
        """Retorno pelo +Y deve contar mesmo sem atravessar de -X para +X."""
        xy, t = synthetic_corner_return_laps(3)
        before = xy.copy()
        old, old_info = split_start_gate_laps(xy, t, **{
            k:v for k,v in self.corner_options().items()
            if k not in (
                "gate_mode", "return_direction_xy", "return_radius_m",
                "smoothing_window", "approach_lookback_s", "approach_min_m",
                "max_gap_s",
            )
        })
        self.assertEqual(old_info["complete_laps"], 0)
        self.assertEqual(len(old), 1)

        laps, info = split_start_gate_laps(xy, t, **self.corner_options())
        self.assertEqual(info["gate_mode"], "corner_return")
        self.assertEqual(info["complete_laps"], 3)
        self.assertEqual(len(info["crossing_indices_pcnt"]), 3)
        self.assertEqual([lap.complete for lap in laps], [True, True, True, False])
        self.assertEqual(laps[-1].stop, len(xy))
        np.testing.assert_array_equal(xy, before)

    def test_debug_info_reveals_gate_geometry_mismatch_without_false_laps(self):
        """Distingue referência UWB afastada de erro no sentido de chegada."""
        xy, t = synthetic_corner_return_laps(2)
        # Simula deslocamento sistemático da localização (p.ex. por
        # âncora mal cadastrada), sem tocar no PCNT.
        shifted = xy + np.array([1.5, 0.])
        slices, info = split_start_gate_laps(
            shifted, t, **self.corner_options()
        )
        self.assertEqual(info["complete_laps"], 0)
        self.assertEqual(info["potential_arrivals"], 0)
        self.assertEqual(info["samples_inside_gate_radius"], 0)
        self.assertGreater(info["closest_gate_distance_m"], .5)
        self.assertFalse(info["first_segment_near_gate"])
        self.assertEqual(info["valid_xy_samples"], len(shifted))
        self.assertGreater(len(info["nearest_gate_by_80s_window"]), 0)
        self.assertEqual(len(slices), 1)

    def test_debug_info_distinguishes_rejected_return_direction(self):
        xy, t = synthetic_corner_return_laps(2)
        opts = self.corner_options()
        opts["return_direction_xy"] = [0., -1.]
        _, info = split_start_gate_laps(xy, t, **opts)
        self.assertGreater(info["samples_inside_gate_radius"], 0)
        self.assertGreater(info["potential_arrivals"], 0)
        self.assertGreater(info["rejected_arrival_direction"], 0)
        self.assertEqual(info["complete_laps"], 0)

    def test_corner_rejects_wrong_approach_side(self):
        """Retorno pelo -Y não corresponde à última reta real do LAB7A."""
        xy, t = synthetic_corner_return_laps(2)
        opts = self.corner_options()
        opts["return_direction_xy"] = [0., -1.]
        laps, info = split_start_gate_laps(xy, t, **opts)
        self.assertEqual(info["complete_laps"], 0)
        self.assertEqual(len(laps), 1)
        self.assertGreater(info["rejected_arrival_direction"], 0)

    def test_corner_with_isolated_uwb_outlier_and_sparse_pcnt_mapping(self):
        xy, t = synthetic_corner_return_laps(2)
        original = xy.copy()
        xy[20] = [9., 12.]
        xy[102] = [-7., 8.]
        ids = np.arange(0, len(xy), 2, dtype=int)
        opts = self.corner_options()
        opts["smoothing_window"] = 5
        laps, info = split_start_gate_laps(
            xy[ids], t[ids], sample_indices=ids,
            total_samples=len(xy), **opts
        )
        self.assertEqual(info["complete_laps"], 2)
        self.assertEqual(laps[-1].stop, len(xy))
        self.assertTrue(all(p.stop <= len(xy) for p in laps))
        self.assertEqual(xy[20, 0], 9.)
        self.assertEqual(xy[102, 0], -7.)
        np.testing.assert_array_equal(original[103:], xy[103:])

    def test_two_clockwise_complete_laps_and_partial_are_counted(self):
        xy, t = synthetic_clockwise_laps(3)
        before = xy.copy()
        slices, info = split_start_gate_laps(xy, t, **self.options())
        self.assertEqual(info["complete_laps"], 3)
        self.assertEqual(len(info["crossing_indices_pcnt"]), 3)
        self.assertEqual(len(slices), 4)
        self.assertTrue(all(part.complete for part in slices[:-1]))
        self.assertFalse(slices[-1].complete)
        self.assertEqual(slices[0].start, 0)
        self.assertEqual(slices[-1].stop, len(xy))
        self.assertEqual(slices[1].start, slices[0].stop - 1)
        np.testing.assert_array_equal(xy, before)

    def test_reverse_crossing_not_counted(self):
        xy = np.array([[0., 0.], [.4, 0.], [1., 0.], [1., -.2],
                       [-.4, -.2], [-.5, 0.], [-.2, 0.]])
        t = np.arange(len(xy), dtype=float) * 10
        laps, info = split_start_gate_laps(
            xy, t, max_step_m=2., **{
                k:v for k,v in self.options().items() if k != "max_step_m"
            })
        self.assertEqual(info["complete_laps"], 0)
        self.assertEqual(len(laps), 1)
        self.assertFalse(laps[0].complete)

    def test_gate_rejects_wrong_lateral_position(self):
        xy, t = synthetic_clockwise_laps(2)
        xy[:, 1] += 1.0
        laps, info = split_start_gate_laps(xy, t, **self.options())
        self.assertEqual(info["complete_laps"], 0)
        self.assertEqual(len(laps), 1)

    def test_jitter_and_teleport_are_not_laps(self):
        # Longo tempo com oscilações na origem; sem afastamento real.
        x = np.tile(np.array([-.04, .04]), 120)
        xy = np.column_stack([x, np.zeros(len(x))])
        t = np.arange(len(x), dtype=float) * .5
        laps, info = split_start_gate_laps(xy, t, **self.options())
        self.assertEqual(info["complete_laps"], 0)
        self.assertEqual(len(laps), 1)

        xy2 = np.array([[0., 0.], [1., -1.], [-.5, 0.], [0.2, 0.]])
        t2 = np.array([0., 10., 20., 30.])
        _, diag = split_start_gate_laps(xy2, t2, **self.options())
        self.assertEqual(diag["crossing_indices_pcnt"], [])
        # O cenário tem lacunas de 10 s, logo o detector pode rejeitar
        # pela duração (max_gap=8 s) ANTES de avaliar o salto espacial.
        self.assertGreater(
            diag["rejected_large_steps"] + diag["rejected_long_gaps"], 0
        )

    def test_uwb_sample_indices_map_to_full_pcnt_session(self):
        xy, t = synthetic_clockwise_laps(2)
        sample_ids = np.arange(0, len(xy), 2, dtype=int)
        laps, info = split_start_gate_laps(
            xy[sample_ids], t[sample_ids],
            sample_indices=sample_ids, total_samples=len(xy),
            **self.options()
        )
        self.assertEqual(info["complete_laps"], 2)
        self.assertEqual(laps[0].start, 0)
        self.assertEqual(laps[-1].stop, len(xy))
        self.assertTrue(all(p.stop <= len(xy) for p in laps))
        self.assertTrue(all(p.start < p.stop for p in laps))

    def test_first_segment_from_distant_point_is_partial(self):
        xy, t = synthetic_clockwise_laps(2)
        # Não houve início da gravação no portão: o primeiro retorno
        # termina um trecho parcial, não uma volta completa.
        xy = xy[15:]
        t = t[15:] - t[15]
        laps, info = split_start_gate_laps(xy, t, **self.options())
        self.assertFalse(info["first_segment_near_gate"])
        self.assertFalse(laps[0].complete)
        self.assertGreaterEqual(info["complete_laps"], 1)

    def test_rejects_bad_timestamps_and_mapping(self):
        xy, t = synthetic_clockwise_laps(1)
        with self.assertRaises(ValueError):
            split_start_gate_laps(xy, t[::-1], **self.options())
        with self.assertRaises(ValueError):
            split_start_gate_laps(xy, t, sample_indices=np.zeros(len(t)), **self.options())
        with self.assertRaises(ValueError):
            split_start_gate_laps(
                xy, t, direction_xy=[0., 0.],
                **{k:v for k,v in self.options().items() if k != "direction_xy"}
            )


class TestStartGateProfile(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.json"
        self.source.write_text(json.dumps({
            "meta": {"example_only": False, "test_id": "LAB7A"},
            "anchors_xy": [[0., 0., 1.], [2., 0., 1.], [0., 2., 1.], [2., 2., 1.]],
            "pcnt_slot_to_anchor_index": [None, 0, 1, 2, 3],
            "robot_2026": {
                "wheel_radius_m": .03425, "wheel_base_m": .229,
                "ticks_per_wheel_rev": 4560,
                "swap_encoder_columns": True,
                "invert_left": True, "invert_right": False,
                "initial_x_m": .15, "initial_y_m": 1.38,
                "initial_theta_deg": 0,
            },
            "uwb_2026": {"min_valid_anchors": 4, "min_sigma_m": .02},
            "view_2026": {"segmentation": "time_windows", "window_duration_s": 80},
        }), encoding="utf-8")

    def test_creates_isolated_experimental_profile(self):
        original = self.source.read_bytes()
        dest = self.root / "variant.json"
        make_variant([
            "--source", str(self.source), "--out", str(dest),
            "--gate-source", "uwb_lms"
        ])
        source = json.loads(original)
        result = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(result["anchors_xy"], source["anchors_xy"])
        self.assertEqual(result["pcnt_slot_to_anchor_index"], source["pcnt_slot_to_anchor_index"])
        self.assertEqual(result["uwb_2026"], source["uwb_2026"])
        self.assertFalse(result["robot_2026"]["swap_encoder_columns"])
        self.assertFalse(result["robot_2026"]["invert_left"])
        self.assertTrue(result["robot_2026"]["invert_right"])
        self.assertEqual(result["robot_2026"]["left_scale"], 1.)
        self.assertEqual(result["robot_2026"]["right_scale"], 1.)
        view = read_robot2026_view_settings(dest)
        self.assertEqual(view["segmentation"], "start_gate")
        self.assertEqual(view["gate_source"], "uwb_lms")
        self.assertEqual(view["gate_mode"], "corner_return")
        self.assertEqual(view["gate_return_direction_xy"], [0., 1.])
        self.assertEqual(view["gate_smoothing_window"], 5)
        self.assertEqual(view["gate_xy_m"], [.15, 1.38])

    def test_refuses_overwrite_or_example_geometry(self):
        dest = self.root / "variant.json"
        dest.write_text("preserve", encoding="utf-8")
        with self.assertRaises(SystemExit):
            make_variant(["--source", str(self.source), "--out", str(dest)])
        self.assertEqual(dest.read_text(encoding="utf-8"), "preserve")
        source = json.loads(self.source.read_text(encoding="utf-8"))
        source["meta"]["example_only"] = True
        self.source.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaises(SystemExit):
            make_variant([
                "--source", str(self.source),
                "--out", str(self.root / "blocked.json")
            ])
        self.assertFalse((self.root / "blocked.json").exists())

    def test_invalid_gate_settings_do_not_silently_fall_back(self):
        source = json.loads(self.source.read_text(encoding="utf-8"))
        source["view_2026"] = {
            "segmentation": "start_gate",
            "gate_source": "uwb_lms",
            "gate_xy_m": [.15, 1.38],
            "gate_direction_xy": [1, 0],
            "gate_hysteresis_m": .9,
            "gate_departure_m": .8,
        }
        self.source.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "departure"):
            read_robot2026_view_settings(self.source)
        source["view_2026"]["gate_hysteresis_m"] = .12
        source["view_2026"]["gate_direction_xy"] = [0., 0.]
        self.source.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "não pode ser zero"):
            read_robot2026_view_settings(self.source)

    def test_legacy_time_windows_preserved(self):
        view = read_robot2026_view_settings(self.source)
        self.assertEqual(view, {"segmentation": "time_windows", "window_duration_s": 80.})

    def test_actual_ui_wires_config_lms_batch_and_renders_gate(self):
        # Regressão estrutural: impede repetir PR #9, quando a função
        # existia mas não era chamada pelo render principal.
        import inspect
        from pathlib import Path
        base = Path(__file__).resolve().parents[1]
        mode_tree = ast.parse((base / "src/ui/algo_modes/dataset_mode.py").read_text(encoding="utf-8"))
        render_tree = ast.parse((base / "src/ui/algo_modes/dataset_render.py").read_text(encoding="utf-8"))
        klass = next(n for n in mode_tree.body if isinstance(n, ast.ClassDef) and n.name == "DatasetMode")
        methods = {n.name: n for n in klass.body if isinstance(n, ast.FunctionDef)}
        for name, call in [
            ("_apply_robot2026_single_tag_config", "_apply_start_gate_laps"),
            ("_run_batch", "_refresh_uwb_gate_laps_after_batch"),
            ("_refresh_uwb_gate_laps_after_batch", "_apply_start_gate_laps"),
        ]:
            invoked = {n.func.attr for n in ast.walk(methods[name])
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertIn(call, invoked, f"Missing integration {name} -> {call}")
        # Fallback evita o estado sem navegação exibido nas capturas
        # quando nenhuma volta do portão pode ser reconhecida.
        gate_method = methods["_apply_start_gate_laps"]
        fallback_calls = [
            node for node in ast.walk(gate_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "split_time_windows"
        ]
        self.assertTrue(fallback_calls, "Portão sem voltas precisa manter janelas")
        rendered = {n.name for n in render_tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("draw_reference_routes", rendered)
        draw = next(n for n in render_tree.body if isinstance(n, ast.FunctionDef) and n.name=="draw_reference_routes")
        self.assertTrue(any(
            isinstance(n, ast.Constant) and n.value=="Portao"
            for n in ast.walk(draw)
        ))


if __name__ == "__main__":
    unittest.main()
