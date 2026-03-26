from __future__ import annotations
from typing import Any
import os
import json
import math
import numpy as np
import pygame as pg

import src.config as config
from src.simulator import Simulator
from src.uwb.uwb_sim import UwbSimPipeline
from src.uwb.algoritmos_step import ALGORITMOS, NOMES_UI
from src.ui.botton import Button
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_robot
from src.environment.environment import Environment, draw_environment
from src.ui.algo_modes.shared import (
    ALGO_COLORS,
    ALGO_ORDER,
    BLACK,
    BLUE,
    GRAY_D,
    GRAY_L,
    GREEN,
    MODE_DATASET,
    MODE_MONTE_CARLO,
    MODE_STEP,
    ORANGE,
    WHITE,
    default_selected,
    draw_analyzer_panel,
)
from src.analysis.algo_metrics import compute_step_vs_truth_stats, build_ranking_summary



class StepMode:
    def __init__(self, host: Any) -> None:
        self.host = host

        # estado do STEP
        self._sim = None
        self._pipeline = None
        self._localizadores = {}
        self._trail_true = []
        self._trails = {k: [] for k in ALGO_ORDER}
        self._wp_idx = 0
        self._step_count = 0
        self._running = False

        self._anchors_sim = None
        self._waypoints = []
        self._step_stats = None

        # cenário opcional
        self._route_label = ""
        self._anchors_label = ""
        self._map_env = None
        self._map_label = ""

        # dirs
        self.routes_dir = "routes"
        self.anchors_dir = "anchor_sets"
        self.maps_dir = "maps"
        os.makedirs(self.routes_dir, exist_ok=True)
        os.makedirs(self.anchors_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)

        # modal
        self.step_modal_open = False
        self.step_dropdown_route_open = False
        self.step_dropdown_anchors_open = False
        self.step_dropdown_map_open = False

        self.available_routes = []
        self.available_anchors = []
        self.available_maps = []

        self.step_inputs = {}
        self.step_buttons = {}
        self.step_modal_rect = None

        self.route_dropdown_scroll = 0
        self.anchors_dropdown_scroll = 0
        self.map_dropdown_scroll = 0

        # câmera
        self._panning = False
        self._pan_last = None

        self.selected = default_selected()

    def on_enter(self, host: Any) -> None:
        self.host = host
        self.host.mode = MODE_STEP

        self._anchors_sim = getattr(host, "_anchors_sim", None)
        self._waypoints = getattr(host, "_waypoints", [])
        self.selected = getattr(host, "selected", default_selected())

        self.btn_back = host.btn_back
        self.btn_mode = host.btn_mode
        self.btn_start = host.btn_start
        self.btn_clear = host.btn_clear
        self.btn_export = host.btn_export
        self.btn_step_config = host.btn_step_config
        self._btn_algos = host._btn_algos

    # =========================================================
    # LISTAGENS
    # =========================================================

    def _list_route_files(self):
        try:
            files = [f for f in os.listdir(self.routes_dir) if f.lower().endswith(".json")]
            files.sort()
            return files
        except Exception:
            return []

    def _list_anchor_files(self):
        try:
            files = [f for f in os.listdir(self.anchors_dir) if f.lower().endswith(".json")]
            files.sort()
            return files
        except Exception:
            return []

    def _list_map_files(self):
        try:
            files = [f for f in os.listdir(self.maps_dir) if f.lower().endswith(".json")]
            files.sort()
            return files
        except Exception:
            return []

    # =========================================================
    # DROPDOWN HELPERS
    # =========================================================

    def _get_dropdown_state(self, name: str):
        if name == "route":
            return self.step_dropdown_route_open, self.route_dropdown_scroll, self.available_routes
        if name == "anchors":
            return self.step_dropdown_anchors_open, self.anchors_dropdown_scroll, self.available_anchors
        if name == "map":
            return self.step_dropdown_map_open, self.map_dropdown_scroll, self.available_maps
        return False, 0, []

    def _get_dropdown_scroll(self, name: str) -> int:
        if name == "route":
            return self.route_dropdown_scroll
        if name == "anchors":
            return self.anchors_dropdown_scroll
        if name == "map":
            return self.map_dropdown_scroll
        return 0

    def _set_dropdown_scroll(self, name: str, value: int):
        if name == "route":
            self.route_dropdown_scroll = value
        elif name == "anchors":
            self.anchors_dropdown_scroll = value
        elif name == "map":
            self.map_dropdown_scroll = value

    def _scroll_dropdown(self, name: str, delta: int):
        _, scroll, items = self._get_dropdown_state(name)
        rect = self.step_inputs[name]["dropdown_rect"]

        item_h = 26
        max_visible = max(1, rect.h // item_h)
        max_scroll = max(0, len(items) - max_visible)

        new_scroll = max(0, min(max_scroll, scroll + delta))
        self._set_dropdown_scroll(name, new_scroll)

    # =========================================================
    # MODAL
    # =========================================================

    def _open_step_modal(self):
        self.step_modal_open = True
        self.step_dropdown_route_open = False
        self.step_dropdown_anchors_open = False
        self.step_dropdown_map_open = False

        self.route_dropdown_scroll = 0
        self.anchors_dropdown_scroll = 0
        self.map_dropdown_scroll = 0

        self.available_routes = self._list_route_files()
        self.available_anchors = self._list_anchor_files()
        self.available_maps = self._list_map_files()

        sw = self.host.screen.get_width()
        sh = self.host.screen.get_height()

        w, h = 640, 370
        mx = (sw - w) // 2
        my = (sh - h) // 2
        self.step_modal_rect = pg.Rect(mx, my, w, h)

        dd_h = 156

        self.step_inputs = {
            "route": {
                "value": self._route_label if self._route_label else "",
                "rect": pg.Rect(mx + 170, my + 78, 340, 30),
                "dropdown_rect": pg.Rect(mx + 170, my + 110, 340, dd_h),
            },
            "anchors": {
                "value": self._anchors_label if self._anchors_label else "",
                "rect": pg.Rect(mx + 170, my + 138, 340, 30),
                "dropdown_rect": pg.Rect(mx + 170, my + 170, 340, dd_h),
            },
            "map": {
                "value": self._map_label if self._map_label else "",
                "rect": pg.Rect(mx + 170, my + 198, 340, 30),
                "dropdown_rect": pg.Rect(mx + 170, my + 230, 340, dd_h),
            },
        }

        self.step_buttons = {
            "ok": Button((mx + w - 220, my + h - 50, 100, 32), "Aplicar", self.host.font),
            "cancel": Button((mx + w - 110, my + h - 50, 90, 32), "Cancelar", self.host.font),
        }

    def _close_modal(self):
        self.step_modal_open = False
        self.step_dropdown_route_open = False
        self.step_dropdown_anchors_open = False
        self.step_dropdown_map_open = False

    def _apply_step_config(self):
        route_file = self.step_inputs["route"]["value"].strip()
        anchors_file = self.step_inputs["anchors"]["value"].strip()
        map_file = self.step_inputs["map"]["value"].strip()

        if not route_file:
            self.host._set_msg("Selecione uma rota")
            return
        if not anchors_file:
            self.host._set_msg("Selecione as âncoras")
            return

        # rota
        try:
            with open(os.path.join(self.routes_dir, route_file), "r", encoding="utf-8") as f:
                data = json.load(f)
            wps = np.array(data.get("waypoints", []), dtype=float)
            if wps.size == 0 or len(wps) < 2:
                self.host._set_msg("Rota inválida")
                return
            self._waypoints = wps
            self._route_label = route_file
        except Exception as e:
            print(f"[STEP] erro ao carregar rota: {e}")
            self.host._set_msg("Erro ao carregar rota")
            return

        # âncoras
        try:
            with open(os.path.join(self.anchors_dir, anchors_file), "r", encoding="utf-8") as f:
                data = json.load(f)

            anchors_xy = np.array(data.get("anchors_xy", []), dtype=float)
            if anchors_xy.size == 0:
                self.host._set_msg("Arquivo de âncoras vazio")
                return

            if anchors_xy.ndim == 2 and anchors_xy.shape[1] == 2:
                anchors_3xN = np.zeros((3, anchors_xy.shape[0]), dtype=float)
                anchors_3xN[0, :] = anchors_xy[:, 0]
                anchors_3xN[1, :] = anchors_xy[:, 1]
                anchors_3xN[2, :] = 1.0
                self._anchors_sim = anchors_3xN
            elif anchors_xy.ndim == 2 and anchors_xy.shape[1] == 3:
                self._anchors_sim = anchors_xy.T
            else:
                self.host._set_msg("Formato inválido de âncoras")
                return

            self._anchors_label = anchors_file

        except Exception as e:
            print(f"[STEP] erro ao carregar âncoras: {e}")
            self.host._set_msg("Erro ao carregar âncoras")
            return

        # mapa
        self._map_env = None
        self._map_label = ""
        if map_file:
            try:
                self._map_env = Environment.load_json(os.path.join(self.maps_dir, map_file))
                self._map_label = map_file
                if getattr(self, "sim", None) is not None and hasattr(self.sim, "set_environment"):
                    self.sim.set_environment(self.env)
            except Exception as e:
                print(f"[STEP] erro ao carregar mapa: {e}")
                self.host._set_msg("Erro ao carregar mapa")
                return

        # resetar simulação com nova config
        self._stop_sim()
        self._clear_trails()
        self._close_modal()
        self.host._set_msg("STEP configurado")

    def _handle_step_modal_events(self, event) -> bool:
        if not self.step_modal_open:
            return False

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self._close_modal()
                return True

        names = ["route", "anchors", "map"]
        flags = {
            "route": "step_dropdown_route_open",
            "anchors": "step_dropdown_anchors_open",
            "map": "step_dropdown_map_open",
        }
        lists = {
            "route": self.available_routes,
            "anchors": self.available_anchors,
            "map": self.available_maps,
        }

        if event.type == pg.MOUSEWHEEL:
            mouse_pos = pg.mouse.get_pos()
            for name in names:
                inp = self.step_inputs[name]
                is_open = getattr(self, flags[name])
                if is_open and inp["dropdown_rect"].collidepoint(mouse_pos):
                    self._scroll_dropdown(name, -event.y)
                    return True

        if event.type == pg.MOUSEBUTTONDOWN:
            pos = getattr(event, "pos", pg.mouse.get_pos())

            if event.button == 4 or event.button == 5:
                delta = -1 if event.button == 4 else 1
                for name in names:
                    inp = self.step_inputs[name]
                    is_open = getattr(self, flags[name])
                    if is_open and inp["dropdown_rect"].collidepoint(pos):
                        self._scroll_dropdown(name, delta)
                        return True

            if event.button == 1:
                if self.step_buttons["ok"].hit(pos):
                    self._apply_step_config()
                    return True

                if self.step_buttons["cancel"].hit(pos):
                    self._close_modal()
                    return True

                # clique em dropdown aberto
                for name in names:
                    inp = self.step_inputs[name]
                    flag_name = flags[name]
                    is_open = getattr(self, flag_name)

                    if is_open and inp["dropdown_rect"].collidepoint(pos):
                        item_h = 26
                        scroll = self._get_dropdown_scroll(name)
                        items = lists[name]
                        max_visible = max(1, inp["dropdown_rect"].h // item_h)
                        visible_items = items[scroll:scroll + max_visible]

                        idx = (pos[1] - inp["dropdown_rect"].y) // item_h
                        if 0 <= idx < len(visible_items):
                            inp["value"] = visible_items[idx]
                        setattr(self, flag_name, False)
                        return True

                # abre dropdown
                for name in names:
                    inp = self.step_inputs[name]
                    if inp["rect"].collidepoint(pos):
                        self.step_dropdown_route_open = False
                        self.step_dropdown_anchors_open = False
                        self.step_dropdown_map_open = False
                        setattr(self, flags[name], True)
                        return True

                # clique fora fecha tudo
                self.step_dropdown_route_open = False
                self.step_dropdown_anchors_open = False
                self.step_dropdown_map_open = False

        return False

    # =========================================================
    # EVENTOS GERAIS
    # =========================================================

    def handle_events(self, events):
        actions = _actions_default()

        for event in events:
            if event.type == pg.QUIT:
                return _actions_quit()

            if self._handle_step_modal_events(event):
                continue

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._stop_sim()
                    return _actions_menu()
                elif event.key == pg.K_SPACE:
                    self._toggle_run()

            elif event.type == pg.MOUSEBUTTONDOWN:
                pos = getattr(event, "pos", pg.mouse.get_pos())

                if event.button == 4:
                    self.host.cam.zoom_at(pos, 1.15)
                    continue
                elif event.button == 5:
                    self.host.cam.zoom_at(pos, 1 / 1.15)
                    continue
                elif event.button == 2:
                    self._panning = True
                    self._pan_last = pos
                    continue

                elif event.button == 1:
                    if self.btn_back.hit(pos):
                        self._stop_sim()
                        return _actions_menu()

                    elif self.btn_mode.hit(pos):
                        self._stop_sim()
                        self.host._toggle_mode()
                        return actions

                    elif self.btn_step_config.hit(pos):
                        self._open_step_modal()
                        continue

                    elif self.btn_start.hit(pos):
                        self._toggle_run()
                        continue

                    elif self.btn_clear.hit(pos):
                        self._clear_trails()
                        continue

                    elif self.btn_export.hit(pos):
                        if hasattr(self.host, "_export_csv"):
                            self.host._batch_results = self._export_step_as_batch()
                            self.host._export_csv()
                        continue

                    else:
                        for nome, btn in self._btn_algos.items():
                            if btn.hit(pos):
                                self.selected[nome] = not self.selected[nome]
                                self.host.selected = self.selected
                                self.host._refresh_algo_buttons()
                                break

            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 2:
                    self._panning = False
                    self._pan_last = None

            elif event.type == pg.MOUSEMOTION:
                if self._panning and self._pan_last is not None:
                    dx = event.pos[0] - self._pan_last[0]
                    dy = event.pos[1] - self._pan_last[1]
                    self.host.cam.pan_pixels(dx, dy)
                    self._pan_last = event.pos

            elif event.type == pg.MOUSEWHEEL:
                zoom_pos = pg.mouse.get_pos()
                factor = 1.1 if event.y > 0 else 1 / 1.1
                self.host.cam.zoom_at(zoom_pos, factor)

        return actions

    # =========================================================
    # UPDATE
    # =========================================================

    def update(self, dt: float) -> None:
        if not self._running:
            return

        if self._sim is None:
            self._init_step_sim()

        speed_factor = getattr(self.host, "speed_factor", 1)
        for _ in range(speed_factor):
            self._do_step()

        self._step_stats = self._compute_step_stats()

    def _init_step_sim(self) -> None:
        anchors = self._anchors_sim
        if anchors is None or anchors.shape[1] == 0:
            self.host._set_msg("Configure rota e âncoras primeiro")
            self._running = False
            return

        N = anchors.shape[1]
        Q = np.diag([1e-4, 1e-4, 1e-4])
        R = np.eye(2 * N) * (0.05 ** 2)

        self._pipeline = UwbSimPipeline.from_defaults(seed=42, env=self._map_env)

        self._sim = Simulator(
            anchors=anchors,
            baseline=getattr(config, "UWB_BASELINE", 0.65),
            z_c=0.5,
            Q=Q,
            R=R,
            dt=getattr(config, "TIME_STEP", 0.05),
            config=config,
            uwb_pipeline=self._pipeline,
            env=self._map_env,
        )

        if self._pipeline is not None and hasattr(self._pipeline, "set_environment") and self._map_env is not None:
            self._pipeline.set_environment(self._map_env)

        if self._sim is not None and hasattr(self._sim, "set_environment") and self._map_env is not None:
            self._sim.set_environment(self._map_env)

        algos_selecionados = [k for k, v in self.selected.items() if v]
        anchors_Nx3 = anchors.T

        self._localizadores = {}
        for nome in algos_selecionados:
            if nome == "bc_ekf":
                cls = ALGORITMOS["bc_ekf"]
                self._localizadores[nome] = cls(
                    anchors_Nx3,
                    baseline=getattr(config, "UWB_BASELINE", 0.65),
                    z_c=0.5,
                    dt=getattr(config, "TIME_STEP", 0.05),
                    Q=Q,
                    R=R,
                )
            elif nome in ALGORITMOS:
                self._localizadores[nome] = ALGORITMOS[nome](anchors_Nx3)

        self._wp_idx = 0
        self._step_count = 0

        if self.selected.get("lmsp", False) and not hasattr(self._pipeline, "measure_ranges_and_sigmas"):
            self.host._set_msg("LMSP requer pipeline com measure_ranges_and_sigmas()")

    def _do_step(self) -> None:
        if self._sim is None or self._waypoints is None or len(self._waypoints) < 2:
            return

        sim = self._sim
        wp = self._waypoints[self._wp_idx % len(self._waypoints)]
        dx = wp[0] - sim.robot.x
        dy = wp[1] - sim.robot.y
        dist = math.hypot(dx, dy)
        angle_to = math.atan2(dy, dx)
        angle_diff = math.atan2(
            math.sin(angle_to - sim.robot.theta),
            math.cos(angle_to - sim.robot.theta)
        )

        v_cmd = min(dist, 0.30)
        w_cmd = float(np.clip(angle_diff * 2.0, -1.2, 1.2))

        if dist < 0.3:
            self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)

        result = sim.step(v_cmd, w_cmd)
        x_true, y_true, _ = result["true"]
        self._trail_true.append((x_true, y_true))

        meas = self._get_measurement_bundle(sim, result)
        if meas is None:
            return

        z_full = meas["z_full"]              # (2N,)
        d_front = meas["d_front"]            # (N,)
        sig_front = meas["sig_front"]        # (N,) ou None
        sig_full = meas["sig_full"]          # (2N,) ou None

        for nome, loc in self._localizadores.items():
            try:
                pos = None

                if nome == "bc_ekf":
                    if hasattr(loc, "set_odometry"):
                        loc.set_odometry(v_cmd, w_cmd)

                    # tenta com sigmas
                    try:
                        pos = loc.step(z_full, sig_full)
                    except TypeError:
                        # fallback se a implementação do filtro não aceita sigmas
                        pos = loc.step(z_full, None)

                    if pos is None and self._step_count < 5:
                        print("[STEP DEBUG][bc_ekf] retornou None")
                        print("z_full shape:", None if z_full is None else np.shape(z_full))
                        print("sig_full shape:", None if sig_full is None else np.shape(sig_full))

                elif nome == "lmsp":
                    if sig_front is None:
                        if self._step_count < 5:
                            print("[STEP DEBUG][lmsp] sig_front = None")
                        continue

                    d_front_arr = np.asarray(d_front, dtype=float).reshape(-1)
                    sig_front_arr = np.asarray(sig_front, dtype=float).reshape(-1)

                    if len(sig_front_arr) != len(d_front_arr):
                        if self._step_count < 20:
                            print("[STEP DEBUG][lmsp] shape incompatível")
                            print("d_front shape:", d_front_arr.shape)
                            print("sig_front shape:", sig_front_arr.shape)
                        continue

                    # proteção contra sigma zero/negativo
                    sig_front_arr = np.maximum(sig_front_arr, 1e-6)

                    try:
                        pos = loc.step(d_front_arr, sig_front_arr)
                    except TypeError:
                        # algumas versões podem aceitar kwargs diferentes
                        pos = loc.step(d_front_arr, sig_front_arr)

                    if pos is None and self._step_count < 5:
                        print("[STEP DEBUG][lmsp] retornou None")
                        print("d_front shape:", d_front_arr.shape)
                        print("sig_front shape:", sig_front_arr.shape)

                else:
                    pos = loc.step(d_front, None)

                if nome not in self._trails:
                    self._trails[nome] = []

                if pos is not None:
                    pos = np.asarray(pos, dtype=float).reshape(-1)
                    if len(pos) >= 2 and np.isfinite(pos[0]) and np.isfinite(pos[1]):
                        self._trails[nome].append((float(pos[0]), float(pos[1])))
                    else:
                        if self._step_count < 10:
                            print(f"[STEP DEBUG][{nome}] posição inválida:", pos)

            except Exception as e:
                print(f"[STEP ERRO][{nome}] {type(e).__name__}: {e}")
                if self._step_count < 10:
                    print("  d_front shape:", None if d_front is None else np.shape(d_front))
                    print("  sig_front shape:", None if sig_front is None else np.shape(sig_front))
                    print("  z_full shape:", None if z_full is None else np.shape(z_full))
                    print("  sig_full shape:", None if sig_full is None else np.shape(sig_full))

        self._step_count += 1

    def _get_measurement_bundle(self, sim: Simulator, result: dict):
        if self._pipeline is None or sim.anchors is None:
            return None

        x, y, th = result["true"]
        x_state = np.array([x, y, th], dtype=float)

        if hasattr(self._pipeline, "measure_ranges_and_sigmas"):
            try:
                d_front, sig_front = self._pipeline.measure_ranges_and_sigmas(
                    x_state=x_state,
                    anchors=sim.anchors,
                    l=sim.l,
                    tag="front",
                    return_meta=False,
                )

                d_rear, sig_rear = self._pipeline.measure_ranges_and_sigmas(
                    x_state=x_state,
                    anchors=sim.anchors,
                    l=sim.l,
                    tag="rear",
                    return_meta=False,
                )

                d_front = np.asarray(d_front, dtype=float).reshape(-1)
                d_rear = np.asarray(d_rear, dtype=float).reshape(-1)

                sig_front = None if sig_front is None else np.asarray(sig_front, dtype=float).reshape(-1)
                sig_rear = None if sig_rear is None else np.asarray(sig_rear, dtype=float).reshape(-1)

                if len(d_front) != len(d_rear):
                    print("[STEP DEBUG] front/rear com tamanhos diferentes")
                    return None

                N = len(d_front)
                z_full = np.empty((2 * N,), dtype=float)
                z_full[0::2] = d_front
                z_full[1::2] = d_rear

                sig_full = None
                if sig_front is not None and sig_rear is not None:
                    if len(sig_front) == N and len(sig_rear) == N:
                        sig_full = np.empty((2 * N,), dtype=float)
                        sig_full[0::2] = sig_front
                        sig_full[1::2] = sig_rear

                return {
                    "z_full": z_full,
                    "d_front": d_front,
                    "sig_front": sig_front,
                    "sig_full": sig_full,
                }

            except Exception as e:
                print(f"[STEP DEBUG] fallback measure() por erro em measure_ranges_and_sigmas: {e}")

        try:
            z_k = self._pipeline.measure([x, y, th], sim.anchors, sim.l, sim.z_c)
            if not isinstance(z_k, np.ndarray):
                return None

            z_k = np.asarray(z_k, dtype=float).reshape(-1)
            d_front = np.asarray(z_k[0::2], dtype=float).reshape(-1)

            return {
                "z_full": z_k,
                "d_front": d_front,
                "sig_front": None,
                "sig_full": None,
            }

        except Exception as e:
            print(f"[STEP DEBUG] erro em measure(): {e}")
            return None

    def _toggle_run(self) -> None:
        if not self._running:
            if self._anchors_sim is None or self._waypoints is None or len(self._waypoints) < 2:
                self.host._set_msg("Configure STEP primeiro")
                return

            if self._sim is None:
                self._init_step_sim()

            self._running = True
            self.btn_start.text = "⏸  Pausar"
            self.btn_start.bg = (255, 245, 230)
            self.btn_start.fg = ORANGE
            self.btn_start.border = ORANGE
        else:
            self._running = False
            self.btn_start.text = "▶  Continuar"
            self.btn_start.bg = (235, 250, 235)
            self.btn_start.fg = GREEN
            self.btn_start.border = GREEN

    def _stop_sim(self) -> None:
        self._running = False
        self._sim = None
        self._pipeline = None
        self._localizadores = {}
        self.btn_start.text = "▶  Iniciar"
        self.btn_start.bg = (235, 250, 235)
        self.btn_start.fg = GREEN
        self.btn_start.border = GREEN

    def _clear_trails(self) -> None:
        self._trail_true = []
        self._trails = {k: [] for k in ALGO_ORDER}
        for loc in self._localizadores.values():
            try:
                loc.reset()
            except Exception:
                pass
        self._step_count = 0
        self._step_stats = None
        self.host._set_msg("Trilhas apagadas")

    # =========================================================
    # DRAW
    # =========================================================

    def draw(self) -> None:
        self.host.screen.fill(WHITE)

        draw_grid(self.host.screen, self.host.cam)

        if self._map_env is not None:
            draw_environment(self.host.screen, self.host.cam, self._map_env)

        if self._anchors_sim is not None and self._anchors_sim.size > 0:
            draw_anchors(self.host.screen, self.host.cam, self._anchors_sim)

        draw_axes(self.host.screen, self.host.cam, self.host.font)

        if self._waypoints is not None and len(self._waypoints) > 1:
            draw_path(
                self.host.screen,
                self.host.cam,
                [tuple(p[:2]) for p in self._waypoints],
                GRAY_D,
                2,
                dashed=True,
            )

        if len(self._trail_true) > 1:
            draw_path(self.host.screen, self.host.cam, self._trail_true, BLACK, 2)

        for nome in ALGO_ORDER:
            if not self.selected.get(nome, False):
                continue
            trail = self._trails.get(nome, [])
            if len(trail) > 1:
                draw_path(self.host.screen, self.host.cam, trail, ALGO_COLORS[nome], 2)

            if trail:
                sx, sy = self.host.cam.world_to_screen(*trail[-1])
                pg.draw.circle(self.host.screen, ALGO_COLORS[nome], (sx, sy), 4)
                pg.draw.circle(self.host.screen, BLACK, (sx, sy), 4, 1)

        if len(self._trail_true) > 0 and self._sim is not None:
            x, y = self._trail_true[-1]
            th = self._sim.robot.theta
            draw_robot(
                self.host.screen,
                self.host.cam,
                x,
                y,
                th,
                BLACK,
                l=getattr(self._sim, "l", 0.325),
            )

        pg.draw.rect(
            self.host.screen,
            GRAY_D,
            pg.Rect(0, 0, self.host.cam.viewport[0], self.host.screen.get_height()),
            1,
        )

        # espelha no host para HUD
        self.host.mode = MODE_STEP
        self.host._sim = self._sim
        self.host._trail_true = self._trail_true
        self.host._trails = self._trails
        self.host._localizadores = self._localizadores
        self.host._step_count = self._step_count
        self.host.selected = self.selected
        self.host._draw_hud()

        if self._step_stats is not None:
            self._draw_analyzer()

        if self.step_modal_open:
            self._draw_step_modal()

    def _draw_step_modal(self):
        screen = self.host.screen
        font = self.host.font
        bigfont = self.host.bigfont

        sw = screen.get_width()
        sh = screen.get_height()

        w, h = 640, 370
        mx = (sw - w) // 2
        my = (sh - h) // 2
        modal_rect = pg.Rect(mx, my, w, h)

        overlay = pg.Surface((sw, sh), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        pg.draw.rect(screen, (245, 245, 248), modal_rect, border_radius=10)
        pg.draw.rect(screen, (80, 80, 90), modal_rect, 2, border_radius=10)

        txt = bigfont.render("Configurar STEP", True, (20, 20, 20))
        screen.blit(txt, (mx + 18, my + 14))

        label_x = mx + 30
        y = my + 82

        entries = [
            ("Rota:", "route"),
            ("Âncoras:", "anchors"),
            ("Mapa:", "map"),
        ]

        for label, key in entries:
            txt = font.render(label, True, (40, 40, 40))
            screen.blit(txt, (label_x, y))
            self._draw_input_box(self.step_inputs[key])
            self._draw_dropdown_arrow(self.step_inputs[key]["rect"])
            y += 60

        if self.step_dropdown_route_open:
            self._draw_dropdown_list("route", self.step_inputs["route"]["dropdown_rect"], self.available_routes)
        if self.step_dropdown_anchors_open:
            self._draw_dropdown_list("anchors", self.step_inputs["anchors"]["dropdown_rect"], self.available_anchors)
        if self.step_dropdown_map_open:
            self._draw_dropdown_list("map", self.step_inputs["map"]["dropdown_rect"], self.available_maps)

        self.step_buttons["ok"].draw(screen)
        self.step_buttons["cancel"].draw(screen)

    def _draw_input_box(self, inp: dict):
        rect = inp["rect"]
        value = inp.get("value", "")

        pg.draw.rect(self.host.screen, (255, 255, 255), rect, border_radius=6)
        pg.draw.rect(self.host.screen, (130, 130, 140), rect, 2, border_radius=6)

        txt = self.host.font.render(str(value), True, (30, 30, 30))
        self.host.screen.blit(txt, (rect.x + 8, rect.y + 5))

    def _draw_dropdown_arrow(self, rect: pg.Rect):
        cx = rect.right - 14
        cy = rect.centery + 1
        pts = [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)]
        pg.draw.polygon(self.host.screen, (80, 80, 80), pts)

    def _draw_dropdown_list(self, name: str, rect: pg.Rect, items: list[str]):
        if not items:
            items = ["(vazio)"]

        item_h = 26
        max_visible = max(1, rect.h // item_h)
        scroll = self._get_dropdown_scroll(name)

        visible_items = items[scroll:scroll + max_visible]
        real_h = len(visible_items) * item_h
        draw_rect = pg.Rect(rect.x, rect.y, rect.w, real_h)

        pg.draw.rect(self.host.screen, (255, 255, 255), draw_rect, border_radius=4)
        pg.draw.rect(self.host.screen, (120, 120, 130), draw_rect, 2, border_radius=4)

        mouse_pos = pg.mouse.get_pos()
        y = draw_rect.y

        for item in visible_items:
            item_rect = pg.Rect(draw_rect.x, y, draw_rect.w, item_h)

            if item_rect.collidepoint(mouse_pos):
                pg.draw.rect(self.host.screen, (230, 238, 255), item_rect)

            txt = self.host.font.render(str(item), True, (30, 30, 30))
            self.host.screen.blit(txt, (item_rect.x + 8, item_rect.y + 4))

            pg.draw.line(
                self.host.screen,
                (225, 225, 230),
                (item_rect.x, item_rect.bottom),
                (item_rect.right, item_rect.bottom),
                1
            )
            y += item_h

        if len(items) > max_visible:
            bar_w = 8
            bar_x = draw_rect.right - bar_w - 2
            bar_y = draw_rect.y + 2
            bar_h = draw_rect.h - 4

            pg.draw.rect(self.host.screen, (240, 240, 240), (bar_x, bar_y, bar_w, bar_h))

            thumb_h = max(20, int(bar_h * (max_visible / len(items))))
            max_scroll = len(items) - max_visible
            thumb_y = bar_y
            if max_scroll > 0:
                thumb_y = bar_y + int((scroll / max_scroll) * (bar_h - thumb_h))

            pg.draw.rect(self.host.screen, (160, 160, 170), (bar_x, thumb_y, bar_w, thumb_h), border_radius=3)

    # =========================================================
    # ANALYZER
    # =========================================================

    def _compute_step_stats(self):
        ''' Calcula RMSE e outras métricas comparando as trilhas dos algoritmos com a trilha verdadeira '''
        return compute_step_vs_truth_stats(
            self._trail_true,
            self._trails,
            algo_order=ALGO_ORDER,
        )

    def _step_ranking(self):
        ''' Retorna uma string formatada com o ranking dos algoritmos baseado nas métricas calculadas '''
        return build_ranking_summary(
            self._step_stats,
            selected=self.selected,
            top_k=5,
        )

    def _draw_analyzer(self):
        draw_analyzer_panel(
            screen=self.host.screen,
            font=self.host.font,
            bigfont=self.host.bigfont,
            title="Step Analyzer",
            stats=self._step_stats,
            selected=self.selected,
            box_fill=GRAY_L,
        )

    def _export_step_as_batch(self):
        ranking = self._step_ranking()
        export_data = {}

        for algo, trail in self._trails.items():
            if not trail:
                continue

            export_data[algo] = {
                "posicoes": np.array(trail, dtype=float),
                "rmse_xy": self._step_stats.get(algo, {}).get("rmse") if self._step_stats else None,
                "ranking_row": next(
                    (row for row in ranking if row["algo"] == algo),
                    None
                ),
            }

        return export_data

    def close(self) -> None:
        self._stop_sim()


def _actions_default():
    class _A:
        go_to_menu = False
        quit_app = False
    return _A()


def _actions_quit():
    class _A:
        go_to_menu = False
        quit_app = True
    return _A()


def _actions_menu():
    class _A:
        go_to_menu = True
        quit_app = False
    return _A()