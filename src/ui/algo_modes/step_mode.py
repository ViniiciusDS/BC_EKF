from __future__ import annotations
from typing import Any, Optional
import math
import numpy as np
import pygame as pg

import src.config as config
from src.simulator import Simulator
from src.uwb.uwb_sim import UwbSimPipeline
from src.uwb.algoritmos_step import ALGORITMOS, NOMES_UI

from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_robot

MODE_STEP = "step"
MODE_DATASET = "dataset"
MODE_MONTE_CARLO = "monte_carlo"

WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
BLUE   = (50, 100, 220)
GREEN  = (0, 180, 0)
ORANGE = (255, 150, 0)
GRAY_D = (90, 90, 90)

ALGO_ORDER = ["trilaterate3d", "lms", "lmsp", "gauss_newton", "bc_ekf"]
ALGO_COLORS = {
    "trilaterate3d": (220,60,60),
    "lms": (0, 102, 204),
    "lmsp": (153, 51, 255),
    "gauss_newton": (255, 140, 0),
    "bc_ekf": (0, 170, 120),
}


class StepMode:
    def __init__(self, host: Any) -> None:
        self.host = host
        self._legacy_ref = None

        # estado próprio do STEP
        self._sim = None
        self._pipeline = None
        self._localizadores = {}
        self._trail_true = []
        self._trails = {k: [] for k in ALGO_ORDER}
        self._wp_idx = 0
        self._step_count = 0
        self._running = False

    def on_enter(self, host: Any) -> None:
        self.host = host
        self._legacy_ref = host
        self.host.mode = MODE_STEP

        self._anchors_sim = getattr(host, "_anchors_sim", None)
        self._waypoints = getattr(host, "_waypoints", [])
        self.selected = host.selected

        self.btn_back = host.btn_back
        self.btn_mode = host.btn_mode
        self.btn_start = host.btn_start
        self.btn_clear = host.btn_clear
        self.btn_export = host.btn_export
        self._btn_algos = host._btn_algos

    def handle_events(self, events):
        lg = self._legacy_ref
        if lg is None:
            return _actions_default()

        actions = _actions_default()

        for event in events:
            if event.type == pg.QUIT:
                return _actions_quit()

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._stop_sim()
                    return _actions_menu()
                elif event.key == pg.K_SPACE:
                    self._toggle_run()

            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                if self.btn_back.hit(pos):
                    self._stop_sim()
                    return _actions_menu()

                elif self.btn_mode.hit(pos):
                    self.host._toggle_mode()
                    return actions

                elif self.btn_start.hit(pos):
                    self._toggle_run()

                elif self.btn_clear.hit(pos):
                    self._clear_trails()

                elif self.btn_export.hit(pos):
                    if hasattr(lg, "_export_csv"):
                        lg._export_csv()

                else:
                    for nome, btn in self._btn_algos.items():
                        if btn.hit(pos):
                            self.selected[nome] = not self.selected[nome]
                            if hasattr(lg, "_refresh_algo_buttons"):
                                lg.selected = self.selected
                                lg._refresh_algo_buttons()
                            break

        return actions

    def update(self, dt: float) -> None:
        if not self._running:
            return

        if self._sim is None:
            self._init_step_sim()

        speed_factor = getattr(self._legacy_ref, "speed_factor", 1)
        for _ in range(speed_factor):
            self._do_step()

    def draw(self) -> None:

        self.host.screen.fill(WHITE)
        txt = self.host.font.render("Step Mode", True, (0, 0, 0))
        self.host.screen.blit(txt, (30, 30))


        self.host.screen.fill(WHITE)
        map_w = self.host.cam.viewport[0]

        draw_grid(self.host.screen, self.host.cam)

        if self._sim is not None and self._sim.anchors is not None:
            draw_anchors(self.host.screen, self.host.cam, self._sim.anchors)

        draw_axes(self.host.screen, self.host.cam, self.host.font)

        if len(self._waypoints) > 1:
            draw_path(
                self.host.screen,
                self.host.cam,
                [tuple(p) for p in self._waypoints],
                GRAY_D,
                1,
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
                pg.draw.circle(self.host.screen, ALGO_COLORS[nome], (sx, sy), 5)
                pg.draw.circle(self.host.screen, BLACK, (sx, sy), 5, 1)

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
            pg.Rect(0, 0, map_w, self.host.screen.get_height()),
            1,
        )

        if hasattr(self.host, "_draw_hud"):
            self.host.mode = MODE_STEP
            self.host._trail_true = self._trail_true
            self.host._trails = self._trails
            self.host._localizadores = self._localizadores
            self.host._step_count = self._step_count
            self.host.selected = self.selected
            self.host._sim = self._sim
            self.host._draw_hud()



    def close(self) -> None:
        self._stop_sim()

    # --------------------------------------------------
    # lógica migrada do legacy
    # --------------------------------------------------

    def _init_step_sim(self) -> None:
        anchors = self._anchors_sim
        if anchors is None or anchors.shape[1] == 0:
            anchors = np.array([
                [0.0, 0.0, 1.0],
                [8.0, 0.0, 1.0],
                [8.0, 8.0, 1.0],
                [0.0, 8.0, 1.0],
            ]).T

        N = anchors.shape[1]
        Q = np.diag([1e-4, 1e-4, 1e-4])
        R = np.eye(2 * N) * (0.05 ** 2)

        self._pipeline = UwbSimPipeline.from_defaults(seed=42)

        self._sim = Simulator(
            anchors=anchors,
            baseline=getattr(config, "UWB_BASELINE", 0.65),
            z_c=0.5,
            Q=Q,
            R=R,
            dt=getattr(config, "TIME_STEP", 0.05),
            config=config,
            uwb_pipeline=self._pipeline,
        )

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
        self.host._set_msg("Simulação iniciada")

    def _do_step(self) -> None:
        sim = self._sim

        if len(self._waypoints) > 0:
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
        else:
            v_cmd, w_cmd = 0.0, 0.0

        result = sim.step(v_cmd, w_cmd)

        x_true, y_true, th_true = result["true"]
        self._trail_true.append((x_true, y_true))

        if sim.anchors is not None and sim.anchors.shape[1] > 0:
            anchors_Nx3 = sim.anchors.T
            z_full = self._get_last_zk(sim, result)
            if z_full is not None and len(z_full) == 2 * anchors_Nx3.shape[0]:
                d_front = z_full[0::2]
                dev_arr = None

                for nome, loc in self._localizadores.items():
                    if nome == "bc_ekf":
                        loc.set_odometry(v_cmd, w_cmd)
                        pos = loc.step(z_full, dev_arr)
                    else:
                        pos = loc.step(d_front, dev_arr)

                    if nome not in self._trails:
                        self._trails[nome] = []

                    if pos is not None:
                        self._trails[nome].append((float(pos[0]), float(pos[1])))

        self._step_count += 1

    def _get_last_zk(self, sim: Simulator, result: dict) -> Optional[np.ndarray]:
        if self._pipeline is not None and sim.anchors is not None:
            x, y, th = result["true"]
            z_k = self._pipeline.measure([x, y, th], sim.anchors, sim.l, sim.z_c)
            return z_k if isinstance(z_k, np.ndarray) else None
        return None

    def _toggle_run(self) -> None:
        if not self._running:
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
        self._trails = {k: [] for k in self.selected.keys()}
        for loc in self._localizadores.values():
            loc.reset()
        self._step_count = 0
        self.host._set_msg("Trilhas apagadas")


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