from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict
import pygame as pg
import numpy as np

from src.ui.algo_modes.monte_carlo_mode import MonteCarloMode
from src.ui.algo_modes.step_mode import StepMode
from src.ui.algo_modes.dataset_mode import DatasetMode
from src.ui.botton import Button
from src.trajectory import Trajectory
from src.uwb.algoritmos_step import ALGORITMOS, NOMES_UI

WHITE   = (255, 255, 255)
BLACK   = (0,  0,  0)
GRAY    = (220, 220, 220)
GRAY_D  = (150, 150, 150)
BLUE    = (40,  90,  210)
GREEN   = (0,  180, 0)
ORANGE  = (250, 150, 0)
RED     = (210, 40,  40)
BG_HUD  = (245, 247, 252)

ALGO_COLORS: Dict[str, tuple] = {
    "trilaterate3d": (255, 20,  20),
    "lms":           (138, 0, 196),
    "gauss_newton":  (0, 0, 0),
    "lmsp":          (0, 100, 255),
    "bc_ekf":        (255, 150, 0),
}

ALGO_ORDER = ["trilaterate3d", "lms", "gauss_newton", "lmsp", "bc_ekf"]

MODE_STEP = "step"
MODE_DATASET = "dataset"
MODE_MONTE_CARLO = "monte_carlo"


@dataclass
class AlgoActions:
    go_to_menu: bool = False
    quit_app: bool = False


class AlgoTestScreen:
    """
    Shell (Plano 1):
      - STEP/DATASET: delega pro Legacy (sem mexer na lógica atual)
      - MONTE CARLO: delega pro MonteCarloMode (novo)
    """

    def __init__(
    self,
    screen,
    cam,
    clock,
    font,
    bigfont,
    side_width,
    anchors,
    shared_uwb,
    plot_state=None,
    ):
        self.screen = screen
        self.cam = cam
        self.clock = clock
        self.font = font
        self.bigfont = bigfont
        self.SIDE_W = side_width
        self.shared_uwb = shared_uwb
        self.anchors = anchors
        self.plot_state = plot_state or {}

        self.SW = screen.get_width()
        self.SH = screen.get_height()

        self.MODES = [MODE_DATASET, MODE_STEP, MODE_MONTE_CARLO]
        self.mode = MODE_DATASET

        self.selected = {
            "trilaterate3d": True,
            "lms": True,
            "gauss_newton": True,
            "lmsp": False,
            "bc_ekf": False,
        }

        self.speed_factor = 3
        self._msg = ""
        self._msg_t = 0.0

        # DADOS comuns usados pelos modes
        self._anchors_sim = anchors if (anchors is not None and anchors.shape[1] > 0) else None
        self._waypoints = np.array(
            Trajectory.square(size=6, start=(-3, -3)).waypoints
        )

        # cria os botões
        self._build_hud()

        # cria os modes
        self.step_mode = StepMode(self)
        self.dataset_mode = DatasetMode(self)
        self.mc_mode = MonteCarloMode(self)

        # entra no modo inicial
        if self.mode == MODE_STEP:
            self.step_mode.on_enter(self)
        elif self.mode == MODE_DATASET:
            self.dataset_mode.on_enter(self)
        elif self.mode == MODE_MONTE_CARLO:
            self.mc_mode.on_enter(self)

    def _is_step(self) -> bool:
        return self.mode == MODE_STEP

    def _is_dataset(self) -> bool:
        return self.mode == MODE_DATASET

    def _is_mc(self) -> bool:
        return self.mode == MODE_MONTE_CARLO

    def _build_hud(self) -> None:
        sx = self.screen.get_width() - self.SIDE_W + 12
        y  = 10

        self.btn_back = Button(
            (sx, y, self.SIDE_W - 24, 32),
            "← Menu", self.font, bg=(245,245,245), fg=(80,80,80)
        )
        y += 42

        self.btn_mode = Button(
            (sx, y, self.SIDE_W - 24, 32),
            self._mode_label(), self.bigfont,
            bg=(235, 240, 255), fg=BLUE, border=BLUE
        )
        y += 42

        self._btn_algos: Dict[str, Button] = {}
        for nome in ALGO_ORDER:
            cor = ALGO_COLORS[nome]
            btn = Button(
                (sx, y, self.SIDE_W - 24, 28),
                self._algo_label(nome), self.font,
                bg=self._algo_bg(nome), fg=cor, border=cor
            )
            self._btn_algos[nome] = btn
            y += 34

        y += 6

        self.btn_start = Button(
            (sx, y, self.SIDE_W - 24, 36),
            "▶  Iniciar", self.bigfont,
            bg=(235, 250, 235), fg=GREEN, border=GREEN
        )
        y += 44

        self.btn_clear = Button(
            (sx, y, (self.SIDE_W - 30) // 2, 28),
            "Limpar trilhas", self.font
        )
        self.btn_export = Button(
            (sx + (self.SIDE_W - 30) // 2 + 6, y, (self.SIDE_W - 30) // 2, 28),
            "Exportar CSV", self.font
        )
        y += 40

        self.btn_load_dataset = Button(
            (sx, y, self.SIDE_W - 24, 28),
            "Carregar dataset (.jsonl / .txt)", self.font,
            bg=(240, 240, 255), fg=BLUE, border=BLUE
        )
        y += 38

        self.btn_run_batch = Button(
            (sx, y, self.SIDE_W - 24, 28),
            "Rodar batch", self.font,
            bg=(240, 255, 240), fg=GREEN, border=GREEN
        )

        self._hud_metrics_y = y + 42

    def _toggle_mode(self) -> None:
        current_idx = self.MODES.index(self.mode) if self.mode in self.MODES else 0
        next_idx = (current_idx + 1) % len(self.MODES)
        self.mode = self.MODES[next_idx]
        self.btn_mode.text = self._mode_label()
        self._set_msg(f"Modo: {self.mode.upper()}")

        if self.mode == MODE_MONTE_CARLO:
            self.mc_mode.on_enter(self)
        elif self.mode == MODE_STEP:
            self.step_mode.on_enter(self)
        elif self.mode == MODE_DATASET:
            self.dataset_mode.on_enter(self)

    def _mode_label(self) -> str:
        labels = {
            MODE_DATASET: "Modo: Dataset (batch)",
            MODE_STEP: "Modo: Step (tempo real)",
            MODE_MONTE_CARLO: "Modo: Monte Carlo",
        }
        return labels.get(self.mode, f"Modo: {self.mode}")

    def _algo_label(self, nome: str) -> str:
        tick = "✔" if self.selected.get(nome, False) else "○"
        return f"{tick}  {nome}"

    def _algo_bg(self, nome: str) -> tuple:
        if self.selected.get(nome, False):
            r, g, b = ALGO_COLORS[nome]
            return (min(255, r + 180), min(255, g + 180), min(255, b + 180))
        return (245, 245, 245)

    def _refresh_algo_buttons(self) -> None:
        for nome, btn in self._btn_algos.items():
            btn.text = self._algo_label(nome)
            btn.bg = self._algo_bg(nome)

    def _set_msg(self, text: str, duration: float = 2.5):
        self._msg = text
        self._msg_t = duration

    def _update_msg(self, dt: float):
        if self._msg_t > 0:
            self._msg_t = max(0.0, self._msg_t - dt)
            if self._msg_t == 0.0:
                self._msg = ""

    def _draw_msg_overlay(self):
        if not self._msg:
            return

        txt = self.font.render(self._msg, True, (255, 255, 255))
        pad = 8
        w = txt.get_width() + 2 * pad
        h = txt.get_height() + 2 * pad

        x = 20
        y = self.screen.get_height() - h - 20

        bg = pg.Surface((w, h), pg.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        self.screen.blit(bg, (x, y))
        self.screen.blit(txt, (x + pad, y + pad))

    def _active_is_mc(self) -> bool:
        return self.mode == MODE_MONTE_CARLO

    def handle_events(self, events) -> AlgoActions:
        ''' '''
        if self.mode == MODE_MONTE_CARLO:
            return self.mc_mode.handle_events(events)
        if self.mode == MODE_STEP:
            return self.step_mode.handle_events(events)
        if self.mode == MODE_DATASET:
            return self.dataset_mode.handle_events(events)
        
        return AlgoActions()

    def update(self, dt: float) -> None:
        ''' '''
        if self.mode == MODE_MONTE_CARLO:
            self.mc_mode.update(dt)
        elif self.mode == MODE_STEP:
            self.step_mode.update(dt)
        elif self.mode == MODE_DATASET:
            self.dataset_mode.update(dt)
        self._draw_msg_overlay()

    def draw(self) -> None:
        if self.mode == MODE_MONTE_CARLO:
            self.mc_mode.draw()
        elif self.mode == MODE_STEP:
            self.step_mode.draw()
        elif self.mode == MODE_DATASET:
            self.dataset_mode.draw()

        self._draw_msg_overlay()

    def _draw_hud(self) -> None:
        map_w = self.cam.viewport[0]
        hud_x = map_w
        hud_w = self.SIDE_W

        # Fundo do HUD
        pg.draw.rect(
            self.screen,
            BG_HUD,
            pg.Rect(hud_x, 0, hud_w, self.screen.get_height())
        )
        pg.draw.line(
            self.screen,
            GRAY_D,
            (hud_x, 0),
            (hud_x, self.screen.get_height()),
            1
        )

        # Botões comuns
        self.btn_back.draw(self.screen)
        self.btn_mode.draw(self.screen)
        for btn in self._btn_algos.values():
            btn.draw(self.screen)

        # Botões por modo
        if self._is_step():
            self.btn_start.draw(self.screen)
            self.btn_clear.draw(self.screen)
            self.btn_export.draw(self.screen)
        elif self._is_dataset():
            self.btn_load_dataset.draw(self.screen)
            self.btn_run_batch.draw(self.screen)

            lbl = self.font.render(
                getattr(self, "_dataset_label", "Nenhum arquivo carregado"),
                True,
                GRAY_D
            )
            self.screen.blit(
                lbl,
                (hud_x + 12, self.btn_run_batch.rect.bottom + 6)
            )

        # Métricas por algoritmo
        y = self._hud_metrics_y + (40 if self.mode == MODE_DATASET else 0)
        y += 12

        for nome in ALGO_ORDER:
            if not self.selected.get(nome, False):
                continue

            cor = ALGO_COLORS[nome]
            rmse_txt = ""

            if self.mode == MODE_STEP:
                locs = getattr(self, "_localizadores", {})
                if nome in locs:
                    loc = locs[nome]
                    trail_true = getattr(self, "_trail_true", [])
                    if trail_true and getattr(loc, "historico", None):
                        n = min(len(trail_true), len(loc.historico))
                        gt = np.array(trail_true[:n])
                        h = np.array(loc.historico[:n])
                        err = h[:, :2] - gt
                        rmse = float(np.sqrt(np.mean(err ** 2)))
                        rmse_txt = f"  RMSE={rmse:.3f}m"

            elif self.mode == MODE_DATASET:
                batch = getattr(self, "_batch_results", None)
                if batch and nome in batch:
                    r = batch[nome]
                    if r.get("rmse_xy") is not None:
                        rmse_txt = f"  RMSE={r['rmse_xy']:.3f}m"

            label = f"{NOMES_UI.get(nome, nome)}{rmse_txt}"
            surf = self.font.render(label, True, cor)

            pg.draw.rect(self.screen, cor, pg.Rect(hud_x + 12, y + 4, 10, 10))
            self.screen.blit(surf, (hud_x + 28, y))
            y += 22

        # Contadores
        if self._is_step():
            step_txt = f"Steps: {getattr(self, '_step_count', 0)}"
            s = self.font.render(step_txt, True, GRAY_D)
            self.screen.blit(s, (hud_x + 12, y + 8))

        elif self.mode == MODE_DATASET and getattr(self, "_batch_dists", None) is not None:
            s = self.font.render(
                f"Amostras: {len(self._batch_dists)}",
                True,
                GRAY_D
            )
            self.screen.blit(s, (hud_x + 12, y + 8))

        # Mensagem temporária
        if getattr(self, "_msg", "") and getattr(self, "_msg_t", 0.0) > 0:
            msg_surf = self.font.render(self._msg, True, BLUE)
            self.screen.blit(
                msg_surf,
                (hud_x + 12, self.screen.get_height() - 30)
            )

    def close(self) -> None:
        try:
            self.step_mode.close()
        except Exception:
            pass
        try:
            self.dataset_mode.close()
        except Exception:
            pass
        try:
            self.mc_mode.close()
        except Exception:
            pass

