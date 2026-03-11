from __future__ import annotations
from typing import Any
import os
import json
import numpy as np
import pygame as pg

from src.ui.drawing import draw_grid, draw_axes, draw_anchors

MODE_DATASET = "dataset"
MODE_STEP = "step"
MODE_MONTE_CARLO = "monte_carlo"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_D = (90, 90, 90)

ALGO_ORDER = ["trilaterate3d", "lms", "lmsp", "gauss_newton", "bc_ekf"]
ALGO_COLORS = {
    "trilaterate3d": (220,60,60),
    "lms": (0, 102, 204),
    "lmsp": (153, 51, 255),
    "gauss_newton": (255, 140, 0),
    "bc_ekf": (0, 170, 120),
}


class DatasetMode:
    def __init__(self, host: Any) -> None:
        self.host = host
        self._legacy_ref = None

        # estado próprio do dataset
        self._dataset_path = None
        self._dataset_label = ""
        self._batch_dists = None
        self._batch_devs = None
        self._batch_results = None
        self._dataset_anchors = None

    def on_enter(self, host: Any) -> None:
        self.host = host
        self._legacy_ref = host
        self.host.mode = MODE_DATASET

        self.selected = getattr(host, "selected", {k: True for k in ALGO_ORDER})

        self._dataset_path = getattr(host, "_dataset_path", None)
        self._dataset_label = getattr(host, "_dataset_label", "")
        self._batch_dists = getattr(host, "_batch_dists", None)
        self._batch_devs = getattr(host, "_batch_devs", None)
        self._batch_results = getattr(host, "_batch_results", None)
        self._dataset_anchors = getattr(host, "_dataset_anchors", None)

        self.btn_back = host.btn_back
        self.btn_mode = host.btn_mode
        self.btn_load_dataset = host.btn_load_dataset
        self.btn_run_batch = host.btn_run_batch
        self.btn_export = host.btn_export
        self._btn_algos = host._btn_algos

    def handle_events(self, events):
        lg = self._legacy_ref
        actions = _actions_default()

        for event in events:
            if event.type == pg.QUIT:
                return _actions_quit()

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return _actions_menu()

            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                if self.btn_back.hit(pos):
                    return _actions_menu()

                elif self.btn_mode.hit(pos):
                    self.host._toggle_mode()
                    self.host.mode = getattr(self.host, "mode", MODE_DATASET)
                    return actions

                elif self.btn_load_dataset.hit(pos):
                    path = None

                    if hasattr(self.host, "_pick_dataset_file"):
                        path = self.host._pick_dataset_file()
                    elif hasattr(self.host, "_pick_txt_file"):
                        path = self.host._pick_txt_file()
                    elif hasattr(self.host, "_pick_jsonl_file"):
                        path = self.host._pick_jsonl_file()

                    if path:
                        self._try_load_dataset(path)
                    continue

                elif self.btn_run_batch.hit(pos):
                    self._run_batch()
                    continue

                elif self.btn_export.hit(pos):
                    if hasattr(lg, "_export_csv"):
                        lg._batch_results = self._batch_results
                        lg._export_csv()
                    continue

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
        pass

    def draw(self) -> None:
        lg = self._legacy_ref
        if lg is None:
            self.host.screen.fill(WHITE)
            txt = self.host.font.render("Dataset Mode", True, BLACK)
            self.host.screen.blit(txt, (30, 30))
            return

        self.host.screen.fill(WHITE)

        draw_grid(self.host.screen, self.host.cam)

        if self._dataset_anchors is not None and self._dataset_anchors.size > 0:
            if self._dataset_anchors.shape[0] == 2:
                anchors_3d = np.zeros((3, self._dataset_anchors.shape[1]))
                anchors_3d[0, :] = self._dataset_anchors[0, :]
                anchors_3d[1, :] = self._dataset_anchors[1, :]
                anchors_3d[2, :] = 1.0
                draw_anchors(self.host.screen, self.host.cam, anchors_3d)
            elif self._dataset_anchors.shape[0] == 3:
                draw_anchors(self.host.screen, self.host.cam, self._dataset_anchors)

        draw_axes(self.host.screen, self.host.cam, self.host.font)

        if self._batch_results is not None and hasattr(lg, "_draw_batch_scatter"):
            lg._batch_results = self._batch_results
            lg.selected = self.selected
            lg._draw_batch_scatter()

        pg.draw.rect(
            self.host.screen,
            GRAY_D,
            pg.Rect(0, 0, self.host.cam.viewport[0], self.host.screen.get_height()),
            1,
        )

        # HUD do dataset reaproveitando o legacy por enquanto
        lg.mode = MODE_DATASET
        lg._dataset_path = self._dataset_path
        lg._dataset_label = self._dataset_label
        lg._batch_dists = self._batch_dists
        lg._batch_devs = self._batch_devs
        lg._batch_results = self._batch_results
        lg._dataset_anchors = self._dataset_anchors
        lg.selected = self.selected
        lg._draw_hud()

    def close(self) -> None:
        pass

    # --------------------------------------------------
    # lógica migrada do legacy
    # --------------------------------------------------

    def _try_load_dataset(self, path: str):
        lg = self._legacy_ref
        if lg is None:
            return

        self._dataset_path = path
        self._dataset_label = os.path.basename(path)

        if path.lower().endswith(".jsonl"):
            self._load_jsonl(path)
        else:
            if hasattr(lg, "_try_load_dataset"):
                lg._try_load_dataset(path)
                self._dataset_label = getattr(lg, "_dataset_label", self._dataset_label)
                self._batch_dists = getattr(lg, "_batch_dists", None)
                self._batch_devs = getattr(lg, "_batch_devs", None)
                self._dataset_anchors = getattr(lg, "_dataset_anchors", None)

        self.host._set_msg(f"Dataset carregado: {self._dataset_label}")

    def _load_jsonl(self, path: str):
        lg = self._legacy_ref
        if lg is None:
            return

        if hasattr(lg, "_load_jsonl"):
            lg._load_jsonl(path)
            self._dataset_label = getattr(lg, "_dataset_label", self._dataset_label)
            self._batch_dists = getattr(lg, "_batch_dists", None)
            self._batch_devs = getattr(lg, "_batch_devs", None)
            self._dataset_anchors = getattr(lg, "_dataset_anchors", None)

    def _run_batch(self):
        lg = self._legacy_ref
        if lg is None:
            return

        lg.selected = self.selected
        lg._batch_dists = self._batch_dists
        lg._batch_devs = self._batch_devs
        lg._dataset_anchors = self._dataset_anchors

        if hasattr(lg, "_run_batch"):
            lg._run_batch()
            self._batch_results = getattr(lg, "_batch_results", None)

        self.host._set_msg("Batch executado")


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