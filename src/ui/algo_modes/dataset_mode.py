from __future__ import annotations
from typing import Any
import os
import json
import numpy as np
import pygame as pg

from src.ui.botton import Button
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path
from src.environment import Environment, draw_environment
from src.uwb.algoritmos_estaticos import carregar_ensaio_lab, run_batch
from src.ui.algo_modes.shared import (
    ALGO_ORDER,
    ALGO_COLORS,
    MODE_DATASET,
    MODE_MONTE_CARLO,
    MODE_STEP,
    default_selected,
    draw_analyzer_panel,
)
from src.analysis.algo_metrics import compute_dataset_cluster_stats, build_ranking_summary


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_D = (90, 90, 90)
GRAY_L = (235, 235, 240)
RED = (200, 40, 40)



class DatasetMode:
    def __init__(self, host: Any) -> None:
        self.host = host

        # dataset
        self._dataset_path = None
        self._dataset_label = ""
        self._batch_dists = None
        self._batch_devs = None
        self._batch_results = None
        self._dataset_anchors = None   # (N,3)
        self._dataset_truth = None
        self._dataset_stats = None

        # cenário opcional
        self._route_waypoints = None
        self._route_label = ""
        self._map_env = None
        self._map_label = ""

        # dirs
        self.dataset_dir = os.path.join("resultados", "datasets")
        self.anchors_dir = "anchor_sets"
        self.routes_dir = "routes"
        self.maps_dir = "maps"

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.anchors_dir, exist_ok=True)
        os.makedirs(self.routes_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)

        # modal
        self.dataset_modal_open = False
        self.dataset_dropdown_data_open = False
        self.dataset_dropdown_anchors_open = False
        self.dataset_dropdown_route_open = False
        self.dataset_dropdown_map_open = False

        self.available_datasets = []
        self.available_anchors = []
        self.available_routes = []
        self.available_maps = []

        self.dataset_inputs = {}
        self.dataset_buttons = {}
        self.dataset_modal_rect = None

        # scroll dos dropdowns
        self.dataset_dropdown_scroll = 0
        self.anchors_dropdown_scroll = 0
        self.route_dropdown_scroll = 0
        self.map_dropdown_scroll = 0

        # câmera
        self._panning = False
        self._pan_last = None

        self.selected = default_selected()

    def on_enter(self, host: Any) -> None:
        self.host = host
        self.host.mode = MODE_DATASET

        self.selected = getattr(host, "selected", default_selected())

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

    # =========================================================
    # LISTAGENS
    # =========================================================

    def _list_dataset_files(self):
        try:
            files = [
                f for f in os.listdir(self.dataset_dir)
                if f.lower().endswith(".txt") or f.lower().endswith(".jsonl")
            ]
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

    def _list_route_files(self):
        try:
            files = [f for f in os.listdir(self.routes_dir) if f.lower().endswith(".json")]
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
    # MODAL
    # =========================================================

    def _open_dataset_modal(self):
        self.dataset_modal_open = True
        self.dataset_dropdown_data_open = False
        self.dataset_dropdown_anchors_open = False
        self.dataset_dropdown_route_open = False
        self.dataset_dropdown_map_open = False

        self.available_datasets = self._list_dataset_files()
        self.available_anchors = self._list_anchor_files()
        self.available_routes = self._list_route_files()
        self.available_maps = self._list_map_files()

        sw = self.host.screen.get_width()
        sh = self.host.screen.get_height()

        # modal maior
        w, h = 660, 430
        mx = (sw - w) // 2
        my = (sh - h) // 2
        self.dataset_modal_rect = pg.Rect(mx, my, w, h)

        dd_h = 156

        self.dataset_inputs = {
            "dataset": {
                "value": os.path.basename(self._dataset_path) if self._dataset_path else "",
                "rect": pg.Rect(mx + 180, my + 78, 350, 30),
                "dropdown_rect": pg.Rect(mx + 180, my + 110, 350, dd_h),
            },
            "anchors": {
                "value": "",
                "rect": pg.Rect(mx + 180, my + 138, 350, 30),
                "dropdown_rect": pg.Rect(mx + 180, my + 170, 350, dd_h),
            },
            "route": {
                "value": self._route_label if self._route_label else "",
                "rect": pg.Rect(mx + 180, my + 198, 350, 30),
                "dropdown_rect": pg.Rect(mx + 180, my + 230, 350, dd_h),
            },
            "map": {
                "value": self._map_label if self._map_label else "",
                "rect": pg.Rect(mx + 180, my + 258, 350, 30),
                "dropdown_rect": pg.Rect(mx + 180, my + 290, 350, dd_h),
            },
        }

        self.dataset_buttons = {
            "ok": Button((mx + w - 220, my + h - 50, 100, 32), "Carregar", self.host.font),
            "cancel": Button((mx + w - 110, my + h - 50, 90, 32), "Cancelar", self.host.font),
        }

        self.dataset_dropdown_scroll = 0
        self.anchors_dropdown_scroll = 0
        self.route_dropdown_scroll = 0
        self.map_dropdown_scroll = 0

    def _close_modal(self):
        self.dataset_modal_open = False
        self.dataset_dropdown_data_open = False
        self.dataset_dropdown_anchors_open = False
        self.dataset_dropdown_route_open = False
        self.dataset_dropdown_map_open = False

    def _apply_dataset_config(self):
        dataset_file = self.dataset_inputs["dataset"]["value"].strip()
        anchors_file = self.dataset_inputs["anchors"]["value"].strip()
        route_file = self.dataset_inputs["route"]["value"].strip()
        map_file = self.dataset_inputs["map"]["value"].strip()

        if not dataset_file:
            self.host._set_msg("Selecione um dataset")
            return

        # 1) dataset
        dataset_path = os.path.join(self.dataset_dir, dataset_file)
        self._try_load_dataset(dataset_path)

        # 2) anchors
        if anchors_file:
            try:
                with open(os.path.join(self.anchors_dir, anchors_file), "r", encoding="utf-8") as f:
                    data = json.load(f)

                anchors_xy = np.array(data.get("anchors_xy", []), dtype=float)
                if anchors_xy.size == 0:
                    self.host._set_msg("Arquivo de âncoras vazio")
                    return

                if anchors_xy.ndim == 2 and anchors_xy.shape[1] == 2:
                    anchors_nx3 = np.zeros((anchors_xy.shape[0], 3), dtype=float)
                    anchors_nx3[:, 0] = anchors_xy[:, 0]
                    anchors_nx3[:, 1] = anchors_xy[:, 1]
                    anchors_nx3[:, 2] = 1.0
                    self._dataset_anchors = anchors_nx3
                elif anchors_xy.ndim == 2 and anchors_xy.shape[1] == 3:
                    self._dataset_anchors = anchors_xy
                else:
                    self.host._set_msg("Formato inválido de âncoras")
                    return

                if self._batch_dists is not None:
                    n_dataset = self._batch_dists.shape[1]
                    if self._dataset_anchors.shape[0] > n_dataset:
                        self._dataset_anchors = self._dataset_anchors[:n_dataset]

            except Exception as e:
                print(f"[DATASET] erro ao carregar âncoras: {e}")
                self.host._set_msg("Erro ao carregar âncoras")
                return

        # 3) route
        self._route_waypoints = None
        self._route_label = ""
        if route_file:
            try:
                with open(os.path.join(self.routes_dir, route_file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                wps = np.array(data.get("waypoints", []), dtype=float)
                if wps.size > 0:
                    self._route_waypoints = wps
                    self._route_label = route_file
            except Exception as e:
                print(f"[DATASET] erro ao carregar rota: {e}")
                self.host._set_msg("Erro ao carregar rota")
                return

        # 4) map
        self._map_env = None
        self._map_label = ""
        if map_file:
            try:
                map_path = os.path.join(self.maps_dir, map_file)
                self._map_env = Environment.load_json(map_path)
                self._map_label = map_file
            except Exception as e:
                print(f"[DATASET] erro ao carregar mapa: {e}")
                self.host._set_msg("Erro ao carregar mapa")
                return

        self._close_modal()
        self.host._set_msg("Dataset configurado")

    def _handle_dataset_modal_events(self, event) -> bool:
        if not self.dataset_modal_open:
            return False

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self._close_modal()
                return True

        names = ["dataset", "anchors", "route", "map"]
        flags = {
            "dataset": "dataset_dropdown_data_open",
            "anchors": "dataset_dropdown_anchors_open",
            "route": "dataset_dropdown_route_open",
            "map": "dataset_dropdown_map_open",
        }
        lists = {
            "dataset": self.available_datasets,
            "anchors": self.available_anchors,
            "route": self.available_routes,
            "map": self.available_maps,
        }

        # scroll por mouse wheel dentro de dropdown aberto
        if event.type == pg.MOUSEWHEEL:
            mouse_pos = pg.mouse.get_pos()
            for name in names:
                inp = self.dataset_inputs[name]
                is_open = getattr(self, flags[name])
                if is_open and inp["dropdown_rect"].collidepoint(mouse_pos):
                    # wheel up => sobe lista
                    self._scroll_dropdown(name, -event.y)
                    return True

        if event.type == pg.MOUSEBUTTONDOWN:
            pos = getattr(event, "pos", pg.mouse.get_pos())

            # scroll antigo do pygame
            if event.button == 4 or event.button == 5:
                delta = -1 if event.button == 4 else 1
                for name in names:
                    inp = self.dataset_inputs[name]
                    is_open = getattr(self, flags[name])
                    if is_open and inp["dropdown_rect"].collidepoint(pos):
                        self._scroll_dropdown(name, delta)
                        return True

            if event.button == 1:
                if self.dataset_buttons["ok"].hit(pos):
                    self._apply_dataset_config()
                    return True

                if self.dataset_buttons["cancel"].hit(pos):
                    self._close_modal()
                    return True

                # primeiro tenta clique em dropdown aberto
                for name in names:
                    inp = self.dataset_inputs[name]
                    flag_name = flags[name]
                    is_open = getattr(self, flag_name)

                    if is_open:
                        if inp["dropdown_rect"].collidepoint(pos):
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

                # clique nas caixas abre o dropdown correspondente
                for name in names:
                    inp = self.dataset_inputs[name]
                    if inp["rect"].collidepoint(pos):
                        self.dataset_dropdown_data_open = False
                        self.dataset_dropdown_anchors_open = False
                        self.dataset_dropdown_route_open = False
                        self.dataset_dropdown_map_open = False
                        setattr(self, flags[name], True)
                        return True

                # clique fora fecha tudo
                self.dataset_dropdown_data_open = False
                self.dataset_dropdown_anchors_open = False
                self.dataset_dropdown_route_open = False
                self.dataset_dropdown_map_open = False

        return False
    
    def _get_dropdown_state(self, name: str):
        if name == "dataset":
            return self.dataset_dropdown_data_open, self.dataset_dropdown_scroll, self.available_datasets
        if name == "anchors":
            return self.dataset_dropdown_anchors_open, self.anchors_dropdown_scroll, self.available_anchors
        if name == "route":
            return self.dataset_dropdown_route_open, self.route_dropdown_scroll, self.available_routes
        if name == "map":
            return self.dataset_dropdown_map_open, self.map_dropdown_scroll, self.available_maps
        return False, 0, []

    def _set_dropdown_open(self, name: str, value: bool):
        if name == "dataset":
            self.dataset_dropdown_data_open = value
        elif name == "anchors":
            self.dataset_dropdown_anchors_open = value
        elif name == "route":
            self.dataset_dropdown_route_open = value
        elif name == "map":
            self.dataset_dropdown_map_open = value

    def _get_dropdown_scroll(self, name: str) -> int:
        if name == "dataset":
            return self.dataset_dropdown_scroll
        if name == "anchors":
            return self.anchors_dropdown_scroll
        if name == "route":
            return self.route_dropdown_scroll
        if name == "map":
            return self.map_dropdown_scroll
        return 0

    def _set_dropdown_scroll(self, name: str, value: int):
        if name == "dataset":
            self.dataset_dropdown_scroll = value
        elif name == "anchors":
            self.anchors_dropdown_scroll = value
        elif name == "route":
            self.route_dropdown_scroll = value
        elif name == "map":
            self.map_dropdown_scroll = value

    def _scroll_dropdown(self, name: str, delta: int):
        _, scroll, items = self._get_dropdown_state(name)
        rect = self.dataset_inputs[name]["dropdown_rect"]

        item_h = 26
        max_visible = max(1, rect.h // item_h)
        max_scroll = max(0, len(items) - max_visible)

        new_scroll = max(0, min(max_scroll, scroll + delta))
        self._set_dropdown_scroll(name, new_scroll)

    # =========================================================
    # EVENTS
    # =========================================================

    def handle_events(self, events):
        actions = _actions_default()

        for event in events:
            if event.type == pg.QUIT:
                return _actions_quit()

            if self._handle_dataset_modal_events(event):
                continue

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return _actions_menu()

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
                        return _actions_menu()

                    elif self.btn_mode.hit(pos):
                        self.host._toggle_mode()
                        return actions

                    elif self.btn_load_dataset.hit(pos):
                        self._open_dataset_modal()
                        continue

                    elif self.btn_run_batch.hit(pos):
                        self._run_batch()
                        continue

                    elif self.btn_export.hit(pos):
                        if hasattr(self.host, "_export_csv"):
                            self.host._batch_results = self._batch_results
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
    # UPDATE / DRAW
    # =========================================================

    def update(self, dt: float) -> None:
        pass

    def draw(self) -> None:
        self.host.screen.fill(WHITE)

        draw_grid(self.host.screen, self.host.cam)

        # mapa
        if self._map_env is not None:
            draw_environment(self.host.screen, self.host.cam, self._map_env)

        # rota real
        if self._route_waypoints is not None and len(self._route_waypoints) > 1:
            route_pts = [tuple(p[:2]) for p in self._route_waypoints]
            draw_path(self.host.screen, self.host.cam, route_pts, (80, 80, 80), 2, dashed=True)

        # âncoras
        if self._dataset_anchors is not None and self._dataset_anchors.size > 0:
            anchors_3xN = self._dataset_anchors.T
            draw_anchors(self.host.screen, self.host.cam, anchors_3xN)

        draw_axes(self.host.screen, self.host.cam, self.host.font)

        # resultados: rota estimada + pontos
        if self._batch_results is not None:
            for algo in ALGO_ORDER:
                if not self.selected.get(algo, False):
                    continue
                if algo not in self._batch_results:
                    continue

                res = self._batch_results[algo]
                pos = res.get("posicoes", None)
                if pos is None:
                    continue

                pos = np.asarray(pos, dtype=float)
                if pos.ndim != 2 or pos.shape[1] < 2:
                    continue

                valid = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
                pos = pos[valid]
                if len(pos) == 0:
                    continue

                color = ALGO_COLORS.get(algo, BLACK)

                # trajetória estimada
                pts = [tuple(p[:2]) for p in pos]
                if len(pts) > 1:
                    draw_path(self.host.screen, self.host.cam, pts, color, 2)

                # pontos estimados
                for px, py in pts:
                    sx, sy = self.host.cam.world_to_screen(px, py)
                    pg.draw.circle(self.host.screen, color, (sx, sy), 3)
                    pg.draw.circle(self.host.screen, BLACK, (sx, sy), 3, 1)

        pg.draw.rect(
            self.host.screen,
            GRAY_D,
            pg.Rect(0, 0, self.host.cam.viewport[0], self.host.screen.get_height()),
            1,
        )

        self.host.mode = MODE_DATASET
        self.host._dataset_path = self._dataset_path
        self.host._dataset_label = self._dataset_label
        self.host._batch_dists = self._batch_dists
        self.host._batch_devs = self._batch_devs
        self.host._batch_results = self._batch_results
        self.host._dataset_anchors = self._dataset_anchors
        self.host.selected = self.selected

        if hasattr(self.host, "_draw_hud"):
            self.host._draw_hud()

        if self._batch_results is not None:
            self._draw_analyzer()

        if self.dataset_modal_open:
            self._draw_dataset_modal()

    def _draw_dataset_modal(self):
        screen = self.host.screen
        font = self.host.font
        bigfont = self.host.bigfont

        sw = screen.get_width()
        sh = screen.get_height()

        w, h = 660, 430
        mx = (sw - w) // 2
        my = (sh - h) // 2
        modal_rect = pg.Rect(mx, my, w, h)

        overlay = pg.Surface((sw, sh), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        pg.draw.rect(screen, (245, 245, 248), modal_rect, border_radius=10)
        pg.draw.rect(screen, (80, 80, 90), modal_rect, 2, border_radius=10)

        txt = bigfont.render("Configurar Dataset", True, (20, 20, 20))
        screen.blit(txt, (mx + 18, my + 14))

        label_x = mx + 30
        y = my + 82

        entries = [
            ("Dataset:", "dataset"),
            ("Âncoras:", "anchors"),
            ("Rota:", "route"),
            ("Mapa:", "map"),
        ]

        for label, key in entries:
            txt = font.render(label, True, (40, 40, 40))
            screen.blit(txt, (label_x, y))
            self._draw_input_box(self.dataset_inputs[key])
            self._draw_dropdown_arrow(self.dataset_inputs[key]["rect"])
            y += 60

        # desenhar dropdown por último, acima de tudo
        if self.dataset_dropdown_data_open:
            self._draw_dropdown_list("dataset", self.dataset_inputs["dataset"]["dropdown_rect"], self.available_datasets)
        if self.dataset_dropdown_anchors_open:
            self._draw_dropdown_list("anchors", self.dataset_inputs["anchors"]["dropdown_rect"], self.available_anchors)
        if self.dataset_dropdown_route_open:
            self._draw_dropdown_list("route", self.dataset_inputs["route"]["dropdown_rect"], self.available_routes)
        if self.dataset_dropdown_map_open:
            self._draw_dropdown_list("map", self.dataset_inputs["map"]["dropdown_rect"], self.available_maps)

        self.dataset_buttons["ok"].draw(screen)
        self.dataset_buttons["cancel"].draw(screen)

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

        # scrollbar visual
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
    # LOADERS
    # =========================================================

    def _try_load_dataset(self, path: str):
        self._dataset_path = path
        self._dataset_label = os.path.basename(path)
        self._dataset_stats = None

        try:
            if path.lower().endswith(".jsonl"):
                self._load_jsonl(path)
            else:
                try:
                    dists, devs = carregar_ensaio_lab(path)
                except Exception:
                    dists, devs = self._load_sim_txt_dataset(path)

                self._batch_dists = dists
                self._batch_devs = devs

            self.host._set_msg(f"Dataset carregado: {self._dataset_label}")

        except Exception as e:
            print(f"[DATASET] erro ao carregar dataset: {e}")
            self._batch_dists = None
            self._batch_devs = None
            self.host._set_msg("Erro ao carregar dataset")

    def _load_sim_txt_dataset(self, path: str):
        rows = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue
                if line.startswith("#"):
                    continue

                if ";" in line:
                    parts = [p.strip() for p in line.split(";") if p.strip()]
                elif "," in line:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                else:
                    parts = [p.strip() for p in line.split() if p.strip()]

                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue

                rows.append(vals)

        if not rows:
            raise ValueError(f"Nenhuma linha válida encontrada em {path}")

        data = np.array(rows, dtype=float)

        if data.shape[1] % 2 != 0:
            raise ValueError(
                f"Número de colunas inválido: {data.shape[1]} "
                f"(esperado par: dist,sigma,dist,sigma,...)"
            )

        dists = data[:, 0::2]
        devs = data[:, 1::2]
        return dists, devs

    def _load_jsonl(self, path: str):
        rows = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "z_k" in obj and obj["z_k"]:
                    rows.append(obj["z_k"])

        if not rows:
            raise ValueError("JSONL sem medições z_k válidas")

        dists_2N = np.array(rows, dtype=float)
        dists = dists_2N[:, 0::2]

        self._batch_dists = dists
        self._batch_devs = None

    def _run_batch(self):
        if self._batch_dists is None:
            self.host._set_msg("Carregue um dataset primeiro")
            return

        if self._dataset_anchors is None:
            self.host._set_msg("Selecione as âncoras primeiro")
            return

        algos = [k for k, v in self.selected.items() if v and k != "bc_ekf"]

        if not algos:
            self.host._set_msg("Selecione pelo menos um algoritmo")
            return

        try:
            n_dataset = self._batch_dists.shape[1]
            anchors = self._dataset_anchors
            if anchors.shape[0] > n_dataset:
                anchors = anchors[:n_dataset]

            devs = self._batch_devs
            if devs is not None and devs.shape[1] != n_dataset:
                devs = devs[:, :n_dataset]

            print("[DATASET DEBUG]")
            print("anchors shape:", anchors.shape)
            print("dists shape:", self._batch_dists.shape if self._batch_dists is not None else None)
            print("devs shape:", devs.shape if devs is not None else None)
            print("algos:", algos)

            self._batch_results = run_batch(
                anchors_Nx3=anchors,
                distances=self._batch_dists,
                deviations=devs,
                algoritmos=algos,
                p_true=None,
            )

            if not self._batch_results:
                self.host._set_msg("Batch executado, mas vazio")
                return

            self._dataset_stats = self._compute_dataset_stats(self._batch_results)

            ranking = self._dataset_ranking()
            for algo in self._batch_results:
                self._batch_results[algo]["ranking_row"] = next(
                    (row for row in ranking if row["algo"] == algo),
                    None
                )

            self.host._set_msg("Batch executado")

        except Exception as e:
            print(f"[DATASET] erro no batch: {e}")
            self._batch_results = None
            self._dataset_stats = None
            self.host._set_msg("Erro no batch")

    # =========================================================
    # ANALYZER
    # =========================================================

    def _compute_dataset_stats(self, results: dict):
        return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)
    
    def _dataset_ranking(self):
        return build_ranking_summary(self._dataset_stats, top_k=5)

    def _draw_analyzer(self):
        draw_analyzer_panel(
            screen=self.host.screen,
            font=self.host.font,
            bigfont=self.host.bigfont,
            title="Dataset Analyzer",
            stats=self._dataset_stats,
        )

    def close(self) -> None:
        pass


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