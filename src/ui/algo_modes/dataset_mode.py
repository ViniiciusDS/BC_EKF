from __future__ import annotations
from typing import Any
import os
import json
import numpy as np
import pygame as pg
from pathlib import Path
import csv

from src.ui.botton import Button
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_text
from src.environment.environment import Environment, draw_environment
from src.uwb.algoritmos_estaticos import carregar_ensaio_lab, run_batch
from src.ui.algo_modes.shared import (
    ALGO_ORDER,
    ALGO_COLORS,
    MODE_DATASET,
    MODE_MONTE_CARLO,
    MODE_STEP,
    default_selected,
    draw_analyzer_panel,
    load_anchors_from_json,
    load_route_from_json,
    load_map_from_json,
)
from src.analysis.algo_metrics import compute_dataset_cluster_stats, build_ranking_summary
from src.odometry import (
    EncoderConfig,
    DifferentialDriveConfig,
    build_dataset_from_encoder_and_uwb,
    build_range_sigma_matrices,
    extract_odometry_path,
    load_and_validate_encoder_file,
)
from src.ui.ui_elements import TextBoxDropdown


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
        self._dataset_route = None

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

        self._real_encoder_file = ""
        self._real_uwb_file = ""
        self._real_dataset = None
        self._real_aligned_rows = None
        self._real_odom_path = []
        self._real_range_matrix = None
        self._real_sigma_matrix = None
        self._real_timestamps = []
        self._real_anchor_ids = []

        self._real_drive_cfg = DifferentialDriveConfig(
            wheel_radius_m=0.03,
            wheel_base_m=0.16,
            encoder=EncoderConfig(ticks_per_wheel_rev=600.0),
        )

        self._dataset_source = "default"   # "default" | "real_encoder_uwb"

        self._real_encoder_input = None
        self._real_uwb_input = None

        self._btn_load_real_rect = pg.Rect(0, 0, 190, 32)
        self._btn_run_real_rect = pg.Rect(0, 0, 190, 32)

        self._real_encoder_input = TextBoxDropdown(
            rect=(0, 0, 320, 28),
            font=self.host.font if hasattr(self, "host") and self.host else None,
            options=[],
            placeholder="Arquivo encoder (.csv/.txt)",
        )

        self._real_uwb_input = TextBoxDropdown(
            rect=(0, 0, 320, 28),
            font=self.host.font if hasattr(self, "host") and self.host else None,
            options=[],
            placeholder="Arquivo UWB (.csv/.txt)",
        )   

        self.dataset_source_type = "simulated"   # "simulated" | "real_encoder_uwb"

        self.simulated_dataset_kind = "Front"   # "Front" | "Rear" | "BC"
        self.available_simulated_dataset_kinds = ["Front", "Rear", "BC"]
        self.dataset_dropdown_sim_kind_open = False

        self._bc_ekf_data = None

        # dropdowns de dataset real
        self.dataset_dropdown_source_open = False
        self.dataset_dropdown_real_encoder_open = False
        self.dataset_dropdown_real_uwb_open = False

        self.available_dataset_sources = [
            "Simulado",
            "Real (encoder + UWB)",
        ]

        self.available_real_encoder_files = []
        self.available_real_uwb_files = []

        self.real_encoder_dropdown_scroll = 0
        self.real_uwb_dropdown_scroll = 0

        self.real_data_dir = os.path.join("src", "odometry")

        self.selected = default_selected()

    def on_enter(self, host: Any) -> None:
        from pathlib import Path

        base = Path(__file__).resolve().parents[3]

        if self._real_encoder_input is not None and not self._real_encoder_input.text.strip():
            self._real_encoder_input.set_text(str(base / "src/odometry/encoder_square.csv"))

        if self._real_uwb_input is not None and not self._real_uwb_input.text.strip():
            self._real_uwb_input.set_text(str(base / "src/odometry/uwb_square.csv"))
            
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

    def _list_real_encoder_files(self):
        try:
            files = [
                f for f in os.listdir(self.real_data_dir)
                if f.lower().endswith(".csv") or f.lower().endswith(".txt")
            ]
            files.sort()
            return [f for f in files if "encoder" in f.lower()]
        except Exception:
            return []


    def _list_real_uwb_files(self):
        try:
            files = [
                f for f in os.listdir(self.real_data_dir)
                if f.lower().endswith(".csv") or f.lower().endswith(".txt")
            ]
            files.sort()
            return [f for f in files if "uwb" in f.lower()]
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
        self.available_real_encoder_files = self._list_real_encoder_files()
        self.available_real_uwb_files = self._list_real_uwb_files()

        sw = self.host.screen.get_width()
        sh = self.host.screen.get_height()

        # modal maior
        w, h = 780, 640
        mx = (sw - w) // 2
        my = (sh - h) // 2
        self.dataset_modal_rect = pg.Rect(mx, my, w, h)

        top_y = my + 78
        left_label_x = mx + 28
        left_input_x = mx + 220
        row_h = 60
        input_w = 440
        input_h = 30
        drop_h = 128

        self.dataset_inputs = {
            "source": {
                "value": "Simulado" if self.dataset_source_type == "simulated" else "Real (encoder + UWB)",
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
            "sim_kind": {
                "value": self.simulated_dataset_kind,
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
            "dataset": {
                "value": os.path.basename(self._dataset_path) if self._dataset_path else "",
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
            "real_encoder": {
                "value": os.path.basename(self._real_encoder_file) if self._real_encoder_file else "",
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
            "real_uwb": {
                "value": os.path.basename(self._real_uwb_file) if self._real_uwb_file else "",
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
            "anchors": {
                "value": os.path.basename(self._anchors_path) if getattr(self, "_anchors_path", "") else "",
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
            "route": {
                "value": self._route_label if self._route_label else "",
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
            "map": {
                "value": self._map_label if self._map_label else "",
                "label_pos": (0, 0),
                "rect": pg.Rect(0, 0, input_w, input_h),
                "dropdown_rect": pg.Rect(0, 0, input_w, drop_h),
            },
        }

        self._reflow_dataset_modal_inputs()

        btn_y = my + h - 48

        self.dataset_buttons["ok"] = Button(
            (mx + w - 220, btn_y, 90, 30),
            "Carregar",
            self.host.font,
        )

        self.dataset_buttons["cancel"] = Button(
            (mx + w - 115, btn_y, 90, 30),
            "Cancelar",
            self.host.font,
        )

        self.dataset_dropdown_scroll = 0
        self.anchors_dropdown_scroll = 0
        self.route_dropdown_scroll = 0
        self.map_dropdown_scroll = 0
        self.dataset_dropdown_source_open = False
        self.dataset_dropdown_real_encoder_open = False
        self.dataset_dropdown_real_uwb_open = False
        self.real_encoder_dropdown_scroll = 0
        self.real_uwb_dropdown_scroll = 0

    def _reflow_dataset_modal_inputs(self) -> None:
        if self.dataset_modal_rect is None or not self.dataset_inputs:
            return

        mx = self.dataset_modal_rect.x
        my = self.dataset_modal_rect.y

        top_y = my + 78
        left_label_x = mx + 28
        left_input_x = mx + 220
        row_h = 60
        input_w = 440
        input_h = 30
        drop_h = 128

        source_value = self.dataset_inputs["source"]["value"].strip()
        if source_value == "Real (encoder + UWB)":
            order = ["source", "real_encoder", "real_uwb", "anchors", "map"]
        else:
            order = ["source", "sim_kind", "dataset", "anchors", "route", "map"]

        for row_idx, key in enumerate(order):
            y = top_y + row_h * row_idx
            self.dataset_inputs[key]["label_pos"] = (left_label_x, y + 6)
            self.dataset_inputs[key]["rect"] = pg.Rect(left_input_x, y, input_w, input_h)
            self.dataset_inputs[key]["dropdown_rect"] = pg.Rect(left_input_x, y + 32, input_w, drop_h)

    def _close_modal(self):
        self.dataset_modal_open = False
        self.dataset_dropdown_data_open = False
        self.dataset_dropdown_anchors_open = False
        self.dataset_dropdown_route_open = False
        self.dataset_dropdown_map_open = False

    def _apply_dataset_config(self):
        dataset_source = self.dataset_inputs["source"]["value"].strip()
        dataset_file = self.dataset_inputs["dataset"]["value"].strip()
        real_encoder_file = self.dataset_inputs["real_encoder"]["value"].strip()
        real_uwb_file = self.dataset_inputs["real_uwb"]["value"].strip()
        anchors_file = self.dataset_inputs["anchors"]["value"].strip()
        route_file = self.dataset_inputs["route"]["value"].strip()
        map_file = self.dataset_inputs["map"]["value"].strip()
        sim_kind = self.dataset_inputs["sim_kind"]["value"].strip()

        self._bc_ekf_data = None

        self.dataset_source_type = (
            "real_encoder_uwb" if dataset_source == "Real (encoder + UWB)" else "simulated"
        )

        if self.dataset_source_type == "simulated":
            if not dataset_file:
                self.host._set_msg("Selecione um dataset simulado")
                return

            if sim_kind not in ("Front", "Rear", "BC"):
                self.host._set_msg("Selecione o tipo do dataset simulado")
                return

            self.simulated_dataset_kind = sim_kind

            # 1) dataset
            dataset_path = os.path.join(self.dataset_dir, dataset_file)
            self._try_load_dataset(dataset_path)

            # 2) âncoras
            if anchors_file:
                anchors_path = os.path.join(self.anchors_dir, anchors_file)
                if not self._load_anchors(anchors_path):
                    return

            # 3) rota
            if route_file:
                route_path = os.path.join(self.routes_dir, route_file)
                if not self._try_load_route(route_path):
                    return

            # 4) mapa
            if map_file:
                map_path = os.path.join(self.maps_dir, map_file)
                if not self._try_load_map(map_path):
                    return

            print(
                "[DATASET_CONFIG_SIM]",
                "sim_kind=", self.simulated_dataset_kind,
                "dataset_shape=",
                None if self._batch_dists is None else self._batch_dists.shape,
                "anchors_shape=",
                None if self._dataset_anchors is None else self._dataset_anchors.shape,
                "route_shape=",
                None if self._dataset_route is None else np.asarray(self._dataset_route).shape,
            )
            
            # 5) prepara BC-EKF somente no final, com tudo carregado
            self._bc_ekf_data = None
            if self.simulated_dataset_kind == "BC":
                self._prepare_bc_ekf_data_for_simulated_bc()

        elif self.dataset_source_type == "real_encoder_uwb":
            if not real_encoder_file:
                self.host._set_msg("Selecione um arquivo de encoder")
                return
            if not real_uwb_file:
                self.host._set_msg("Selecione um arquivo UWB")
                return

            if anchors_file:
                anchors_path = os.path.join(self.anchors_dir, anchors_file)
                if not self._load_anchors(anchors_path):
                    return

            if map_file:
                map_path = os.path.join(self.maps_dir, map_file)
                if not self._try_load_map(map_path):
                    return

            encoder_path = os.path.join(self.real_data_dir, real_encoder_file)
            uwb_path = os.path.join(self.real_data_dir, real_uwb_file)
            self._load_real_encoder_uwb_dataset(encoder_path, uwb_path)


        self._close_modal()
        self.host._set_msg("Dataset configurado")

    def _load_anchors(self, anchors_path: str) -> bool:
        ''' Carrega âncoras e valida compatibilidade com dataset carregado (se houver) '''
        try:
            self._dataset_anchors, _ = load_anchors_from_json(anchors_path)

            if self._batch_dists is not None:
                n_dataset = int(self._batch_dists.shape[1])
                n_anchors = int(self._dataset_anchors.shape[0])

                # Caso especial: dataset simulado BC usa 2 colunas por âncora
                if self.dataset_source_type == "simulated" and self.simulated_dataset_kind == "BC":
                    if n_dataset != 2 * n_anchors:
                        self.host._set_msg(
                            f"Incompatibilidade: dataset BC possui {n_dataset} colunas, "
                            f"mas o layout possui {n_anchors} âncoras (esperado {2 * n_anchors})"
                        )
                        return False
                else:
                    if n_anchors != n_dataset:
                        self.host._set_msg(
                            f"Incompatibilidade: dataset possui {n_dataset} colunas de âncora, "
                            f"mas o layout possui {n_anchors} âncoras"
                        )
                        return False

            return True

        except ValueError as e:
            print(f"[DATASET] erro ao carregar âncoras: {e}")
            self.host._set_msg(f"Erro ao carregar âncoras: {str(e)}")
            return False
        except Exception as e:
            print(f"[DATASET] erro ao carregar âncoras: {e}")
            self.host._set_msg("Erro ao carregar âncoras")
            return False

    def _try_load_route(self, route_path: str) -> bool:
        try:
            self._dataset_route, self._route_label = load_route_from_json(route_path)
            self._route_waypoints = self._dataset_route
            return True
        except ValueError as e:
            print(f"[DATASET] erro ao carregar rota: {e}")
            self.host._set_msg(f"Erro ao carregar rota: {str(e)}")
            return False
        except Exception as e:
            print(f"[DATASET] erro ao carregar rota: {e}")
            self.host._set_msg("Erro ao carregar rota")
            return False

    def _try_load_map(self, map_path: str) -> bool:
        try:
            self._map_env, self._map_label = load_map_from_json(map_path)
            return True
        except Exception as e:
            print(f"[DATASET] erro ao carregar mapa: {e}")
            self.host._set_msg("Erro ao carregar mapa")
            return False

    def _handle_dataset_modal_events(self, event) -> bool:
        if not self.dataset_modal_open:
            return False

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self._close_modal()
                return True

        source_value = self.dataset_inputs["source"]["value"].strip()

        names = ["source", "anchors", "map"]

        if source_value == "Real (encoder + UWB)":
            names.extend(["real_encoder", "real_uwb"])
        else:
            names.extend(["sim_kind", "dataset", "route"])
        
        flags = {
            "source": "dataset_dropdown_source_open",
            "sim_kind": "dataset_dropdown_sim_kind_open",
            "dataset": "dataset_dropdown_data_open",
            "real_encoder": "dataset_dropdown_real_encoder_open",
            "real_uwb": "dataset_dropdown_real_uwb_open",
            "anchors": "dataset_dropdown_anchors_open",
            "route": "dataset_dropdown_route_open",
            "map": "dataset_dropdown_map_open",
        }

        lists = {
            "source": self.available_dataset_sources,
            "dataset": self.available_datasets,
            "sim_kind": self.available_simulated_dataset_kinds,
            "real_encoder": self.available_real_encoder_files,
            "real_uwb": self.available_real_uwb_files,
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
                                if name == "source":
                                    self._reflow_dataset_modal_inputs()
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
                        self.dataset_dropdown_source_open = False
                        self.dataset_dropdown_real_encoder_open = False
                        self.dataset_dropdown_real_uwb_open = False
                        setattr(self, flags[name], True)
                        return True

                # clique fora fecha tudo
                self.dataset_dropdown_data_open = False
                self.dataset_dropdown_sim_kind_open = False
                self.dataset_dropdown_anchors_open = False
                self.dataset_dropdown_route_open = False
                self.dataset_dropdown_map_open = False
                self.dataset_dropdown_source_open = False
                self.dataset_dropdown_real_encoder_open = False
                self.dataset_dropdown_real_uwb_open = False

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
                
                elif event.key == pg.K_F9:
                    try:
                        self._load_demo_real_dataset()
                    except Exception as e:
                        self.host._set_msg(f"Erro ao carregar dataset real: {e}")

                elif event.key == pg.K_F10:
                    try:
                        if self._batch_dists is None:
                            raise ValueError("Nenhum dataset batch carregado")
                        self._run_batch()
                    except Exception as e:
                        self.host._set_msg(f"Erro ao rodar batch real: {e}")

            elif event.type == pg.MOUSEBUTTONDOWN:
                pos = getattr(event, "pos", pg.mouse.get_pos())
                mx, my = pos

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
        if self._real_encoder_input is not None:
            self._real_encoder_input.update(dt)

        if self._real_uwb_input is not None:
            self._real_uwb_input.update(dt)
        pass

    def draw(self) -> None:
        self.host.screen.fill(WHITE)

        draw_grid(self.host.screen, self.host.cam)

        # mapa
        if self._map_env is not None:
            draw_environment(self.host.screen, self.host.cam, self._map_env)

        route_to_draw = None
        route_color = (80, 80, 80)
        route_width = 2
        route_dashed = True

        if self._dataset_source == "real_encoder_uwb" and self._dataset_route is not None:
            route_to_draw = self._dataset_route
            route_color = (0, 0, 0)
            route_dashed = False
        elif self._dataset_route is not None:
            route_to_draw = self._dataset_route
        elif self._route_waypoints is not None:
            route_to_draw = self._route_waypoints

        if route_to_draw is not None and len(route_to_draw) > 1:
            route_pts = [tuple(p[:2]) for p in route_to_draw]
            draw_path(
                self.host.screen,
                self.host.cam,
                route_pts,
                route_color,
                route_width,
                dashed=route_dashed,
            )

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

        self._reflow_dataset_modal_inputs()

        sw = screen.get_width()
        sh = screen.get_height()

        w, h = 780, 640
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

        entries = [("Fonte:", "source")]

        source_value = self.dataset_inputs["source"]["value"].strip()

        if source_value == "Real (encoder + UWB)":
            entries.extend([
                ("Encoder real:", "real_encoder"),
                ("UWB real:", "real_uwb"),
                ("Âncoras:", "anchors"),
                ("Mapa:", "map"),
            ])
        else:
            entries.extend([
                ("Tipo de simulação:", "sim_kind"),
                ("Dataset simulado:", "dataset"),
                ("Âncoras:", "anchors"),
                ("Rota:", "route"),
                ("Mapa:", "map"),
            ])

        for label, key in entries:
            label_x, label_y = self.dataset_inputs[key]["label_pos"]
            rect = self.dataset_inputs[key]["rect"]

            draw_text(screen, label, label_x, label_y, font, color=(60, 60, 60))
            self._draw_dropdown_box(screen, rect, self.dataset_inputs[key]["value"], font)
            self._draw_dropdown_arrow(rect)

        source_value = self.dataset_inputs["source"]["value"].strip()

        self.dataset_buttons["ok"].draw(screen)
        self.dataset_buttons["cancel"].draw(screen)

        if self.dataset_dropdown_source_open:
            self._draw_dropdown_list("source", self.dataset_inputs["source"]["dropdown_rect"], self.available_dataset_sources)

        if source_value == "Real (encoder + UWB)":
            if self.dataset_dropdown_real_encoder_open:
                self._draw_dropdown_list("real_encoder", self.dataset_inputs["real_encoder"]["dropdown_rect"], self.available_real_encoder_files)
            if self.dataset_dropdown_real_uwb_open:
                self._draw_dropdown_list("real_uwb", self.dataset_inputs["real_uwb"]["dropdown_rect"], self.available_real_uwb_files)
            if self.dataset_dropdown_anchors_open:
                self._draw_dropdown_list("anchors", self.dataset_inputs["anchors"]["dropdown_rect"], self.available_anchors)
            if self.dataset_dropdown_map_open:
                self._draw_dropdown_list("map", self.dataset_inputs["map"]["dropdown_rect"], self.available_maps)
        else:
            if self.dataset_dropdown_sim_kind_open:
                self._draw_dropdown_list("sim_kind", self.dataset_inputs["sim_kind"]["dropdown_rect"], self.available_simulated_dataset_kinds)
            if self.dataset_dropdown_data_open:
                self._draw_dropdown_list("dataset", self.dataset_inputs["dataset"]["dropdown_rect"], self.available_datasets)
            if self.dataset_dropdown_anchors_open:
                self._draw_dropdown_list("anchors", self.dataset_inputs["anchors"]["dropdown_rect"], self.available_anchors)
            if self.dataset_dropdown_route_open:
                self._draw_dropdown_list("route", self.dataset_inputs["route"]["dropdown_rect"], self.available_routes)
            if self.dataset_dropdown_map_open:
                self._draw_dropdown_list("map", self.dataset_inputs["map"]["dropdown_rect"], self.available_maps)

    def _draw_dropdown_box(self, screen: pg.Surface, rect: pg.Rect, value: str, font: pg.font.Font) -> None:
        pg.draw.rect(screen, (255, 255, 255), rect, border_radius=6)
        pg.draw.rect(screen, (130, 130, 140), rect, 2, border_radius=6)

        txt = font.render(str(value), True, (30, 30, 30))
        screen.blit(txt, (rect.x + 8, rect.y + 5))

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

        algos_to_run = [a for a in ALGO_ORDER if self.selected.get(a, False)]

        # BC-EKF só faz sentido com dataset simulado do tipo BC
        if self.dataset_source_type == "simulated":
            if self.simulated_dataset_kind != "BC": 
                if "bc_ekf" in algos_to_run:
                    algos_to_run = [a for a in algos_to_run if a != "bc_ekf"]
                    self.host._set_msg("BC-EKF ignorado: só roda com dataset simulado do tipo BC")
            else:
                if "bc_ekf" in algos_to_run and self._bc_ekf_data is None:
                    algos_to_run = [a for a in algos_to_run if a != "bc_ekf"]
                    self.host._set_msg("BC-EKF ignorado: dados BC não foram preparados corretamente")

        if not algos_to_run:
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

            if self.dataset_source_type == "simulated" and self.simulated_dataset_kind == "BC":
                if "bc_ekf" in algos_to_run and self._bc_ekf_data is None:
                    self.host._set_msg(
                        "BC-EKF não foi preparado corretamente para este dataset BC"
                    )

            print(
                "[RUN_BATCH]",
                "dataset_source_type=", self.dataset_source_type,
                "simulated_dataset_kind=", self.simulated_dataset_kind,
                "bc_ekf_data_is_none=", self._bc_ekf_data is None,
                "batch_shape=", None if self._batch_dists is None else self._batch_dists.shape,
            )

            self._batch_results = run_batch(
                anchors_Nx3=anchors,
                distances=self._batch_dists,
                deviations=devs,
                algoritmos=algos_to_run,
                p_true=None,
                bc_ekf_data=self._bc_ekf_data,
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

            
    def _load_simple_uwb_file(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo UWB não encontrado: {path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8-sig")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                raise ValueError(f"Arquivo UWB vazio: {path}")

            # tenta csv com vírgula primeiro
            if "," in lines[0]:
                reader = csv.DictReader(lines, delimiter=",")
                return list(reader)

            # tenta delimitado por espaço
            header = lines[0].split()
            rows = []
            for line in lines[1:]:
                values = line.split()
                if len(values) != len(header):
                    raise ValueError(f"Linha UWB inválida: {line}")
                rows.append(dict(zip(header, values)))
            return rows

        raise ValueError(f"Formato UWB não suportado: {suffix}")
    
    def _load_real_encoder_uwb_dataset(
        self,
        encoder_file: str,
        uwb_file: str,
    ):
        """
        Carrega encoder + UWB reais, constrói dataset alinhado
        e prepara estruturas compatíveis com o Dataset Mode.
        """
        encoder_samples = load_and_validate_encoder_file(encoder_file)
        uwb_rows = self._load_simple_uwb_file(uwb_file)

        dataset = build_dataset_from_encoder_and_uwb(
            encoder_samples,
            uwb_rows,
            self._real_drive_cfg,
            clamp=True,
        )

        matrices = build_range_sigma_matrices(dataset["aligned_rows"])
        odom_path = extract_odometry_path(dataset["aligned_rows"])

        self._real_encoder_file = str(encoder_file)
        self._real_uwb_file = str(uwb_file)
        self._real_dataset = dataset
        self._real_aligned_rows = dataset["aligned_rows"]
        self._real_odom_path = odom_path
        self._real_range_matrix = matrices["ranges"]
        self._real_sigma_matrix = matrices["sigmas"]
        self._real_timestamps = matrices["timestamps_s"]
        self._real_anchor_ids = matrices["anchor_ids"]

        ok = self._apply_real_dataset_to_batch_state()
        if not ok:
            return

        self.host._set_msg(
            f"Dataset real carregado: {len(dataset['aligned_rows'])} medições alinhadas"
        )
        
    def _real_dataset_as_numpy(self):
        """
        Converte o dataset real consolidado em arrays NumPy compatíveis
        com o fluxo batch do Dataset Mode.
        """
        import numpy as np

        if self._real_range_matrix is None:
            return None

        dists = np.asarray(self._real_range_matrix, dtype=float)
        devs = np.asarray(self._real_sigma_matrix, dtype=float)
        route = np.asarray(self._real_odom_path, dtype=float) if self._real_odom_path else None

        return {
            "dists": dists,
            "devs": devs,
            "route": route,
            "timestamps_s": np.asarray(self._real_timestamps, dtype=float),
            "anchor_ids": np.asarray(self._real_anchor_ids, dtype=int),
        }
    
    def _load_demo_real_dataset(self):
        from pathlib import Path

        base = Path(__file__).resolve().parents[3]

        self._load_real_encoder_uwb_dataset(
            base / "src/odometry/encoder_square.csv",
            base / "src/odometry/uwb_square.csv",
        )

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

    def _draw_real_dataset_status(self):
        if self._real_dataset is None:
            return

        # Keep status aligned with the analyzer/sidebar block defaults.
        status_x = 20
        status_y = 350
        txt = self.host.font.render(
            f"Odom real: {len(self._real_odom_path)} pts",
            True,
            (40, 40, 40),
        )
        self.host.screen.blit(txt, (status_x, status_y))

        if self._dataset_source == "real_encoder_uwb":
            source_txt = self.host.font.render(
                "Fonte: encoder + UWB real",
                True,
                (20, 80, 180),
            )
            self.host.screen.blit(source_txt, (status_x, status_y + 30))

    def _draw_real_dataset_controls(self):
        draw_text(
            self.host.screen,
            "Dataset real (encoder + UWB)",
            20,
            420,
            self.host.bigfont,
            color=(20, 20, 20),
        )

        if self._real_encoder_input is not None:
            self._real_encoder_input.rect.topleft = (20, 455)
            self._real_encoder_input.rect.size = (320, 28)
            self._real_encoder_input.draw(self.host.screen)

        draw_text(
            self.host.screen,
            "Encoder:",
            20,
            438,
            self.host.font,
            color=(50, 50, 50),
        )

        if self._real_uwb_input is not None:
            self._real_uwb_input.rect.topleft = (20, 515)
            self._real_uwb_input.rect.size = (320, 28)
            self._real_uwb_input.draw(self.host.screen)

        draw_text(
            self.host.screen,
            "UWB:",
            20,
            498,
            self.host.font,
            color=(50, 50, 50),
        )

        self._btn_load_real_rect.topleft = (20, 560)
        self._btn_load_real_rect.size = (160, 32)
        self._draw_button(
            self.host.screen,
            self._btn_load_real_rect,
            "Carregar real",
            self.host.font,
        )

        self._btn_run_real_rect.topleft = (190, 560)
        self._btn_run_real_rect.size = (150, 32)
        self._draw_button(
            self.host.screen,
            self._btn_run_real_rect,
            "Rodar batch",
            self.host.font,
        )

        if self._dataset_source == "real_encoder_uwb":
            draw_text(
                self.host.screen,
                f"Fonte: REAL ({len(self._real_odom_path)} poses odom)",
                20,
                600,
                self.host.font,
                color=(20, 80, 180),
            )

            draw_text(
                self.host.screen,
                f"Âncoras no dataset: {len(self._real_anchor_ids)}",
                20,
                620,
                self.host.font,
                color=(40, 40, 40),
            )

    def _draw_button(self, screen: pg.Surface, rect: pg.Rect, label: str, font: pg.font.Font) -> None:
        pg.draw.rect(screen, (220, 220, 220), rect, border_radius=4)
        pg.draw.rect(screen, BLACK, rect, 1, border_radius=4)
        txt = font.render(label, True, BLACK)
        screen.blit(txt, (rect.x + 10, rect.y + (rect.height - txt.get_height()) // 2))

    def _apply_real_dataset_to_batch_state(self):
        """
        Injeta o dataset real consolidado no mesmo estado interno usado
        pelo fluxo batch tradicional do Dataset Mode.
        """
        import numpy as np

        payload = self._real_dataset_as_numpy()
        if payload is None:
            raise ValueError("Nenhum dataset real carregado")

        self._batch_dists = np.asarray(payload["dists"], dtype=float)
        self._batch_devs = np.asarray(payload["devs"], dtype=float)
        self._dataset_route = payload["route"]
        self._dataset_label = (
            f"REAL | enc={Path(self._real_encoder_file).name} | "
            f"uwb={Path(self._real_uwb_file).name}"
        )
        self._dataset_source = "real_encoder_uwb"

        if not self._validate_real_dataset_against_loaded_anchors():
            return False

        self.host._set_msg(
            f"Dataset real aplicado ao batch: {self._batch_dists.shape[0]} amostras, "
            f"{self._batch_dists.shape[1]} âncoras"
        )
        return True

    def _validate_real_dataset_against_loaded_anchors(self) -> bool:
        """
        Verifica se o número de colunas do dataset real bate com as âncoras já carregadas.
        Em caso de erro, mostra mensagem na UI e retorna False.
        """
        if self._batch_dists is None:
            return True

        if self._dataset_anchors is None:
            self.host._set_msg(
                "Dataset real carregado, mas nenhum layout de âncoras foi carregado"
            )
            return False

        n_cols = int(self._batch_dists.shape[1])
        n_anchors = int(self._dataset_anchors.shape[0])

        if n_cols != n_anchors:
            self.host._set_msg(
                f"Incompatibilidade: dataset real possui {n_cols} âncoras, "
                f"mas o layout carregado tem {n_anchors}"
            )
            return False

        return True

    def _load_real_dataset_from_inputs(self):
        encoder_file = ""
        uwb_file = ""

        if self._real_encoder_input is not None:
            encoder_file = self._real_encoder_input.text.strip()

        if self._real_uwb_input is not None:
            uwb_file = self._real_uwb_input.text.strip()

        if not encoder_file:
            raise ValueError("Informe o arquivo de encoder")
        if not uwb_file:
            raise ValueError("Informe o arquivo UWB")

        self._load_real_encoder_uwb_dataset(encoder_file, uwb_file)

    def _run_real_dataset_batch(self):
        if self._dataset_source != "real_encoder_uwb":
            raise ValueError("Nenhum dataset real foi carregado")

        if self._batch_dists is None:
            raise ValueError("Dataset real não foi aplicado ao estado batch")

        self._run_batch()


    def _route_xy_to_pose_xytheta(self, route_xy):
        '''Converte uma rota de pontos XY em uma sequência de poses XYTheta, 
        onde Theta é a orientação calculada entre os pontos consecutivos. 
        Para o último ponto, a orientação é copiada do penúltimo para evitar valores indefinidos.'''
        import numpy as np

        route_xy = np.asarray(route_xy, dtype=float)
        if route_xy.ndim != 2 or route_xy.shape[1] < 2 or len(route_xy) == 0:
            raise ValueError("Rota inválida para BC-EKF")

        poses = np.zeros((len(route_xy), 3), dtype=float)
        poses[:, :2] = route_xy[:, :2]

        if len(route_xy) == 1:
            return poses

        for i in range(len(route_xy) - 1):
            dx = route_xy[i + 1, 0] - route_xy[i, 0]
            dy = route_xy[i + 1, 1] - route_xy[i, 1]
            poses[i, 2] = np.arctan2(dy, dx)

        poses[-1, 2] = poses[-2, 2]
        return poses


    def _pose_xytheta_to_vw(self, poses_xytheta, T):
        '''Converte uma sequência de poses XYTheta em um caminho de odometria linear e angular (v, w) usando diferenças finitas.
        A velocidade linear v é calculada como a distância entre poses consecutivas dividida pelo intervalo'''
        import numpy as np

        poses = np.asarray(poses_xytheta, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError("Poses inválidas para BC-EKF")

        M = poses.shape[0]
        odom = np.zeros((2, M), dtype=float)

        if M < 2:
            return odom

        for k in range(1, M):
            dx = poses[k, 0] - poses[k - 1, 0]
            dy = poses[k, 1] - poses[k - 1, 1]
            ds = float(np.hypot(dx, dy))

            dtheta = poses[k, 2] - poses[k - 1, 2]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

            odom[0, k] = ds / T
            odom[1, k] = dtheta / T

        return odom
    

    def _prepare_bc_ekf_data_for_simulated_bc(self):
        '''Prepara os dados do dataset para o formato esperado pelo BC-EKF,
         assumindo que as medições são de um cenário simulado com layout conhecido de âncoras e rota.
        Verifica consistência dos dados, converte a rota em odometria e extrai'''
        import numpy as np
        import src.config as config

        print(
            "[BC_EKF_PREP_ENTER]",
            "batch_dists_is_none=", self._batch_dists is None,
            "anchors_is_none=", self._dataset_anchors is None,
            "route_is_none=", self._dataset_route is None,
            "sim_kind=", self.simulated_dataset_kind,
        )

        if self._batch_dists is None:
            self._bc_ekf_data = None
            return

        if self._dataset_anchors is None:
            self._bc_ekf_data = None
            return

        full_dists = np.asarray(self._batch_dists, dtype=float)
        full_devs = np.asarray(self._batch_devs, dtype=float) if self._batch_devs is not None else None

        if full_dists.ndim != 2:
            self._bc_ekf_data = None
            return

        n_cols = full_dists.shape[1]
        if n_cols % 2 != 0:
            self.host._set_msg("Dataset BC inválido: número de colunas deve ser par (front/rear)")
            self._bc_ekf_data = None
            return

        n_anchors = self._dataset_anchors.shape[0]
        if n_cols != 2 * n_anchors:
            self.host._set_msg(
                f"Dataset BC incompatível: possui {n_cols} colunas, mas o layout tem {n_anchors} âncoras"
            )
            self._bc_ekf_data = None
            return

        if self._dataset_route is None:
            self.host._set_msg("Dataset BC requer rota para gerar odometria do EKF")
            self._bc_ekf_data = None
            return

        poses = self._route_xy_to_pose_xytheta(self._dataset_route)
        T = float(getattr(config, "TIME_STEP", 0.05))
        odometry_noisy = self._pose_xytheta_to_vw(poses, T)

        z_hist = full_dists.T  # (2N, M)

        sigma_uwb = float(np.nanmedian(full_devs)) if full_devs is not None else float(np.sqrt(0.0025))
        if not np.isfinite(sigma_uwb):
            sigma_uwb = float(np.sqrt(0.0025))

        M = full_dists.shape[0]

        if poses.shape[0] != M:
            self.host._set_msg(
                f"BC-EKF inválido: rota possui {poses.shape[0]} amostras, mas dataset possui {M}"
            )
            self._bc_ekf_data = None
            return

        if odometry_noisy.shape[1] != M:
            self.host._set_msg(
                f"BC-EKF inválido: odometria possui {odometry_noisy.shape[1]} passos, mas dataset possui {M}"
            )
            self._bc_ekf_data = None
            return

        if z_hist.shape[1] != M:
            self.host._set_msg(
                f"BC-EKF inválido: z_hist possui {z_hist.shape[1]} passos, mas dataset possui {M}"
            )
            self._bc_ekf_data = None
            return

        if z_hist.shape[0] != 2 * n_anchors:
            self.host._set_msg(
                f"BC-EKF inválido: z_hist possui {z_hist.shape[0]} linhas, esperado {2 * n_anchors}"
            )
            self._bc_ekf_data = None
            return

        self._bc_ekf_data = {
            "T": T,
            "odometry_noisy": odometry_noisy,
            "z_hist": z_hist,
            "l": float(getattr(config, "WHEEL_BASE", 0.65)) / 2.0,
            "z_c": float(getattr(config, "TAG_HEIGHT", 0.5)),
            "sigma_uwb": sigma_uwb,
        }

        print(
            "[BC_EKF_PREP_OK]",
            "z_hist=", z_hist.shape,
            "odometry_noisy=", odometry_noisy.shape,
            "anchors=", self._dataset_anchors.shape,
        )
        
        # para os algoritmos clássicos, usa apenas as colunas da tag frontal
        self._batch_dists = full_dists[:, 0::2]
        if full_devs is not None:
            self._batch_devs = full_devs[:, 0::2]

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