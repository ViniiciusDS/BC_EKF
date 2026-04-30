from __future__ import annotations
from typing import Any
import os
import json
import numpy as np
import pygame as pg
from pathlib import Path
import csv
import re

from src import config
from src.ui.botton import Button
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_text
from src.environment.environment import draw_environment
from src.uwb.algoritmos_estaticos import carregar_ensaio_lab, run_batch
from src.ui.algo_modes.shared import (
    ALGO_ORDER,
    ALGO_COLORS,
    MODE_DATASET,
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
from src.ui.legend_overlay import draw_legend_overlay


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_D = (90, 90, 90)



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
        self._anchors_uwb_ids = None
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

        self._real_encoder_use_distance_columns = bool(
            getattr(config, "REAL_ENCODER_USE_DISTANCE_COLUMNS", True)
        )
        self._real_encoder_distance_unit_scale = float(
            getattr(config, "REAL_ENCODER_DISTANCE_UNIT_SCALE", 0.01)
        )
        self._real_encoder_swap_lr = bool(
            getattr(config, "REAL_ENCODER_SWAP_LR", False)
        )
        self._real_encoder_invert_left = bool(
            getattr(config, "REAL_ENCODER_INVERT_LEFT", False)
        )
        self._real_encoder_invert_right = bool(
            getattr(config, "REAL_ENCODER_INVERT_RIGHT", False)
        )

        self._real_drive_cfg = DifferentialDriveConfig(
            wheel_radius_m=float(getattr(config, "WHEEL_RADIUS", 0.0325)),
            wheel_base_m=float(getattr(config, "WHEEL_BASE", 0.16)),
            encoder=EncoderConfig(
                ticks_per_wheel_rev=float(getattr(config, "ENCODER_TICKS_PER_REV", 1320.0))
            ),
        )

        self._dataset_source = "default"   # "default" | "real_encoder_uwb"

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

        self.simulated_dataset_kind = "Front"   # "Front" | "Rear" | "MID" | "BC"
        self.available_simulated_dataset_kinds = ["Front", "Rear", "MID", "BC"]
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

        self.real_data_dir = os.path.join("resultados", "datasets")

        self.show_analyzer = True
        self.btn_toggle_analyzer = Button(
            (0, 0, 190, 32),
            "Ocultar Analyzer",
            self.host.font if self.host else None,
        )

        self.show_legend_overlay = False
        self._legend_close_rect = None
        self.btn_toggle_legend = Button(
            (0, 0, 190, 32),
            "Mostrar Legenda",
            self.host.font if self.host else None,
        )

        self.selected = default_selected()

    def on_enter(self, host: Any) -> None:
        base = Path(__file__).resolve().parents[3]

        if self._real_encoder_input is not None and not self._real_encoder_input.text.strip():
            self._real_encoder_input.set_text(str(base / "resultados" / "datasets" / "encoder_square.csv"))

        if self._real_uwb_input is not None and not self._real_uwb_input.text.strip():
            self._real_uwb_input.set_text(str(base / "resultados" / "datasets" / "uwb_square.csv"))

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

    # ------------------------------------------------------------------
    # HELPERS INTERNOS
    # ------------------------------------------------------------------

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

    def _close_all_dataset_dropdowns(self) -> None:
        self.dataset_dropdown_data_open = False
        self.dataset_dropdown_anchors_open = False
        self.dataset_dropdown_route_open = False
        self.dataset_dropdown_map_open = False
        self.dataset_dropdown_source_open = False
        self.dataset_dropdown_sim_kind_open = False
        self.dataset_dropdown_real_encoder_open = False
        self.dataset_dropdown_real_uwb_open = False

    def _open_dataset_modal(self):
        self.dataset_modal_open = True
        self._close_all_dataset_dropdowns()

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
        self._close_all_dataset_dropdowns()

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

            if sim_kind not in ("Front", "Rear", "MID", "BC"):
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
        """Carrega âncoras e valida compatibilidade com dataset carregado (se houver).
        Também lê, opcionalmente, o mapeamento dos IDs reais do UWB.
        """
        try:
            self._dataset_anchors, _ = load_anchors_from_json(anchors_path)

            # guarda caminho e tenta ler metadados extras do JSON bruto
            self._anchors_path = anchors_path
            self._anchors_uwb_ids = None

            try:
                raw = json.loads(Path(anchors_path).read_text(encoding="utf-8"))
                raw_ids = raw.get("anchor_ids_uwb", None)

                if raw_ids is not None:
                    parsed_ids = []
                    for x in raw_ids:
                        sx = str(x).strip()
                        if sx.lower().startswith("da"):
                            sx = sx[2:]
                        parsed_ids.append(int(sx))
                    self._anchors_uwb_ids = parsed_ids

                    print("[ANCHORS_UWB_IDS]", self._anchors_uwb_ids)
            except Exception as meta_err:
                print(f"[DATASET] aviso ao ler anchor_ids_uwb: {meta_err}")

            if self._batch_dists is not None:
                n_dataset = int(self._batch_dists.shape[1])
                n_anchors = int(self._dataset_anchors.shape[0])

                # Caso especial: dataset simulado BC usa 2 colunas por âncora
                if self.dataset_source_type == "simulated" and self.simulated_dataset_kind == "BC":
                    if n_dataset != 2 * n_anchors:
                        self.host._set_msg(
                            f"Dataset BC inválido: esperado front+rear ({2 * n_anchors} colunas), recebido {n_dataset}"
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
                        self._close_all_dataset_dropdowns()
                        setattr(self, flags[name], True)
                        return True

                # clique fora fecha tudo
                self._close_all_dataset_dropdowns()

        return False
    
    def _get_dropdown_state(self, name: str):
        if name == "source":
            return self.dataset_dropdown_source_open, 0, self.available_dataset_sources
        if name == "sim_kind":
            return self.dataset_dropdown_sim_kind_open, 0, self.available_simulated_dataset_kinds
        if name == "dataset":
            return self.dataset_dropdown_data_open, self.dataset_dropdown_scroll, self.available_datasets
        if name == "real_encoder":
            return self.dataset_dropdown_real_encoder_open, self.real_encoder_dropdown_scroll, self.available_real_encoder_files
        if name == "real_uwb":
            return self.dataset_dropdown_real_uwb_open, self.real_uwb_dropdown_scroll, self.available_real_uwb_files
        if name == "anchors":
            return self.dataset_dropdown_anchors_open, self.anchors_dropdown_scroll, self.available_anchors
        if name == "route":
            return self.dataset_dropdown_route_open, self.route_dropdown_scroll, self.available_routes
        if name == "map":
            return self.dataset_dropdown_map_open, self.map_dropdown_scroll, self.available_maps
        return False, 0, []

    def _get_dropdown_scroll(self, name: str) -> int:
        if name == "source":
            return 0
        if name == "sim_kind":
            return 0
        if name == "dataset":
            return self.dataset_dropdown_scroll
        if name == "real_encoder":
            return self.real_encoder_dropdown_scroll
        if name == "real_uwb":
            return self.real_uwb_dropdown_scroll
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
        elif name == "real_encoder":
            self.real_encoder_dropdown_scroll = value
        elif name == "real_uwb":
            self.real_uwb_dropdown_scroll = value
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
    # EVENTOS
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
                    if self.show_legend_overlay:
                        self.show_legend_overlay = False
                        continue
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
                    if self.show_legend_overlay and self._legend_close_rect is not None:
                        if self._legend_close_rect.collidepoint(pos):
                            self.show_legend_overlay = False
                            continue

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

                    elif self.btn_toggle_analyzer.hit(pos):
                        self.show_analyzer = not self.show_analyzer
                        return actions

                    elif self.btn_toggle_legend.hit(pos):
                        self.show_legend_overlay = not self.show_legend_overlay
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

    # ------------------------------------------------------------------
    # DRAW
    # ------------------------------------------------------------------

    # =========================================================
    # UPDATE / DRAW
    # =========================================================

    def update(self, dt: float) -> None:
        if self._real_encoder_input is not None:
            self._real_encoder_input.update(dt)

        if self._real_uwb_input is not None:
            self._real_uwb_input.update(dt)

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

        # botão de toggle do analyzer (painel lateral)
        sidebar_x = self.host.cam.viewport[0] + 16
        screen_h = self.host.screen.get_height()
        legend_y = screen_h - 44
        toggle_y = legend_y - 40
        self.btn_toggle_analyzer.rect.topleft = (sidebar_x, toggle_y)
        self.btn_toggle_analyzer.rect.size = (190, 32)
        self.btn_toggle_analyzer.text = "Ocultar Analyzer" if self.show_analyzer else "Mostrar Analyzer"
        self.btn_toggle_analyzer.draw(self.host.screen)

        self.btn_toggle_legend.rect.topleft = (sidebar_x, legend_y)
        self.btn_toggle_legend.rect.size = (190, 32)
        self.btn_toggle_legend.text = "Ocultar Legenda" if self.show_legend_overlay else "Mostrar Legenda"
        self.btn_toggle_legend.draw(self.host.screen)

        if self._batch_results is not None and self.show_analyzer:
            self._draw_analyzer()

        if self.dataset_modal_open:
            self._draw_dataset_modal()

        if self.show_legend_overlay:
            self._legend_close_rect = draw_legend_overlay(
                self.host.screen,
                self.host.font,
                self.host.bigfont,
                selected=self.selected,
            )
        else:
            self._legend_close_rect = None

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
                    self.host._set_msg(
                        f"BC-EKF ignorado: tipo '{self.simulated_dataset_kind}' não fornece front+rear"
                    )
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
                    algos_to_run = [a for a in algos_to_run if a != "bc_ekf"]
                    self.host._set_msg("BC-EKF ignorado: dados BC simulados não foram preparados")

            if self.dataset_source_type == "real_encoder_uwb":
                if "bc_ekf" in algos_to_run and self._bc_ekf_data is None:
                    algos_to_run = [a for a in algos_to_run if a != "bc_ekf"]
                    self.host._set_msg("BC-EKF real ignorado: dados BC não foram preparados")

            print(
                "[RUN_BATCH]",
                "dataset_source_type=", self.dataset_source_type,
                "simulated_dataset_kind=", self.simulated_dataset_kind,
                "bc_ekf_data_is_none=", self._bc_ekf_data is None,
                "batch_shape=", None if self._batch_dists is None else self._batch_dists.shape,
            )

            p_true = self._get_batch_ground_truth_xy()

            self._batch_results = run_batch(
                anchors_Nx3=anchors,
                distances=self._batch_dists,
                deviations=devs,
                algoritmos=algos_to_run,
                p_true=p_true,
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

    def _get_batch_ground_truth_xy(self):
        """
        Resolve a trajetória de referência para métricas do batch.
        Em datasets simulados, prioriza sidecar *_traj.csv; fallback para rota carregada.
        Retorna sempre (M, 2).
        """
        if self._batch_dists is None:
            return None

        if self.dataset_source_type != "simulated":
            return None

        m = int(self._batch_dists.shape[0])

        def _to_m2(arr_like):
            arr = np.asarray(arr_like, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != m or arr.shape[1] < 2:
                return None
            return arr[:, :2]

        if self._dataset_path:
            traj_sidecar = self._guess_sampled_traj_sidecar(self._dataset_path)
            if traj_sidecar is not None:
                try:
                    sampled_route = np.asarray(self._load_sampled_traj_csv(traj_sidecar), dtype=float)
                    p_true = _to_m2(sampled_route)
                    if p_true is not None:
                        return p_true
                except Exception as e:
                    print(f"[DATASET] falha ao carregar sidecar para p_true: {e}")

        if self._dataset_route is not None:
            p_true = _to_m2(self._dataset_route)
            if p_true is not None:
                return p_true

        return None

    def _moving_average(self, x: np.ndarray, window: int) -> np.ndarray:
        """Suavização simples para estimar tendência local da série."""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x.copy()

        window = max(3, int(window))
        if window % 2 == 0:
            window += 1

        pad = window // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(xp, kernel, mode="valid")


    def _estimate_local_sigma_series(
        self,
        values,
        smooth_window: int = 9,
        sigma_window: int = 15,
        min_sigma: float = 0.01,
    ) -> np.ndarray:
        """
        Estima sigma local a partir do resíduo em relação a uma série suavizada.
        Útil quando o log real não traz desvio padrão medido pelo hardware.
        """
        x = np.asarray(values, dtype=float)
        if x.size == 0:
            return np.asarray([], dtype=float)

        trend = self._moving_average(x, smooth_window)
        resid = x - trend

        sigma_window = max(5, int(sigma_window))
        if sigma_window % 2 == 0:
            sigma_window += 1

        pad = sigma_window // 2
        sigmas = np.zeros_like(x, dtype=float)

        for i in range(x.size):
            a = max(0, i - pad)
            b = min(x.size, i + pad + 1)
            chunk = resid[a:b]

            if chunk.size <= 1:
                s = min_sigma
            else:
                s = float(np.std(chunk, ddof=1))
                if not np.isfinite(s):
                    s = min_sigma

            sigmas[i] = max(min_sigma, s)

        return sigmas


    def _get_real_anchor_id_map(self, detected_ids):
        """
        Retorna o mapa dos IDs reais do UWB para os índices internos 0..N-1
        na mesma ordem do arquivo de âncoras.

        Exemplo:
        anchor_ids_uwb = [6, 3, 9, 7, 8]
        -> {6: 0, 3: 1, 9: 2, 7: 3, 8: 4}
        """
        detected_ids = [int(x) for x in detected_ids]

        # Caso ideal: veio explícito no JSON de âncoras
        if self._anchors_uwb_ids is not None:
            if len(self._anchors_uwb_ids) != len(self._dataset_anchors):
                raise ValueError(
                    f"anchor_ids_uwb possui {len(self._anchors_uwb_ids)} IDs, "
                    f"mas o arquivo de âncoras possui {len(self._dataset_anchors)} posições"
                )

            mapping = {int(real_id): idx for idx, real_id in enumerate(self._anchors_uwb_ids)}

            missing = [aid for aid in detected_ids if aid not in mapping]
            if missing:
                raise ValueError(
                    f"IDs do UWB não encontrados em anchor_ids_uwb: {missing}. "
                    f"Esperado algo compatível com {self._anchors_uwb_ids}"
                )

            return mapping

        # Fallback automático: ordena IDs detectados e associa na ordem do layout
        # útil para teste, mas o ideal é usar anchor_ids_uwb no JSON.
        if self._dataset_anchors is not None and len(detected_ids) == len(self._dataset_anchors):
            detected_sorted = sorted(detected_ids)
            mapping = {aid: idx for idx, aid in enumerate(detected_sorted)}
            print("[REAL_UWB_AUTO_ID_MAP]", mapping)
            return mapping

        raise ValueError(
            "Não foi possível montar o mapeamento das âncoras reais. "
            "Adicione 'anchor_ids_uwb' no JSON de âncoras."
        )

    def _normalize_real_uwb_rows(self, rows):
        """
        Converte logs reais brutos para o formato interno esperado pelo pipeline.

        Casos suportados:
        1) Já padronizado:
        - timestamp, anchor_id, range, sigma
        - timestamp, anchor_id, range_front, sigma_front, range_rear, sigma_rear
        => devolve como está

        2) Formato bruto por coluna, exemplo:
        timestamp,Da6_t1,Da6_t2
        timestamp,Da6_t1,Da6_t2,Da7_t1,Da7_t2,...
        => estima sigma e devolve em formato BC:
            timestamp, anchor_id, range_front, sigma_front, range_rear, sigma_rear

        IMPORTANTE:
        o anchor_id devolvido já é remapeado para o índice interno 0..N-1
        conforme a ordem do arquivo de âncoras e do campo anchor_ids_uwb.
        """
        if not rows:
            return rows

        sample = rows[0]
        keys = [str(k).strip() for k in sample.keys()]
        keys_lower = {k.lower() for k in keys}

        # Caso já padronizado
        if (
            {"timestamp", "anchor_id", "range", "sigma"} <= keys_lower
            or {"timestamp", "anchor_id", "range_front", "sigma_front", "range_rear", "sigma_rear"} <= keys_lower
            or {"timestamp_s", "anchor_id", "range", "sigma"} <= keys_lower
            or {"timestamp_s", "anchor_id", "range_front", "sigma_front", "range_rear", "sigma_rear"} <= keys_lower
        ):
            return rows

        # Detecta colunas do tipo Da6_t1 / Da6_t2 / Da10_t1 ...
        pair_map = {}
        for col in keys:
            m = re.match(r"^[Dd][Aa](\d+)_t([12])$", col.strip())
            if m:
                real_anchor_id = int(m.group(1))
                tag_idx = int(m.group(2))  # 1=front, 2=rear
                pair_map.setdefault(real_anchor_id, {})[tag_idx] = col

        if not pair_map:
            return rows

        ts_key = "timestamp" if "timestamp" in keys_lower else "timestamp_s"

        detected_ids = sorted(pair_map.keys())
        id_map = self._get_real_anchor_id_map(detected_ids)

        print("[REAL_UWB_ID_MAP]", id_map)

        # Estima sigma por série/coluna
        sigma_by_col = {}
        for real_anchor_id, cols in pair_map.items():
            for _, col in cols.items():
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[col]))
                    except Exception:
                        vals.append(np.nan)

                arr = np.asarray(vals, dtype=float)
                valid = np.isfinite(arr)

                sig = np.full(arr.shape, 0.01, dtype=float)
                if np.any(valid):
                    sig_valid = self._estimate_local_sigma_series(arr[valid])
                    sig[valid] = sig_valid

                sigma_by_col[col] = sig

        # ---------------------------------------------------------
        # Normaliza timestamps do UWB para segundos, começando em 0.
        # Logs STM normalmente vêm em ms: 89, 350, 611, ...
        # Encoder normalizado já está em segundos: 0.0, 0.1, ...
        # ---------------------------------------------------------
        raw_ts = []
        for row in rows:
            try:
                raw_ts.append(float(row[ts_key]))
            except Exception:
                raw_ts.append(np.nan)

        raw_ts = np.asarray(raw_ts, dtype=float)
        valid_ts = raw_ts[np.isfinite(raw_ts)]

        if valid_ts.size == 0:
            raise ValueError("UWB sem timestamps válidos")

        t0_raw = float(valid_ts[0])
        span_raw = float(np.nanmax(valid_ts) - np.nanmin(valid_ts))

        # Se o span é grande, assume ms. Ex.: 0..65000 ms.
        # Se já estiver em segundos, mantém em segundos.
        ts_scale = 0.001 if span_raw > 1000.0 else 1.0

        ts_norm = (raw_ts - t0_raw) * ts_scale

        print(
            "[REAL_UWB_TIME]",
            "t0_raw=", t0_raw,
            "span_raw=", span_raw,
            "scale=", ts_scale,
            "first_s=", float(ts_norm[np.isfinite(ts_norm)][0]),
            "last_s=", float(ts_norm[np.isfinite(ts_norm)][-1]),
        )

        # Constrói formato BC interno já remapeado
        normalized = []
        for i, row in enumerate(rows):
            try:
                ts = float(ts_norm[i])
            except Exception:
                continue

            # percorre na ordem física definida pelo layout
            for real_anchor_id in detected_ids:
                cols = pair_map[real_anchor_id]
                col_t1 = cols.get(1)
                col_t2 = cols.get(2)

                if col_t1 is None or col_t2 is None:
                    continue

                try:
                    r1 = float(row[col_t1])
                    r2 = float(row[col_t2])
                except Exception:
                    continue

                if not (np.isfinite(r1) and np.isfinite(r2)):
                    continue

                normalized.append({
                    "timestamp": ts,
                    "anchor_id": int(id_map[real_anchor_id]),   # remapeado para 0..N-1
                    "range_front": float(r1),
                    "sigma_front": float(sigma_by_col[col_t1][i]),
                    "range_rear": float(r2),
                    "sigma_rear": float(sigma_by_col[col_t2][i]),
                })

        if not normalized:
            raise ValueError("Não foi possível normalizar o log UWB real para o formato interno")

        print(
            "[REAL_UWB_NORMALIZE]",
            "rows_in=", len(rows),
            "rows_out=", len(normalized),
            "detected_ids=", detected_ids,
            "mapped_ids=", [id_map[k] for k in detected_ids],
        )

        return normalized
    
    def _resample_pose_path_to_length(self, poses, M: int):
        """
        Reamostra uma trajetória (N,3) para M amostras.
        Usado para casar odometria com o número de amostras UWB.
        """
        poses = np.asarray(poses, dtype=float)

        if poses.ndim != 2 or poses.shape[0] == 0:
            return poses

        if poses.shape[0] == M:
            return poses.copy()

        idx = np.linspace(0, poses.shape[0] - 1, M)
        out = np.zeros((M, poses.shape[1]), dtype=float)

        base_idx = np.arange(poses.shape[0])
        for d in range(poses.shape[1]):
            out[:, d] = np.interp(idx, base_idx, poses[:, d])

        if out.shape[1] >= 3:
            out[:, 2] = np.unwrap(out[:, 2])
            out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))

        return out

    def _load_simple_table_file(self, path: str):
        """
        Lê CSV/TXT simples e devolve lista de dicts.
        Aceita delimitador por vírgula, ponto e vírgula ou whitespace.

        Também suporta logs SEM cabeçalho no formato bruto do ESP32:
        contador_direita;contador_esquerda;velocidade_direita;velocidade_esquerda;
        distancia_direita;distancia_esquerda;millis
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        suffix = path.suffix.lower()

        if suffix in (".csv", ".txt"):
            text = path.read_text(encoding="utf-8-sig")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                raise ValueError(f"Arquivo vazio: {path}")

            header = lines[0]

            # -------------------------------------------------
            # 1) Caso com cabeçalho explícito
            # -------------------------------------------------
            header_lower = header.lower()
            if any(k in header_lower for k in ["contador", "millis", "timestamp", "tempo"]):
                if ";" in header:
                    reader = csv.DictReader(lines, delimiter=";")
                    return list(reader)
                if "," in header:
                    reader = csv.DictReader(lines, delimiter=",")
                    return list(reader)

                cols = header.split()
                rows = []
                for line in lines[1:]:
                    vals = line.split()
                    if len(vals) != len(cols):
                        continue
                    rows.append(dict(zip(cols, vals)))
                return rows

            # -------------------------------------------------
            # 2) Caso sem cabeçalho: log bruto do ESP32
            # ordem do main.cpp:
            # contador_direita;contador_esquerda;velocidade_direita;velocidade_esquerda;
            # distancia_direita;distancia_esquerda;millis
            # -------------------------------------------------
            sample_delim = ";" if ";" in header else ("," if "," in header else None)

            if sample_delim is not None:
                first_parts = [p.strip() for p in header.split(sample_delim)]
                if len(first_parts) == 7:
                    cols = [
                        "contador_direita",
                        "contador_esquerda",
                        "velocidade_direita",
                        "velocidade_esquerda",
                        "distancia_direita",
                        "distancia_esquerda",
                        "millis",
                    ]
                    rows = []
                    for line in lines:
                        vals = [p.strip() for p in line.split(sample_delim)]
                        if len(vals) != 7:
                            continue
                        rows.append(dict(zip(cols, vals)))
                    return rows

            # -------------------------------------------------
            # 3) Fallback whitespace
            # -------------------------------------------------
            cols = header.split()
            rows = []
            for line in lines[1:]:
                vals = line.split()
                if len(vals) != len(cols):
                    continue
                rows.append(dict(zip(cols, vals)))
            return rows

        raise ValueError(f"Formato não suportado: {suffix}")


    def _find_encoder_columns(self, rows):
        """
        Detecta automaticamente as colunas do log do ESP32.

        Formato do main.cpp:
        contador_direita ; contador_esquerda ; velocidade_direita ; velocidade_esquerda ;
        distancia_direita ; distancia_esquerda ; millis
        """
        if not rows:
            raise ValueError("Arquivo de encoder vazio")

        keys = list(rows[0].keys())
        keys_norm = {k: k.strip().lower() for k in keys}

        def pick(*patterns):
            for raw, norm in keys_norm.items():
                for p in patterns:
                    if p in norm:
                        return raw
            return None

        col_right_count = pick(
            "contador_direita", "count_right", "right_count", "pulsos_direita"
        )
        col_left_count = pick(
            "contador_esquerda", "count_left", "left_count", "pulsos_esquerda"
        )
        col_right_dist = pick(
            "distancia_direita", "dist_right", "right_dist", "distance_right"
        )
        col_left_dist = pick(
            "distancia_esquerda", "dist_left", "left_dist", "distance_left"
        )
        col_time_ms = pick(
            "millis", "timestamp", "tempo", "time_ms"
        )

        if col_time_ms and col_right_count and col_left_count:
            return {
                "right_count": col_right_count,
                "left_count": col_left_count,
                "right_dist": col_right_dist,
                "left_dist": col_left_dist,
                "time_ms": col_time_ms,
            }

        raise ValueError(
            "Não foi possível detectar colunas de encoder. "
            "Esperado contador_direita, contador_esquerda e millis."
        )

    def _apply_real_odom_initial_pose(self, poses):
        """
        Aplica pose inicial à odometria real.

        A odometria reconstruída pelo encoder normalmente nasce em um referencial local.
        Esta função:
        1) desloca a rota para começar em (0,0);
        2) rotaciona pela orientação inicial configurada;
        3) translada para a posição inicial real do robô.
        """
        arr = np.asarray(poses, dtype=float)

        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            return arr

        out = arr.copy()

        x0 = float(getattr(config, "REAL_ODOM_INITIAL_X", 0.0))
        y0 = float(getattr(config, "REAL_ODOM_INITIAL_Y", 0.0))
        th0 = np.deg2rad(float(getattr(config, "REAL_ODOM_INITIAL_THETA_DEG", 0.0)))

        # origem local da odometria
        p0 = out[0, :2].copy()
        xy_local = out[:, :2] - p0

        c = np.cos(th0)
        s = np.sin(th0)

        R = np.array([
            [c, -s],
            [s,  c],
        ], dtype=float)

        xy_global = xy_local @ R.T
        out[:, 0] = xy_global[:, 0] + x0
        out[:, 1] = xy_global[:, 1] + y0

        # se tiver theta, soma orientação inicial
        if out.shape[1] >= 3:
            out[:, 2] = out[:, 2] + th0
            out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))

        print(
            "[REAL_ODOM_INITIAL_POSE]",
            "x0=", x0,
            "y0=", y0,
            "theta_deg=", float(getattr(config, "REAL_ODOM_INITIAL_THETA_DEG", 0.0)),
            "first_xy=", out[0, :2],
        )

        return out

    def _refresh_real_config_from_config(self):
        """
        Recarrega parâmetros reais do config.py sempre que um dataset real é carregado.
        Evita precisar recriar a tela/classe para aplicar mudanças no config.
        """
        self._real_encoder_use_distance_columns = bool(
            getattr(config, "REAL_ENCODER_USE_DISTANCE_COLUMNS", True)
        )
        self._real_encoder_distance_unit_scale = float(
            getattr(config, "REAL_ENCODER_DISTANCE_UNIT_SCALE", 0.01)
        )
        self._real_encoder_swap_lr = bool(
            getattr(config, "REAL_ENCODER_SWAP_LR", False)
        )
        self._real_encoder_invert_left = bool(
            getattr(config, "REAL_ENCODER_INVERT_LEFT", False)
        )
        self._real_encoder_invert_right = bool(
            getattr(config, "REAL_ENCODER_INVERT_RIGHT", False)
        )

        self._real_drive_cfg = DifferentialDriveConfig(
            wheel_radius_m=float(getattr(config, "WHEEL_RADIUS", 0.0325)),
            wheel_base_m=float(getattr(config, "WHEEL_BASE", 0.16)),
            encoder=EncoderConfig(
                ticks_per_wheel_rev=float(getattr(config, "ENCODER_TICKS_PER_REV", 1320.0))
            ),
        )

        print(
            "[REAL_CONFIG]",
            "wheel_radius=", self._real_drive_cfg.wheel_radius_m,
            "wheel_base=", self._real_drive_cfg.wheel_base_m,
            "ticks_per_rev=", self._real_drive_cfg.encoder.ticks_per_wheel_rev,
            "use_dist=", self._real_encoder_use_distance_columns,
            "dist_scale=", self._real_encoder_distance_unit_scale,
            "swap_lr=", self._real_encoder_swap_lr,
            "inv_l=", self._real_encoder_invert_left,
            "inv_r=", self._real_encoder_invert_right,
        )

    def _reset_real_dataset_state(self):
        """
        Limpa estados derivados de um carregamento real anterior.
        Evita reaproveitar rota, resultados ou BC-EKF de outro ensaio.
        """
        self._real_dataset = None
        self._real_aligned_rows = None
        self._real_odom_path = None
        self._real_range_matrix = None
        self._real_sigma_matrix = None
        self._real_timestamps = []
        self._real_anchor_ids = []

        self._batch_dists = None
        self._batch_devs = None
        self._batch_results = None
        self._dataset_stats = None
        self._bc_ekf_data = None

        # rota real deve ser reconstruída do zero a cada carregamento
        self._dataset_route = None

    def _normalize_real_encoder_rows(self, rows):
        """
        Converte log incremental do ESP32 em ticks acumulados no formato:
        timestamp,left_ticks,right_ticks

        Pode usar:
        1) contadores incrementais; ou
        2) distancia_direita/distancia_esquerda calculadas pelo ESP32.

        O modo por distância é útil porque o ESP32 já calcula o deslocamento
        incremental de cada roda. Nesse caso, convertemos a distância acumulada
        para 'ticks equivalentes', de modo que o loader existente continue
        funcionando sem alterações.
        """
        cols = self._find_encoder_columns(rows)

        use_dist = (
            self._real_encoder_use_distance_columns
            and cols.get("right_dist") is not None
            and cols.get("left_dist") is not None
        )

        left_acc_ticks = 0.0
        right_acc_ticks = 0.0
        left_acc_m = 0.0
        right_acc_m = 0.0

        out = []
        t0 = None
        prev_t = None

        wheel_radius = float(self._real_drive_cfg.wheel_radius_m)
        ticks_per_rev = float(self._real_drive_cfg.encoder.ticks_per_wheel_rev)

        meters_per_tick = (2.0 * np.pi * wheel_radius) / ticks_per_rev
        if meters_per_tick <= 0:
            raise ValueError("meters_per_tick inválido")

        for row in rows:
            try:
                t_ms = float(row[cols["time_ms"]])
            except Exception:
                continue

            if not np.isfinite(t_ms):
                continue

            if prev_t is not None and t_ms < prev_t:
                continue
            prev_t = t_ms

            if t0 is None:
                t0 = t_ms

            try:
                dr_count = float(row[cols["right_count"]])
                dl_count = float(row[cols["left_count"]])
            except Exception:
                dr_count = 0.0
                dl_count = 0.0

            if use_dist:
                try:
                    dr_m = float(row[cols["right_dist"]]) * self._real_encoder_distance_unit_scale
                    dl_m = float(row[cols["left_dist"]]) * self._real_encoder_distance_unit_scale
                except Exception:
                    continue
            else:
                dr_m = None
                dl_m = None

            # Troca esquerda/direita, se necessário
            if self._real_encoder_swap_lr:
                dr_count, dl_count = dl_count, dr_count
                if use_dist:
                    dr_m, dl_m = dl_m, dr_m

            # Inversão de sinal, se necessário
            if self._real_encoder_invert_right:
                dr_count = -dr_count
                if use_dist:
                    dr_m = -dr_m

            if self._real_encoder_invert_left:
                dl_count = -dl_count
                if use_dist:
                    dl_m = -dl_m

            if use_dist:
                right_acc_m += dr_m
                left_acc_m += dl_m

                right_acc_ticks = right_acc_m / meters_per_tick
                left_acc_ticks = left_acc_m / meters_per_tick
            else:
                right_acc_ticks += dr_count
                left_acc_ticks += dl_count

            out.append({
                "timestamp": (t_ms - t0) / 1000.0,
                "left_ticks": left_acc_ticks,
                "right_ticks": right_acc_ticks,
            })

        if not out:
            raise ValueError("Não foi possível normalizar o log do encoder real")

        print(
            "[REAL_ENCODER_NORMALIZE]",
            "mode=", "distance_columns" if use_dist else "count_columns",
            "rows_in=", len(rows),
            "rows_out=", len(out),
            "first_ts_s=", out[0]["timestamp"],
            "last_ts_s=", out[-1]["timestamp"],
            "wheel_radius=", wheel_radius,
            "wheel_base=", float(self._real_drive_cfg.wheel_base_m),
            "ticks_per_rev=", ticks_per_rev,
            "meters_per_tick=", meters_per_tick,
            "swap_lr=", self._real_encoder_swap_lr,
            "invert_left=", self._real_encoder_invert_left,
            "invert_right=", self._real_encoder_invert_right,
        )

        return out


    def _write_normalized_encoder_temp_csv(self, normalized_rows):
        """
        Salva o encoder normalizado em arquivo temporário dentro de resultados/datasets,
        para reutilizar o loader existente.
        """
        temp_path = Path(self.real_data_dir) / "_encoder_real_normalized_tmp.csv"
        with temp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "left_ticks", "right_ticks"])
            writer.writeheader()
            writer.writerows(normalized_rows)
        return temp_path


    def _load_and_normalize_real_encoder_file(self, encoder_file: str):
        """
        Detecta se o encoder já está no formato aceito.
        Se não estiver, tenta adaptar o log bruto do ESP32.
        """
        # tenta o fluxo atual primeiro
        try:
            return load_and_validate_encoder_file(encoder_file)
        except Exception as first_error:
            print(f"[REAL_ENCODER] formato padrão falhou: {first_error}")

        raw_rows = self._load_simple_table_file(encoder_file)
        normalized_rows = self._normalize_real_encoder_rows(raw_rows)
        temp_csv = self._write_normalized_encoder_temp_csv(normalized_rows)

        # agora reaproveita o loader já existente do projeto
        return load_and_validate_encoder_file(temp_csv)

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
        self._reset_real_dataset_state()
        self._refresh_real_config_from_config()

        encoder_samples = self._load_and_normalize_real_encoder_file(encoder_file)

        uwb_rows_raw = self._load_simple_uwb_file(uwb_file)
        uwb_rows = self._normalize_real_uwb_rows(uwb_rows_raw)

        is_bc = self._is_bc_uwb_rows(uwb_rows)

        if is_bc:
            expanded_rows = self._expand_bc_uwb_rows_if_needed(uwb_rows)

            dataset = build_dataset_from_encoder_and_uwb(
                encoder_samples,
                expanded_rows,
                self._real_drive_cfg,
                clamp=True,
            )

            matrices = build_range_sigma_matrices(dataset["aligned_rows"])

            self._real_encoder_file = str(encoder_file)
            self._real_uwb_file = str(uwb_file)
            self._real_dataset = dataset
            self._real_aligned_rows = dataset["aligned_rows"]
            self._real_range_matrix = matrices["ranges"]
            self._real_sigma_matrix = matrices["sigmas"]
            self._real_timestamps = matrices["timestamps_s"]
            self._real_anchor_ids = matrices["anchor_ids"]

            # Prepara BC e define a rota visual a partir da odometria processada
            self._prepare_bc_ekf_data_for_real_bc(encoder_samples, uwb_rows)

            payload = self._real_dataset_as_numpy()
            if payload is None:
                raise ValueError("Falha ao converter dataset real para batch")

            self._batch_dists = np.asarray(payload["dists"], dtype=float)
            self._batch_devs = np.asarray(payload["devs"], dtype=float)

            self._dataset_label = (
                f"REAL | enc={Path(self._real_encoder_file).name} | "
                f"uwb={Path(self._real_uwb_file).name}"
            )
            self._dataset_source = "real_encoder_uwb"

            if not self._validate_real_dataset_against_loaded_anchors():
                return

            self.host._set_msg(f"Dataset real BC carregado: {len(uwb_rows)} linhas UWB")
            return

        # Caso não BC
        dataset = build_dataset_from_encoder_and_uwb(
            encoder_samples,
            uwb_rows,
            self._real_drive_cfg,
            clamp=True,
        )

        matrices = build_range_sigma_matrices(dataset["aligned_rows"])
        odom_path = extract_odometry_path(dataset["aligned_rows"])
        odom_path = self._apply_real_odom_initial_pose(odom_path)

        self._real_encoder_file = str(encoder_file)
        self._real_uwb_file = str(uwb_file)
        self._real_dataset = dataset
        self._real_aligned_rows = dataset["aligned_rows"]
        self._real_odom_path = odom_path
        self._real_range_matrix = matrices["ranges"]
        self._real_sigma_matrix = matrices["sigmas"]
        self._real_timestamps = matrices["timestamps_s"]
        self._real_anchor_ids = matrices["anchor_ids"]
        self._dataset_route = odom_path

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
        if self._real_range_matrix is None:
            return None

        dists = np.asarray(self._real_range_matrix, dtype=float)
        devs = np.asarray(self._real_sigma_matrix, dtype=float)

        route = None
        if self._real_odom_path is not None:
            route_arr = np.asarray(self._real_odom_path, dtype=float)
            if route_arr.ndim == 2 and route_arr.shape[0] > 0 and route_arr.shape[1] >= 2:
                route = route_arr

        return {
            "dists": dists,
            "devs": devs,
            "route": route,
            "timestamps_s": np.asarray(self._real_timestamps, dtype=float),
            "anchor_ids": np.asarray(self._real_anchor_ids, dtype=int),
        }
    
    def _load_demo_real_dataset(self):
        base = Path(__file__).resolve().parents[3]

        self._load_real_encoder_uwb_dataset(
            base / "resultados" / "datasets" / "encoder_square.csv",
            base / "resultados" / "datasets" / "uwb_square.csv",
        )

    # =========================================================
    # ANALYZER
    # =========================================================

    def _compute_dataset_stats(self, results: dict):
        """
        Calcula métricas reais contra a trajetória de referência (ground truth)
        e devolve no formato esperado por shared.draw_analyzer_panel /
        shared.draw_boxplot_panel.
        """
        p_true = self._get_batch_ground_truth_xy()
        if p_true is None:
            return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)

        stats = {}

        for algo in ALGO_ORDER:
            if algo not in results:
                continue

            pos = results[algo].get("posicoes", None)
            if pos is None:
                continue

            pos = np.asarray(pos, dtype=float)
            if pos.ndim != 2 or pos.shape[1] < 2:
                continue

            n = min(len(pos), len(p_true))
            pos_xy = pos[:n, :2]
            truth_xy = p_true[:n, :2]

            valid = (
                np.isfinite(pos_xy[:, 0]) & np.isfinite(pos_xy[:, 1]) &
                np.isfinite(truth_xy[:, 0]) & np.isfinite(truth_xy[:, 1])
            )
            pos_xy = pos_xy[valid]
            truth_xy = truth_xy[valid]

            if len(pos_xy) == 0:
                continue

            err_xy = pos_xy - truth_xy
            err_pos = np.linalg.norm(err_xy, axis=1)

            # mesmo RMSE principal usado no batch/hud
            rmse_val = results[algo].get("rmse_xy", None)
            if rmse_val is None:
                rmse_val = float(np.sqrt(np.mean(err_pos ** 2)))

            q1, median, q3 = np.percentile(err_pos, [25, 50, 75])

            stats[algo] = {
                "rmse": float(rmse_val),
                "mae": float(np.mean(err_pos)),
                "max": float(np.max(err_pos)),
                "max_err": float(np.max(err_pos)),   # compatibilidade extra
                "min": float(np.min(err_pos)),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "errors": err_pos,
            }

        return stats
    
    def _dataset_ranking(self):
        return build_ranking_summary(self._dataset_stats, top_k=5)

    def _draw_analyzer(self):
        draw_analyzer_panel(
            screen=self.host.screen,
            font=self.host.font,
            bigfont=self.host.bigfont,
            title="Dataset Analyzer",
            stats=self._dataset_stats,
            selected=self.selected,
            x=10,
            y=40,
            w=500,
            h=380,
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

        # só usa a rota do payload se ainda não houver rota externa carregada
        if self._dataset_route is None:
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
                f"UWB/layout incompatíveis: {n_cols} vs {n_anchors} âncoras"
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
    

    def _prepare_bc_ekf_data_for_real_bc(self, encoder_samples, uwb_rows):
        """
        Prepara o BC-EKF para dataset real BC.

        Esta versão:
        - usa os anchor_id já remapeados para 0..N-1;
        - remove alinhamento automático com _dataset_route;
        - aplica pose inicial apenas uma vez;
        - usa T real a partir dos timestamps UWB;
        - descarta timestamps incompletos de forma explícita.
        """
        if self._dataset_anchors is None:
            self.host._set_msg("BC-EKF real requer âncoras carregadas")
            self._bc_ekf_data = None
            return

        if not uwb_rows:
            self.host._set_msg("BC-EKF real: UWB vazio")
            self._bc_ekf_data = None
            return

        n_anchors = int(self._dataset_anchors.shape[0])

        # timestamps únicos do UWB já devem estar em segundos
        ts_sorted_all = sorted({float(r["timestamp"]) for r in uwb_rows})
        M_all = len(ts_sorted_all)

        z_hist = np.full((2 * n_anchors, M_all), np.nan, dtype=float)
        sigma_vals = []

        ts_to_idx = {t: k for k, t in enumerate(ts_sorted_all)}

        for row in uwb_rows:
            try:
                t = float(row["timestamp"])
                aid = int(row["anchor_id"])
            except Exception:
                continue

            if aid < 0 or aid >= n_anchors:
                continue

            k = ts_to_idx[t]
            j = aid

            try:
                z_hist[2 * j, k] = float(row["range_front"])
                z_hist[2 * j + 1, k] = float(row["range_rear"])

                sf = float(row["sigma_front"])
                sr = float(row["sigma_rear"])

                if np.isfinite(sf):
                    sigma_vals.append(sf)
                if np.isfinite(sr):
                    sigma_vals.append(sr)

            except Exception:
                continue

        # Mantém somente instantes com todas as medições front/rear de todas as âncoras
        complete_mask = ~np.isnan(z_hist).any(axis=0)
        dropped = int(np.count_nonzero(~complete_mask))

        if not np.any(complete_mask):
            self.host._set_msg("BC real inválido: nenhum timestamp UWB completo")
            self._bc_ekf_data = None
            return

        z_hist = z_hist[:, complete_mask]
        ts_sorted = np.asarray(ts_sorted_all, dtype=float)[complete_mask]
        M = int(z_hist.shape[1])

        if dropped > 0:
            print(
                "[REAL_BC_UWB_FILTER]",
                "timestamps_in=", M_all,
                "timestamps_valid=", M,
                "timestamps_dropped=", dropped,
            )

        # ---------------------------------------------------------
        # Trajetória do encoder:
        # 1) reconstrói odometria local;
        # 2) reamostra para o número de instantes UWB válidos;
        # 3) aplica pose inicial uma única vez.
        # ---------------------------------------------------------
        poses = self._build_pose_path_from_encoder_samples(encoder_samples)

        if poses is None or len(poses) == 0:
            self.host._set_msg("BC-EKF real: odometria vazia")
            self._bc_ekf_data = None
            return

        poses = self._resample_pose_path_to_length(poses, M)
        poses = self._apply_real_odom_initial_pose(poses)

        self._dataset_route = poses.copy()
        self._route_waypoints = poses[:, :2].copy()
        self._real_odom_path = poses.copy()

        # T real do UWB
        if M > 1:
            dt = np.diff(ts_sorted)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            T = float(np.median(dt)) if dt.size > 0 else float(getattr(config, "TIME_STEP", 0.05))
        else:
            T = float(getattr(config, "TIME_STEP", 0.05))

        if not np.isfinite(T) or T <= 0:
            T = float(getattr(config, "TIME_STEP", 0.05))

        odometry_noisy = self._pose_xytheta_to_vw(poses, T)

        sigma_uwb = float(np.nanmedian(np.asarray(sigma_vals, dtype=float))) if sigma_vals else float(getattr(config, "UWB_NOISE_STD", 0.05))
        if not np.isfinite(sigma_uwb) or sigma_uwb <= 0:
            sigma_uwb = float(getattr(config, "UWB_NOISE_STD", 0.05))

        self._bc_ekf_data = {
            "T": T,
            "odometry_noisy": odometry_noisy,
            "z_hist": z_hist,
            "l": float(getattr(config, "TAG_BASELINE", 0.25)) / 2.0,
            "z_c": float(getattr(config, "TAG_HEIGHT", 0.20)),
            "sigma_uwb": sigma_uwb,
            "x0": np.asarray(poses[0], dtype=float).reshape(3,),
        }

        # Para os algoritmos clássicos, usa apenas a tag frontal
        self._batch_dists = z_hist.T[:, 0::2]
        self._batch_devs = np.full_like(self._batch_dists, sigma_uwb, dtype=float)

        # Mantém matrizes reais coerentes com o batch atual
        self._real_range_matrix = self._batch_dists.copy()
        self._real_sigma_matrix = self._batch_devs.copy()
        self._real_timestamps = ts_sorted.tolist()
        self._real_anchor_ids = list(range(n_anchors))

        print(
            "[REAL_BC_EKF_PREP_OK]",
            "z_hist=", z_hist.shape,
            "odom=", odometry_noisy.shape,
            "poses=", poses.shape,
            "batch_dists=", self._batch_dists.shape,
            "T=", T,
            "sigma_uwb=", sigma_uwb,
            "x0=", poses[0],
            "dropped=", dropped,
        )
    
    def _guess_sampled_traj_sidecar(self, dataset_path: str) -> str | None:
        '''Dado o caminho de um dataset, tenta adivinhar se existe um arquivo CSV de trajetória amostrada associado,
          seguindo a convenção de nomeação: base do dataset + "_traj.csv".'''
        base, _ = os.path.splitext(dataset_path)
        candidate = base + "_traj.csv"
        return candidate if os.path.exists(candidate) else None


    def _load_sampled_traj_csv(self, path: str):
        '''Carrega um CSV simples de trajetória amostrada, com colunas x,y,theta (pode ter outras colunas, mas essas são obrigatórias).
        Retorna array (M, 3) com os dados de pose.'''
        rows = []

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(2048)
            f.seek(0)

            # caso CSV com vírgula
            if "," in sample:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        x = float(row["x"])
                        y = float(row["y"])
                        th = float(row["theta"])
                    except Exception:
                        continue
                    rows.append([x, y, th])

            else:
                # fallback: separado por espaço
                header = f.readline().strip().split()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    vals = line.split()
                    if len(vals) < 4:
                        continue
                    try:
                        _, x, y, th = map(float, vals[:4])
                    except Exception:
                        continue
                    rows.append([x, y, th])

        if not rows:
            raise ValueError(f"Trajetória amostrada vazia: {path}")

        return np.asarray(rows, dtype=float)
    
    def _expand_bc_uwb_rows_if_needed(self, rows):
        """
        Se o arquivo UWB estiver em formato BC:
        timestamp, anchor_id, range_front, sigma_front, range_rear, sigma_rear

        expande cada linha em duas linhas simples compatíveis com o builder atual:
        - uma linha FRONT
        - uma linha REAR

        Se já estiver no formato simples (range/sigma), devolve como está.
        """
        if not rows:
            return rows

        sample = rows[0]
        keys = {str(k).strip().lower() for k in sample.keys()}

        is_bc = (
            "range_front" in keys and
            "sigma_front" in keys and
            "range_rear" in keys and
            "sigma_rear" in keys
        )

        if not is_bc:
            return rows

        expanded = []
        for row in rows:
            ts = row.get("timestamp", row.get("timestamp_s", ""))
            aid = row.get("anchor_id", row.get("anchor", row.get("id", "")))

            expanded.append({
                "timestamp": ts,
                "anchor_id": aid,
                "range": row["range_front"],
                "sigma": row["sigma_front"],
                "tag": "front",
            })
            expanded.append({
                "timestamp": ts,
                "anchor_id": aid,
                "range": row["range_rear"],
                "sigma": row["sigma_rear"],
                "tag": "rear",
            })

        return expanded
    
    def _is_bc_uwb_rows(self, rows) -> bool:
        if not rows:
            return False
        sample = rows[0]
        keys = {str(k).strip().lower() for k in sample.keys()}
        return (
            "range_front" in keys and
            "sigma_front" in keys and
            "range_rear" in keys and
            "sigma_rear" in keys
        )
    
    def _build_pose_path_from_encoder_samples(self, encoder_samples):
        """
        Reconstrói uma trajetória XYTheta a partir dos samples do encoder.
        Funciona tanto para EncoderSample (atributos) quanto para dict.
        Retorna array (M, 3).
        """
        import numpy as np

        def _get(obj, key):
            if isinstance(obj, dict):
                return obj[key]
            return getattr(obj, key)

        cfg = self._real_drive_cfg
        poses = []

        x = 0.0
        y = 0.0
        th = 0.0

        poses.append([x, y, th])

        meters_per_tick = (2.0 * np.pi * float(cfg.wheel_radius_m)) / float(cfg.encoder.ticks_per_wheel_rev)

        for i in range(1, len(encoder_samples)):
            prev = encoder_samples[i - 1]
            cur = encoder_samples[i]

            dl_ticks = float(_get(cur, "left_ticks")) - float(_get(prev, "left_ticks"))
            dr_ticks = float(_get(cur, "right_ticks")) - float(_get(prev, "right_ticks"))

            dl = dl_ticks * meters_per_tick
            dr = dr_ticks * meters_per_tick

            ds = 0.5 * (dl + dr)
            dth = (dr - dl) / float(cfg.wheel_base_m)

            th_mid = th + 0.5 * dth
            x += ds * np.cos(th_mid)
            y += ds * np.sin(th_mid)
            th += dth

            poses.append([float(x), float(y), float(th)])

        return np.asarray(poses, dtype=float)
    
    def _align_path_to_reference(self, path_xytheta, ref_xy):
        """
        Alinha rigidamente a trajetória odométrica ao referencial global
        usando ajuste por múltiplos pontos (rotação + translação, sem escala).
        """
        import numpy as np

        path = np.asarray(path_xytheta, dtype=float).copy()
        ref = np.asarray(ref_xy, dtype=float)

        if path.ndim != 2 or path.shape[1] < 2:
            return path
        if ref.ndim != 2 or ref.shape[1] < 2:
            return path
        if len(path) < 2 or len(ref) < 2:
            return path

        # usa a mesma quantidade de amostras em ambos
        M = min(len(path), len(ref))
        P = path[:M, :2]
        Q = ref[:M, :2]

        # centróides
        p_mean = P.mean(axis=0)
        q_mean = Q.mean(axis=0)

        P0 = P - p_mean
        Q0 = Q - q_mean

        # Kabsch 2D (sem escala)
        H = P0.T @ Q0
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # garante rotação própria (sem reflexão)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # aplica alinhamento a toda trajetória
        xy = path[:, :2] - p_mean[None, :]
        xy = (R @ xy.T).T
        xy = xy + q_mean[None, :]

        # ângulo da rotação aplicada
        dth = float(np.arctan2(R[1, 0], R[0, 0]))

        path[:, :2] = xy
        path[:, 2] = path[:, 2] + dth
        return path

    def _guess_real_traj_sidecar(self, uwb_path: str) -> str | None:
        base, _ = os.path.splitext(uwb_path)
        candidate = base + "_traj.csv"
        return candidate if os.path.exists(candidate) else None

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