from __future__ import annotations
from typing import Any
import os
import numpy as np
import pygame as pg
from pathlib import Path
import re
import csv

from src import config
from src.ui.botton import Button
from src.uwb.algoritmos_estaticos import run_batch
from src.ui.algo_modes.shared import (
    ALGO_ORDER,
    MODE_DATASET,
    default_selected,
    default_algorithm_variant_state,
    algorithm_active_key,
    algorithm_result_alias,
    cycle_algorithm_variant,
    algorithm_button_label,
)
from src.analysis.algo_metrics import (
    compute_dataset_cluster_stats,
    build_ranking_summary,
    compute_track_vs_polyline_stats,
    compute_track_vs_synced_reference_stats,
    compute_track_vs_sampled_reference_stats,
    resample_polyline,
)
from src.odometry import (
    EncoderConfig,
    DifferentialDriveConfig,
    load_and_validate_encoder_file,
)
from src.ui.algo_modes.dataset_bc_prep import (
    BcPrepResult,
    guess_sampled_traj_sidecar,
    load_sampled_traj_csv,
    pose_xytheta_to_vw,
    prepare_real_bc_ekf_data,
    prepare_simulated_bc_ekf_data,
    route_xy_to_pose_xytheta,
)
from src.ui.algo_modes.dataset_real_pipeline import (
    load_real_encoder_uwb_dataset,
)
from src.ui.algo_modes.dataset_render import (
    draw_dataset_mode,
    draw_dataset_analyzer,
)
from src.ui.algo_modes.dataset_modal import DatasetConfigModal
from src.ui.algo_modes.dataset_sim_pipeline import (
    SimPipelineResult,
    load_simulated_dataset_file,
    normalize_simulated_dataset_by_kind,
    normalize_simulated_kind,
)
from src.ui.algo_modes.dataset_io import (
    apply_anchors_to_dataset_mode,
    apply_route_to_dataset_mode,
    apply_map_to_dataset_mode,
)


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
        self.dataset_modal = DatasetConfigModal(self)
        self.dataset_modal_open = False

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

        self.dataset_source_type = "simulated"   # "simulated" | "real_encoder_uwb"

        self.simulated_dataset_kind = "Front"   # "Front" | "Rear" | "Mid" | "BC"
        self.available_simulated_dataset_kinds = ["Front", "Rear", "Mid", "BC"]
        self.dataset_dropdown_sim_kind_open = False

        self._bc_ekf_data = None

        self.available_dataset_sources = [
            "Simulado",
            "Real (encoder + UWB)",
        ]


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

        self.metric_mode = "reference_route"

        self.available_metric_modes = [
            "reference_route",      # compara com rota de referência (RMSE)
            "reference_synced",     # compara com rota de referência, mas sincroniza por tempo (RMSE)
            "encoder_route",        # compara com rota extraída do encoder (RMSE)
            "cluster",              # compara com clusters de posição (distância média ao cluster mais próximo)
        ]

        self.btn_metric_mode = Button(
            (0, 0, 190, 32),
            "Métrica: Rota ref.",
            self.host.font if self.host else None,
        )

        self.selected = default_selected()

        self.selected_algo_variants = default_algorithm_variant_state()

        self._reference_route_display = None   # waypoints originais do JSON
        self._reference_route_dense = None     # rota reamostrada para métricas

        self._real_encoder_samples = None
        self._real_uwb_rows = None


    def on_enter(self, host: Any) -> None:

        self.host = host
        self.host.mode = MODE_DATASET

        self.selected = getattr(host, "selected", default_selected())

        self.selected_algo_variants = getattr(
            host,
            "selected_algo_variants",
            default_algorithm_variant_state()
        )

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
    # MODAL
    # =========================================================

    def _apply_dataset_config(self):
        """
        Compatibilidade com o modal antigo.
        Pode ser removida depois que o DatasetConfigModal estiver estável.
        """
        return self._apply_dataset_config_from_modal_values()

    def _apply_dataset_config_from_modal_values(self):
        """
        Ponte entre o novo DatasetConfigModal e o fluxo atual de carregamento.
        """
        dataset_file = getattr(self, "_modal_dataset_file", "")
        real_encoder_file = getattr(self, "_modal_real_encoder_file", "")
        real_uwb_file = getattr(self, "_modal_real_uwb_file", "")
        anchor_file = getattr(self, "_modal_anchor_file", "")
        route_file = getattr(self, "_modal_route_file", "")
        map_file = getattr(self, "_modal_map_file", "")
        simulated_kind = getattr(
            self,
            "_modal_simulated_kind",
            getattr(self, "simulated_dataset_kind", "Front")
        )

        # A partir daqui, reaproveite a lógica que já existe em _apply_dataset_config().
        return self._apply_dataset_config_values(
            dataset_file=dataset_file,
            real_encoder_file=real_encoder_file,
            real_uwb_file=real_uwb_file,
            anchor_file=anchor_file,
            route_file=route_file,
            map_file=map_file,
            simulated_kind=simulated_kind,
        )

    def _apply_dataset_config_values(
            self,
            *,
            dataset_file="",
            real_encoder_file="",
            real_uwb_file="",
            anchor_file="",
            route_file="",
            map_file="",
            simulated_kind="Front",
        ):
        """
        Núcleo do fluxo de carregamento de dataset extraído de _apply_dataset_config().
        Recebe os nomes de arquivo/paths a partir do modal novo e reaplica a lógica
        existente sem depender diretamente dos `dataset_inputs`.
        """
        # decide tipo a partir dos arquivos passados: prioridade para real se houver arquivos reais
        is_real = bool(real_encoder_file or real_uwb_file)
        self._bc_ekf_data = None

        if is_real:
            self.dataset_source_type = "real_encoder_uwb"
        else:
            self.dataset_source_type = "simulated"

        # Simulado
        if self.dataset_source_type == "simulated":
            if not dataset_file:
                self.host._set_msg("Selecione um dataset simulado")
                return

            self.simulated_dataset_kind = normalize_simulated_kind(
                simulated_kind or getattr(self, "simulated_dataset_kind", "Front")
            )

            if self.simulated_dataset_kind not in ("Front", "Rear", "Mid", "BC"):
                self.host._set_msg("Selecione o tipo do dataset simulado")
                return False

            # 1) dataset
            dataset_path = os.path.join(self.dataset_dir, dataset_file)
            if not self._try_load_dataset(dataset_path):
                return False

            # 2) âncoras
            if anchor_file:
                anchors_path = os.path.join(self.anchors_dir, anchor_file)
                if not self._load_anchors(anchors_path):
                    return False

            # 3) rota
            if route_file:
                route_path = os.path.join(self.routes_dir, route_file)
                if not self._try_load_route(route_path):
                    return False

                # No modo simulado, a rota selecionada também pode ser usada
                # como referência/fallback para preparar o BC-EKF quando não
                # existir sidecar *_traj.csv.
                if self._route_waypoints is not None:
                    self._dataset_route = np.asarray(self._route_waypoints, dtype=float).copy()

            # 4) mapa
            if map_file:
                map_path = os.path.join(self.maps_dir, map_file)
                if not self._try_load_map(map_path):
                    return False

            # 5) normaliza pelo tipo Front/Rear/Mid/BC
            if not self._normalize_simulated_dataset_by_kind():
                return False

            # 6) se for BC, prepara BC-EKF
            if self.simulated_dataset_kind == "BC":
                self._prepare_bc_ekf_data_for_simulated_bc()

                if self._bc_ekf_data is None:
                    self.host._set_msg("Falha ao preparar BC-EKF para dataset BC")
                    return False

            # 7) checagem final depois de reduzir para N colunas
            if self._batch_dists is None or self._dataset_anchors is None:
                self.host._set_msg("Dataset ou âncoras não carregados")
                return False

            n_cols = int(self._batch_dists.shape[1])
            n_anchors = int(np.asarray(self._dataset_anchors).shape[0])

            if n_cols != n_anchors:
                self.host._set_msg(
                    f"Incompatibilidade: dataset possui {n_cols} colunas de âncora, "
                    f"mas o layout possui {n_anchors} âncoras"
                )
                return False

            self._dataset_label = (
                f"SIM | tipo={self.simulated_dataset_kind} | "
                f"dataset={os.path.basename(dataset_file)}"
            )

            self.host._set_msg(
                f"Dataset simulado configurado: tipo={self.simulated_dataset_kind}, "
                f"{self._batch_dists.shape[0]} amostras"
            )

        # Real (encoder + UWB)
        else:
            if not real_encoder_file:
                self.host._set_msg("Selecione um arquivo de encoder")
                return False
            if not real_uwb_file:
                self.host._set_msg("Selecione um arquivo UWB")
                return False

            if anchor_file:
                anchors_path = os.path.join(self.anchors_dir, anchor_file)
                if not self._load_anchors(anchors_path):
                    return False

            if route_file:
                route_path = os.path.join(self.routes_dir, route_file)
                if not self._try_load_route(route_path):
                    return False

            if map_file:
                map_path = os.path.join(self.maps_dir, map_file)
                if not self._try_load_map(map_path):
                    return False

            encoder_path = os.path.join(self.real_data_dir, real_encoder_file)
            uwb_path = os.path.join(self.real_data_dir, real_uwb_file)

            if not self._load_real_encoder_uwb_dataset(encoder_path, uwb_path):
                return False

        self.dataset_modal.close()

        if self.dataset_source_type == "real_encoder_uwb":
            if self._batch_dists is not None:
                self.host._set_msg(
                    f"Dataset real configurado: {self._batch_dists.shape[0]} amostras"
                )
        else:
            self.host._set_msg("Dataset configurado")

    def _refresh_dataset_algo_buttons(self):
        """
        Atualiza texto dos botões laterais considerando variantes.
        """
        if not hasattr(self, "_btn_algos") or self._btn_algos is None:
            return

        for algo in ALGO_ORDER:
            btn = self._btn_algos.get(algo)

            if btn is None:
                continue

            label = algorithm_button_label(
                algo,
                self.selected,
                self.selected_algo_variants,
            )

            # O Button do projeto usa .text.
            # Mantemos .label também por segurança.
            if hasattr(btn, "text"):
                btn.text = label
            if hasattr(btn, "label"):
                btn.label = label


    def _cycle_sidebar_algorithm(self, algo_key: str):
        """
        Clique em botão de algoritmo.

        Para algoritmos sem variantes:
            liga/desliga.

        Para algoritmos com variantes:
            off -> variante 1 -> variante 2 -> ... -> off.
        """
        cycle_algorithm_variant(
            algo_key,
            self.selected,
            self.selected_algo_variants,
        )

        self.host.selected = self.selected
        self.host.selected_algo_variants = self.selected_algo_variants

        if hasattr(self.host, "_refresh_algo_buttons"):
            self.host._refresh_algo_buttons()

        self._refresh_dataset_algo_buttons()

        self.host._set_msg(
            f"{algo_key}: "
            f"{algorithm_button_label(algo_key, self.selected, self.selected_algo_variants)}"
        )


    def _resolve_algos_to_run(self):
        """
        Converte os botões laterais em nomes concretos para run_batch().
        """
        algos = []

        for base_key in ALGO_ORDER:
            concrete_key = algorithm_active_key(
                base_key,
                self.selected,
                self.selected_algo_variants,
            )

            if concrete_key is not None:
                algos.append(concrete_key)

        return algos


    def _remap_batch_results_to_sidebar_keys(self, raw_results: dict):
        """
        O run_batch retorna chaves concretas:
            trilat_geo_sang2019

        Mas o render/analyzer trabalham melhor com os grupos do sidebar:
            trilaterate3d

        Então remapeamos variantes para o grupo original.
        """
        if not isinstance(raw_results, dict):
            return raw_results

        out = {}

        for concrete_key, result in raw_results.items():
            base_key = algorithm_result_alias(concrete_key)

            if isinstance(result, dict):
                result = dict(result)
                result["algo_key_actual"] = concrete_key
                result["algo_label"] = algorithm_button_label(
                    base_key,
                    self.selected,
                    self.selected_algo_variants,
                )

            out[base_key] = result

        return out

    def _metric_mode_label(self, mode=None):
        mode = mode or self.metric_mode

        labels = {
            "reference_route": "Rota próxima",
            "reference_synced": "Rota sinc.",
            "encoder_route": "Encoder",
            "cluster": "Cluster",
        }

        return labels.get(mode, str(mode))


    def _cycle_metric_mode(self):
        if not hasattr(self, "available_metric_modes"):
            self.available_metric_modes = [
                "reference_route",
                "reference_synced",
                "encoder_route",
                "cluster",
            ]

        try:
            idx = self.available_metric_modes.index(self.metric_mode)
        except ValueError:
            idx = 0

        self.metric_mode = self.available_metric_modes[
            (idx + 1) % len(self.available_metric_modes)
        ]

        # Recalcula o analyzer se já houver resultado
        if self._batch_results is not None:
            self._dataset_stats = self._compute_dataset_stats(self._batch_results)

            ranking = self._dataset_ranking()

            for algo in self._batch_results:
                if isinstance(self._batch_results[algo], dict):
                    self._batch_results[algo]["ranking_row"] = next(
                        (row for row in ranking if row.get("algo") == algo),
                        None
                    )

        self.host._set_msg(f"Métrica: {self._metric_mode_label()}")



    def _load_anchors(self, anchors_path: str) -> bool:
        return apply_anchors_to_dataset_mode(self, anchors_path)

    def _try_load_route(self, route_path: str) -> bool:
        return apply_route_to_dataset_mode(self, route_path)
    
    def _try_load_dataset(self, path: str) -> bool:
        """
        Carrega somente o arquivo bruto do dataset simulado.
        A normalização por Front/Rear/Mid/BC acontece depois,
        quando as âncoras já estiverem carregadas.
        """
        self._dataset_path = path
        self._dataset_label = os.path.basename(path)
        self._dataset_stats = None
        self._batch_results = None

        try:
            dists, devs = load_simulated_dataset_file(path)

            self._batch_dists = dists
            self._batch_devs = devs

            self.host._set_msg(f"Dataset carregado: {self._dataset_label}")
            return True

        except Exception as e:
            print(f"[DATASET] erro ao carregar dataset: {e}")
            self._batch_dists = None
            self._batch_devs = None
            self.host._set_msg("Erro ao carregar dataset")
            return False

    def _try_load_map(self, map_path: str) -> bool:
        return apply_map_to_dataset_mode(self, map_path)

    # =========================================================
    # EVENTOS
    # =========================================================

    def handle_events(self, events):
        actions = _actions_default()

        for event in events:
            if event.type == pg.QUIT:
                return _actions_quit()

            if self.dataset_modal_open:
                if self.dataset_modal.handle_event(event):
                    continue

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    if self.show_legend_overlay:
                        self.show_legend_overlay = False
                        continue
                    return _actions_menu()
                
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
                        self.dataset_modal.open()
                        continue

                    elif self.btn_run_batch.hit(pos):
                        self._run_batch()
                        continue

                    elif self.btn_export.hit(pos):
                        if hasattr(self.host, "_export_csv"):
                            self.host._batch_results = self._batch_results
                            self.host._export_csv()
                        continue
                    
                    elif self.btn_metric_mode.hit(pos):
                        self._cycle_metric_mode()
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
                                self._cycle_sidebar_algorithm(nome)
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
        pass

    def draw(self):
        # Primeiro desenha mundo, status, analyzer, legenda e limpa painel lateral
        draw_dataset_mode(self)

        self._refresh_dataset_algo_buttons()

        # Depois desenha os botões principais por cima
        self.btn_back.draw(self.host.screen)
        self.btn_mode.draw(self.host.screen)
        self.btn_load_dataset.draw(self.host.screen)
        self.btn_run_batch.draw(self.host.screen)

        if hasattr(self, "btn_export") and self.btn_export is not None:
            self.btn_export.draw(self.host.screen)

        for algo in ALGO_ORDER:
            if algo in self._btn_algos:
                self._btn_algos[algo].draw(self.host.screen)

        if self.dataset_modal_open:
            self.dataset_modal.draw()

    def _normalize_simulated_dataset_by_kind(self):
        """
        Wrapper para normalização do dataset simulado.
        """
        result = normalize_simulated_dataset_by_kind(
            batch_dists=self._batch_dists,
            batch_devs=self._batch_devs,
            dataset_anchors=self._dataset_anchors,
            simulated_kind=self.simulated_dataset_kind,
            cfg=config,
        )

        if not result.ok:
            self.host._set_msg(result.message)
            return False

        self._batch_dists = result.batch_dists
        self._batch_devs = result.batch_devs
        self.simulated_dataset_kind = result.simulated_kind

        return True

    # =========================================================
    # LOADERS
    # =========================================================

    def _run_batch(self):
        '''Executa os algoritmos selecionados no batch atual.
        Verifica pré-requisitos (dataset, âncoras, tipo) e prepara dados'''
        if self._batch_dists is None:
            self.host._set_msg("Carregue um dataset primeiro")
            return

        if self._dataset_anchors is None:
            self.host._set_msg("Selecione as âncoras primeiro")
            return

        algos_to_run = self._resolve_algos_to_run()

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

            p_true = self._get_batch_ground_truth_xy()

            raw_results = run_batch(
                anchors_Nx3=anchors,
                distances=self._batch_dists,
                deviations=devs,
                algoritmos=algos_to_run,
                p_true=p_true,
                bc_ekf_data=self._bc_ekf_data,
            )

            self._batch_results = self._remap_batch_results_to_sidebar_keys(raw_results)

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
            traj_sidecar = guess_sampled_traj_sidecar(self._dataset_path)
            if traj_sidecar is not None:
                try:
                    sampled_route = np.asarray(load_sampled_traj_csv(traj_sidecar), dtype=float)
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
            if any(k in header_lower for k in [
                "contador", "millis", "timestamp", "tempo",
                "contdir", "contesq", "tmp", "disdir", "diseq"
            ]):
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
            "contador_direita", "contdir", "count_right", "right_count", "pulsos_direita"
        )

        col_left_count = pick(
            "contador_esquerda", "contesq", "count_left", "left_count", "pulsos_esquerda"
        )

        col_right_dist = pick(
            "distancia_direita", "disdir", "dist_right", "right_dist", "distance_right"
        )

        col_left_dist = pick(
            "distancia_esquerda", "diseq", "disesq", "dist_left", "left_dist", "distance_left"
        )

        col_time_ms = pick(
            "millis", "tmp", "timestamp", "tempo", "time_ms"
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


    def _reset_real_dataset_state(self):
        """
        Limpa estados derivados de um carregamento real anterior.
        Não limpa rota de referência, mapa ou âncoras.
        """
        self._real_dataset = None
        self._real_aligned_rows = None
        self._real_odom_path = None
        self._real_range_matrix = None
        self._real_sigma_matrix = None
        self._real_timestamps = []
        self._real_anchor_ids = []

        self._real_encoder_samples = None
        self._real_uwb_rows = None

        self._batch_dists = None
        self._batch_devs = None
        self._batch_results = None
        self._dataset_stats = None
        self._bc_ekf_data = None

        # Esta é a odometria reconstruída do dataset real.
        self._dataset_route = None

        # Não limpar:
        # self._route_waypoints
        # self._route_label
        # self._reference_route_display
        # self._reference_route_dense

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


        return out

    def _apply_dataset_config_from_modal_values(self):
        """
        Ponte entre o DatasetConfigModal e o fluxo atual de carregamento.
        """
        dataset_file = getattr(self, "_modal_dataset_file", "")
        real_encoder_file = getattr(self, "_modal_real_encoder_file", "")
        real_uwb_file = getattr(self, "_modal_real_uwb_file", "")
        anchor_file = getattr(self, "_modal_anchor_file", "")
        route_file = getattr(self, "_modal_route_file", "")
        map_file = getattr(self, "_modal_map_file", "")
        sim_kind = getattr(self, "_modal_simulated_kind", self.simulated_dataset_kind)

        self.simulated_dataset_kind = sim_kind or "Front"

        return self._apply_dataset_config_values(
            dataset_file=dataset_file,
            real_encoder_file=real_encoder_file,
            real_uwb_file=real_uwb_file,
            anchor_file=anchor_file,
            route_file=route_file,
            map_file=map_file,
            simulated_kind=self.simulated_dataset_kind,
        )

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

    def _draw_dashed_line_world(
        self,
        p0,
        p1,
        color=(140, 140, 140),
        width=2,
        dash_px=12,
        gap_px=8,
    ):
        p0_screen = np.array(
            self.host.cam.world_to_screen(float(p0[0]), float(p0[1])),
            dtype=float,
        )
        p1_screen = np.array(
            self.host.cam.world_to_screen(float(p1[0]), float(p1[1])),
            dtype=float,
        )

        vec = p1_screen - p0_screen
        length = float(np.linalg.norm(vec))

        if length < 1e-9:
            return

        direction = vec / length
        step = dash_px + gap_px
        t = 0.0

        while t < length:
            a = p0_screen + direction * t
            b = p0_screen + direction * min(t + dash_px, length)

            pg.draw.line(
                self.host.screen,
                color,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                width,
            )

            t += step


    def _draw_reference_route(self):
        if self._reference_route_display is None:
            return

        pts = np.asarray(self._reference_route_display, dtype=float)

        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            return

        for p0, p1 in zip(pts[:-1], pts[1:]):
            self._draw_dashed_line_world(
                p0,
                p1,
                color=(130, 130, 130),
                width=2,
                dash_px=12,
                gap_px=8,
            )

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
    
    def _load_real_encoder_uwb_dataset(self, encoder_path, uwb_path):
        """
        Carrega dataset real encoder + UWB usando pipeline externa.
        Depois prepara BC-EKF quando possível.
        Retorna True se o dataset foi realmente aplicado ao estado batch.
        """
        self._reset_real_dataset_state()

        anchor_uwb_ids = (
            getattr(self, "_anchor_uwb_ids", None)
            or getattr(self, "_anchors_uwb_ids", None)
            or getattr(self, "_dataset_anchor_uwb_ids", None)
            or getattr(self, "_dataset_uwb_ids", None)
            or getattr(self, "anchor_uwb_ids", None)
        )

        if anchor_uwb_ids is None and hasattr(self, "_dataset_anchor_meta"):
            meta = getattr(self, "_dataset_anchor_meta", None)
            if isinstance(meta, dict):
                anchor_uwb_ids = (
                    meta.get("anchor_ids_uwb")
                    or meta.get("uwb_ids")
                    or meta.get("anchor_ids")
                )

        result = load_real_encoder_uwb_dataset(
            encoder_path=encoder_path,
            uwb_path=uwb_path,
            dataset_anchors=self._dataset_anchors,
            anchor_uwb_ids=anchor_uwb_ids,
            cfg=config,
        )

        if not self._apply_real_pipeline_result(result):
            return False

        if self._batch_dists is None:
            self.host._set_msg("Dataset real carregado, mas sem matriz de distâncias")
            return False

        if self._batch_dists.size == 0:
            self.host._set_msg("Dataset real carregado, mas matriz de distâncias vazia")
            return False

        if self._dataset_anchors is None:
            self.host._set_msg("Dataset real carregado, mas sem arquivo de âncoras")
            return False

        n_cols = int(self._batch_dists.shape[1])
        n_anchors = int(self._dataset_anchors.shape[0])

        if n_cols != n_anchors:
            self.host._set_msg(
                f"UWB/layout incompatíveis: dataset tem {n_cols} colunas, "
                f"mas o layout possui {n_anchors} âncoras"
            )
            return False

        # Prepara BC-EKF quando houver dados suficientes.
        if self._real_encoder_samples is not None and self._real_uwb_rows is not None:
            self._prepare_bc_ekf_data_for_real_bc(
                self._real_encoder_samples,
                self._real_uwb_rows,
            )

        self.dataset_source_type = "real_encoder_uwb"
        self.simulated_dataset_kind = "Front"

        self._dataset_label = (
            f"REAL | enc={os.path.basename(str(encoder_path))} | "
            f"uwb={os.path.basename(str(uwb_path))}"
        )

        self.host._set_msg(
            f"Dataset real configurado: {self._batch_dists.shape[0]} amostras, "
            f"{self._batch_dists.shape[1]} âncoras"
        )
        
        return True
        
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
    
    # =========================================================
    # ANALYZER
    # =========================================================

    def _compute_dataset_stats(self, results: dict):
        """
        Calcula estatísticas do Dataset Analyzer.

        Modos:
        - reference_route: distância até a rota de referência;
        - reference_synced: rota de referência reamostrada ponto a ponto;
        - encoder_route: comparação com odometria reconstruída;
        - cluster: dispersão relativa.
        """
        if self.dataset_source_type == "real_encoder_uwb":
            reference_route = (
                self._reference_route_display
                if self._reference_route_display is not None
                else self._route_waypoints
            )

            if self.metric_mode == "reference_route":
                stats = compute_track_vs_polyline_stats(results, reference_route, algo_order=ALGO_ORDER)
                if stats:
                    return stats

                return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)

            if self.metric_mode == "reference_synced":
                stats = compute_track_vs_synced_reference_stats(results, reference_route, algo_order=ALGO_ORDER)
                if stats:
                    return stats

                return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)

            if self.metric_mode == "encoder_route":
                encoder_ref = self._real_odom_path if self._real_odom_path is not None else self._dataset_route
                stats = compute_track_vs_sampled_reference_stats(results, encoder_ref, algo_order=ALGO_ORDER)
                if stats:
                    return stats

                return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)

            return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)

        p_true = self._get_batch_ground_truth_xy()

        if p_true is None:
            return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)

        stats = compute_track_vs_sampled_reference_stats(
            results,
            p_true,
            algo_order=ALGO_ORDER,
        )

        if stats:
            return stats

        return compute_dataset_cluster_stats(results, algo_order=ALGO_ORDER)
    
    def _dataset_ranking(self):
        """
        Gera ranking a partir de _dataset_stats.
        Formato principal: dict indexado por nome do algoritmo.
        """
        if self._dataset_stats is None:
            return []

        if isinstance(self._dataset_stats, dict):
            try:
                return build_ranking_summary(self._dataset_stats, top_k=5)
            except Exception as e:
                print("[DATASET] erro ao gerar ranking:", e)

                rows = []
                for algo, row in self._dataset_stats.items():
                    if not isinstance(row, dict):
                        continue

                    try:
                        rmse = float(row.get("rmse", np.inf))
                    except Exception:
                        rmse = np.inf

                    if not np.isfinite(rmse):
                        continue

                    r = dict(row)
                    r["algo"] = algo
                    r["rmse"] = rmse
                    rows.append(r)

                rows.sort(key=lambda r: r.get("rmse", np.inf))

                for idx, row in enumerate(rows, start=1):
                    row["rank"] = idx

                return rows[:5]

        # Compatibilidade com versões anteriores que retornavam lista
        if isinstance(self._dataset_stats, list):
            rows = []

            for row in self._dataset_stats:
                if not isinstance(row, dict):
                    continue
                if "algo" not in row:
                    continue

                try:
                    rmse = float(row.get("rmse", np.inf))
                except Exception:
                    rmse = np.inf

                if not np.isfinite(rmse):
                    continue

                r = dict(row)
                r["rmse"] = rmse
                rows.append(r)

            rows.sort(key=lambda r: r.get("rmse", np.inf))

            for idx, row in enumerate(rows, start=1):
                row["rank"] = idx

            return rows[:5]

        return []

    def _draw_analyzer(self):
        draw_dataset_analyzer(self)

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

        if self.dataset_source_type == "real_encoder_uwb":
            if self._reference_route_display is not None and len(self._reference_route_display) >= 2:
                self._reference_route_dense = resample_polyline(
                    self._reference_route_display,
                    len(self._batch_dists)
                )
            else:
                self._reference_route_dense = None

        # só usa a rota do payload se ainda não houver rota externa carregada
        if self._dataset_route is None:
            self._dataset_route = payload["route"]

        self._dataset_label = (
            f"REAL | enc={Path(self._real_encoder_file).name} | "
            f"uwb={Path(self._real_uwb_file).name}"
        )

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

    def _route_xy_to_pose_xytheta(self, route_xy):
        '''Converte uma rota de referência dada como sequência de pontos XY em uma sequência de poses XYTheta,'''
        return route_xy_to_pose_xytheta(route_xy)


    def _pose_xytheta_to_vw(self, poses_xytheta, T):
        return pose_xytheta_to_vw(poses_xytheta, T)

    def _apply_bc_prep_result(self, result: BcPrepResult):
        """
        Aplica no estado da tela o resultado vindo de dataset_bc_prep.py.
        """
        if result is None or not result.ok:
            self._bc_ekf_data = None
            msg = result.message if result is not None else "Falha ao preparar BC-EKF"
            self.host._set_msg(msg)
            return False

        self._bc_ekf_data = result.bc_ekf_data

        if result.batch_dists is not None:
            self._batch_dists = result.batch_dists

        if result.batch_devs is not None:
            self._batch_devs = result.batch_devs

        if result.dataset_route is not None:
            self._dataset_route = result.dataset_route

        if result.real_odom_path is not None:
            self._real_odom_path = result.real_odom_path

        if result.real_range_matrix is not None:
            self._real_range_matrix = result.real_range_matrix

        if result.real_sigma_matrix is not None:
            self._real_sigma_matrix = result.real_sigma_matrix

        if result.real_timestamps is not None:
            self._real_timestamps = result.real_timestamps

        if result.real_anchor_ids is not None:
            self._real_anchor_ids = result.real_anchor_ids

        return True

    def _apply_real_pipeline_result(self, result):
        """
        Aplica ao estado da tela o resultado do pipeline real.
        """
        if result is None or not result.ok:
            msg = result.message if result is not None else "Falha ao carregar dataset real"
            self.host._set_msg(msg)
            return False

        self._real_dataset = result.real_dataset
        self._real_aligned_rows = result.real_aligned_rows

        self._batch_dists = None
        self._batch_devs = None

        if result.batch_dists is not None:
            self._batch_dists = np.asarray(result.batch_dists, dtype=float)

        if result.batch_devs is not None:
            self._batch_devs = np.asarray(result.batch_devs, dtype=float)

        if self._batch_dists is None:
            self.host._set_msg("Pipeline real não retornou batch_dists")
            return False

        if self._batch_devs is None:
            self._batch_devs = np.full_like(
                self._batch_dists,
                float(getattr(config, "UWB_NOISE_STD", 0.05)),
                dtype=float,
            )

        self._real_range_matrix = (
            np.asarray(result.real_range_matrix, dtype=float)
            if result.real_range_matrix is not None
            else self._batch_dists.copy()
        )

        self._real_sigma_matrix = (
            np.asarray(result.real_sigma_matrix, dtype=float)
            if result.real_sigma_matrix is not None
            else self._batch_devs.copy()
        )

        self._real_timestamps = result.real_timestamps or []
        self._real_anchor_ids = result.real_anchor_ids or []

        self._real_encoder_samples = result.encoder_samples
        self._real_uwb_rows = result.uwb_rows

        return True

    def _prepare_bc_ekf_data_for_simulated_bc(self):
        result = prepare_simulated_bc_ekf_data(
            batch_dists=self._batch_dists,
            batch_devs=self._batch_devs,
            dataset_anchors=self._dataset_anchors,
            dataset_path=self._dataset_path,
            dataset_route=self._dataset_route,
            route_waypoints=self._route_waypoints,
            cfg=config,
            resample_polyline_fn=resample_polyline,
        )

        self._apply_bc_prep_result(result)


    def _prepare_bc_ekf_data_for_real_bc(self, encoder_samples, uwb_rows):
        result = prepare_real_bc_ekf_data(
            encoder_samples=encoder_samples,
            uwb_rows=uwb_rows,
            dataset_anchors=self._dataset_anchors,
            cfg=config,
            build_pose_path_fn=self._build_pose_path_from_encoder_samples,
            resample_pose_path_fn=self._resample_pose_path_to_length,
            apply_initial_pose_fn=self._apply_real_odom_initial_pose,
        )

        self._apply_bc_prep_result(result)

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