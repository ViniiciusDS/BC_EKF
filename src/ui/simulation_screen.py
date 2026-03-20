# src/ui/simulation_screen.py

from __future__ import annotations
import math
from typing import Optional, List, Tuple, Any, Dict
from dataclasses import dataclass
import numpy as np
import pygame as pg
import os
import json

import src.config as config
from src.utils import push_plot_data, RunLogger, start_plot_process, stop_plot_process
from src.environment.environment import draw_environment
from src.trajectory import Trajectory  
from src.ui.ui_elements import TextBoxDropdown, ToggleRow 
from src.control.waypoint_controller import waypoint_controller
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_robot, draw_text
from src.ui.botton import Button
from src.uwb.shared_state import SharedUwbState
from src.uwb.node_params_serialization import shared_state_to_dict, dict_to_node_params, upgrade_anchors_file_format, validate_anchors_data
from src.experiments.dataset_export import export_uwb_dataset_from_sim, DatasetConfig                            

BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
GREEN   = (0, 180, 0)
BLUE    = (50, 100, 220)
ORANGE  = (255, 150, 0)
PURPLE  = (160, 0, 160)

@dataclass
class SimActions:
    """Ações solicitadas pela tela de simulação ao main_interactive."""
    go_to_menu: bool = False    # indica que o usuário quer voltar ao menu
    quite_app: bool = False     # indica que o usuário quer fechar o app


class SimulationScreen:
    """
    Encapsula todo o estado e a lógica da simulação (estado SIM).

    O main_interactive fica responsável apenas por:
      - controlar o estado global (MENU / SIM / MAPEDITOR)
      - criar a janela, clock, fontes e câmera
      - delegar: sim_screen.step(dt, events) e sim_screen.draw(...)

    Esta classe cuida de:
      - eventos de teclado/mouse no estado SIM
      - comandos v_cmd / w_cmd (manual e autopilot)
      - gravação de rota (recording / recorded_points / waypoints)
      - âncoras dinâmicas (anchors_dyn)
      - histórico de trajetórias e erros
      - HUD lateral (botões, textboxes, debug)
      - logging em arquivo (RunLogger)
      - envio de dados para gráficos (push_plot_data)
    """ 
        ####        TP INIT       #####
    def __init__(
        self,
        screen: pg.Surface,
        cam: Any,
        clock: pg.time.Clock,
        sim: Any,
        font: pg.font.Font,
        bigfont: pg.font.Font,
        side_width: int,
        plot_state: dict,
        shared_uwb: Optional[SharedUwbState] = None,
    ) -> None:
        
        self.screen = screen
        self.cam = cam
        self.clock = clock
        self.sim = sim
        self.font = font
        self.bigfont = bigfont
        self.SIDE_W = side_width
        self.plot_state = plot_state
        self._last_dt = 0.0

        # -------- estados gerais da simulação --------
        self.autopilot = False
        self.speed_factor = 5
        self.show_debug = False
        self.perfect_motion = False

        # comandos de controle
        self.V_MAX = getattr(config, "V_MAX", 0.6)
        self.W_MAX = getattr(config, "W_MAX", 1.0)
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.accel_lin = 0.02
        self.accel_ang = 0.08

        self.modal_mode = None
        
        self.robot_cfg = {
            "perfect_motion": False,
            "perfect_odometry": False,
            "perfect_uwb": False,
            "perfect_filter_model": False,
            "use_odometry": True,
        }

        self.robot_rows = []

        # gravação de rota e autopilot
        self.recording = False
        self.recorded_points: List[Tuple[float, float]] = []
        self.waypoints: Optional[np.ndarray] = None
        self.wp_idx: int = 0

        # paths para desenhar trilhas
        self.path_true: List[Tuple[float, float]] = []
        self.path_pred: List[Tuple[float, float]] = []
        self.path_est: List[Tuple[float, float]] = []

        # histórico de erros para gráficos
        self.ts_hist: List[float] = []
        self.pos_err_hist: List[float] = []
        self.head_err_hist: List[float] = []

        # métricas instantâneas
        self.pos_err = 0.0
        self.head_err = 0.0
        self.innov_norm: Optional[float] = None
        self.nis: Optional[float] = None
        self.gdop = float("nan")

        # ----------------------------
        # GDOP stats para debug e análise (média, min, max)
        # ----------------------------
        self.gdop = float("nan")
        self._gdop_sum = 0.0
        self._gdop_count = 0
        self._gdop_min = float("inf")
        self._gdop_max = float("-inf")
        self.gdop_avg = float("nan")

        # (opcional) histórico pra plot
        self.gdop_hist = []
        self.gdop_hist_maxlen = 5000

        # logging em arquivo
        self.filelog_on = False
        self.file_logger: Optional[RunLogger] = None

        # UI: botões e textboxes (HUD)
        self.btn_filelog = None
        self.btn_graphs = None
        self.btn_place_anchor = None
        self.textbox_x = None
        self.textbox_y = None

        # logs atuais (trajetórias)
        self.true_traj = np.empty((0, 3))
        self.est_traj  = np.empty((0, 3))
        self.pred_traj = np.empty((0, 3))

        # pan / zoom
        self.panning: bool = False
        self.pan_last: tuple[int, int] | None = None
        
        # --- HUD: Âncoras ---
        self.textbox_x = TextBoxDropdown(pg.Rect(0, 0, 80, 26), self.font, options=[])
        self.textbox_x.text = "0.0"
        self.textbox_x.cursor_pos = len(self.textbox_x.text)

        self.textbox_y = TextBoxDropdown(pg.Rect(0, 0, 80, 26), self.font, options=[])
        self.textbox_y.text = "0.0"
        self.textbox_y.cursor_pos = len(self.textbox_y.text)

        self.btn_place_anchor = Button(
            rect=(0, 0, 190, 30),
            text="Adicionar Âncora (x,y)",
            font=self.font,
            bg=(235, 250, 235),
        )  

        # ---- HUD: ferramentas  ----
        self.btn_filelog = Button(
            rect=(0, 0, 190, 30),
            text="Log arquivo: OFF",
            font=self.font,
            bg=(235, 250, 235),
        )
        self.btn_graphs = Button(
            rect=(0, 0, 190, 30),
            text="Mostrar Gráficos",
            font=self.font,
            bg=(235, 235, 250),
        )

        self.shared_uwb = shared_uwb
    
        # Usar anchors do shared
        if shared_uwb is not None:
            self.anchors_dyn = shared_uwb.anchors_np3()
        else:
            self.anchors_dyn = np.zeros((3, 0))

        # âncoras dinâmicas
        self.anchors_dyn = self.shared_uwb.anchors_np3() if self.shared_uwb is not None else np.zeros((3, 0))

        # ---- Topbar (rotas/âncoras) ----
        self.topbar_h = 36
        self.topbar_rect = pg.Rect(0, 0, self.cam.viewport[0], self.topbar_h)

        self.routes_dir = "routes"
        self.anchors_dir = "anchor_sets"
        os.makedirs(self.routes_dir, exist_ok=True)
        os.makedirs(self.anchors_dir, exist_ok=True)

        self.btn_tb_load_route  = Button((0,0,130,26), "Carregar Rotas", self.font, bg=(235,245,255))
        self.btn_tb_save_route  = Button((0,0,120,26), "Salvar Rotas",   self.font, bg=(235,255,235))
        self.btn_tb_load_anchor = Button((0,0,150,26), "Carregar Âncoras", self.font, bg=(235,245,255))
        self.btn_tb_save_anchor = Button((0,0,140,26), "Salvar Âncoras",   self.font, bg=(235,255,235))

        # ---- Modal (genérico) ----
        self.modal_open = False
        self.modal_mode = None  # "load_route" | "save_route" | "load_anchors" | "save_anchors"
        self.modal_rect = None

        self.modal_dropdown = TextBoxDropdown(pg.Rect(0,0,260,26), self.font, options=[])
        self.modal_namebox  = TextBoxDropdown(pg.Rect(0,0,260,26), self.font, options=[], placeholder="nome")

        self.btn_modal_ok = Button((0,0,90,26), "OK", self.font, bg=(235,250,235))
        self.btn_modal_cancel = Button((0,0,90,26), "Cancelar", self.font, bg=(250,235,235))
        

        self.btn_tb_robot = Button((0,0,110,26), "Robô", self.font, bg=(245,235,255))

        self._pm_seg_i = 0
        self._pm_s = 0.0
        self._pm_vref = 0.6  # velocidade ao longo do caminho quando perfect

        self.btn_gen_dataset = Button((0,0,190,30), "Gerar Dataset", self.font, bg=(245,235,210))
        self.dataset_tag_mode = "mid"   # "front" | "rear" | "mid"
        # --- Modal Robô: seleção de tag do dataset (dropdown) ---
        self.dataset_tag_dropdown = TextBoxDropdown(
            pg.Rect(0, 0, 160, 26),
            self.font,
            options=["mid", "front", "rear"],
        )
        self.dataset_tag_dropdown.set_text(self.dataset_tag_mode)
        self.dataset_tag_dropdown.active = False
        self.dataset_tag_dropdown.dropdown_open = False
        # --- Dataset recording (ao vivo, sem travar UI) ---
        self.dataset_recording = False
        self.dataset_fp = None
        self.dataset_sample_dt = 0.10   # 10 Hz (ajuste como quiser)
        self._dataset_acc = 0.0
        self.dataset_out_path = None




    # ------------------------------------------------------------------
    # Interface principal que o main_interactive vai chamar
    # ------------------------------------------------------------------

    def handle_events(self, events) -> SimActions:
        """
        Processa todos os eventos de pygame para o estado SIM.
        Retorna:
            False -> continuar rodando
            True  -> sair da SIM (voltar ao menu)
        """

        actions = SimActions()

        # garante que hitboxes estejam atualizadas antes dos cliques
        self.layout_hud()
        self._layout_topbar_buttons()

        for event in events:
            if event.type == pg.QUIT:
                # deixe o main decidir sair do programa inteiro
                actions.quite_app = True
                return actions

            # 1) Se modal aberto, ele tem prioridade total
            if self.modal_open:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self._close_modal()
                        continue

                    # repassa teclado pro campo certo

                        # 1) dropdown dataset tag
                        if self.dataset_tag_dropdown:
                            # Se o dropdown estiver aberto, deixe ele processar o clique SEMPRE
                            # (isso permite clicar nas opções "front/rear")
                            if self.dataset_tag_dropdown.dropdown_open:
                                self.dataset_tag_dropdown.handle_event(event)

                                # se depois do clique ele fechou (escolheu algo), desativa foco
                                if not self.dataset_tag_dropdown.dropdown_open:
                                    self.dataset_tag_dropdown.active = False
                                continue

                            # Caso dropdown esteja fechado, clique no campo abre
                            inside = self.dataset_tag_dropdown.rect.collidepoint((mx, my))
                            self.dataset_tag_dropdown.active = inside
                            if inside:
                                self.dataset_tag_dropdown.options_filtered = list(self.dataset_tag_dropdown.options_all)
                                self.dataset_tag_dropdown.dropdown_open = True
                                self.dataset_tag_dropdown.handle_event(event)
                            else:
                                self.dataset_tag_dropdown.dropdown_open = False

                    # passa eventos de teclado para o campo de texto
                    if self.modal_mode in ("save_route", "save_anchors"):
                        if self.modal_namebox.active:
                            self.modal_namebox.handle_event(event)

                    if event.key == pg.K_RETURN:
                        self._modal_confirm()
                    continue

                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos

                    # OK / Cancel
                    if self.btn_modal_ok.hit((mx, my)):
                        self._modal_confirm()
                        continue
                    if self.btn_modal_cancel.hit((mx, my)):
                        self._close_modal()
                        continue

                    # clique no campo (ativa)
                    if self.modal_mode in ("load_route", "load_anchors"):
                        self.modal_dropdown.active = self.modal_dropdown.rect.collidepoint((mx, my))
                        self.modal_namebox.active = False
                        self.modal_dropdown.handle_event(event)
                    else:
                        self.modal_namebox.active = self.modal_namebox.rect.collidepoint((mx, my))
                        self.modal_dropdown.active = False
                        self.modal_namebox.handle_event(event)
                    
                    if self.modal_mode == "robot_cfg":
                        # toggles
                        for row in self.robot_rows:
                            if row.hit((mx,my)):
                                row.toggle()
                                continue

                    # dropdown tag dataset
                    if self.dataset_tag_dropdown:
                        # 1) Se a lista estiver aberta, ela tem prioridade total (permite clicar em front/rear)
                        if self.dataset_tag_dropdown.dropdown_open:
                            self.dataset_tag_dropdown.handle_event(event)

                            # aplica imediatamente no estado (sem depender do OK)
                            chosen = (self.dataset_tag_dropdown.text or "").strip().lower()
                            if chosen in ("front", "rear", "mid"):
                                self.dataset_tag_mode = chosen

                            # se após clique a lista fechou, tira foco
                            if not self.dataset_tag_dropdown.dropdown_open:
                                self.dataset_tag_dropdown.active = False

                            # NÃO fecha o modal neste clique
                            continue

                        # 2) Se lista estiver fechada: clique no campo abre
                        if self.dataset_tag_dropdown.rect.collidepoint((mx, my)):
                            self.dataset_tag_dropdown.active = True
                            self.dataset_tag_dropdown.options_filtered = list(self.dataset_tag_dropdown.options_all)
                            self.dataset_tag_dropdown.dropdown_open = True
                            self.dataset_tag_dropdown.handle_event(event)
                            continue
                        else:
                            self.dataset_tag_dropdown.active = False
                            # não force dropdown_open=False aqui — deixa fechado como já está

                    # --- robot_cfg: dropdown deve capturar clique mesmo fora do modal_rect ---
                    if self.modal_mode == "robot_cfg" and getattr(self, "dataset_tag_dropdown", None):
                        # se dropdown estiver aberto, dê prioridade TOTAL pra ele tratar o clique
                        if self.dataset_tag_dropdown.dropdown_open:
                            self.dataset_tag_dropdown.handle_event(event)
                            # se escolheu uma opção, ele normalmente fecha sozinho
                            if not self.dataset_tag_dropdown.dropdown_open:
                                self.dataset_tag_dropdown.active = False
                            # não fecha o modal neste clique
                            continue

                    # clique fora fecha
                    if self.modal_rect and (not self.modal_rect.collidepoint((mx, my))):
                        self._close_modal()
                    continue

                # enquanto modal aberto, ignora o resto
                continue

            if event.type == pg.KEYDOWN:
                # Check Textboxes first
                consumed = False
                if self.textbox_x and self.textbox_x.handle_event(event):
                    consumed = True
                if not consumed and self.textbox_y and self.textbox_y.handle_event(event):
                    consumed = True
                
                if consumed:
                    # A tecla foi usada pelo TextBox (nmr, backspace, etc)
                    # Não processa como tecla do HUD
                    continue

                if event.key == pg.K_ESCAPE:                    
                    # quando voltar para o MENU (em KEYDOWN K_ESCAPE que troca o state) OU ao encerrar o programa:
                    if self.file_logger:
                        try: self.file_logger.close()
                        except: pass
                        self.file_logger = None
                        self.filelog_on = False
                    stop_plot_process(self.plot_state)

                    if event.key == pg.K_ESCAPE:
                        actions.go_to_menu = True
                        return actions


                elif event.key == pg.K_d:
                    self.show_debug = not self.show_debug

                elif event.key == pg.K_SPACE:
                    self.autopilot = not self.autopilot

                elif event.key == pg.K_LEFTBRACKET:
                    self.speed_factor = max(1, self.speed_factor - 1)

                elif event.key == pg.K_RIGHTBRACKET:
                    self.speed_factor = min(20, self.speed_factor + 1)

                elif event.key == pg.K_c:
                    self.anchors_dyn = np.zeros((3, 0))
                    self.sim.anchors = self.anchors_dyn
                    self.sim.R = np.eye(2) * 1e6
                    self.shared_uwb.anchors_xy = []

                elif event.key == pg.K_r:
                    self.cam.reset_view()

                elif event.key == pg.K_g:
                    self.recording = not self.recording
                    if not self.recording and len(self.recorded_points) > 1:
                        self.waypoints = np.array(self.recorded_points, dtype=float)
                        self.wp_idx = 0
                        self.autopilot = True                    
                        self._pm_seg_i = 0
                        self._pm_s = 0.0
                        self._reset_gdop_stats()

                elif event.key == pg.K_RETURN:
                    if self.recording and len(self.recorded_points) > 1:
                        self.waypoints = np.array(self.recorded_points, dtype=float)
                        self.wp_idx = 0
                        self.autopilot = True
                        self.recording = False

                elif event.key == pg.K_DELETE:
                    self.recorded_points = []
                
                elif event.key == pg.K_p:
                    self.robot_cfg["perfect_motion"] = not self.robot_cfg.get("perfect_motion", False)
                    print(f"[SIM] Perfect motion: {'ON' if self.robot_cfg['perfect_motion'] else 'OFF'}")

                elif event.key == pg.K_1:
                    self.dataset_tag_mode = "front"; print("[SIM] Dataset tag: front")
                elif event.key == pg.K_2:
                    self.dataset_tag_mode = "rear"; print("[SIM] Dataset tag: rear")
                elif event.key == pg.K_3:
                    self.dataset_tag_mode = "mid"; print("[SIM] Dataset tag: mid")
                    

            if event.type == pg.MOUSEBUTTONDOWN:
                mx, my = event.pos

                if event.button == 1:  # botão esquerdo
                    mx, my = event.pos
                    # garante que os rects estão atualizados mesmo antes do draw()
                    self._layout_topbar_buttons()

                    # --- TOPBAR: intercepta clique e não deixa cair no mapa ---
                    if mx < self.cam.viewport[0] and my <= self.topbar_h:
                        print(f"[SIM] click topbar em ({mx},{my})")
                        if self.btn_tb_load_route.hit((mx, my)):
                            print("[SIM] Click: Carregar Rotas")
                            self._open_modal_load("route")
                            self._reset_gdop_stats()
                            continue
                        if self.btn_tb_save_route.hit((mx, my)):
                            print("[SIM] Click: Salvar Rotas")
                            self._open_modal_save("route")
                            continue
                        if self.btn_tb_load_anchor.hit((mx, my)):
                            print("[SIM] Click: Carregar Âncoras")
                            self._open_modal_load("anchors")
                            self._reset_gdop_stats()
                            continue
                        if self.btn_tb_save_anchor.hit((mx, my)):
                            print("[SIM] Click: Salvar Âncoras")
                            self._open_modal_save("anchors")
                            continue
                        if self.btn_tb_robot.hit((mx, my)):
                            self._open_modal_robot()
                            continue

                        # clicou na área da topbar mas não em botão: não faz nada no mapa
                        continue

                    if self.textbox_x and self.textbox_x.rect.collidepoint((mx, my)):
                        self.textbox_x.active = True
                        if self.textbox_y:
                            self.textbox_y.active = False
                        continue

                    if self.textbox_y and self.textbox_y.rect.collidepoint((mx, my)):
                        self.textbox_y.active = True
                        if self.textbox_x:
                            self.textbox_x.active = False
                        continue

                    # clicou fora → perde foco
                    if self.textbox_x:
                        self.textbox_x.active = False
                    if self.textbox_y:
                        self.textbox_y.active = False
                        
                # zoom sempre
                if event.button == 4:
                    self.cam.zoom_at((mx, my), 1.15)
                    continue
                elif event.button == 5:
                    self.cam.zoom_at((mx, my), 1/1.15)
                    continue

                # pan (botão do meio) – independe de mapa ou HUD
                if event.button == 2:
                    self.panning = True
                    self.pan_last = (mx, my)
                    continue

                # -----------------------------------------
                # CLIQUE ESQUERDO (mapa OU HUD)
                # -----------------------------------------
                if event.button == 1:
                    if mx < self.cam.viewport[0]:
                        # >>> MAPA (LMB)
                        wx, wy = self.cam.screen_to_world(mx, my)
                        if self.recording:
                            self.recorded_points.append((wx, wy))
                        else:
                            z = getattr(config, "TAG_HEIGHT", 0.5)
                            self.anchors_dyn = np.hstack([self.anchors_dyn, np.array([[wx], [wy], [z]])])
                            self.sim.anchors = self.anchors_dyn
                            self.sim.R = np.eye(2 * self.anchors_dyn.shape[1]) * 0.0025
                            self.shared_uwb.anchors_xy = [(float(self.anchors_dyn[0,i]), float(self.anchors_dyn[1,i])) for i in range(self.anchors_dyn.shape[1])]
                    else:
                        # >>> HUD (LMB)

                        if self.textbox_x and self.textbox_x.handle_event(event):
                            continue
                        if self.textbox_y and self.textbox_y.handle_event(event):
                            continue

                        if self.btn_filelog and self.btn_filelog.hit((mx, my)):
                            if not self.filelog_on:
                                meta = {
                                    "dt": float(self.sim.dt),
                                    "anchors": self.anchors_dyn[:2,:].T.tolist() if self.anchors_dyn is not None else [],
                                    "z_c": float(getattr(config, "TAG_HEIGHT", 0.5)),
                                    "baseline": float(getattr(config, "WHEEL_BASE", 0.65)),
                                    "route_waypoints": self.waypoints.tolist() if self.waypoints is not None else [],
                                    "config": {
                                        "TIME_STEP": getattr(config, "TIME_STEP", None),
                                        "NOISE_STD_V": getattr(config, "NOISE_STD_V", None),
                                        "NOISE_STD_W": getattr(config, "NOISE_STD_W", None),
                                        "UWB_BIAS_ENABLED": getattr(config, "UWB_BIAS_ENABLED", None),
                                        "UWB_MISALIGNMENT_ENABLED": getattr(config, "UWB_MISALIGNMENT_ENABLED", None),
                                    }
                                }
                                self.file_logger = RunLogger(
                                    out_dir=getattr(config, "LOG_DIR", "resultados/logs"),
                                    run_name=None,
                                    meta=meta,
                                    flush_every_n=getattr(config, "LOG_FLUSH_EVERY_N", 50),
                                )
                                self.filelog_on = True
                                self.btn_filelog.text = "Log arquivo: ON"
                            else:
                                try:
                                    if self.file_logger: self.file_logger.close()
                                finally:
                                    self.file_logger = None
                                    self.filelog_on = False
                                    self.btn_filelog.text = "Log arquivo: OFF"

                        elif self.btn_graphs and self.btn_graphs.hit((mx, my)):
                            # inicia/reativa a janela de gráficos (processo separado)
                            start_plot_process(self.plot_state)

                        if self.btn_place_anchor and self.btn_place_anchor.hit((mx, my)):
                            try:
                                x = float(self.textbox_x.text.replace(",", "."))
                                y = float(self.textbox_y.text.replace(",", "."))
                                z = getattr(config, "TAG_HEIGHT", 0.5)

                                new_anchor = np.array([[x], [y], [z]])
                                self.anchors_dyn = np.hstack([self.anchors_dyn, new_anchor])
                                self.sim.anchors = self.anchors_dyn
                                self.sim.R = np.eye(2 * self.anchors_dyn.shape[1]) * 0.0025
                                self.shared_uwb.anchors_xy = [(float(self.anchors_dyn[0,i]), float(self.anchors_dyn[1,i])) for i in range(self.anchors_dyn.shape[1])]
                                print(f"Âncora adicionada em ({x}, {y})")
                            except ValueError:
                                print("Coordenadas inválidas para âncora.")
                            continue

                        elif self.btn_gen_dataset and self.btn_gen_dataset.hit((mx, my)):
                            # precisa ter rota + anchors
                            if self.waypoints is None or len(self.waypoints) < 2:
                                print("[SIM] Gere uma rota antes.")
                                continue
                            if not self.shared_uwb or len(self.shared_uwb.anchors_xy) == 0:
                                print("[SIM] Posicione âncoras antes.")
                                continue

                            import time
                            stamp = time.strftime("%Y%m%d_%H%M%S")


                            try:
                                self._start_dataset_recording()
                                print(f"[SIM] Dataset salvo em: {self.dataset_out_path}")
                            except Exception as e:
                                print(f"[SIM] Falha ao gerar dataset: {e}")
                            continue

                # -----------------------------------------
                # CLIQUE DIREITO (mapa)
                # -----------------------------------------
                elif event.button == 3 and mx < self.cam.viewport[0]:
                    if self.recording and self.recorded_points:
                        self.recorded_points.pop()
                    else:
                        wx, wy = self.cam.screen_to_world(mx, my)
                        if self.anchors_dyn.shape[1] > 0:
                            dif = self.anchors_dyn[:2, :].T - np.array([wx, wy])[None, :]
                            j = int(np.argmin(np.sum(dif**2, axis=1)))
                            self.anchors_dyn = np.delete(self.anchors_dyn, j, axis=1)
                            self.sim.anchors = self.anchors_dyn
                            self.sim.R = (np.eye(2 * self.anchors_dyn.shape[1]) * 0.0025) if self.anchors_dyn.shape[1] > 0 else np.eye(2) * 1e6
                            self.shared_uwb.anchors_xy = [(float(self.anchors_dyn[0,i]), float(self.anchors_dyn[1,i])) for i in range(self.anchors_dyn.shape[1])]
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 2:
                    self.panning = False

            if event.type == pg.MOUSEMOTION:
                mx, my = event.pos

                if self.panning and self.pan_last is not None:
                    dx = mx - self.pan_last[0]
                    dy = my - self.pan_last[1]
                    self.cam.pan_pixels(dx, dy)
                    self.pan_last = (mx, my)

        return actions
        

    def update(self, dt: float) -> None:
        """
        Atualiza física, controle, histórico e logging.
          - leitura de teclas contínuas (UP/DOWN/LEFT/RIGHT)
          - autopilot (waypoint_controller)
          - sim.step(...)
          - cálculo de erros, hist, push_plot_data(...)
          - logging em arquivo (file_logger.log_step)
        """
        # teclas contínuas (manual)
        keys = pg.key.get_pressed()
        if not self.autopilot:
            if keys[pg.K_UP]:
                self.v_cmd = min(self.V_MAX, self.v_cmd + self.accel_lin)
            elif keys[pg.K_DOWN]:
                self.v_cmd = max(-self.V_MAX, self.v_cmd - self.accel_lin)
            else:
                self.v_cmd *= 0.90
            if keys[pg.K_LEFT]:
                self.w_cmd = max(-self.W_MAX, self.w_cmd - self.accel_ang)
            elif keys[pg.K_RIGHT]:
                self.w_cmd = min(self.W_MAX, self.w_cmd + self.accel_ang)
            else:
                self.w_cmd *= 0.86
        else:
            x_for_ctrl = getattr(self.sim, "x_true", self.sim.x_est)
            self.v_cmd, self.w_cmd, self.wp_idx = waypoint_controller(
                x_for_ctrl, self.waypoints, self.wp_idx,
                v_max=0.25, w_max=0.8
            )

        # física
        noisy = not self.perfect_motion
        cfg = getattr(self, "robot_cfg", {})
        pm = bool(cfg.get("perfect_motion", False))
        po = bool(cfg.get("perfect_odometry", False))
        pu = bool(cfg.get("perfect_uwb", False))
        pf = bool(cfg.get("perfect_filter_model", False))


        for _ in range(self.speed_factor):
            # 1) Se perfect motion estiver ligado, o ground-truth deve vir da rota perfeita
            true_override = None
            if pm:
                # Usa o dt real do simulador (não o dt do pygame), porque você está sub-steppando
                true_override = self._perfect_pose_from_waypoints(self.sim.dt)

            # 2) Controle (recalcula a cada substep)
            if self.autopilot and self.waypoints is not None and len(self.waypoints) > 1:
                # quando pm=True, controle deve enxergar o "true" perfeito (override)
                x_for_ctrl = true_override if (pm and true_override is not None) else getattr(self.sim, "x_true", self.sim.x_est)

                self.v_cmd, self.w_cmd, self.wp_idx = waypoint_controller(
                    x_for_ctrl, self.waypoints, self.wp_idx,
                    v_max=0.25, w_max=0.8
                )

            use_odo = bool(cfg.get("use_odometry", True))

            self.sim.step(
                self.v_cmd, self.w_cmd,
                perfect_motion=pm, perfect_odometry=po,
                perfect_uwb=pu, perfect_filter_model=pf,
                true_override=true_override,
                use_odometry=use_odo,
            )

        # GDOP (usa pose estimada por padrão; pode trocar por x_true se quiser)
        try:
            self.gdop = self.sim.compute_gdop(self.sim.x_est[0], self.sim.x_est[1])
        except Exception:
            self.gdop = float("nan")

        # acumula stats se for finito
        if np.isfinite(self.gdop):
            self._gdop_sum += float(self.gdop)
            self._gdop_count += 1
            self._gdop_min = min(self._gdop_min, float(self.gdop))
            self._gdop_max = max(self._gdop_max, float(self.gdop))
            self.gdop_avg = self._gdop_sum / max(1, self._gdop_count)

            # (opcional) histórico pra plot
            self.gdop_hist.append(float(self.gdop))
            if len(self.gdop_hist) > self.gdop_hist_maxlen:
                self.gdop_hist.pop(0)

        if self.waypoints is not None and self.wp_idx >= len(self.waypoints):
            self.autopilot = False
            self.v_cmd = 0.0
            self.w_cmd = 0.0
        innov_norm = None
        nis = None

        if getattr(self.sim, 'last_debug', None) and self.sim.last_debug['innov'] is not None:
            y = self.sim.last_debug['innov']
            S = self.sim.last_debug['S']
            try:
                nis = float(y.T @ np.linalg.inv(S) @ y)   # NIS: consistência da medição
            except:
                nis = None
            innov_norm = float(np.linalg.norm(y))
            
        # logs
        self.true_traj, self.est_traj = self.sim.get_logs()
        pred_traj = np.array(self.sim.history_pred)

        if len(self.true_traj) > 0:
            p = (self.true_traj[-1, 0], self.true_traj[-1, 1])
            if not self.path_true or p != self.path_true[-1]:
                self.path_true.append(p)
        if len(pred_traj) > 0:
            self.path_pred.append((pred_traj[-1, 0], pred_traj[-1, 1]))
        if len(self.est_traj) > 0:
            self.path_est.append((self.est_traj[-1, 0], self.est_traj[-1, 1]))
        self.path_true = self.path_true[-2500:]
        self.path_pred = self.path_pred[-2500:]
        self.path_est = self.path_est[-2500:]

        # erros atuais
        pos_err = 0.0
        head_err = 0.0
        if len(self.true_traj) > 0 and len(self.est_traj) > 0:
            pos_err = float(np.linalg.norm(self.true_traj[-1, :2] - self.est_traj[-1, :2]))
            dth = (self.true_traj[-1, 2] - self.est_traj[-1, 2])
            dth = math.atan2(math.sin(dth), math.cos(dth))
            head_err = abs(dth * 180.0 / math.pi)

        # atualiza séries e envia ao processo de plot
        if len(self.true_traj) > 0 and len(self.est_traj) > 0:
            t_now = (len(self.ts_hist)) * self.sim.dt
            self.ts_hist.append(t_now); self.pos_err_hist.append(pos_err); self.head_err_hist.append(head_err)
            push_plot_data(self.plot_state, self.ts_hist, self.pos_err_hist, self.head_err_hist)

        # logging em arquivo (se ativo)
        if self.filelog_on and self.file_logger and len(self.true_traj) > 0 and len(self.est_traj) > 0:
            true_state = self.true_traj[-1, :]
            est_state  = self.est_traj[-1, :]
            pred_state = self.sim.last_debug.get('x_pred', None) if getattr(self.sim, 'last_debug', None) else None
            v_meas, w_meas = getattr(self.sim, 'last_meas', (float('nan'), float('nan')))
            try:
                self.file_logger.log_step(
                    true_state=true_state,
                    pred_state=pred_state,   # pode ser None; o RunLogger já trata
                    est_state=est_state,
                    v_cmd=self.v_cmd, w_cmd=self.w_cmd,
                    v_meas=v_meas, w_meas=w_meas,
                    pos_err=pos_err, heading_err_deg=head_err
                )
            except Exception as e:
                # evita quebrar a simulação por erro de I/O
                pass
        self.pos_err = pos_err
        self.head_err = head_err
        self.innov_norm = innov_norm
        self.nis = nis
        self._last_dt = dt

        # ----------------------------
        # Dataset recording (ao vivo)
        # ----------------------------
        if self.dataset_recording and self.dataset_fp and self.shared_uwb and self.shared_uwb.pipeline:
            self._dataset_acc += float(self.sim.dt)

            while self._dataset_acc >= self.dataset_sample_dt:
                self._dataset_acc -= self.dataset_sample_dt

                anchors = self.anchors_dyn
                if anchors is None or anchors.shape[1] == 0:
                    continue

                l = float(getattr(self.sim, "l", 0.325))
                x_state = np.array(getattr(self.sim, "x_true", self.sim.x_est), dtype=float)
                tag = getattr(self, "dataset_tag_mode", "mid")

                ranges_m, sigmas_m = self.shared_uwb.pipeline.measure_ranges_and_sigmas(
                    x_state=x_state,
                    anchors=anchors,
                    l=l,
                    tag=tag
                )

                # linha: r0 s0 r1 s1 ... (igual dataset real)
                row = []
                for r, s in zip(ranges_m.tolist(), sigmas_m.tolist()):
                    row.append(f"{float(r):.6f}")
                    row.append(f"{float(s):.6f}")
                self.dataset_fp.write(" ".join(row) + "\n")

            # terminou rota? fecha arquivo
            if self.waypoints is not None and self.wp_idx >= len(self.waypoints):
                self._stop_dataset_recording()

    def draw(self) -> None:
        """
        Desenha o mapa, trilhas, robôs e HUD lateral.
        É aqui que entra:
          - fill(WHITE)
          - draw_grid, draw_environment, draw_axes, draw_anchors
          - draw_path(...) trilhas
          - draw_robot(...)
          - indicador de waypoint
          - HUD lateral (FPS, erros, botões, textboxes, etc)
        """
        # desenho
        self.screen.fill(WHITE)

        # --- TOPBAR (no MAPA) ---
        cam_w = self.cam.viewport[0]
        self.topbar_rect = pg.Rect(0, 0, cam_w, self.topbar_h)
        pg.draw.rect(self.screen, (235,235,235), self.topbar_rect)
        pg.draw.line(self.screen, (190,190,190), (0,self.topbar_h), (cam_w,self.topbar_h), 1)

        # layout botões
        self._layout_topbar_buttons()

        self.btn_tb_load_route.draw(self.screen)
        self.btn_tb_save_route.draw(self.screen)
        self.btn_tb_load_anchor.draw(self.screen)
        self.btn_tb_save_anchor.draw(self.screen)
        self.btn_tb_robot.draw(self.screen)

        # mapa (esquerda)
        map_rect = pg.Rect(0, self.topbar_h, cam_w, self.cam.viewport[1] - self.topbar_h)
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(map_rect)
        pg.draw.rect(self.screen, WHITE, map_rect)
        draw_grid(self.screen, self.cam)
        draw_environment(self.screen, self.cam, self.sim.env)
        draw_axes(self.screen, self.cam, self.font)
        draw_anchors(self.screen, self.cam, self.anchors_dyn)

        # rota planejada
        if self.waypoints is not None and len(self.waypoints) > 1:
            draw_path(self.screen, self.cam, self.waypoints, BLACK, 2, dashed=True)
            # destaque do waypoint atual
            if self.autopilot and 0 <= self.wp_idx < len(self.waypoints):
                sx, sy = self.cam.world_to_screen(*self.waypoints[self.wp_idx])
                pg.draw.circle(self.screen, GREEN, (sx, sy), 6)
                pg.draw.circle(self.screen, BLACK, (sx, sy), 6, 1)
        # rota gravada em curso (pontos)
        if self.recording and len(self.recorded_points) > 0:
            for pt in self.recorded_points:
                sx, sy = self.cam.world_to_screen(*pt)
                pg.draw.circle(self.screen, PURPLE, (sx, sy), 4)

        # trilhas
        draw_path(self.screen, self.cam, self.path_true, BLACK, 2)
        draw_path(self.screen, self.cam, self.path_pred, BLUE, 2, dashed=True)
        draw_path(self.screen, self.cam, self.path_est, ORANGE, 2, dashed=True)

        # robôs
        robot_l = getattr(self.sim, 'l', 0.325)   # metade do baseline
        if len(self.true_traj) > 0:
            xr, yr, tr = self.true_traj[-1]
            draw_robot(self.screen, self.cam, xr, yr, tr, BLACK, l=robot_l)
        if len(self.est_traj) > 0:
            xe, ye, te = self.est_traj[-1]
            draw_robot(self.screen, self.cam, xe, ye, te, ORANGE, l=robot_l)

        # --- Overlay: indicador de waypoint no canto inferior-direito do MAPA ---
        if self.waypoints is not None and len(self.waypoints) > 0:
            wp_current = min(self.wp_idx + 1, len(self.waypoints))
            label = f"WP: {wp_current}/{len(self.waypoints)}"
            img = self.font.render(label, True, BLACK)

            pad = 6
            tx = self.cam.viewport[0] - img.get_width() - pad - 8
            ty = self.cam.viewport[1] - img.get_height() - pad - 8

            card = pg.Surface((img.get_width() + 2*pad, img.get_height() + 2*pad), pg.SRCALPHA)
            card.fill((255, 255, 255, 210))  # branco com alpha
            card.blit(img, (pad, pad))
            self.screen.blit(card, (tx - pad, ty - pad))
            pg.draw.rect(self.screen, BLACK, (tx - pad, ty - pad, card.get_width(), card.get_height()), 1)

        self.screen.set_clip(prev_clip)
        # HUD lateral (direita)
        self._draw_hud()

        if self.modal_open:
            self._draw_modal()
    
    def setup_hud_elements(
        self,
        btn_filelog,
        btn_graphs,
        textbox_x,
        textbox_y,
        btn_place_anchor,
        ):
        '''Configura referências aos elementos do HUD (botões, textboxes) para uso em eventos e desenho.
        Deve ser chamado pelo main_interactive após criar os botões/textboxes.'''
        self.btn_filelog = btn_filelog
        self.btn_graphs = btn_graphs
        self.textbox_x = textbox_x
        self.textbox_y = textbox_y
        self.btn_place_anchor = btn_place_anchor

    def layout_hud(self):
        """Atualiza posições (rects) do HUD antes de processar eventos e antes de desenhar."""
        cam_w = self.cam.viewport[0]
        sidebar_x = cam_w + 16
        y = 14
        LINE_H = 18

        # Header
        y += 30

        # Métricas: você desenha 7 linhas e depois +14
        y += LINE_H * 7 + 14

        # Controles: você desenha 7 linhas e depois +10
        y += LINE_H * 7 + 10

        # Âncoras (x,y): título + 22
        y += 22  # "Posicionar âncora (x, y):"
        # linha dos textboxes
        if self.textbox_x and self.textbox_y:
            self.textbox_x.rect.topleft = (sidebar_x, y)
            self.textbox_y.rect.topleft = (sidebar_x + self.textbox_x.rect.w + 10, y)
            y += self.textbox_x.rect.h + 8

        if self.btn_place_anchor:
            self.btn_place_anchor.rect.topleft = (sidebar_x, y)
            y += self.btn_place_anchor.rect.h + 16

        # Debug EKF (se tiver)
        if self.show_debug:
            y += (LINE_H * 4 + 10)

        # Ferramentas
        y += 26  # título "Ferramentas:" ocupa um pouco (mesmo offset do draw)
        if self.btn_filelog:
            self.btn_filelog.rect.topleft = (sidebar_x, y)
            y += 44

        if self.btn_graphs:
            self.btn_graphs.rect.topleft = (sidebar_x, y)
            y += 44

        if self.btn_gen_dataset:
            self.btn_gen_dataset.rect.topleft = (sidebar_x, y)
            y += 44
    # ------------------------------------------------------------------
    # Métodos auxiliares (internos)
    # ------------------------------------------------------------------

    def _handle_mouse_button_down(self, event: pg.event.Event) -> None:
        """Clique de mouse (zoom, pan, add/rem anchors, HUD)."""
        pass

    def _handle_mouse_motion(self, event: pg.event.Event) -> None:
        """Pan com botão do meio, se ativo."""
        pass

    def close(self):
        """Fecha logger e gráficos ao sair da SIM."""
        if self.file_logger:
            try:
                self.file_logger.close()
            except Exception:
                pass
            self.file_logger = None
            self.filelog_on = False

        # encerra janela de gráficos em tempo real
        from src.utils import stop_plot_process
        stop_plot_process(self.plot_state)
        
    def _draw_hud(self):
        '''
        Desenha o HUD lateral direito, com métricas, botões e textboxes.
        '''
        cam_w = self.cam.viewport[0]
        cam_h = self.cam.viewport[1]

        pg.draw.rect(
            self.screen,
            (245, 245, 245),
            (cam_w, 0, self.SIDE_W, cam_h)
        )

        sidebar_x = cam_w + 16
        y = 14
        LINE_H = 18

        # Header
        self._draw_hud_header(sidebar_x, y, LINE_H)
        y += 30

        # Métricas
        self._draw_hud_metrics(sidebar_x, y, LINE_H)
        y += LINE_H * 7 + 14

        # Controles
        self._draw_hud_controls(sidebar_x, y, LINE_H)
        y += LINE_H * 7 + 10

        # Âncoras (x, y)
        self._draw_hud_anchor_tools(sidebar_x, y)
        y += 90

        # Debug EKF
        if self.show_debug:
            self._draw_hud_debug(sidebar_x, y, LINE_H)
            y += LINE_H * 4 + 10

        # Ferramentas
        self._draw_hud_tools(sidebar_x, y)

    def _draw_hud_header(self, x, y, LINE_H):
        '''Desenha o título do HUD e separador abaixo.'''
        draw_text(self.screen, "BC-EKF — Simulador", x, y, self.bigfont)

    def _draw_hud_metrics(self, x, y, LINE_H):
        '''Desenha as métricas atuais (FPS, velocidade, erros) no HUD lateral.'''
        draw_text(self.screen, f"FPS: {self.clock.get_fps():5.1f}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Speed x: {self.speed_factor}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Autopilot: {'ON' if self.autopilot else 'OFF'} (SPACE)", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Âncoras: {self.anchors_dyn.shape[1]}", x, y, self.font)
        y += LINE_H
        gdop_txt = "inf" if (self.gdop == float("inf")) else (f"{self.gdop:.3f}" if np.isfinite(self.gdop) else "N/A")
        avg_txt = f"{self.gdop_avg:.3f}" if np.isfinite(self.gdop_avg) else "N/A"
        draw_text(self.screen, f"GDOP: {gdop_txt} | avg: {avg_txt}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Erro Pos (m): {self.pos_err:.3f}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Erro Heading (°): {self.head_err:.2f}", x, y, self.font)
        y += LINE_H
        pm = bool(self.robot_cfg.get("perfect_motion", False))
        draw_text(self.screen, f"Perfect Motion: {'ON' if pm else 'OFF'}", x, y, self.font)

    
    def _draw_hud_tools(self, x, y):
        '''Desenha as ferramentas (botões) do HUD lateral. O texto dos botões é atualizado dinamicamente (ex: log ON/OFF).'''
        draw_text(self.screen, "Ferramentas:", x, y, self.bigfont)
        y += 26

        # atualiza texto dinamicamente
        if self.btn_filelog:
            self.btn_filelog.text = "Log arquivo: ON" if self.filelog_on else "Log arquivo: OFF"
            self.btn_filelog.draw(self.screen)

        if self.btn_graphs:
            self.btn_graphs.draw(self.screen)

        if self.btn_gen_dataset:
            self.btn_gen_dataset.draw(self.screen)

    def _draw_hud_controls(self, x, y, LINE_H):
        '''Desenha as instruções de controle (teclas) no HUD lateral.'''
        draw_text(self.screen, "Controles:", x, y, self.bigfont); y += LINE_H
        draw_text(self.screen, "↑/↓ acel. linear   ←/→ acel. angular", x, y, self.font); y += LINE_H
        draw_text(self.screen, "[ / ] velocidade simulação", x, y, self.font); y += LINE_H
        draw_text(self.screen, "Scroll: zoom  |  Botão do meio: pan", x, y, self.font); y += LINE_H
        draw_text(self.screen, "C: limpar âncoras  |  B: âncoras padrão", x, y, self.font); y += LINE_H
        draw_text(self.screen, "ESC: voltar ao menu", x, y, self.font); y += LINE_H
        draw_text(self.screen, "P: perfect motion (sem ruído)", x, y, self.font); y += LINE_H

    def _draw_hud_anchor_tools(self, x, y):
        '''Desenha as ferramentas de posicionamento de âncoras (instruções, textboxes, botão) no HUD lateral.'''
        draw_text(self.screen, "Posicionar âncora (x, y):", x, y, self.font)
        y += 22

        if self.textbox_x and self.textbox_y and self.btn_place_anchor:
            dt = getattr(self, "_last_dt", 0.0)

            # NÃO mover rect aqui — o layout_hud() já posicionou
            self.textbox_x.update(dt)
            self.textbox_x.draw(self.screen)

            self.textbox_y.update(dt)
            self.textbox_y.draw(self.screen)

            self.btn_place_anchor.draw(self.screen)
    
    def _draw_hud_debug(self, x, y, LINE_H):
        '''Desenha métricas de debug do EKF (norma da inovação, NIS) no HUD lateral, se show_debug estiver ativo.'''
        draw_text(self.screen, "DEBUG EKF", x, y, self.bigfont); y += LINE_H

        if self.innov_norm is not None:
            draw_text(self.screen, f"||innov||: {self.innov_norm:.3f}", x, y, self.font)
        else:
            draw_text(self.screen, "||innov||: n/a", x, y, self.font)
        y += LINE_H

        if self.nis is not None:
            draw_text(self.screen, f"NIS: {self.nis:.3f}", x, y, self.font)
        else:
            draw_text(self.screen, "NIS: n/a", x, y, self.font)
        y += LINE_H

        draw_text(self.screen, "D: mostra/oculta debug", x, y, self.font)

    def _list_json_files(self, folder: str):
        '''Retorna lista de arquivos .json em um diretório, ordenada alfabeticamente. Retorna lista vazia se houver erro 
        (ex: diretório não existe).'''
        try:
            files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
            files.sort()
            return files
        except Exception:
            return []

    def _open_modal_load(self, which: str):
        '''Abre o modal de carregamento, preenchendo a dropdown com os arquivos disponíveis.
         O parâmetro "which" indica se é para carregar rotas ou âncoras, e ajusta o título e a lista de arquivos de acordo.'''
        self.modal_open = True
        self.modal_mode = "load_route" if which == "route" else "load_anchors"

        folder = self.routes_dir if which == "route" else self.anchors_dir
        opts = self._list_json_files(folder)

        self.modal_dropdown.options_all = list(opts)
        self.modal_dropdown.options_filtered = list(opts)

        # abre e ativa (pra já poder usar setinha/teclado)
        self.modal_dropdown.active = True
        self.modal_dropdown.dropdown_open = True if opts else False

        self.modal_namebox.active = False

    def _open_modal_save(self, which: str):
        '''Abre o modal de salvamento, preparando o campo de texto para o nome do arquivo. 
        O parâmetro "which" indica se é para salvar rotas ou âncoras, e ajusta o título de acordo.'''
        self.modal_open = True
        self.modal_mode = "save_route" if which == "route" else "save_anchors"

        self.modal_namebox.set_text("")
        self.modal_namebox.active = True
        self.modal_namebox.dropdown_open = False

        self.modal_dropdown.active = False
        self.modal_dropdown.dropdown_open = False

    def _close_modal(self):
        '''Fecha o modal de carregamento/salvamento, limpando estado e desativando elementos.'''
        self.modal_open = False
        self.modal_mode = None
        self.modal_dropdown.active = False
        self.modal_namebox.active = False

    def _apply_loaded_anchors(self, anchors_xy):
        '''Aplica as âncoras carregadas do arquivo, atualizando o estado compartilhado, a simulação e o desenho.'''
        # anchors_xy: list[(x,y)]
        if not self.shared_uwb:
            return

        # atualiza shared IN-PLACE pra manter referência
        self.shared_uwb.anchors_xy[:] = [(float(x), float(y)) for (x,y) in anchors_xy]
        self.shared_uwb.reindex_anchor_params()
        self.shared_uwb.sync_pipeline_from_state()

        # atualiza sim + draw
        self.anchors_dyn = self.shared_uwb.anchors_np3()
        self.sim.anchors = self.anchors_dyn

        # atualiza R conforme N
        n = self.anchors_dyn.shape[1]
        self.sim.R = (np.eye(2*n) * 0.0025) if n > 0 else (np.eye(2) * 1e6)

    def _save_anchors_to_file(self, name: str):
        """Salva âncoras + parâmetros completos."""
        if not self.shared_uwb:
            return
        
        # Usa o método to_dict() do shared_uwb
        payload = self.shared_uwb.to_dict()
        
        # Remove campos desnecessários
        payload.pop("seed", None)  # seed é do pipeline, não das âncoras
        
        path = os.path.join(self.anchors_dir, name + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        print(f"[SIM] Âncoras + parâmetros salvos: {path}")
        

    def _load_anchors_from_file(self, filename: str):
        """Carrega âncoras + parâmetros completos."""
        path = os.path.join(self.anchors_dir, filename)
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Carrega no shared_uwb usando from_dict()
            if self.shared_uwb:
                # Mantém seed atual
                current_seed = self.shared_uwb.pipeline.seed
                
                # Atualiza campos
                self.shared_uwb.anchors_xy = data.get("anchors_xy", [])
                
                if "tag_params" in data:
                    self.shared_uwb.tag_params = dict_to_node_params(data["tag_params"])
                
                if "anchor_params" in data:
                    self.shared_uwb.anchor_params = {
                        int(k): dict_to_node_params(v)
                        for k, v in data["anchor_params"].items()
                    }
                
                # Reaplica seed e sincroniza
                self.shared_uwb.pipeline.seed = current_seed
                self.shared_uwb.reindex_anchor_params()
                self.shared_uwb.sync_pipeline_from_state()
                
                # Atualiza visualização
                self.anchors_dyn = self.shared_uwb.anchors_np3()
            
            print(f"[SIM] Âncoras + parâmetros carregados: {path}")
        
        except Exception as e:
            print(f"[SIM] Erro ao carregar âncoras: {e}")

    def _save_route_to_file(self, name: str):
        '''Salva a rota atual (waypoints) em um arquivo JSON, com o nome fornecido.'''
        name = name.strip()
        if not name:
            return
        if not name.lower().endswith(".json"):
            name += ".json"

        wps = []
        if self.waypoints is not None and len(self.waypoints) > 0:
            wps = [[float(x), float(y)] for (x,y) in np.array(self.waypoints).tolist()]

        payload = {
            "waypoints": wps,
            "robot_config":  self.robot_cfg,
            "meta": {"count": len(wps)},
        }

        path = os.path.join(self.routes_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[SIM] Rota salva em: {path}")
        

    def _export_shared_uwb_complete(self, name: str):
        '''Exporta shared_uwb COMPLETO (incluindo seed) para backup.'''
        name = name.strip()
        if not name.lower().endswith(".json"):
            name += ".json"
        
        if not self.shared_uwb:
            return
        
        # Exporta tudo (incluindo seed)
        payload = shared_state_to_dict(self.shared_uwb)
        
        path = os.path.join(self.anchors_dir, "backups", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        print(f"[SIM] Backup completo salvo: {path}")

    def _load_route_from_file(self, filename: str):
        '''Carrega uma rota de um arquivo JSON, atualizando os waypoints, configuração do robô e ativando o autopilot.'''
        if not filename:
            return
        path = os.path.join(self.routes_dir, filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            wps = data.get("waypoints", [])

            cfg = data.get("robot_config", {})
            if isinstance(cfg, dict):
                self.robot_cfg.update(cfg)
                self._apply_robot_config_to_sim()

            if len(wps) >= 2:
                self.waypoints = np.array(wps, dtype=float)
                self.wp_idx = 0
                self.autopilot = True
            else:
                self.waypoints = None
                self.wp_idx = 0
                self.autopilot = False
                self._pm_seg_i = 0
                self._pm_s = 0.0

            print(f"[SIM] Rota carregada: {path}")

        except Exception as e:
            print(f"[SIM] Falha ao carregar rota ({path}): {e}")

    def _draw_modal(self):
        ''' Desenha o modal de carregamento/salvamento, 
        com backdrop, caixa, título, conteúdo (dropdown ou textbox) e botões OK/Cancel.'''
        cam_w = self.cam.viewport[0]
        cam_h = self.cam.viewport[1]

        # --- dimensões ---
        w = 460

        # altura dinâmica pro robot_cfg (rows + dropdown + botões)
        if self.modal_mode == "robot_cfg":
            rows_h = len(self.robot_rows) * 30
            h = 70 + rows_h + 40 + 55   # topo + rows + dropdown + área botões
        else:
            h = 170

        x = cam_w//2 - w//2
        y = self.topbar_h + 30
        self.modal_rect = pg.Rect(x, y, w, h)

        dt = getattr(self, "_last_dt", 0.0)
        self.modal_dropdown.update(dt)
        self.modal_namebox.update(dt)
        if getattr(self, "dataset_tag_dropdown", None):
            self.dataset_tag_dropdown.update(dt)

        overlay = pg.Surface((cam_w, cam_h), pg.SRCALPHA)
        overlay.fill((0,0,0,80))
        self.screen.blit(overlay, (0,0))

        pg.draw.rect(self.screen, (252,252,252), self.modal_rect)
        pg.draw.rect(self.screen, (170,170,170), self.modal_rect, 1)

        # --- ROBOT CFG ---
        if self.modal_mode == "robot_cfg":
            title = "Configurações do robô"
            self.screen.blit(self.bigfont.render(title, True, (20,20,20)), (x+12, y+10))

            cx = x + 20
            cy = y + 55

            # toggles
            for i, row in enumerate(self.robot_rows):
                row.rect.topleft = (cx, cy + i*30)
                row.box.topleft = (row.rect.x, row.rect.y + 2)
                row.draw(self.screen)

            # dropdown abaixo dos toggles
            dd_y = cy + len(self.robot_rows)*30 + 10
            self.screen.blit(self.font.render("Dataset Tag:", True, (20,20,20)), (cx, dd_y))

            if getattr(self, "dataset_tag_dropdown", None):
                self.dataset_tag_dropdown.rect.topleft = (cx + 120, dd_y - 3)
                self.dataset_tag_dropdown.draw(self.screen)

            # botões sempre no rodapé do modal
            by = y + h - 38
            self.btn_modal_ok.rect.topleft = (x + w - 200, by)
            self.btn_modal_cancel.rect.topleft = (x + w - 100, by)
            self.btn_modal_ok.draw(self.screen)
            self.btn_modal_cancel.draw(self.screen)
            return

        # --- resto (load/save route/anchors)  ---
        title = "Modal"
        if self.modal_mode == "load_route": title = "Carregar rota"
        if self.modal_mode == "save_route": title = "Salvar rota"
        if self.modal_mode == "load_anchors": title = "Carregar âncoras"
        if self.modal_mode == "save_anchors": title = "Salvar âncoras"
        self.screen.blit(self.bigfont.render(title, True, (20,20,20)), (x+12, y+10))

        cx = x + 12
        cy = y + 55
        if self.modal_mode in ("load_route", "load_anchors"):
            self.screen.blit(self.font.render("Arquivo:", True, (20,20,20)), (cx, cy))
            self.modal_dropdown.rect.topleft = (cx+70, cy-3)
            self.modal_dropdown.draw(self.screen)
        else:
            self.screen.blit(self.font.render("Nome:", True, (20,20,20)), (cx, cy))
            self.modal_namebox.rect.topleft = (cx+70, cy-3)
            self.modal_namebox.draw(self.screen)

        by = y + h - 38
        self.btn_modal_ok.rect.topleft = (x + w - 200, by)
        self.btn_modal_cancel.rect.topleft = (x + w - 100, by)
        self.btn_modal_ok.draw(self.screen)
        self.btn_modal_cancel.draw(self.screen)

        # conteúdo
        cx = x + 12
        cy = y + 55

        # botões
        by = y + h - 38
        self.btn_modal_ok.rect.topleft = (x + w - 200, by)
        self.btn_modal_cancel.rect.topleft = (x + w - 100, by)
        self.btn_modal_ok.draw(self.screen)
        self.btn_modal_cancel.draw(self.screen)

        if self.modal_mode in ("load_route", "load_anchors"):
            self.screen.blit(self.font.render("Arquivo:", True, (20,20,20)), (cx, cy))
            self.modal_dropdown.rect.topleft = (cx+70, cy-3)
            self.modal_dropdown.draw(self.screen)
        else:
            self.screen.blit(self.font.render("Nome:", True, (20,20,20)), (cx, cy))
            self.modal_namebox.rect.topleft = (cx+70, cy-3)
            self.modal_namebox.draw(self.screen)


    def _modal_confirm(self):
        '''Lógica executada ao clicar em OK no modal, dependendo do modo atual (carregar/salvar rota/âncoras).'''
        if self.modal_mode == "load_route":
            fn = self.modal_dropdown.text.strip()
            self._load_route_from_file(fn)
            self._close_modal()
            return

        if self.modal_mode == "save_route":
            name = self.modal_namebox.text.strip()
            self._save_route_to_file(name)
            self._close_modal()
            return

        if self.modal_mode == "load_anchors":
            fn = self.modal_dropdown.text.strip()
            self._load_anchors_from_file(fn)
            self._close_modal()
            return

        if self.modal_mode == "save_anchors":
            name = self.modal_namebox.text.strip()
            self._save_anchors_to_file(name)
            self._close_modal()
            return
        
        if self.modal_mode == "robot_cfg":
            # grava do UI pro estado
            self.robot_cfg["perfect_motion"] = self.robot_rows[0].value
            self.robot_cfg["perfect_odometry"] = self.robot_rows[1].value
            self.robot_cfg["perfect_uwb"] = self.robot_rows[2].value
            self.robot_cfg["perfect_filter_model"] = self.robot_rows[3].value
            self.robot_cfg["use_odometry"] = self.robot_rows[4].value

            if self.robot_cfg["perfect_uwb"]:
                self.robot_cfg["perfect_filter_model"] = True
                self.robot_rows[3].value = True

            # salva dropdown da tag do dataset
            if self.dataset_tag_dropdown:
                chosen = (self.dataset_tag_dropdown.text or "").strip().lower()
                if chosen in ("front", "rear", "mid"):
                    self.dataset_tag_mode = chosen
                else:
                    self.dataset_tag_mode = "mid"
                    self.dataset_tag_dropdown.set_text("mid")

            # aplica imediatamente no sim (próximo passo)
            self._apply_robot_config_to_sim()

            self._close_modal()
            return
                
    def _layout_topbar_buttons(self):
        '''Define as posições dos botões na topbar do mapa. 
        Deve ser chamado no draw antes de desenhar os botões,
        para garantir que os rects estejam atualizados para detecção de clique.'''
        x = 10
        y = 5
        gap = 10
        self.btn_tb_load_route.rect.topleft  = (x, y); x += self.btn_tb_load_route.rect.w + gap
        self.btn_tb_save_route.rect.topleft  = (x, y); x += self.btn_tb_save_route.rect.w + gap
        self.btn_tb_load_anchor.rect.topleft = (x, y); x += self.btn_tb_load_anchor.rect.w + gap
        self.btn_tb_save_anchor.rect.topleft = (x, y); x += self.btn_tb_save_anchor.rect.w + gap
        self.btn_tb_robot.rect.topleft = (x, y)
    
    def _open_modal_robot(self):
        '''Abre o modal de configuração do robô, que permite alternar opções como "perfect motion", "perfect odometry", etc.'''
        self.modal_open = True
        self.modal_mode = "robot_cfg"

        # cria as linhas (posições reais serão ajustadas no _draw_modal)
        self.robot_rows = [
            ToggleRow("Perfect Motion", (0,0,340,26), self.font, self.robot_cfg["perfect_motion"]),
            ToggleRow("Perfect Odometry", (0,0,340,26), self.font, self.robot_cfg["perfect_odometry"]),
            ToggleRow("Perfect UWB", (0,0,340,26), self.font, self.robot_cfg["perfect_uwb"]),
            ToggleRow("Perfect Filter Model", (0,0,340,26), self.font, self.robot_cfg["perfect_filter_model"]),
            ToggleRow("Use Odometry for UWB", (0,0,340,26), self.font, self.robot_cfg.get("use_odometry", True)),
        ]

        # dropdown dataset tag
        if self.dataset_tag_dropdown:
            opts = ["mid", "front", "rear"]
            self.dataset_tag_dropdown.options_all = list(opts)
            self.dataset_tag_dropdown.options_filtered = list(opts) 
            self.dataset_tag_dropdown.set_text(self.dataset_tag_mode)
            self.dataset_tag_dropdown.active = False
            self.dataset_tag_dropdown.dropdown_open = False

    def _apply_robot_config_to_sim(self) -> None:
        """
        Aplica as configs do robô ao estado da simulação.
        Quem usa essas flags é o SimulationScreen.update() ao chamar self.sim.step(...).
        Então aqui só garante consistência e reseta históricos quando muda modo.
        """

        self.perfect_motion = bool(self.robot_cfg.get("perfect_motion", False))
        
        # garante chaves
        for k in ("perfect_motion", "perfect_odometry", "perfect_uwb", "perfect_filter_model"):
            if k not in self.robot_cfg:
                self.robot_cfg[k] = False

        pm = bool(self.robot_cfg["perfect_motion"])
        po = bool(self.robot_cfg["perfect_odometry"])
        pu = bool(self.robot_cfg["perfect_uwb"])
        pf = bool(self.robot_cfg["perfect_filter_model"])

        print(f"[SIM] Robot cfg aplicado: motion={pm}, odo={po}, uwb={pu}, filter={pf}")

        self.sim.history_true.clear()
        self.sim.history_est.clear()
        self.sim.history_pred.clear()
        self.path_true.clear()
        self.path_pred.clear()
        self.path_est.clear()
        self.ts_hist.clear()
        self.pos_err_hist.clear()
        self.head_err_hist.clear()

    def _perfect_pose_from_waypoints(self, dt: float):
        '''Calcula a pose "perfeita" do robô seguindo os waypoints, avançando ao longo da rota com velocidade constante vref.'''
        if self.waypoints is None or len(self.waypoints) < 2:
            return None

        wps = self.waypoints
        i = self._pm_seg_i
        s = self._pm_s

        remaining = self._pm_vref * dt

        while remaining > 0 and i < len(wps) - 1:
            p0 = wps[i]
            p1 = wps[i+1]
            d = float(np.linalg.norm(p1 - p0))
            if d < 1e-9:
                i += 1
                s = 0.0
                continue

            left = d - s
            step = min(left, remaining)
            s += step
            remaining -= step

            if s >= d - 1e-9:
                i += 1
                s = 0.0

        # clamp
        if i >= len(wps) - 1:
            i = len(wps) - 2
            s = float(np.linalg.norm(wps[i+1] - wps[i]))

        p0 = wps[i]
        p1 = wps[i+1]
        d = float(np.linalg.norm(p1 - p0))
        t = 0.0 if d < 1e-9 else (s / d)
        pos = (1 - t) * p0 + t * p1

        theta = float(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

        self._pm_seg_i = i
        self._pm_s = s
        # wp_idx pro HUD (bolinha verde)
        self.wp_idx = min(i + 1, len(wps) - 1)

        return np.array([float(pos[0]), float(pos[1]), theta], dtype=float)

    def _reset_gdop_stats(self):
        self._gdop_sum = 0.0
        self._gdop_count = 0
        self._gdop_min = float("inf")
        self._gdop_max = float("-inf")
        self.gdop_avg = float("nan")
        self.gdop_hist.clear()

    def _reset_robot_to_route_start(self):
        """Posiciona o robô no primeiro waypoint e reinicia índices da rota."""
        if self.waypoints is None or len(self.waypoints) < 2:
            return False

        p0 = self.waypoints[0]
        p1 = self.waypoints[1]
        theta0 = float(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

        # move o robô (ground truth)
        if hasattr(self.sim, "robot"):
            self.sim.robot.x = float(p0[0])
            self.sim.robot.y = float(p0[1])
            self.sim.robot.theta = theta0

        if hasattr(self.sim, "x_true"):
            self.sim.x_true = np.array([float(p0[0]), float(p0[1]), theta0], dtype=float)

        # reinicia autopilot/índices
        self.wp_idx = 0
        self.autopilot = True
        self.v_cmd = 0.0
        self.w_cmd = 0.0

        # reset do perfect-motion progress (se estiver usando)
        self._pm_seg_i = 0
        self._pm_s = 0.0

        return True

    def _start_dataset_recording(self):
        """Inicia gravação de dataset e reinicia a rota."""
        if self.dataset_recording:
            return

        if self.waypoints is None or len(self.waypoints) < 2:
            print("[SIM] Falha ao gerar dataset: não há rota (grave/carregue uma rota).")
            return
        if not self.shared_uwb or not getattr(self.shared_uwb, "pipeline", None):
            print("[SIM] Falha ao gerar dataset: shared_uwb/pipeline não disponível.")
            return

        # reinicia rota SEMPRE (evita dataset vazio)
        ok = self._reset_robot_to_route_start()
        if not ok:
            print("[SIM] Falha ao gerar dataset: rota inválida.")
            return

        # reseed dos ruídos (cada dataset diferente)
        try:
            np.random.seed(None)
        except Exception:
            pass

        # arquivo de saída
        import time, os
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = getattr(self, "dataset_tag_mode", "mid")
        out_dir = os.path.join("resultados", "datasets")
        os.makedirs(out_dir, exist_ok=True)
        self.dataset_out_path = os.path.join(out_dir, f"dataset_{tag}_{stamp}.txt")

        self.dataset_fp = open(self.dataset_out_path, "w", encoding="utf-8")
        self.dataset_recording = True
        self._dataset_acc = 0.0

        print(f"[SIM] Gerando dataset (ao vivo): {self.dataset_out_path}")

    def _stop_dataset_recording(self):
        """Finaliza gravação."""
        if self.dataset_fp:
            try:
                self.dataset_fp.close()
            except Exception:
                pass
        self.dataset_fp = None
        self.dataset_recording = False
        print(f"[SIM] Dataset finalizado: {self.dataset_out_path}")