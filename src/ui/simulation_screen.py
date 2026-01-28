# src/ui/simulation_screen.py

from __future__ import annotations
import math
from typing import Optional, List, Tuple, Any, Dict
from dataclasses import dataclass
import numpy as np
import pygame as pg
import src.config as config
from src.scenarios import anchors_tectrol
from src.utils import push_plot_data, RunLogger, start_plot_process, stop_plot_process
from src.environment import draw_environment
from src.trajectory import Trajectory  
from src.ui.ui_elements import TextBoxDropdown  
from src.control.waypoint_controller import waypoint_controller
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_robot, draw_text
from src.ui.botton import Button

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

        # comandos de controle
        self.V_MAX = getattr(config, "V_MAX", 0.6)
        self.W_MAX = getattr(config, "W_MAX", 1.0)
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.accel_lin = 0.02
        self.accel_ang = 0.08

        # âncoras dinâmicas
        self.anchors_dyn = np.zeros((3, 0))

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

        for event in events:
            if event.type == pg.QUIT:
                # deixe o main decidir sair do programa inteiro
                actions.quite_app = True
                return actions

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

                    state = STATE_MENU


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

                elif event.key == pg.K_b:
                    self.anchors_dyn = anchors_tectrol.copy()
                    self.sim.anchors = self.anchors_dyn
                    self.sim.R = np.eye(2 * self.anchors_dyn.shape[1]) * 0.0025

                elif event.key == pg.K_r:
                    self.cam.reset_view()

                elif event.key == pg.K_g:
                    self.recording = not self.recording
                    if not self.recording and len(self.recorded_points) > 1:
                        self.waypoints = np.array(self.recorded_points, dtype=float)
                        self.wp_idx = 0
                        self.autopilot = True

                elif event.key == pg.K_RETURN:
                    if self.recording and len(self.recorded_points) > 1:
                        self.waypoints = np.array(self.recorded_points, dtype=float)
                        self.wp_idx = 0
                        self.autopilot = True
                        self.recording = False

                elif event.key == pg.K_DELETE:
                    self.recorded_points = []
                    

            if event.type == pg.MOUSEBUTTONDOWN:
                mx, my = event.pos

                if event.button == 1:  # botão esquerdo
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
                    else:
                        # >>> HUD (LMB)

                        if self.textbox_x and self.textbox_x.handle_event(event):
                            continue
                        if self.textbox_y and self.textbox_y.handle_event(event):
                            continue

                        if self.btn_filelog and self.btn_filelog.hit((mx, my)):
                            if not self.filelog_on:
                                from src.utils import RunLogger
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
                                print(f"Âncora adicionada em ({x}, {y})")
                            except ValueError:
                                print("Coordenadas inválidas para âncora.")
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
            self.v_cmd, self.w_cmd, self.wp_idx = waypoint_controller(self.sim.x_est, self.waypoints, self.wp_idx, v_max=0.25, w_max=0.8)

        # física
        for _ in range(self.speed_factor):
            self.sim.step(self.v_cmd, self.w_cmd, noisy=True)
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
        # mapa (esquerda)
        map_rect = pg.Rect(0, 0, self.cam.viewport[0], self.cam.viewport[1])
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
        if len(self.true_traj) > 0:
            xr, yr, tr = self.true_traj[-1]
            draw_robot(self.screen, self.cam, xr, yr, tr, BLACK)
        if len(self.est_traj) > 0:
            xe, ye, te = self.est_traj[-1]
            draw_robot(self.screen, self.cam, xe, ye, te, ORANGE)

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

        # HUD lateral (direita)
        self._draw_hud()

    # ------------------------------------------------------------------
    # Métodos auxiliares (internos)
    # ------------------------------------------------------------------

    def _handle_keydown(self, event: pg.event.Event) -> Optional[bool]:
        """Lógica de teclas únicas (ESC, D, SPACE, [, ], C, B, R, G, ENTER, DEL)."""
        # Aqui você pode copiar o miolo de:
        #   elif event.type == pg.KEYDOWN:
        #       if textbox_x ... (consumed)
        #       if event.key == pg.K_ESCAPE: ...
        #       elif event.key == pg.K_d: ...
        #       ...
        return None

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
        cam_w = self.cam.viewport[0]
        cam_h = self.cam.viewport[1]

        pg.draw.rect(
            self.screen,
            (245, 245, 245),
            (cam_w, 0, self.SIDE_W, cam_h)
        )

        sidebar_x = cam_w + 16
        y = 18
        LINE_H = 22

        # Header
        self._draw_hud_header(sidebar_x, y, LINE_H)
        y += 34

        # Métricas
        self._draw_hud_metrics(sidebar_x, y, LINE_H)
        y += LINE_H * 6 + 10

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
        draw_text(self.screen, "BC-EKF — Simulador", x, y, self.bigfont)

    def _draw_hud_metrics(self, x, y, LINE_H):
        draw_text(self.screen, f"FPS: {self.clock.get_fps():5.1f}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Speed x: {self.speed_factor}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Autopilot: {'ON' if self.autopilot else 'OFF'} (SPACE)", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Âncoras: {self.anchors_dyn.shape[1]}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Erro Pos (m): {self.pos_err:.3f}", x, y, self.font)
        y += LINE_H
        draw_text(self.screen, f"Erro Heading (°): {self.head_err:.2f}", x, y, self.font)
    
    def _draw_hud_tools(self, x, y):
        draw_text(self.screen, "Ferramentas:", x, y, self.bigfont)
        y += 26
        self.btn_filelog = Button(
            rect=(0, 0, 190, 30),
            text="Log arquivo: ON" if self.filelog_on else "Log arquivo: OFF",
            font=self.font,
            bg=(235, 250, 235),
        )
        self.btn_graphs = Button(
            rect=(0, 0, 190, 30),
            text="Mostrar Gráficos",
            font=self.font,
            bg=(235, 235, 250),
        )
        
        if self.btn_filelog:
            self.btn_filelog.rect.topleft = (x, y)
            self.btn_filelog.draw(self.screen)

        y += 44

        if self.btn_graphs:
            self.btn_graphs.rect.topleft = (x, y)
            self.btn_graphs.draw(self.screen)

    def _draw_hud_controls(self, x, y, LINE_H):
        draw_text(self.screen, "Controles:", x, y, self.bigfont); y += LINE_H
        draw_text(self.screen, "↑/↓ acel. linear   ←/→ acel. angular", x, y, self.font); y += LINE_H
        draw_text(self.screen, "[ / ] velocidade simulação", x, y, self.font); y += LINE_H
        draw_text(self.screen, "Scroll: zoom  |  Botão do meio: pan", x, y, self.font); y += LINE_H
        draw_text(self.screen, "C: limpar âncoras  |  B: âncoras padrão", x, y, self.font); y += LINE_H
        draw_text(self.screen, "ESC: voltar ao menu", x, y, self.font)

    def _draw_hud_anchor_tools(self, x, y):
        draw_text(self.screen, "Posicionar âncora (x, y):", x, y, self.font)
        y += 22

        if self.textbox_x and self.textbox_y and self.btn_place_anchor:
            dt = getattr(self, "_last_dt", 0.0)
            self.textbox_x.rect.topleft = (x, y)
            self.textbox_x.update(dt)
            self.textbox_x.draw(self.screen)

            self.textbox_y.rect.topleft = (x + self.textbox_x.rect.w + 10, y)
            self.textbox_y.update(dt)
            self.textbox_y.draw(self.screen)

            y += self.textbox_x.rect.h + 8
            self.btn_place_anchor.rect.topleft = (x, y)
            self.btn_place_anchor.draw(self.screen)
    
    def _draw_hud_debug(self, x, y, LINE_H):
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

    def setup_hud_elements(
        self,
        btn_filelog,
        btn_graphs,
        textbox_x,
        textbox_y,
        btn_place_anchor,
    ):
        self.btn_filelog = btn_filelog
        self.btn_graphs = btn_graphs
        self.textbox_x = textbox_x
        self.textbox_y = textbox_y
        self.btn_place_anchor = btn_place_anchor

    def layout_hud(self):
        """Atualiza posições dos elementos do HUD (rects) antes de processar eventos."""
        cam_w = self.cam.viewport[0]
        sidebar_x = cam_w + 16

        # você precisa manter o MESMO y usado no draw
        y = 18
        y += 34                 # header
        y += 22*6 + 10          # metrics (ajuste se seu layout mudou)
        y += 22*6 + 10          # controls (ajuste também)

        # âncoras
        y += 22                 # título "Posicionar âncora"
        y += 22                 # linha abaixo do título

        if self.textbox_x and self.textbox_y:
            self.textbox_x.rect.topleft = (sidebar_x, y)
            self.textbox_y.rect.topleft = (sidebar_x + self.textbox_x.rect.w + 10, y)

            y += self.textbox_x.rect.h + 8

        if self.btn_place_anchor:
            self.btn_place_anchor.rect.topleft = (sidebar_x, y)

