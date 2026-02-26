# src/ui/algo_test_screen.py
"""
Tela de comparação de algoritmos de localização UWB.

Dois modos:
  STEP    — robô se move em tempo real, cada medição UWB alimenta todos os
             algoritmos selecionados simultaneamente. Trilhas coloridas no mapa.
  DATASET — carrega um .jsonl (simulação gravada) ou .txt (ensaio real do lab)
             e roda todos os algoritmos em batch, exibindo resultados.

Algoritmos disponíveis (registry em algoritmo_step.py):
  trilaterate3d, lms, gauss_newton, lmsp, bc_ekf

Integração com main_interactive.py:
  state = STATE_ALGO_TEST  →  algo_screen.handle_events / update / draw
  ESC                      →  retorna STATE_MENU via actions.go_to_menu
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pygame as pg

from src.ui.botton import Button
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_robot, draw_text
from src.uwb.algoritmos_step import (
    ALGORITMOS, NOMES_UI, criar_localizadores, LocalizadorBase
)
from src.uwb.algoritmos_estaticos import (
    ANCHORS_LAB_CBA, SALA_LAB, carregar_ensaio_lab, run_batch
)
from src.uwb.uwb_sim import UwbSimPipeline
from src.simulator import Simulator
from src.trajectory import Trajectory
import src.config as config

# Cores ############################################################################
WHITE   = (255, 255, 255)
BLACK   = (20,  20,  20)
GRAY    = (220, 220, 220)
GRAY_D  = (150, 150, 150)
BLUE    = (40,  90,  210)
GREEN   = (40,  160, 60)
ORANGE  = (250, 140, 30)
RED     = (210, 40,  40)
BG_HUD  = (245, 247, 252)

# Cores dos algoritmos 
ALGO_COLORS: Dict[str, tuple] = {
    "trilaterate3d": (220, 50,  50),   # vermelho
    "lms":           (160, 30,  180),  # magenta
    "gauss_newton":  (30,  30,  30),   # preto
    "lmsp":          (30,  80,  200),  # azul
    "bc_ekf":        (255, 140, 0),    # laranja
}

ALGO_ORDER = ["trilaterate3d", "lms", "gauss_newton", "lmsp", "bc_ekf"]

MODE_STEP    = "step"
MODE_DATASET = "dataset"


@dataclass
class AlgoActions:
    go_to_menu: bool = False
    quit_app:   bool = False


# ###########################################################################
class AlgoTestScreen:
    """
    Tela de comparação de algoritmos de localização.

    Responsabilidades:
      - Gerenciar modo (STEP / DATASET)
      - Selecionar quais algoritmos rodar
      - No modo STEP: andar com o robô, gerar UWB, alimentar algoritmos
      - No modo DATASET: carregar arquivo, rodar batch, exibir resultados
      - Desenhar mapa com trilhas coloridas por algoritmo
      - HUD lateral com controles e métricas
    """

    def __init__(
        self,
        screen:     pg.Surface,
        cam:        Any,
        clock:      pg.time.Clock,
        font:       pg.font.Font,
        bigfont:    pg.font.Font,
        side_width: int,
        anchors:    Optional[np.ndarray] = None,
    ) -> None:
        self.screen     = screen
        self.cam        = cam
        self.clock      = clock
        self.font       = font
        self.bigfont    = bigfont
        self.SIDE_W     = side_width

        # Modo atual: STEP ou DATASET
        self.mode: str = MODE_STEP

        # Algoritmos selecionados
        # Padrão: todos exceto bc_ekf (que exige odometria)
        self.selected: Dict[str, bool] = {
            "trilaterate3d": True,
            "lms":           True,
            "gauss_newton":  True,
            "lmsp":          False,   # exige desvio padrão; ligado só se disponível
            "bc_ekf":        False,   # requer odometria; separado
        }

        # Âncoras 
        # Usa âncoras passadas (simulação) ou preset do lab (dataset)
        self._anchors_sim = anchors if (anchors is not None and anchors.shape[1] > 0) \
                            else None
        self._anchors_lab = ANCHORS_LAB_CBA          # (8,3) — preset lab CBA

        # Estado STEP 
        self._sim:         Optional[Simulator] = None
        self._pipeline:    Optional[UwbSimPipeline] = None
        self._localizadores: Dict[str, LocalizadorBase] = {}
        self._running:     bool = False
        self._step_count:  int  = 0

        # Trajetória verdadeira e estimativas por algoritmo
        self._trail_true: List[tuple] = []
        self._trails:     Dict[str, List[tuple]] = {k: [] for k in ALGO_ORDER}

        # Estado DATASET 
        self._dataset_path:  Optional[str] = None
        self._batch_dists:   Optional[np.ndarray] = None   # (M, N)
        self._batch_devs:    Optional[np.ndarray] = None   # (M, N) ou None
        self._batch_results: Optional[dict] = None          # de run_batch()
        self._dataset_label: str = "Nenhum arquivo carregado"
        self._dataset_anchors: Optional[np.ndarray] = None  # (N,3) para batch

        # Mensagem de status 
        self._msg:   str   = ""
        self._msg_t: float = 0.0

        # Waypoints padrão (modo STEP) 
        self._waypoints = np.array(
            Trajectory.square(size=6, start=(-3, -3)).waypoints
        )
        self._wp_idx: int = 0
        self.speed_factor: int = 3

        # Botões HUD 
        self._build_hud()

    #########################################################################
    # Construção do HUD
    #########################################################################
    def _build_hud(self) -> None:
        sx = self.screen.get_width() - self.SIDE_W + 12
        y  = 10

        self.btn_back = Button(
            (sx, y, self.SIDE_W - 24, 32),
            "← Menu",  self.font, bg=(245,245,245), fg=(80,80,80)
        )
        y += 42

        # Modo
        self.btn_mode = Button(
            (sx, y, self.SIDE_W - 24, 32),
            self._mode_label(), self.bigfont,
            bg=(235, 240, 255), fg=BLUE, border=BLUE
        )
        y += 42

        # Botões de algoritmo (toggles)
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

        # Controles de execução
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

        # Dataset
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
        y += 42

        self._hud_metrics_y = y   # posição onde começam as métricas dinâmicas

    ###########################################################################
    # Handle events
    ###########################################################################
    def handle_events(self, events: list) -> AlgoActions:
        actions = AlgoActions()

        for event in events:
            if event.type == pg.QUIT:
                actions.quit_app = True

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._stop_sim()
                    actions.go_to_menu = True
                elif event.key == pg.K_SPACE:
                    self._toggle_run()

            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                if self.btn_back.hit(pos):
                    self._stop_sim()
                    actions.go_to_menu = True

                elif self.btn_mode.hit(pos):
                    self._toggle_mode()

                elif self.btn_start.hit(pos):
                    self._toggle_run()

                elif self.btn_clear.hit(pos):
                    self._clear_trails()

                elif self.btn_export.hit(pos):
                    self._export_csv()

                elif self.btn_load_dataset.hit(pos):
                    self._try_load_dataset()

                elif self.btn_run_batch.hit(pos):
                    self._run_batch()

                else:
                    for nome, btn in self._btn_algos.items():
                        if btn.hit(pos):
                            self.selected[nome] = not self.selected[nome]
                            self._refresh_algo_buttons()
                            break

        return actions

    ####################################
    # Update (modo STEP)
    ####################################
    def update(self, dt: float) -> None:
        if self.mode != MODE_STEP or not self._running:
            return

        if self._sim is None:
            self._init_step_sim()

        # Executa N sub-steps por frame (speed_factor)
        for _ in range(self.speed_factor):
            self._do_step()

        # Decai mensagem de status
        if self._msg_t > 0:
            self._msg_t -= dt

    def _do_step(self) -> None:
        """Um passo da simulação: move robô, mede UWB, alimenta algoritmos."""
        sim = self._sim

        # Controlador de waypoint simples
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

        # Step do simulador (move robô + gera z_k pelo pipeline TWR)
        result = sim.step(v_cmd, w_cmd, noisy=True)

        x_true, y_true, th_true = result["true"]
        self._trail_true.append((x_true, y_true))

        # Medições UWB geradas pelo pipeline: z_k tem shape (2*N,)
        # Para algoritmos estáticos: usa só tag frontal → z_k[0::2], shape (N,)
        if sim.anchors is not None and sim.anchors.shape[1] > 0:
            # Re-gera distâncias para a posição verdadeira com ruído
            # (já está em sim.last_meas_meta mas z_k está guardado internamente)
            anchors_Nx3 = sim.anchors.T      # (N, 3)
            z_full = self._get_last_zk(sim, result)  # (2*N,) ou None
            if z_full is not None and len(z_full) == 2 * anchors_Nx3.shape[0]:
                d_front = z_full[0::2]   # (N,) — distância tag frontal → âncora i
                dev_arr = None           # simulador não exporta desvios ainda

                for nome, loc in self._localizadores.items():
                    if nome == "bc_ekf":
                        # BC-EKF usa z_full (2*N) e odometria
                        loc.set_odometry(v_cmd, w_cmd)
                        pos = loc.step(z_full, dev_arr)
                    else:
                        pos = loc.step(d_front, dev_arr)

                    if pos is not None:
                        self._trails[nome].append((float(pos[0]), float(pos[1])))

        self._step_count += 1

    def _get_last_zk(self, sim: Simulator, result: dict) -> Optional[np.ndarray]:
        """Recupera z_k da última medição do simulador."""
        # O Simulator não expõe z_k diretamente; re-gera via pipeline
        if self._pipeline is not None and sim.anchors is not None:
            x, y, th = result["true"]
            z_k = self._pipeline.measure([x, y, th], sim.anchors, sim.l, sim.z_c)
            # measure() retorna só z_k quando return_meta=False, shape (2*N,)
            # mas a assinatura atual retorna tuple quando return_meta=False também?
            # Corrigindo: return_meta=False → retorna ndarray
            return z_k if isinstance(z_k, np.ndarray) else None
        return None

    ######################################
    # Draw
    ######################################
    def draw(self) -> None:
        self.screen.fill(WHITE)
        map_w = self.cam.viewport[0]

        # Mapa de fundo: grid + eixos
        draw_grid(self.screen, self.cam)

        # Âncoras ativas
        if self.mode == MODE_STEP and self._sim is not None and self._sim.anchors is not None:
            draw_anchors(self.screen, self.cam, self._sim.anchors)
        elif self.mode == MODE_DATASET and self._dataset_anchors is not None:
            # converte (N,3) → (3,N) para draw_anchors
            draw_anchors(self.screen, self.cam, self._dataset_anchors.T)

        draw_axes(self.screen, self.cam, self.font)

        # Waypoints (modo step)
        if self.mode == MODE_STEP and len(self._waypoints) > 1:
            draw_path(self.screen, self.cam,
                      [tuple(p) for p in self._waypoints],
                      GRAY_D, 1, dashed=True)

        # Trilha verdadeira
        if len(self._trail_true) > 1:
            draw_path(self.screen, self.cam, self._trail_true, BLACK, 2)

        # Trilhas por algoritmo
        for nome in ALGO_ORDER:
            if not self.selected.get(nome, False):
                continue
            trail = self._trails.get(nome, [])
            if len(trail) > 1:
                draw_path(self.screen, self.cam, trail, ALGO_COLORS[nome], 2)

            # Ponto atual (último)
            if trail:
                sx, sy = self.cam.world_to_screen(*trail[-1])
                pg.draw.circle(self.screen, ALGO_COLORS[nome], (sx, sy), 5)
                pg.draw.circle(self.screen, BLACK, (sx, sy), 5, 1)

        # Robô verdadeiro (modo step)
        if self.mode == MODE_STEP and len(self._trail_true) > 0:
            x, y = self._trail_true[-1]
            th = self._sim.robot.theta if self._sim else 0.0
            draw_robot(self.screen, self.cam, x, y, th, BLACK,
                       l=getattr(self._sim, 'l', 0.325) if self._sim else 0.325)

        # Resultados batch (modo dataset): pontos dispersos
        if self.mode == MODE_DATASET and self._batch_results:
            self._draw_batch_scatter()

        # Borda da área do mapa
        pg.draw.rect(self.screen, GRAY_D,
                     pg.Rect(0, 0, map_w, self.screen.get_height()), 1)

        # HUD lateral
        self._draw_hud()

        pg.display.flip()

    def _draw_batch_scatter(self) -> None:
        """Pontos do batch coloridos por algoritmo."""
        for nome, res in self._batch_results.items():
            if not self.selected.get(nome, False):
                continue
            pos = res["posicoes"]   # (M, 3)
            cor = ALGO_COLORS.get(nome, GRAY_D)
            for i in range(0, len(pos), max(1, len(pos)//500)):
                sx, sy = self.cam.world_to_screen(pos[i, 0], pos[i, 1])
                pg.draw.circle(self.screen, cor, (sx, sy), 2)

    def _draw_hud(self) -> None:
        map_w = self.cam.viewport[0]
        hud_x = map_w
        hud_w = self.SIDE_W

        # Fundo do HUD
        pg.draw.rect(self.screen, BG_HUD,
                     pg.Rect(hud_x, 0, hud_w, self.screen.get_height()))
        pg.draw.line(self.screen, GRAY_D,
                     (hud_x, 0), (hud_x, self.screen.get_height()), 1)

        # Botões
        self.btn_back.draw(self.screen)
        self.btn_mode.draw(self.screen)
        for btn in self._btn_algos.values():
            btn.draw(self.screen)

        # Oculta botões de dataset no modo step e vice-versa
        if self.mode == MODE_STEP:
            self.btn_start.draw(self.screen)
            self.btn_clear.draw(self.screen)
            self.btn_export.draw(self.screen)
        else:
            self.btn_load_dataset.draw(self.screen)
            self.btn_run_batch.draw(self.screen)
            # Label do arquivo carregado
            lbl = self.font.render(self._dataset_label, True, GRAY_D)
            self.screen.blit(lbl, (
                hud_x + 12,
                self.btn_run_batch.rect.bottom + 6
            ))

        # Métricas por algoritmo
        y = self._hud_metrics_y + (40 if self.mode == MODE_DATASET else 0)
        y += 12
        for nome in ALGO_ORDER:
            if not self.selected.get(nome, False):
                continue
            cor = ALGO_COLORS[nome]
            n_pts = len(self._trails.get(nome, []))

            # RMSE se temos ground truth (step)
            rmse_txt = ""
            if self.mode == MODE_STEP and self._localizadores.get(nome):
                loc = self._localizadores[nome]
                if self._trail_true and loc.historico:
                    # Alinha comprimentos
                    n = min(len(self._trail_true), len(loc.historico))
                    gt = np.array(self._trail_true[:n])
                    h  = np.array(loc.historico[:n])
                    err = h[:, :2] - gt
                    rmse = float(np.sqrt(np.mean(err**2)))
                    rmse_txt = f"  RMSE={rmse:.3f}m"
            elif self.mode == MODE_DATASET and self._batch_results and nome in self._batch_results:
                r = self._batch_results[nome]
                if r["rmse_xy"] is not None:
                    rmse_txt = f"  RMSE={r['rmse_xy']:.3f}m"

            label = f"{NOMES_UI.get(nome, nome)}{rmse_txt}"
            surf = self.font.render(label, True, cor)

            # Quadrado de cor
            pg.draw.rect(self.screen, cor, pg.Rect(hud_x + 12, y + 4, 10, 10))
            self.screen.blit(surf, (hud_x + 28, y))
            y += 22

        # Contador de steps / amostras
        if self.mode == MODE_STEP:
            step_txt = f"Steps: {self._step_count}"
            s = self.font.render(step_txt, True, GRAY_D)
            self.screen.blit(s, (hud_x + 12, y + 8))
        elif self.mode == MODE_DATASET and self._batch_dists is not None:
            s = self.font.render(f"Amostras: {len(self._batch_dists)}", True, GRAY_D)
            self.screen.blit(s, (hud_x + 12, y + 8))

        # Mensagem temporária
        if self._msg and self._msg_t > 0:
            msg_surf = self.font.render(self._msg, True, BLUE)
            self.screen.blit(msg_surf, (
                hud_x + 12,
                self.screen.get_height() - 30
            ))

    ###########################################
    # Inicialização e controle do modo STEP
    ###########################################
    def _init_step_sim(self) -> None:
        """Cria Simulator + UwbSimPipeline + localizadores."""
        anchors = self._anchors_sim
        if anchors is None or anchors.shape[1] == 0:
            # Fallback: 4 âncoras em quadrado 8x8
            anchors = np.array([
                [0.0, 0.0, 1.0],
                [8.0, 0.0, 1.0],
                [8.0, 8.0, 1.0],
                [0.0, 8.0, 1.0],
            ]).T  # (3, 4)

        N = anchors.shape[1]
        Q = np.diag([1e-4, 1e-4, 1e-4])
        R = np.eye(2 * N) * (0.05**2)

        self._pipeline = UwbSimPipeline.from_defaults(seed=42)

        self._sim = Simulator(
            anchors     = anchors,
            baseline    = getattr(config, "UWB_BASELINE", 0.65),
            z_c         = 0.5,
            Q           = Q,
            R           = R,
            dt          = getattr(config, "TIME_STEP", 0.05),
            config      = config,
            uwb_pipeline = self._pipeline,
        )

        # Cria localizadores apenas para os algoritmos selecionados
        algos_selecionados = [k for k, v in self.selected.items() if v]
        anchors_Nx3 = anchors.T   # (N, 3)

        self._localizadores = {}
        for nome in algos_selecionados:
            if nome == "bc_ekf":
                cls = ALGORITMOS["bc_ekf"]
                self._localizadores[nome] = cls(
                    anchors_Nx3,
                    baseline = getattr(config, "UWB_BASELINE", 0.65),
                    z_c = 0.5,
                    dt  = getattr(config, "TIME_STEP", 0.05),
                    Q   = Q,
                    R   = R,
                )
            elif nome in ALGORITMOS:
                self._localizadores[nome] = ALGORITMOS[nome](anchors_Nx3)

        self._wp_idx     = 0
        self._step_count = 0
        self._set_msg("Simulação iniciada")

    def _toggle_run(self) -> None:
        if not self._running:
            if self._sim is None:
                self._init_step_sim()
            self._running = True
            self.btn_start.text = "⏸  Pausar"
            self.btn_start.bg   = (255, 245, 230)
            self.btn_start.fg   = ORANGE
            self.btn_start.border = ORANGE
        else:
            self._running = False
            self.btn_start.text = "▶  Continuar"
            self.btn_start.bg   = (235, 250, 235)
            self.btn_start.fg   = GREEN
            self.btn_start.border = GREEN

    def _stop_sim(self) -> None:
        self._running = False
        self._sim     = None
        self._pipeline = None
        self._localizadores = {}
        self.btn_start.text = "▶  Iniciar"
        self.btn_start.bg   = (235, 250, 235)
        self.btn_start.fg   = GREEN
        self.btn_start.border = GREEN

    def _clear_trails(self) -> None:
        self._trail_true = []
        self._trails = {k: [] for k in ALGO_ORDER}
        for loc in self._localizadores.values():
            loc.reset()
        self._step_count = 0
        self._set_msg("Trilhas apagadas")

    ################################################
    # Modo DATASET
    ################################################
    def _try_load_dataset(self) -> None:
        """
        Tenta carregar um arquivo de dataset.
        Procura primeiro por .txt do lab, depois por .jsonl gravado.
        """
        # Tentativa 1: arquivo do ensaio real
        lab_candidates = [
            "Testes_LAB/Testes_LAB/ENSAIO_10_05/Dados_mesclados.txt",
            "datasets/Dados_mesclados.txt",
            "Dados_mesclados.txt",
        ]
        for path in lab_candidates:
            if os.path.isfile(path):
                try:
                    dists, devs = carregar_ensaio_lab(path)
                    self._batch_dists   = dists
                    self._batch_devs    = devs
                    self._dataset_path  = path
                    self._dataset_label = f"Lab: {os.path.basename(path)} ({len(dists)} amostras)"
                    self._dataset_anchors = self._anchors_lab   # (8,3)
                    self.selected["lmsp"] = True               # lab tem desvios
                    self._refresh_algo_buttons()
                    self._set_msg(f"Carregado: {os.path.basename(path)}")
                    return
                except Exception as e:
                    self._set_msg(f"Erro ao carregar lab: {e}")
                    return

        # Tentativa 2: procura jsonl mais recente em datasets/
        jsonl_dir = "datasets"
        if os.path.isdir(jsonl_dir):
            files = sorted(
                [f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")],
                reverse=True
            )
            if files:
                path = os.path.join(jsonl_dir, files[0])
                try:
                    self._load_jsonl(path)
                    return
                except Exception as e:
                    self._set_msg(f"Erro ao carregar jsonl: {e}")
                    return

        self._set_msg("Nenhum dataset encontrado (lab .txt ou .jsonl)")

    def _load_jsonl(self, path: str) -> None:
        """Carrega dataset .jsonl gravado pelo simulador."""
        import json
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "z_k" in obj and obj["z_k"]:
                    rows.append(obj["z_k"])

        if not rows:
            self._set_msg("JSONL sem medições z_k válidas")
            return

        dists_2N = np.array(rows, dtype=float)    # (M, 2*N)
        # Pega só tag frontal → (M, N)
        dists = dists_2N[:, 0::2]
        self._batch_dists   = dists
        self._batch_devs    = None
        self._dataset_path  = path
        self._dataset_label = f"Sim: {os.path.basename(path)} ({len(dists)} amostras)"
        self._dataset_anchors = (
            self._anchors_sim.T if self._anchors_sim is not None
            else self._anchors_lab
        )
        self.selected["lmsp"] = False   # sem desvios no jsonl
        self._refresh_algo_buttons()
        self._set_msg(f"Carregado: {os.path.basename(path)}")

    def _run_batch(self) -> None:
        if self._batch_dists is None:
            self._set_msg("Carregue um dataset primeiro")
            return
        if self._dataset_anchors is None:
            self._set_msg("Âncoras não definidas")
            return

        algos = [k for k, v in self.selected.items()
                 if v and k != "bc_ekf"]   # bc_ekf precisa odometria

        if not algos:
            self._set_msg("Selecione pelo menos um algoritmo")
            return

        t0 = time.perf_counter()
        try:
            self._batch_results = run_batch(
                anchors_Nx3 = self._dataset_anchors,
                distances   = self._batch_dists,
                deviations  = self._batch_devs,
                algoritmos  = algos,
                p_true      = None,    # sem ground truth no lab
            )
            dt = time.perf_counter() - t0
            n = len(self._batch_dists)
            self._set_msg(f"Batch concluído: {n} amostras em {dt:.2f}s")
        except Exception as e:
            self._set_msg(f"Erro no batch: {e}")

    #################################
    # Export CSV
    #################################
    def _export_csv(self) -> None:
        os.makedirs("resultados", exist_ok=True)
        ts = int(time.time())
        rows = []

        # Ground truth
        for i, (x, y) in enumerate(self._trail_true):
            rows.append({"step": i, "algoritmo": "ground_truth", "x": x, "y": y})

        # Algoritmos
        for nome in ALGO_ORDER:
            for i, (x, y) in enumerate(self._trails.get(nome, [])):
                rows.append({"step": i, "algoritmo": nome, "x": x, "y": y})

        if not rows:
            self._set_msg("Nada para exportar")
            return

        path = f"resultados/algo_compare_{ts}.csv"
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "algoritmo", "x", "y"])
            writer.writeheader()
            writer.writerows(rows)

        self._set_msg(f"Exportado: {os.path.basename(path)}")

    #################################
    # Auxiliares de UI
    #################################
    def _toggle_mode(self) -> None:
        self._stop_sim()
        self._batch_results = None
        self.mode = MODE_DATASET if self.mode == MODE_STEP else MODE_STEP
        self.btn_mode.text = self._mode_label()
        self._set_msg(f"Modo: {self.mode.upper()}")

    def _mode_label(self) -> str:
        return f"Modo: {'Step (tempo real)' if self.mode == MODE_STEP else 'Dataset (batch)'}"

    def _algo_label(self, nome: str) -> str:
        tick = "✔" if self.selected[nome] else "○"
        return f"{tick}  {NOMES_UI.get(nome, nome)}"

    def _algo_bg(self, nome: str) -> tuple:
        if self.selected[nome]:
            r, g, b = ALGO_COLORS[nome]
            return (min(255, r + 180), min(255, g + 180), min(255, b + 180))
        return (245, 245, 245)

    def _refresh_algo_buttons(self) -> None:
        for nome, btn in self._btn_algos.items():
            btn.text   = self._algo_label(nome)
            btn.bg     = self._algo_bg(nome)

    def _set_msg(self, msg: str, duration: float = 3.0) -> None:
        self._msg   = msg
        self._msg_t = duration

    def close(self) -> None:
        self._stop_sim()