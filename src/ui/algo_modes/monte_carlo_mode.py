from __future__ import annotations
from typing import Any
import pygame as pg
import threading
from copy import deepcopy
import os
import json
import csv
import numpy as np
from pathlib import Path
import time

from src.experiments.monte_carlo_runner import MonteCarloRunner, MonteCarloConfig
from src.uwb.algoritmos_step import NOMES_UI
from src.ui.botton import Button
from src.ui.drawing import draw_anchors, draw_axes, draw_grid, draw_path, draw_robot, draw_text
from src.environment.environment import Environment, draw_environment
from src.ui.algo_modes.shared import (
    ALGO_COLORS,
    ALGO_ORDER,
    MODE_DATASET,
    MODE_MONTE_CARLO,
    MODE_STEP,
    WHITE,
    default_selected,
)



class MonteCarloMode:
    """
    Plano 1:
    - Desenho do Monte Carlo já fica fora do legacy
    - Estado ainda vem do legacy (mc_config, mc_results, mc_progress, botões etc.)
    - Eventos/update ainda passam pelo legacy por enquanto
    """
        ################## TP INIT   ##################
    def __init__(self, host: Any) -> None:
        self.host = host
        self._legacy_ref = None

        # Estado Próprio do MC 
        self.mc_running = False
        self.mc_progress = None
        self.mc_results = None
        self.mc_accumulated_points = {}
        self.mc_thread = None

        # Configuração do MC
        self.mc_config = None

        # estado do modal/config do MC 
        self.mc_config_modal_open = False
        self.mc_dropdown_route_open = False
        self.mc_dropdown_anchors_open = False

        self.mc_available_routes = []
        self.mc_available_anchors = []

        self.mc_algo_checkboxes = {}
        self.mc_config_inputs = {}
        self.mc_config_buttons = {}
        
        self._mc_camera_initialized = False
        self._mc_preview_cache = {}

        # botões próprios do modo MC
        font = host.font
        self.btn_mc_run = Button((0, 0, 200, 40), "Executar MC", font, bg=(70, 140, 90))
        self.btn_mc_config = Button((0, 0, 200, 32), "Configurar", font, bg=(75, 95, 135))
        self.btn_mc_export = Button((0, 0, 200, 32), "Exportar", font, bg=(120, 95, 60))

        # diretórios próprios do MC
        self.routes_dir = "routes"
        self.anchors_dir = "anchor_sets"
        self.maps_dir = "maps"
        os.makedirs(self.routes_dir, exist_ok=True)
        os.makedirs(self.anchors_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)

        self.mc_available_maps = []

        self.mc_dropdown_map_open = False

    def on_enter(self, host: Any) -> None:
        self.host = host
        self._legacy_ref = host
        self.host.mode = MODE_MONTE_CARLO

        # configuração própria do MC
        if self.mc_config is None:
            if hasattr(host, "mc_config") and host.mc_config is not None:
                from copy import deepcopy
                self.mc_config = deepcopy(host.mc_config)
            else:
                self.mc_config = MonteCarloConfig(
                    route_file="",
                    anchors_file="",
                    map_file="",
                    algoritmos=[],
                    seeds=[1, 2, 3, 4, 5],
                )

        if not hasattr(self.mc_config, "map_file"):
            self.mc_config.map_file = ""

        # sincroniza botões/estado comum a partir do host
        self.selected = default_selected()

        self.btn_back = host.btn_back
        self.btn_mode = host.btn_mode

    def handle_events(self, events):
        lg = self._legacy_ref
        if lg is None:
            return _actions_default()

        for event in events:
            # fechar app
            if event.type == pg.QUIT:
                return _actions_quit()

            # ESC -> voltar ao menu principal
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return _actions_menu()

            # modal config tem prioridade
            if self._handle_mc_config_modal_events(event):
                continue

            # câmera do MC
            if self._handle_mc_camera_events(event):
                continue

            # mouse / botões do MC
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                # botão de trocar modo (usa o do legacy por enquanto)
                if hasattr(lg, "btn_mode") and lg.btn_mode.hit(pos):
                    lg._toggle_mode()
                    self.host.mode = getattr(lg, "mode", MODE_MONTE_CARLO)
                    return _actions_default()

                # botão executar
                if self.btn_mc_run.hit(pos):
                    if not self.mc_running:
                        self._start_monte_carlo()
                    continue

                # botão configurar
                if self.btn_mc_config.hit(pos):
                    self._open_mc_config_modal()
                    continue

                # botão exportar
                if self.mc_results is not None and self.btn_mc_export.hit(pos):
                    self._export_mc_results()
                    continue

        return _actions_default()

    def update(self, dt: float) -> None:
        # no momento, o progresso vem da thread do runner
        # então não precisamos delegar pro legacy.update()
        pass

    def draw(self) -> None:
        lg = self._legacy_ref
        if lg is None:
            self.host.screen.fill((245, 247, 250))
            txt = self.host.font.render("Monte Carlo Mode", True, (0, 0, 0))
            self.host.screen.blit(txt, (30, 30))
            return

        # Fundo claro no conteúdo principal
        self.host.screen.fill((245, 247, 250))

        # Sidebar escura à direita
        sidebar_rect = pg.Rect(lg.SW - lg.SIDE_W, 0, lg.SIDE_W, lg.SH)
        pg.draw.rect(self.host.screen, (60, 60, 70), sidebar_rect)

        # título
        txt = lg.bigfont.render("Monte Carlo - Comparação de Algoritmos", True, (20, 20, 20))
        self.host.screen.blit(txt, (20, 20))

        # sidebar
        self._draw_mc_sidebar()

        # conteúdo principal
        if self.mc_results is not None:
            self._draw_mc_results()
        elif self.mc_running:
            self._draw_mc_running_with_map()
        else:
            self._draw_mc_welcome()

        # modal
        if self.mc_config_modal_open:
            self._draw_mc_config_modal()

        # botão de trocar modo do legacy (enquanto ainda usa ele)
        if hasattr(lg, "btn_mode"):
            lg.btn_mode.draw(self.host.screen)

    def close(self) -> None:
        # por enquanto não interrompe thread à força,
        # apenas solta referências quando possível
        self.mc_running = False

    # =========================================================
    # DRAW METHODS MIGRADOS DO LEGACY
    # =========================================================

    def _draw_mc_sidebar(self):
        ''''Desenha a sidebar do Monte Carlo, usando os dados de configuração em lg.mc_config'''
        lg = self._legacy_ref
        x = lg.SW - lg.SIDE_W + 16
        y = 60

        # título
        txt = lg.bigfont.render("Configuração", True, WHITE)
        lg.screen.blit(txt, (x, y))
        y += 35

        # separador
        pg.draw.line(lg.screen, (100, 100, 110), (x, y), (x + lg.SIDE_W - 40, y), 1)
        y += 15
        
        # rota
        txt = lg.font.render("Rota:", True, (180, 180, 180))
        lg.screen.blit(txt, (x, y))
        y += 20
        txt = lg.font.render(f"  {self.mc_config.route_file}", True, (220, 220, 220))
        lg.screen.blit(txt, (x, y))
        y += 25

        # âncoras
        txt = lg.font.render("Âncoras:", True, (180, 180, 180))
        lg.screen.blit(txt, (x, y))
        y += 20
        txt = lg.font.render(f"  {self.mc_config.anchors_file}", True, (220, 220, 220))
        lg.screen.blit(txt, (x, y))
        y += 25

        # mapa
        txt = lg.font.render("Mapa:", True, (180, 180, 180))
        lg.screen.blit(txt, (x, y))
        y += 20
        txt = lg.font.render(f"  {self.mc_config.map_file}", True, (220, 220, 220))
        lg.screen.blit(txt, (x, y))
        y += 25

        # algoritmos
        txt = lg.font.render("Algoritmos:", True, (180, 180, 180))
        lg.screen.blit(txt, (x, y))
        y += 20

        # algoritmos na ordem definida por ALGO_ORDER
        algos_ordenados = [a for a in ALGO_ORDER if a in self.mc_config.algoritmos]
        for algo in algos_ordenados:
            cor = ALGO_COLORS.get(algo, (200, 200, 200))
            nome_ui = NOMES_UI.get(algo, algo)
            txt = lg.font.render(f"  • {nome_ui}", True, cor)
            lg.screen.blit(txt, (x, y))
            y += 20
        y += 5

        # seeds
        txt = lg.font.render(f"Seeds: {len(self.mc_config.seeds)}", True, (180, 180, 180))
        lg.screen.blit(txt, (x, y))
        y += 20
        seeds_str = ", ".join(str(s) for s in self.mc_config.seeds[:3])
        if len(self.mc_config.seeds) > 3:
            seeds_str += ", ..."
        txt = lg.font.render(f"  [{seeds_str}]", True, (220, 220, 220))
        lg.screen.blit(txt, (x, y))
        y += 30

        preview_w = lg.SIDE_W - 32
        preview_h = 150
        self._draw_route_preview(x, y, preview_w, preview_h)
        y += preview_h + 10

        # separador
        pg.draw.line(lg.screen, (100, 100, 110), (x, y), (x + lg.SIDE_W - 40, y), 1)
        y += 20

        # botões
        btn_w = lg.SIDE_W - 32

        self.btn_mc_run.rect = pg.Rect(x, y, btn_w, 40)
        self.btn_mc_run.draw(lg.screen)
        y += 48

        self.btn_mc_config.rect = pg.Rect(x, y, btn_w, 32)
        self.btn_mc_config.draw(lg.screen)
        y += 40

        if self.mc_results is not None:
            self.btn_mc_export.rect = pg.Rect(x, y, btn_w, 32)
            self.btn_mc_export.draw(lg.screen)

    def _draw_mc_welcome(self):
        '''Desenha a tela de boas-vindas do Monte Carlo, com instruções básicas.'''
        lg = self._legacy_ref
        cx = (lg.SW - lg.SIDE_W) // 2
        cy = lg.SH // 2

        txt = lg.bigfont.render("Configure e execute o Monte Carlo", True, (100, 100, 100))
        lg.screen.blit(txt, (cx - txt.get_width() // 2, cy))

        # instruções
        y = cy + 40
        instructions = [
            "1. Verifique a configuração no sidebar →",
            "2. Clique em 'Executar MC' para iniciar",
            "3. Aguarde o processamento",
            "4. Veja os resultados quando terminar",
        ]

        for inst in instructions:
            txt = lg.font.render(inst, True, (120, 120, 120))
            lg.screen.blit(txt, (cx - 200, y))
            y += 25

    def _draw_mc_running(self):
        lg = self._legacy_ref

        # layout básico: mapa à esquerda, informações de progresso à direita
        map_w = lg.SW - lg.SIDE_W - 400
        map_x = 50
        map_y = 80
        map_h = lg.SH - 160

        info_x = map_x + map_w + 30
        info_y = map_y

        # desenha prévia da rota no mapa
        self._draw_route_preview(map_x, map_y, map_w, map_h)

        # Informações de progresso à direita
        cy = info_y + 50

        # status
        txt = lg.bigfont.render("Executando...", True, (50, 50, 50))
        lg.screen.blit(txt, (info_x, cy - 30))

        if self.mc_progress:
            # runs completadas
            txt = lg.font.render(
                f"Run {self.mc_progress.completed_runs}/{self.mc_progress.total_runs}",
                True,
                (100, 100, 100),
            )
            lg.screen.blit(txt, (info_x, cy))
            cy += 25

            # seed atual
            txt = lg.font.render(f"Seed: {self.mc_progress.current_seed}", True, (120, 120, 120))
            lg.screen.blit(txt, (info_x, cy))
            cy += 35

        # barra de progresso
        bar_w = 300
        bar_h = 30
        bx = info_x
        by = cy

        pg.draw.rect(lg.screen, (220, 220, 220), (bx, by, bar_w, bar_h))

        if self.mc_progress:
            fill_w = int(bar_w * self.mc_progress.progress)
            pg.draw.rect(lg.screen, (50, 180, 50), (bx, by, fill_w, bar_h))

            pct = f"{self.mc_progress.progress * 100:.1f}%"
            txt = lg.font.render(pct, True, WHITE if fill_w > 50 else (100, 100, 100))
            lg.screen.blit(txt, (bx + bar_w // 2 - txt.get_width() // 2, by + 6))

        pg.draw.rect(lg.screen, (100, 100, 100), (bx, by, bar_w, bar_h), 2)
        cy += bar_h + 20

        # ETA
        if self.mc_progress and self.mc_progress.eta_seconds > 0:
            eta_min = int(self.mc_progress.eta_seconds // 60)
            eta_sec = int(self.mc_progress.eta_seconds % 60)
            txt = lg.font.render(f"ETA: {eta_min}min {eta_sec}s", True, (100, 100, 100))
            lg.screen.blit(txt, (info_x, cy))
            cy += 25

        # tempo decorrido
        if self.mc_progress:
            elapsed_min = int(self.mc_progress.elapsed_time // 60)
            elapsed_sec = int(self.mc_progress.elapsed_time % 60)
            txt = lg.font.render(f"Tempo: {elapsed_min}min {elapsed_sec}s", True, (100, 100, 100))
            lg.screen.blit(txt, (info_x, cy))

    def _draw_mc_progress(self):
        '''Desenha a barra de progresso do Monte Carlo, usando os dados em lg.mc_progress'''
        self._draw_mc_running()

    def _draw_mc_results(self):
        ''' Desenha a tela de resultados do Monte Carlo, usando os dados em lg.mc_results'''
        lg = self._legacy_ref
        if self.mc_results is None:
            return

        # layout básico: tabela de resultados à esquerda, gráfico de RMSE à direita
        table_x = 100
        table_y = 100

        chart_x = 450
        chart_y = 80
        chart_w = lg.SW - lg.SIDE_W - chart_x - 50
        chart_h = 300

        # título
        txt = lg.bigfont.render("Resultados do Monte Carlo", True, (20, 20, 20))
        lg.screen.blit(txt, (table_x, table_y - 40))

        # tabela de estatísticas
        stats = self.mc_results.estatisticas()

        y = table_y

        # cabeçalho da tabela
        txt = lg.font.render("Algoritmo", True, (80, 80, 80))
        lg.screen.blit(txt, (table_x, y))

        txt = lg.font.render("RMSE (m)", True, (80, 80, 80))
        lg.screen.blit(txt, (table_x + 150, y))

        txt = lg.font.render("Desvio", True, (80, 80, 80))
        lg.screen.blit(txt, (table_x + 240, y))

        y += 25
        pg.draw.line(lg.screen, (200, 200, 200), (table_x, y), (table_x + 350, y), 1)
        y += 10

        # dados de cada algoritmo
        for algo, st in stats.items():
            cor = ALGO_COLORS.get(algo, (100, 100, 100))

            txt = lg.font.render(NOMES_UI.get(algo, algo), True, cor)
            lg.screen.blit(txt, (table_x, y))

            txt = lg.font.render(f"{st['rmse_xy_mean']:.3f}", True, (50, 50, 50))
            lg.screen.blit(txt, (table_x + 150, y))

            txt = lg.font.render(f"±{st['rmse_xy_std']:.3f}", True, (50, 50, 50))
            lg.screen.blit(txt, (table_x + 240, y))

            y += 25

        # tempo total
        y += 15
        exec_min = int(self.mc_results.execution_time_s // 60)
        exec_sec = int(self.mc_results.execution_time_s % 60)
        txt = lg.font.render(f"Tempo total: {exec_min}min {exec_sec}s", True, (100, 100, 100))
        lg.screen.blit(txt, (table_x, y))

        # gráfico de RMSE à direita
        self._draw_rmse_chart(chart_x, chart_y, chart_w, chart_h, stats)

    def _draw_mc_config_modal(self):
        '''Desenha o modal de configuração do Monte Carlo, usando os dados em lg.mc_config_inputs, lg.mc_algo_checkboxes etc.'''
        lg = self._legacy_ref
        if lg is None or not self.mc_config_modal_open:
            return

        screen = lg.screen
        font = lg.font
        bigfont = lg.bigfont

        # fundo escurecido
        overlay = pg.Surface((lg.SW, lg.SH), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        screen.blit(overlay, (0, 0))

        # modal
        modal_w = 560
        modal_h = 470
        mx = (lg.SW - modal_w) // 2
        my = (lg.SH - modal_h) // 2
        modal_rect = pg.Rect(mx, my, modal_w, modal_h)

        pg.draw.rect(screen, (245, 245, 248), modal_rect, border_radius=10)
        pg.draw.rect(screen, (80, 80, 90), modal_rect, 2, border_radius=10)

        # título
        txt = bigfont.render("Configurar Monte Carlo", True, (20, 20, 20))
        screen.blit(txt, (mx + 18, my + 14))

        # labels
        label_x = mx + 30
        value_x = mx + 180
        y = my + 75

        # rota
        txt = font.render("Rota:", True, (40, 40, 40))
        screen.blit(txt, (label_x, y))
        self._draw_input_box(self.mc_config_inputs["route"])
        self._draw_dropdown_arrow(self.mc_config_inputs["route"]["rect"])
        y += 50

        # âncoras
        txt = font.render("Âncoras:", True, (40, 40, 40))
        screen.blit(txt, (label_x, y))
        self._draw_input_box(self.mc_config_inputs["anchors"])
        self._draw_dropdown_arrow(self.mc_config_inputs["anchors"]["rect"])
        y += 50

        # mapa
        txt = font.render("Mapa:", True, (40, 40, 40))
        screen.blit(txt, (label_x, y))
        self._draw_input_box(self.mc_config_inputs["map"])
        self._draw_dropdown_arrow(self.mc_config_inputs["map"]["rect"])
        y += 50

        # seeds
        txt = font.render("Nº Seeds:", True, (40, 40, 40))
        screen.blit(txt, (label_x, y))
        self._draw_input_box(self.mc_config_inputs["n_seeds"])
        y += 55

        # algoritmos
        txt = font.render("Algoritmos:", True, (40, 40, 40))
        screen.blit(txt, (label_x, y))

        algo_y = y
        for i, algo in enumerate(ALGO_ORDER):
            if algo not in self.mc_algo_checkboxes:
                continue

            rect = self.mc_algo_checkboxes[algo]
            checked = algo in self.mc_config.algoritmos
            cor = ALGO_COLORS.get(algo, (120, 120, 120))

            pg.draw.rect(self.host.screen, (255, 255, 255), rect)
            pg.draw.rect(self.host.screen, (60, 60, 60), rect, 2)

            if checked:
                inner = rect.inflate(-6, -6)
                pg.draw.rect(self.host.screen, cor, inner)

            nome = NOMES_UI.get(algo, algo)
            txt = self._legacy_ref.font.render(nome, True, cor if checked else (60, 60, 60))
            self.host.screen.blit(txt, (rect.right + 10, rect.y - 1))

        # dropdowns abertos
        if self.mc_dropdown_route_open:
            self._draw_dropdown_list(
                self.mc_config_inputs["route"]["dropdown_rect"],
                self.mc_available_routes,
            )

        if self.mc_dropdown_anchors_open:
            self._draw_dropdown_list(
                self.mc_config_inputs["anchors"]["dropdown_rect"],
                self.mc_available_anchors,
            )

        if self.mc_dropdown_map_open:
            self._draw_dropdown_list(
                self.mc_config_inputs["map"]["dropdown_rect"],
                self.mc_available_maps,
            )

        # botões
        self.mc_config_buttons["save"].draw(screen)
        self.mc_config_buttons["cancel"].draw(screen)

    def _draw_input_box(self, inp: dict):
        '''Desenha uma caixa de input com base no dicionário de configuração (value, active, rect)'''
        lg = self._legacy_ref
        if lg is None:
            return

        rect = inp["rect"]
        active = inp.get("active", False)
        value = inp.get("value", "")

        bg = (255, 255, 255)
        border = (70, 130, 255) if active else (130, 130, 140)

        pg.draw.rect(lg.screen, bg, rect, border_radius=6)
        pg.draw.rect(lg.screen, border, rect, 2, border_radius=6)

        txt = lg.font.render(str(value), True, (30, 30, 30))
        lg.screen.blit(txt, (rect.x + 8, rect.y + 5))

    def _draw_dropdown_arrow(self, rect: pg.Rect):
        '''Desenha um triângulo indicando que o campo tem dropdown, posicionado à direita do input box'''
        lg = self._legacy_ref
        if lg is None:
            return

        cx = rect.right - 14
        cy = rect.centery + 1
        pts = [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)]
        pg.draw.polygon(lg.screen, (80, 80, 80), pts)

    def _draw_dropdown_list(self, rect: pg.Rect, items: list[str]):
        '''Desenha a lista de dropdown abaixo do input box, usando os itens fornecidos'''
        lg = self._legacy_ref
        if lg is None:
            return

        if not items:
            items = ["(vazio)"]

        item_h = 25
        visible_items = items[: max(1, rect.h // item_h)]
        real_h = len(visible_items) * item_h
        draw_rect = pg.Rect(rect.x, rect.y, rect.w, real_h)

        pg.draw.rect(lg.screen, (255, 255, 255), draw_rect, border_radius=4)
        pg.draw.rect(lg.screen, (120, 120, 130), draw_rect, 2, border_radius=4)

        y = draw_rect.y
        for item in visible_items:
            item_rect = pg.Rect(draw_rect.x, y, draw_rect.w, item_h)
            txt = lg.font.render(str(item), True, (30, 30, 30))
            lg.screen.blit(txt, (item_rect.x + 8, item_rect.y + 4))

            pg.draw.line(
                lg.screen,
                (225, 225, 230),
                (item_rect.x, item_rect.bottom),
                (item_rect.right, item_rect.bottom),
                1
            )
            y += item_h

    # =========================================================
    # COMMANDS / ACTIONS
    # =========================================================

    def _open_mc_config_modal(self):
        lg = self._legacy_ref
        if lg is None:
            return

        self.mc_config_modal_open = True
        self.mc_dropdown_route_open = False
        self.mc_dropdown_anchors_open = False
        self.mc_dropdown_map_open = False

        # arquivos disponíveis
        self.mc_available_routes = self._list_json_files(self.routes_dir)
        self.mc_available_anchors = self._list_json_files(self.anchors_dir)
        self.mc_available_maps = self._list_map_files()

        # inputs do modal
        route_file = getattr(self.mc_config, "route_file", "")
        anchors_file = getattr(self.mc_config, "anchors_file", "")
        map_file = getattr(self.mc_config, "map_file", "")
        n_seeds = str(len(getattr(self.mc_config, "seeds", [])))

        # layout base do modal
        modal_w = 560
        modal_h = 420
        mx = (lg.SW - modal_w) // 2
        my = (lg.SH - modal_h) // 2

        self.mc_config_inputs = {
            "route": {
                "value": route_file,
                "active": False,
                "rect": pg.Rect(mx + 180, my + 70, 260, 28),
                "dropdown_rect": pg.Rect(mx + 180, my + 98, 260, min(150, max(25, 25 * max(1, len(self.mc_available_routes)))))
            },
            "anchors": {
                "value": anchors_file,
                "active": False,
                "rect": pg.Rect(mx + 180, my + 120, 260, 28),
                "dropdown_rect": pg.Rect(mx + 180, my + 148, 260, min(150, max(25, 25 * max(1, len(self.mc_available_anchors)))))
            },
            "map": {
                "value": map_file,
                "active": False,
                "rect": pg.Rect(mx + 180, my + 170, 260, 28),
                "dropdown_rect": pg.Rect(mx + 180, my + 198, 260, min(150, max(25, 25 * max(1, len(self.mc_available_maps)))))
            },
            "n_seeds": {
                "value": n_seeds,
                "active": False,
                "rect": pg.Rect(mx + 180, my + 220, 120, 28)
            },
        }

        # checkboxes algoritmos
        self.mc_algo_checkboxes = {}
        algo_y = my + 275
        for i, algo in enumerate(ALGO_ORDER):
            self.mc_algo_checkboxes[algo] = pg.Rect(mx + 180, algo_y + i * 28, 18, 18)

        # botões
        self.mc_config_buttons = {
            "save": Button((mx + modal_w - 210, my + modal_h - 48, 90, 30), "Salvar", lg.font, bg=(235, 250, 235)),
            "cancel": Button((mx + modal_w - 105, my + modal_h - 48, 90, 30), "Cancelar", lg.font, bg=(250, 235, 235)),
        }


    def _save_mc_config(self):
        '''Salva as configurações do Monte Carlo a partir dos inputs do modal, atualizando self.mc_config e espelhando no legacy.'''
        if self.mc_config is None:
            return

        self.mc_config.route_file = self.mc_config_inputs["route"]["value"]
        self.mc_config.anchors_file = self.mc_config_inputs["anchors"]["value"]
        self.mc_config.map_file = self.mc_config_inputs["map"]["value"]

        try:
            n_seeds = int(self.mc_config_inputs["n_seeds"]["value"])
            n_seeds = max(1, n_seeds)
        except Exception:
            n_seeds = 5

        self.mc_config.seeds = list(range(1, n_seeds + 1))
        self.mc_config.algoritmos = [a for a in ALGO_ORDER if a in self.mc_config.algoritmos]

        self.mc_config_modal_open = False
        self.mc_dropdown_route_open = False
        self.mc_dropdown_anchors_open = False

    def _export_mc_results(self):
        if self.mc_results is None:
            self._set_msg("Sem resultados para exportar")
            return

        out_dir = os.path.join("resultados", "monte_carlo")
        os.makedirs(out_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(out_dir, f"mc_results_{timestamp}.csv")

        try:
            stats = self.mc_results.estatisticas()

            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["algoritmo", "rmse_xy_mean", "rmse_xy_std"])

                for algo in ALGO_ORDER:
                    if algo not in stats:
                        continue
                    st = stats[algo]
                    writer.writerow([
                        algo,
                        f"{float(st['rmse_xy_mean']):.6f}",
                        f"{float(st['rmse_xy_std']):.6f}",
                    ])

            self._set_msg(f"Resultados exportados: {os.path.basename(out_csv)}")
            

        except Exception as e:
            self._set_msg("Erro ao exportar resultados")

    def _start_monte_carlo(self):
        """Inicia execução Monte Carlo em thread separada."""
        lg = self._legacy_ref
        if lg is None:
            return

        lg._mc_camera_initialized = False
        self._mc_preview_cache = {}

        if not hasattr(lg, "shared_uwb") or lg.shared_uwb is None:
            self._set_msg("Erro: shared_uwb não configurado")
            return

        if self.mc_running:
            print("[MC] Já está rodando")
            return

        self.mc_running = True
        self.mc_progress = None
        self.mc_results = None
        self.mc_accumulated_points = {}

        def progress_callback(progress):
            self.mc_progress = progress

            if progress.intermediate_data and "positions" in progress.intermediate_data:
                positions = progress.intermediate_data["positions"]

                for algo, points in positions.items():
                    if algo not in self.mc_accumulated_points:
                        self.mc_accumulated_points[algo] = []

                    self.mc_accumulated_points[algo].extend(points)

        def run_in_thread():
            try:
                runner = MonteCarloRunner(
                    self.mc_config,
                    lg.shared_uwb,
                    progress_callback=progress_callback,
                )
                results = runner.run()
                self.mc_results = results
            except Exception as e:
                print(f"[MC] ERRO: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.mc_running = False

        self.mc_thread = threading.Thread(target=run_in_thread, daemon=True)
        self.mc_thread.start()

    def _set_msg(self, text: str):
        if hasattr(self.host, "_set_msg"):
            self.host._set_msg(text)       

    ############################################################
    # helpers para lidar com eventos específicos do Monte Carlo
    ############################################################
    def _handle_mc_config_modal_events(self, event):
        """Processa eventos do modal de configuração do Monte Carlo."""
        lg = self._legacy_ref
        if lg is None or not self.mc_config_modal_open:
            return False

        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos

            # Botões
            if self.mc_config_buttons["save"].hit(pos):
                self._save_mc_config()
                self.mc_config_modal_open = False
                self._set_msg("Configuração salva!")
                return True

            elif self.mc_config_buttons["cancel"].hit(pos):
                self.mc_config_modal_open = False
                self.mc_dropdown_route_open = False
                self.mc_dropdown_anchors_open = False
                return True

            route_inp = self.mc_config_inputs["route"]
            anchors_inp = self.mc_config_inputs["anchors"]
            map_inp = self.mc_config_inputs["map"]

            # dropdown rota aberto
            if self.mc_dropdown_route_open:
                if route_inp["dropdown_rect"].collidepoint(pos):
                    item_h = 25
                    idx = (pos[1] - route_inp["dropdown_rect"].y) // item_h
                    if 0 <= idx < len(self.mc_available_routes):
                        route_inp["value"] = self.mc_available_routes[idx]
                        self.mc_dropdown_route_open = False
                    return True
                else:
                    self.mc_dropdown_route_open = False

            # dropdown âncoras aberto
            if self.mc_dropdown_anchors_open:
                if anchors_inp["dropdown_rect"].collidepoint(pos):
                    item_h = 25
                    idx = (pos[1] - anchors_inp["dropdown_rect"].y) // item_h
                    if 0 <= idx < len(self.mc_available_anchors):
                        anchors_inp["value"] = self.mc_available_anchors[idx]
                        self.mc_dropdown_anchors_open = False
                    return True
                else:
                    self.mc_dropdown_anchors_open = False

            # dropdown mapa aberto
            if getattr(self, "mc_dropdown_map_open", False):
                if map_inp["dropdown_rect"].collidepoint(pos):
                    item_h = 25
                    idx = (pos[1] - map_inp["dropdown_rect"].y) // item_h
                    if 0 <= idx < len(self.mc_available_maps):
                        map_inp["value"] = self.mc_available_maps[idx]
                        self.mc_dropdown_map_open = False
                    return True
                else:
                    self.mc_dropdown_map_open = False

            # abrir dropdowns
            if not self.mc_dropdown_route_open and not self.mc_dropdown_anchors_open:
                if route_inp["rect"].collidepoint(pos):
                    self.mc_dropdown_route_open = True
                    self.mc_dropdown_anchors_open = False
                    return True

                if anchors_inp["rect"].collidepoint(pos):
                    self.mc_dropdown_anchors_open = True
                    self.mc_dropdown_route_open = False
                    return True
                
                if map_inp["rect"].collidepoint(pos):
                    self.mc_dropdown_map_open = True
                    self.mc_dropdown_route_open = False
                    self.mc_dropdown_anchors_open = False
                    return True

            # checkboxes algoritmos
            for algo, check_rect in self.mc_algo_checkboxes.items():
                if check_rect.collidepoint(pos):
                    if algo in self.mc_config.algoritmos:
                        self.mc_config.algoritmos.remove(algo)
                    else:
                        self.mc_config.algoritmos.append(algo)
                    return True

            # campo seeds
            seeds_inp = self.mc_config_inputs["n_seeds"]
            if seeds_inp["rect"].collidepoint(pos):
                for k in self.mc_config_inputs:
                    self.mc_config_inputs[k]["active"] = False
                seeds_inp["active"] = True
                return True

        elif event.type == pg.KEYDOWN:
            seeds_inp = self.mc_config_inputs["n_seeds"]
            if seeds_inp["active"]:
                if event.key == pg.K_BACKSPACE:
                    seeds_inp["value"] = seeds_inp["value"][:-1]
                elif event.key == pg.K_RETURN:
                    seeds_inp["active"] = False
                elif event.unicode.isdigit():
                    seeds_inp["value"] += event.unicode
                return True

        return False

    def _handle_mc_camera_events(self, event) -> bool:
        """Processa câmera do modo Monte Carlo."""
        lg = self._legacy_ref
        if lg is None:
            return False

        # Arrastar com botão do meio
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 2:
                lg._mc_camera_dragging = True
                lg._mc_camera_drag_start = event.pos
                return True

            elif event.button == 1 and pg.key.get_mods() & pg.KMOD_SHIFT:
                lg._mc_camera_dragging = True
                lg._mc_camera_drag_start = event.pos
                return True

        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 2 or (event.button == 1 and getattr(lg, "_mc_camera_dragging", False)):
                lg._mc_camera_dragging = False
                return True

        elif event.type == pg.MOUSEMOTION:
            if getattr(lg, "_mc_camera_dragging", False):
                dx_pixels = event.pos[0] - lg._mc_camera_drag_start[0]
                dy_pixels = event.pos[1] - lg._mc_camera_drag_start[1]

                # usa a API real da câmera
                lg.cam.pan_pixels(dx_pixels, dy_pixels)

                # atualiza referência para arrasto contínuo
                lg._mc_camera_drag_start = event.pos
                return True

        elif event.type == pg.MOUSEWHEEL:
            map_w = lg.SW - lg.SIDE_W
            map_h = lg.SH

            # zoom no centro do mapa
            zoom_pos = (map_w // 2, map_h // 2)
            factor = 1.1 if event.y > 0 else 1 / 1.1
            lg.cam.zoom_at(zoom_pos, factor)
            return True

        return False
    
    def _mc_get_camera_center(self):
        """Retorna o centro atual da câmera em coordenadas do mundo."""
        lg = self._legacy_ref
        if lg is None:
            return (0.0, 0.0)

        map_w = lg.SW - lg.SIDE_W
        map_h = lg.SH
        return lg.cam.screen_to_world(map_w / 2, map_h / 2)

    def _mc_set_camera_center(self, world_x: float, world_y: float):
        """Centraliza a câmera em uma posição do mundo usando a API real da câmera."""
        lg = self._legacy_ref
        if lg is None:
            return

        map_w = lg.SW - lg.SIDE_W
        map_h = lg.SH

        target_sx = map_w / 2
        target_sy = map_h / 2

        if not isinstance(lg.cam.pan, list):
            lg.cam.pan = [float(lg.cam.pan[0]), float(lg.cam.pan[1])]

        lg.cam.pan = [
            target_sx - (lg.cam.cx + world_x * lg.cam.scale),
            target_sy - (lg.cam.cy - world_y * lg.cam.scale),
        ]

    def _draw_route_preview(self, x: int, y: int, w: int, h: int):
        lg = self._legacy_ref
        if lg is None or self.mc_config is None:
            return

        rect = pg.Rect(x, y, w, h)
        pg.draw.rect(lg.screen, (42, 44, 52), rect, border_radius=8)
        pg.draw.rect(lg.screen, (90, 92, 104), rect, 1, border_radius=8)

        # tenta carregar rota e âncoras usando helpers já existentes no legacy
        waypoints = None
        anchors_xy = None
        map_data = None

        if hasattr(lg, "_load_route_preview_data"):
            try:
                waypoints, anchors_xy = lg._load_route_preview_data(
                    self.mc_config.route_file,
                    self.mc_config.anchors_file
                )
                if hasattr(self.mc_config, "map_file"):
                    map_data = lg._load_map_data(self.mc_config.map_file)

            except Exception:
                waypoints, anchors_xy, map_data = None, None, None
        else:
            # fallback: tenta ler dos diretórios padrão
            try:
                import os, json, numpy as np

                route_path = os.path.join(self.routes_dir, self.mc_config.route_file)
                if os.path.exists(route_path):
                    with open(route_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    wps = data.get("waypoints", [])
                    if wps:
                        waypoints = np.array(wps, dtype=float)

                anchor_path = os.path.join(self.anchors_dir, self.mc_config.anchors_file)
                if os.path.exists(anchor_path):
                    with open(anchor_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    anchors_xy = data.get("anchors_xy", [])

                maps_path = os.path.join(self.maps_dir, self.mc_config.map_file)
                if os.path.exists(maps_path):
                    with open(maps_path, "r", encoding="utf-8") as f:
                        map_data = json.load(f)
                    
            except Exception:
                waypoints, anchors_xy, map_data = None, None, None

        # se não há nada, desenha placeholder
        if waypoints is None and anchors_xy is None:
            txt = lg.font.render("Sem preview disponível", True, (180, 180, 185))
            lg.screen.blit(txt, (x + 12, y + h // 2 - 8))
            return

        # coleta pontos para escala
        pts = []
        if waypoints is not None and len(waypoints) > 0:
            pts.extend([(float(px), float(py)) for px, py in waypoints[:, :2]])
        if anchors_xy:
            pts.extend([(float(ax), float(ay)) for ax, ay in anchors_xy])

        if not pts:
            txt = lg.font.render("Sem dados", True, (180, 180, 185))
            lg.screen.blit(txt, (x + 12, y + h // 2 - 8))
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        dx = max(max_x - min_x, 1e-6)
        dy = max(max_y - min_y, 1e-6)

        pad = 16
        sx = (w - 2 * pad) / dx
        sy = (h - 2 * pad) / dy
        s = min(sx, sy)

        def map_pt(px, py):
            mx = x + pad + (px - min_x) * s
            my = y + h - pad - (py - min_y) * s
            return int(mx), int(my)

        # desenha rota
        if waypoints is not None and len(waypoints) > 1:
            route_pts = [map_pt(px, py) for px, py in waypoints[:, :2]]
            if len(route_pts) >= 2:
                pg.draw.lines(lg.screen, (220, 220, 220), False, route_pts, 2)

            # início/fim
            pg.draw.circle(lg.screen, (80, 220, 120), route_pts[0], 5)
            pg.draw.circle(lg.screen, (220, 120, 80), route_pts[-1], 5)

        # desenha âncoras
        if anchors_xy:
            for ax, ay in anchors_xy:
                mx, my = map_pt(float(ax), float(ay))
                pg.draw.circle(lg.screen, (120, 180, 255), (mx, my), 4)
                pg.draw.circle(lg.screen, (20, 20, 20), (mx, my), 4, 1)

        # desenha mapa
        if map_data:
            # lógica para desenhar o mapa, se aplicável
            pass

    def _draw_rmse_chart(self, x, y, width, height, stats):
        """Desenha gráfico de barras com RMSE por algoritmo."""
        lg = self._legacy_ref
        if lg is None or not stats:
            return

        pg.draw.rect(self.host.screen, (250, 250, 250), pg.Rect(x, y, width, height))
        pg.draw.rect(self.host.screen, (200, 200, 200), pg.Rect(x, y, width, height), 1)

        txt = lg.font.render("RMSE por Algoritmo", True, (50, 50, 50))
        self.host.screen.blit(txt, (x + 10, y + 10))

        algos = []
        rmse_values = []
        std_values = []

        for algo in ALGO_ORDER:
            if algo not in stats:
                continue

            rmse = stats[algo]["rmse_xy_mean"]
            std = stats[algo]["rmse_xy_std"]

            if np.isnan(rmse) or np.isnan(std) or rmse < 0:
                print(f"[Chart] Ignorando {algo}: RMSE={rmse}, STD={std}")
                continue

            algos.append(algo)
            rmse_values.append(rmse)
            std_values.append(std)

        if not rmse_values:
            txt = lg.font.render("Sem dados válidos para exibir", True, (150, 150, 150))
            self.host.screen.blit(txt, (x + width // 2 - txt.get_width() // 2, y + height // 2))
            return

        max_rmse = max(rmse_values) * 1.2

        chart_area_x = x + 60
        chart_area_y = y + 50
        chart_area_w = width - 80
        chart_area_h = height - 80

        pg.draw.line(
            self.host.screen,
            (100, 100, 100),
            (chart_area_x, chart_area_y + chart_area_h),
            (chart_area_x + chart_area_w, chart_area_y + chart_area_h),
            2,
        )
        pg.draw.line(
            self.host.screen,
            (100, 100, 100),
            (chart_area_x, chart_area_y),
            (chart_area_x, chart_area_y + chart_area_h),
            2,
        )

        n = len(algos)
        if n == 0:
            return

        bar_width = (chart_area_w - 20) // n
        bar_spacing = 5

        for i, algo in enumerate(algos):
            rmse = rmse_values[i]
            std = std_values[i]
            cor = ALGO_COLORS.get(algo, (100, 100, 100))

            bar_h = int((rmse / max_rmse) * chart_area_h)
            bar_x = chart_area_x + 10 + i * bar_width + bar_spacing
            bar_y = chart_area_y + chart_area_h - bar_h
            bar_w = bar_width - 2 * bar_spacing

            pg.draw.rect(self.host.screen, cor, pg.Rect(bar_x, bar_y, bar_w, bar_h))
            pg.draw.rect(self.host.screen, (50, 50, 50), pg.Rect(bar_x, bar_y, bar_w, bar_h), 1)

            if not np.isnan(std) and std > 0:
                std_h = int((std / max_rmse) * chart_area_h)
                std_y = max(bar_y - std_h, chart_area_y)
                pg.draw.line(
                    self.host.screen,
                    (50, 50, 50),
                    (bar_x + bar_w // 2, bar_y),
                    (bar_x + bar_w // 2, std_y),
                    2,
                )
                cap_w = 6
                pg.draw.line(
                    self.host.screen,
                    (50, 50, 50),
                    (bar_x + bar_w // 2 - cap_w, std_y),
                    (bar_x + bar_w // 2 + cap_w, std_y),
                    2,
                )

            # valor no topo
            txt = lg.font.render(f"{rmse:.2f}", True, (50, 50, 50))
            txt_x = bar_x + bar_w // 2 - txt.get_width() // 2
            txt_y = max(bar_y - 20, chart_area_y + 5)
            self.host.screen.blit(txt, (txt_x, txt_y))

            nome_curto = NOMES_UI.get(algo, algo)[:10]
            txt = lg.font.render(nome_curto, True, (80, 80, 80))
            txt_x = bar_x + bar_w // 2 - txt.get_width() // 2
            txt_y = chart_area_y + chart_area_h + 5
            self.host.screen.blit(txt, (txt_x, txt_y))

        txt = lg.font.render("RMSE (m)", True, (80, 80, 80))
        self.host.screen.blit(txt, (x + 5, y + height // 2))

    def _list_json_files(self, folder: str):
        try:
            files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
            files.sort()
            return files
        except Exception:
            return []
    
    def _draw_mc_running_with_map(self):
        """Tela de progresso com mapa - estilo simulation."""
        lg = self._legacy_ref
        if lg is None:
            return

        self.host.screen.fill((255, 255, 255))

        map_w = lg.SW - lg.SIDE_W
        map_h = lg.SH
        lg.cam.viewport = (map_w, map_h)

        if not hasattr(self, "_mc_camera_initialized"):
            self._mc_camera_initialized = False
        if not hasattr(self, "_mc_preview_cache"):
            self._mc_preview_cache = {}

        data = self._load_mc_preview_data()

        if data is None:
            # sem rota/âncoras = não consegue desenhar mapa do MC
            txt = lg.font.render("Falha ao carregar rota/âncoras do MC", True, (120, 30, 30))
            self.host.screen.blit(txt, (40, 80))
            return

        if not self._mc_camera_initialized:
            waypoints = data["waypoints"]
            anchors = data["anchors"]

            env = data.get("env")

            points = [waypoints[:, :2], anchors]

            if env is not None and hasattr(env, "obstacles"):
                for obs in env.obstacles:
                    points.append(np.array([obs.p0, obs.p1], dtype=float))

            all_points = np.vstack(points)

            min_x, min_y = all_points.min(axis=0)
            max_x, max_y = all_points.max(axis=0)

            margin = 2.0
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            range_x = (max_x - min_x) + 2 * margin
            range_y = (max_y - min_y) + 2 * margin

            scale_x = (map_w - 40) / range_x
            scale_y = (map_h - 40) / range_y

            lg.cam.scale = min(scale_x, scale_y) * 0.9
            self._mc_set_camera_center(center_x, center_y)

            self._mc_camera_initialized = True

        map_rect = pg.Rect(0, 0, map_w, map_h)
        prev_clip = self.host.screen.get_clip()
        self.host.screen.set_clip(map_rect)

        pg.draw.rect(self.host.screen, (255, 255, 255), map_rect)

        draw_grid(self.host.screen, lg.cam)

        if data.get("env") is not None:
            from src.environment.environment import draw_environment
            draw_environment(self.host.screen, lg.cam, data["env"])

        draw_axes(self.host.screen, lg.cam, lg.font)

        if data:
            waypoints = data["waypoints"]
            anchors = data["anchors"]

            anchors_3d = np.zeros((3, len(anchors)))
            anchors_3d[0, :] = anchors[:, 0]
            anchors_3d[1, :] = anchors[:, 1]
            anchors_3d[2, :] = 1.0
            draw_anchors(self.host.screen, lg.cam, anchors_3d)

            if len(waypoints) > 1:
                draw_path(
                    self.host.screen,
                    lg.cam,
                    [tuple(wp[:2]) for wp in waypoints],
                    (20, 20, 20),
                    3,
                    dashed=True,
                )

            if self.mc_progress:
                pos_info = self._get_mc_current_position()
                if pos_info:
                    current_idx = pos_info["current_idx"]
                    if current_idx > 0:
                        trail_so_far = [tuple(wp[:2]) for wp in waypoints[: current_idx + 1]]
                        if len(trail_so_far) > 1:
                            draw_path(self.host.screen, lg.cam, trail_so_far, (255, 150, 0), 3)

                    x, y, theta = pos_info["x"], pos_info["y"], pos_info["theta"]
                    draw_robot(self.host.screen, lg.cam, x, y, theta, (0, 0, 0), l=0.325)

                    sx, sy = lg.cam.world_to_screen(x, y)
                    pg.draw.circle(self.host.screen, (0, 180, 0), (int(sx), int(sy)), 8)
                    pg.draw.circle(self.host.screen, (0, 0, 0), (int(sx), int(sy)), 8, 2)

        if self.mc_accumulated_points:
            for algo, points in self.mc_accumulated_points.items():
                color = ALGO_COLORS.get(algo, (128, 128, 128))
                for point in points:
                    if len(point) >= 2:
                        px, py = point[0], point[1]
                        if not (np.isfinite(px) and np.isfinite(py)):
                            continue
                        screen_pos = lg.cam.world_to_screen(px, py)
                        if screen_pos:
                            sx, sy = int(screen_pos[0]), int(screen_pos[1])
                            pg.draw.circle(self.host.screen, color, (sx, sy), 4)
                            pg.draw.circle(self.host.screen, (255, 255, 255), (sx, sy), 4, 1)

        self.host.screen.set_clip(prev_clip)

        sidebar_x = lg.SW - lg.SIDE_W
        sidebar_w = lg.SIDE_W

        pg.draw.rect(self.host.screen, (50, 50, 55), pg.Rect(sidebar_x, 0, sidebar_w, lg.SH))

        cy = 50
        txt = lg.bigfont.render("Monte Carlo", True, (255, 255, 255))
        self.host.screen.blit(txt, (sidebar_x + 20, cy))
        cy += 60

        txt = lg.font.render("Executando...", True, (200, 200, 200))
        self.host.screen.blit(txt, (sidebar_x + 20, cy))
        cy += 40

        if self.mc_progress:
            txt = lg.font.render(
                f"Run {self.mc_progress.completed_runs}/{self.mc_progress.total_runs}",
                True,
                (180, 180, 180),
            )
            self.host.screen.blit(txt, (sidebar_x + 20, cy))
            cy += 25

            txt = lg.font.render(f"Seed: {self.mc_progress.current_seed}", True, (160, 160, 160))
            self.host.screen.blit(txt, (sidebar_x + 20, cy))
            cy += 35

        bar_w = sidebar_w - 40
        bar_h = 30
        bx = sidebar_x + 20
        by = cy

        pg.draw.rect(self.host.screen, (70, 70, 75), (bx, by, bar_w, bar_h))

        if self.mc_progress:
            fill_w = int(bar_w * self.mc_progress.progress)
            pg.draw.rect(self.host.screen, (50, 180, 50), (bx, by, fill_w, bar_h))

            pct = f"{self.mc_progress.progress * 100:.1f}%"
            txt = lg.font.render(pct, True, (255, 255, 255))
            self.host.screen.blit(txt, (bx + bar_w // 2 - txt.get_width() // 2, by + 6))

        pg.draw.rect(self.host.screen, (100, 100, 100), (bx, by, bar_w, bar_h), 2)
        cy += bar_h + 20

        if self.mc_progress and self.mc_progress.eta_seconds > 0:
            eta_min = int(self.mc_progress.eta_seconds // 60)
            eta_sec = int(self.mc_progress.eta_seconds % 60)
            txt = lg.font.render(f"ETA: {eta_min}min {eta_sec}s", True, (160, 160, 160))
            self.host.screen.blit(txt, (sidebar_x + 20, cy))
            cy += 25

        if self.mc_progress:
            elapsed_min = int(self.mc_progress.elapsed_time // 60)
            elapsed_sec = int(self.mc_progress.elapsed_time % 60)
            txt = lg.font.render(f"Tempo: {elapsed_min}min {elapsed_sec}s", True, (160, 160, 160))
            self.host.screen.blit(txt, (sidebar_x + 20, cy))
            

    def _get_mc_current_position(self):
        """Calcula posição atual do robô baseado no progresso."""
        if not self.mc_progress:
            return None

        data = self._load_mc_preview_data()
        if not data or "waypoints" not in data:
            return None
        
        waypoints = data['waypoints']
        if len(waypoints) < 2:
            return None
        
        # Progresso de 0.0 (início) a 1.0 (fim)
        progress = self.mc_progress.progress
        
        # Calcula índice do waypoint atual
        total_waypoints = len(waypoints)
        current_idx = int(progress * (total_waypoints - 1))
        current_idx = min(current_idx, total_waypoints - 1)
        
        # Interpola entre waypoints
        if current_idx < total_waypoints - 1:
            t = (progress * (total_waypoints - 1)) - current_idx
            wp1 = waypoints[current_idx]
            wp2 = waypoints[current_idx + 1]
            
            x = wp1[0] + t * (wp2[0] - wp1[0])
            y = wp1[1] + t * (wp2[1] - wp1[1])
            
            # Calcula orientação (direção do movimento)
            dx = wp2[0] - wp1[0]
            dy = wp2[1] - wp1[1]
            theta = np.arctan2(dy, dx) if abs(dx) > 1e-6 or abs(dy) > 1e-6 else 0.0
        else:
            # Último waypoint
            x, y = waypoints[-1][0], waypoints[-1][1]
            theta = 0.0
        
        return {
            'x': x,
            'y': y,
            'theta': theta,
            'current_idx': current_idx,
            'total_waypoints': total_waypoints
        }
    
    def _load_mc_preview_data(self):
        """
        Carrega rota e âncoras do MC e mantém em cache.
        Retorna dict com:
            {
                "waypoints": np.ndarray,
                "anchors": np.ndarray
            }
        ou None em caso de falha.
        """
        if self.mc_config is None:
            return None

        if not hasattr(self, "_mc_preview_cache"):
            self._mc_preview_cache = {}

        cache_key = f"{self.mc_config.route_file}_{self.mc_config.anchors_file}"

        if cache_key in self._mc_preview_cache:
            return self._mc_preview_cache[cache_key]

        try:
            route_path = Path(self.routes_dir) / self.mc_config.route_file
            anchors_path = Path(self.anchors_dir) / self.mc_config.anchors_file

            if not route_path.exists():
                print(f"[MC Preview] rota não encontrada: {route_path}")
                return None

            if not anchors_path.exists():
                print(f"[MC Preview] âncoras não encontradas: {anchors_path}")
                return None

            with open(route_path, "r", encoding="utf-8") as f:
                route_data = json.load(f)

            with open(anchors_path, "r", encoding="utf-8") as f:
                anchor_data = json.load(f)

            map_env = None
            if getattr(self.mc_config, "map_file", ""):
                map_path = Path(self.maps_dir) / self.mc_config.map_file
                if map_path.exists():
                    map_env = Environment.load_json(str(map_path))

            waypoints = np.array(route_data.get("waypoints", []), dtype=float)
            anchors_xy = np.array(anchor_data.get("anchors_xy", []), dtype=float)

            if waypoints.size == 0:
                print(f"[MC Preview] waypoints vazios em: {route_path}")
                return None

            if anchors_xy.size == 0:
                print(f"[MC Preview] anchors_xy vazio em: {anchors_path}")
                return None

            data = {
                "waypoints": waypoints,
                "anchors": anchors_xy,
                "env": map_env
            }

            self._mc_preview_cache[cache_key] = data
            return data

        except Exception as e:
            print(f"[MC Preview] erro ao carregar preview: {e}")
            return None
        
    def _list_map_files(self):
        try:
            files = [f for f in os.listdir(self.maps_dir) if f.lower().endswith(".json")]
            files.sort()
            return files
        except Exception:
            return []

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