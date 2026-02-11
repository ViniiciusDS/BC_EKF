# src/ui/uwb_test_screen.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pygame as pg
import math
import numpy as np
import os

from src.ui.drawing import draw_axes, draw_grid
from src.ui.ui_elements import TextBoxDropdown
from src.ui.botton import Button
from src.uwb.ranging_model import RangingConfig, UwbRangingModel
from src.uwb.twr_protocols import AntennaDelayModel, ClockModel, TWRConfig, TWRMode, DS_TWR_Protocol, SS_TWR_Protocol
from src.uwb.dataset import UwbFrame, RangeSample, UwbDatasetLogger, UwbReplay
from src.uwb.node_params import NodeParams

# Cores 
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

# Ações possíveis na tela de testes UWB
@dataclass
class UwbActions:
    go_to_menu: bool = False
    quite_app: bool = False


class UwbTestScreen:
    """
    Tela/aba de Testes UWB.

    Padrão igual às outras telas:
      - handle_events(events) -> UwbActions
      - update(dt)
      - draw()
      - close()
    """

    def __init__(
        self,
        screen: pg.Surface,
        cam: Any,
        clock: pg.time.Clock,
        font: pg.font.Font,
        bigfont: pg.font.Font,
        side_width: int,
    ) -> None:
        self.screen = screen
        self.cam = cam
        self.clock = clock
        self.font = font
        self.bigfont = bigfont
        self.SIDE_W = side_width

        self.seed = 123  # semente para RNG (reprodutibilidade)

        # ===== Estado UWB Test =====
        self.anchors: list[tuple[float, float]] = []  # (x,y) em coordenadas de mundo
        self.tag_pos: tuple[float, float] = (0.0, 0.0)

        # parâmetros de interação
        self.remove_radius_m: float = 0.6  # raio em metros para remover âncora mais próxima

        # pan/zoom (mapa)
        self.panning = False
        self.pan_last = (0, 0)

        # ===== HUD / UI Elements =====
        self.textbox_ax = TextBoxDropdown(pg.Rect(0, 0, 90, 26), self.font, options=[], placeholder="x")
        self.textbox_ax.set_text("0.0")

        self.textbox_ay = TextBoxDropdown(pg.Rect(0, 0, 90, 26), self.font, options=[], placeholder="y")
        self.textbox_ay.set_text("0.0")

        self.btn_add_anchor_xy = Button(
            rect=(0, 0, 190, 30),
            text="Adicionar âncora (x,y)",
            font=self.font,
            bg=(235, 250, 235),
        )

        self.btn_clear_anchors = Button(
            rect=(0, 0, 190, 30),
            text="Limpar âncoras",
            font=self.font,
            bg=(250, 235, 235),
        )

        # ===== Lista rolável de âncoras =====
        self.anchor_scroll: int = 0          # índice inicial (topo da lista)
        self.anchor_visible: int = 5         # quantos aparecem
        self.anchor_line_h: int = 18
        self.anchor_list_rect: pg.Rect | None = None  # definido no layout_hud()

        # ===== Configuração do modelo de ranging UWB =====
        self.ranging_cfg = RangingConfig(dt=0.10)
        self.ranging = UwbRangingModel(self.ranging_cfg, seed=self.seed)

        self._tick_acc = 0.0
        self.last_ranges = []  # lista de dicts/resultados p/ HUD

        # ===== HUD: dt entre medições =====
        self.textbox_dt = TextBoxDropdown(pg.Rect(0, 0, 90, 26), self.font, options=[], placeholder="dt (s)")
        self.textbox_dt.set_text(f"{self.ranging_cfg.dt:.2f}")

        self.btn_apply_dt = Button(
            rect=(0, 0, 120, 26),
            text="Aplicar dt",
            font=self.font,
            bg=(235, 235, 250),
        )

        # ===== Lista rolável de Ranges =====
        self.ranges_scroll: int = 0
        self.ranges_visible: int = 6
        self.ranges_line_h: int = 18
        self.ranges_list_rect: pg.Rect | None = None

        # ===== Toggle para exibir ranges =====
        self.show_ranges: bool = False

        # ===== Protocolo TWR =====
        self.twr_cfg = TWRConfig(mode=TWRMode.DS_TWR)
        self.twr_ds = DS_TWR_Protocol(self.twr_cfg, seed=self.seed)
        self.twr_ss = SS_TWR_Protocol(self.twr_cfg, seed=self.seed)
        self.twr = self.twr_ds  # ativo

        # ===== Step de ajustes (para hotkeys) =====
        self.ppm_step = 1.0          # 1 ppm por tecla
        self.delay_step_ns = 1.0     # 1 ns por tecla

        # ===== Logger / Replay =====
        self.logger = UwbDatasetLogger()
        self.replay: UwbReplay | None = None
        self.use_replay: bool = False
        self.sim_time_s: float = 0.0

        # caminho padrão 
        self.last_saved_path: str | None = None

        # ===== params por tag/âncora =====
        self.tag_params = NodeParams()
        self.anchor_params: dict[int, NodeParams] = {}  # key = anchor_id (índice)

        # seleção de âncora (pra editar no painel)
        self.selected_anchor: int | None = None

        # duplo clique na lista de âncoras (seleção/edição)
        self._last_anchor_click_ms: int = 0
        self._last_anchor_click_id: int | None = None
        self._double_click_ms: int = 350

        # editor modal de parâmetros da âncora selecionada
        self.anchor_editor_open: bool = False
        self.anchor_editor_id: int | None = None
        self.anchor_editor_rect: pg.Rect | None = None

        self.textbox_a_ppm = TextBoxDropdown(pg.Rect(0, 0, 90, 26), self.font, options=[], placeholder="ppm")
        self.textbox_a_tx = TextBoxDropdown(pg.Rect(0, 0, 90, 26), self.font, options=[], placeholder="tx (ns)")
        self.textbox_a_rx = TextBoxDropdown(pg.Rect(0, 0, 90, 26), self.font, options=[], placeholder="rx (ns)")
        self.textbox_a_bias = TextBoxDropdown(pg.Rect(0, 0, 90, 26), self.font, options=[], placeholder="bias (m)")

        self.btn_anchor_apply = Button(
            rect=(0, 0, 90, 26),
            text="Aplicar",
            font=self.font,
            bg=(235, 250, 235),
        )
        self.btn_anchor_close = Button(
            rect=(0, 0, 90, 26),
            text="Fechar",
            font=self.font,
            bg=(250, 235, 235),
        )

        # ===== Seed/Reprodutibilidade =====
        self.textbox_seed = TextBoxDropdown
        



    def handle_events(self, events) -> UwbActions:
        actions = UwbActions()

        self.layout_hud()

        for event in events:
            if event.type == pg.QUIT:
                actions.quite_app = True
                return actions
            
            cam_w = self.cam.viewport[0]

            # ============================
            # MODAL: editor de âncora
            # ============================
            if self.anchor_editor_open:
                # 1) Teclado: se algum textbox do editor está ativo, ele precisa receber KEYDOWN
                if event.type == pg.KEYDOWN:
                    for tb in (self.textbox_a_ppm, self.textbox_a_tx, self.textbox_a_rx, self.textbox_a_bias):
                        if getattr(tb, "active", False):
                            tb.handle_event(event)
                            break
                    # ESC fecha o modal
                    if event.key == pg.K_ESCAPE:
                        self._close_anchor_editor()
                    # Se o modal está aberto, não deixa o resto do sistema “roubar” teclas
                    continue

                # 2) Mouse: clique para focar nos textboxes / botões
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos

                    # primeiro: clicar nos textboxes dá foco
                    consumed = False
                    for tb in (self.textbox_a_ppm, self.textbox_a_tx, self.textbox_a_rx, self.textbox_a_bias):
                        before = getattr(tb, "active", False)
                        tb.handle_event(event)
                        after = getattr(tb, "active", False)
                        if after and not before:
                            consumed = True
                            break
                        # ou: se o clique foi dentro do rect dele, já consideramos consumido
                        if tb.rect.collidepoint((mx, my)):
                            consumed = True
                            break
                    if consumed:
                        continue

                    # botões
                    if self.btn_anchor_apply.hit((mx, my)):
                        self._apply_anchor_editor()
                        continue
                    if self.btn_anchor_close.hit((mx, my)):
                        self._close_anchor_editor()
                        continue

                    # clique fora fecha
                    if self.anchor_editor_rect and (not self.anchor_editor_rect.collidepoint((mx, my))):
                        self._close_anchor_editor()
                        continue

                    # se clicou dentro mas não em textbox/botão, não propaga
                    continue

                # mouse wheel / outros eventos: não propaga enquanto modal está aberto
                continue

            # ===== ZOOM NO MAPA (scroll) =====
            if event.type == pg.MOUSEWHEEL:
                mx, my = pg.mouse.get_pos()

                # Se tiver na lista de âncoras: desca a lista
                if self.anchor_list_rect and self.anchor_list_rect.collidepoint((mx, my)):
                    self._scroll_anchor_list(-event.y)  # wheel up = +1 no event.y
                    continue
                
                # Se tiver na lista de ranges: desca a lista
                if self.ranges_list_rect and self.ranges_list_rect.collidepoint((mx, my)):
                    self._scroll_ranges_list(-event.y)  # wheel up = +1 no event.y
                    continue
                
                # Se tiver no mapa: zoom
                if mx < cam_w:
                    factor = 1.15 if event.y > 0 else 1/1.15
                    self.cam.zoom_at((mx, my), factor)
                    continue

            # compatibilidade wheel (button 4/5)
            if event.type == pg.MOUSEBUTTONDOWN and event.button in (4, 5):
                mx, my = event.pos

                # Só permite interação no MAPA (lado esquerdo)
                if mx < cam_w:
                    self.cam.zoom_at((mx, my), 1.15 if event.button == 4 else 1/1.15)
                    continue
                
                # Se tiver na lista de ranges: desca a lista  
                if self.ranges_list_rect and self.ranges_list_rect.collidepoint((mx, my)):
                    self._scroll_ranges_list(-1 if event.button == 4 else +1)
                    continue

                # Se tiver na lista de âncoras: desca a lista
                if self.anchor_list_rect and self.anchor_list_rect.collidepoint((mx, my)):
                    self._scroll_anchor_list(-1 if event.button == 4 else +1)
                    continue


            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    actions.go_to_menu = True
                    return actions

                # TextBoxes primeiro 
                consumed = False
                if self.textbox_ax.handle_event(event):
                    consumed = True
                if (not consumed) and self.textbox_ay.handle_event(event):
                    consumed = True
                if (not consumed) and self.textbox_dt.handle_event(event):
                    consumed = True

                # ENTER aplica dt se dt box estiver ativa (ou se acabou de confirmar)
                if consumed:
                    # se o textbox_dt confirmou com ENTER, ele desativa e retornou True
                    if event.key == pg.K_RETURN and (not self.textbox_dt.active):
                        self._apply_dt_from_box()
                    continue

                # Hotkeys para toggles
                if event.key == pg.K_n:
                    self.ranging_cfg.noise_enabled = not self.ranging_cfg.noise_enabled
                    print(f"[UWB] Noise: {'ON' if self.ranging_cfg.noise_enabled else 'OFF'}")

                elif event.key == pg.K_l:
                    # alterna NLOS “ligado/desligado” via probabilidade
                    self.ranging_cfg.nlos_prob = 0.0 if self.ranging_cfg.nlos_prob > 0 else 0.15
                    print(f"[UWB] NLOS prob: {self.ranging_cfg.nlos_prob:.2f}")

                elif event.key == pg.K_p:
                    # alterna dropout
                    self.ranging_cfg.dropout_prob = 0.0 if self.ranging_cfg.dropout_prob > 0 else 0.05
                    print(f"[UWB] Dropout prob: {self.ranging_cfg.dropout_prob:.2f}")

                elif event.key == pg.K_q:
                    # alterna quantização
                    self.ranging_cfg.quantize_step = None if self.ranging_cfg.quantize_step else 0.01
                    q = self.ranging_cfg.quantize_step
                    print(f"[UWB] Quantize: {'OFF' if q is None else f'{q:.3f}m'}")

                elif event.key == pg.K_h:
                    # toggle de exibição do painel de ranges
                    self.show_ranges = not self.show_ranges
                
                elif event.key == pg.K_t:
                    # alterna protocolo TWR
                    new_mode = TWRMode.SS_TWR if self.twr_cfg.mode == TWRMode.DS_TWR else TWRMode.DS_TWR
                    self._set_protocol(new_mode)
                
                elif event.key == pg.K_1:
                    self._add_ppm("tag", -self.ppm_step)
                elif event.key == pg.K_2:
                    self._add_ppm("tag", +self.ppm_step)
                elif event.key == pg.K_3:
                    self._add_ppm("anchor", -self.ppm_step)
                elif event.key == pg.K_4:
                    self._add_ppm("anchor", +self.ppm_step)
                elif event.key == pg.K_5:
                    which = "rx" if (pg.key.get_mods() & pg.KMOD_SHIFT) else "tx"
                    self._add_delay_ns("tag", -self.delay_step_ns, which=which)

                elif event.key == pg.K_6:
                    which = "rx" if (pg.key.get_mods() & pg.KMOD_SHIFT) else "tx"
                    self._add_delay_ns("tag", +self.delay_step_ns, which=which)

                elif event.key == pg.K_7:
                    which = "rx" if (pg.key.get_mods() & pg.KMOD_SHIFT) else "tx"
                    self._add_delay_ns("anchor", -self.delay_step_ns, which=which)

                elif event.key == pg.K_8:
                    which = "rx" if (pg.key.get_mods() & pg.KMOD_SHIFT) else "tx"
                    self._add_delay_ns("anchor", +self.delay_step_ns, which=which)

                elif event.key == pg.K_r:
                    # R: start/stop recording
                    if not self.logger.enabled:
                        self.logger.start()
                        print("[UWB] REC ON")
                    else:
                        self.logger.stop()
                        print("[UWB] REC OFF")

                elif event.key == pg.K_s:
                    # S: save dataset
                    out_dir = "datasets"
                    os.makedirs(out_dir, exist_ok=True)
                    path = os.path.join(out_dir, f"uwb_dataset_{int(pg.time.get_ticks())}.jsonl")
                    self.last_saved_path = self.logger.save_jsonl(path)
                    print(f"[UWB] Saved: {self.last_saved_path}")

                elif event.key == pg.K_o:
                    # O: load last saved and enable replay
                    if self.last_saved_path:
                        frames = UwbDatasetLogger.load_jsonl(self.last_saved_path)
                        self.replay = UwbReplay(frames)
                        self.replay.play()
                        self.use_replay = True
                        print(f"[UWB] Replay ON ({len(frames)} frames)")

                elif event.key == pg.K_i:
                    # I: toggle replay on/off (mantém carregado)
                    if self.replay:
                        self.use_replay = not self.use_replay
                        print(f"[UWB] Replay {'ON' if self.use_replay else 'OFF'}")


                
            
            # ===== PAN COM BOTÃO DO MEIO =====
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 2:
                mx, my = event.pos
                if mx < cam_w:
                    self.panning = True
                    self.pan_last = (mx, my)
                    continue

            if event.type == pg.MOUSEBUTTONUP and event.button == 2:
                self.panning = False
                continue

            if event.type == pg.MOUSEMOTION and self.panning:
                mx, my = event.pos
                dx = mx - self.pan_last[0]
                dy = my - self.pan_last[1]
                self.cam.pan_pixels(dx, dy)
                self.pan_last = (mx, my)
                continue
            
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                cam_w = self.cam.viewport[0]

                # clique no HUD
                if mx >= cam_w:

                    # clique na lista de âncoras (seleção / duplo clique para editar)
                    if self.anchor_list_rect and self.anchor_list_rect.collidepoint((mx, my)):
                        idx = self._anchor_list_index_at((mx, my))
                        if idx is not None:
                            now_ms = pg.time.get_ticks()
                            if (self._last_anchor_click_id == idx) and ((now_ms - self._last_anchor_click_ms) <= self._double_click_ms):
                                self._open_anchor_editor(idx)
                            else:
                                self.selected_anchor = idx
                            self._last_anchor_click_id = idx
                            self._last_anchor_click_ms = now_ms
                            continue

                    if self.textbox_ax.handle_event(event):  # foco
                        continue
                    if self.textbox_ay.handle_event(event):
                        continue

                    if self.btn_add_anchor_xy.hit((mx, my)):
                        try:
                            x = float(self.textbox_ax.text.replace(",", "."))
                            y = float(self.textbox_ay.text.replace(",", "."))
                            self.anchors.append((x, y))
                            aid = len(self.anchors) - 1
                            self.anchor_params[aid] = NodeParams()
                            self.selected_anchor = aid
                        except ValueError:
                            print("Coordenadas inválidas.")
                        continue

                    if self.btn_clear_anchors.hit((mx, my)):
                        self.anchors.clear()
                        self.anchor_scroll = 0
                        self.ranges_scroll = 0
                        continue

                    if self.textbox_dt.handle_event(event):
                        continue

                    if self.btn_apply_dt.hit((mx, my)):
                        self._apply_dt_from_box()
                        continue

            if event.type == pg.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # scroll compatível (button 4/5)
                if event.button in (4, 5):
                    if self.anchor_list_rect and self.anchor_list_rect.collidepoint((mx, my)):
                        self._scroll_anchor_list(-1 if event.button == 4 else +1)
                        continue

                # só permite interação no MAPA (lado esquerdo)
                cam_w = self.cam.viewport[0]
                if mx >= cam_w:
                    continue

                wx, wy = self.cam.screen_to_world(mx, my)

                mods = pg.key.get_mods()

                # SHIFT + LMB: move TAG
                if event.button == 1 and (mods & pg.KMOD_SHIFT):
                    self.tag_pos = (wx, wy)
                    continue

                # LMB: adiciona ÂNCORA
                if event.button == 1:
                    self.anchors.append((wx, wy))
                    aid = len(self.anchors) - 1
                    self.anchor_params[aid] = NodeParams()
                    self.selected_anchor = aid
                    continue

                # RMB: remove ÂNCORA mais próxima (se estiver perto)
                if event.button == 3:
                    idx, d = self._find_nearest_anchor(wx, wy)
                    if idx is not None and d <= self.remove_radius_m:
                        self.anchors.pop(idx)
                        # reindexa params pois o índice mudou
                        new_params: dict[int, NodeParams] = {}
                        for new_i in range(len(self.anchors)):
                            # o item antigo era new_i se new_i < idx, senão era new_i+1
                            old_i = new_i if new_i < idx else new_i + 1
                            if old_i in self.anchor_params:
                                new_params[new_i] = self.anchor_params[old_i]
                        self.anchor_params = new_params

                        if self.selected_anchor is not None:
                            if self.selected_anchor == idx:
                                self.selected_anchor = None
                            elif self.selected_anchor > idx:
                                self.selected_anchor -= 1
                    continue

        return actions

    def update(self, dt: float) -> None:
        ''' Atualiza o estado da simulação. Se estiver em replay, avança o tempo simulado e aplica os frames.'''
        self.textbox_ax.update(dt)
        self.textbox_ay.update(dt)
        self.textbox_dt.update(dt)

        # tempo simulado (opcional)
        self.sim_time_s += dt

        # -------- Proteções anti-travamento --------
        # 1) cap no dt do frame (evita alt-tab / troca de tela explodir)
        dt = min(dt, 0.25)  # 250ms máximo por frame

        # 2) dt do tick sempre válido
        tick_dt = float(self.ranging_cfg.dt) if self.ranging_cfg.dt else 0.10
        tick_dt = max(0.01, min(5.0, tick_dt))  # garante >0

        # acumula
        self._tick_acc += dt

        # 3) cap de passos por frame (evita "spiral of death")
        max_steps = 50
        steps = 0

        while self._tick_acc >= tick_dt and steps < max_steps:
            self._tick_acc -= tick_dt
            steps += 1

            if self.use_replay and self.replay:
                fr = self.replay.step()
                if fr is not None:
                    self.tag_pos = (fr.tag_xy[0], fr.tag_xy[1])
                    self.anchors = list(fr.anchors_xy)
                    self.last_ranges = [{
                        "i": rs.anchor_id,
                        "r_true": rs.r_true_m,
                        "r_meas": rs.r_est_m,
                        "nlos": rs.is_nlos,
                        "dropped": rs.dropped,
                        "tof_true": rs.tof_true_s,
                        "tof_est": rs.tof_est_s,
                    } for rs in fr.ranges]
                    self._scroll_ranges_list(0)
                else:
                    self.use_replay = False
            else:
                self._compute_ranges_tick()

        # se estourou o cap, descarta o resto (senão fica travando pra sempre)
        if steps >= max_steps:
            self._tick_acc = 0.0
            print("[UWB] Warning: max_steps hit, dropping accumulated time")
        
        

    def draw(self) -> None:
        # Fundo geral
        self.screen.fill(WHITE)

        # Área do "mapa" (esquerda)
        cam_w = self.cam.viewport[0]
        cam_h = self.cam.viewport[1]
        map_rect = pg.Rect(0, 0, cam_w, cam_h)
        pg.draw.rect(self.screen, WHITE, map_rect)

        # Mapa: grid + eixos
        draw_grid(self.screen, self.cam)
        draw_axes(self.screen, self.cam, self.font)

        # ===== Desenho no mapa =====
        # Tag
        tx, ty = self.tag_pos
        stx, sty = self.cam.world_to_screen(tx, ty)
        pg.draw.circle(self.screen, (250, 160, 60), (stx, sty), 7)    # laranja
        pg.draw.circle(self.screen, BLACK, (stx, sty), 7, 1)

        # Âncoras e linhas até a TAG
        for (ax, ay) in self.anchors:
            sx, sy = self.cam.world_to_screen(ax, ay)
            pg.draw.line(self.screen, (170, 170, 170), (stx, sty), (sx, sy), 1)
            pg.draw.circle(self.screen, (55, 120, 220), (sx, sy), 6)   # azul
            pg.draw.circle(self.screen, BLACK, (sx, sy), 6, 1)

        # Overlay de status (canto superior direito do mapa)
        self._draw_map_overlay()

        # ===== Sidebar (direita) HUD / UI =====
        pg.draw.rect(self.screen, (245, 245, 245), (cam_w, 0, self.SIDE_W, cam_h))
        self.layout_hud()

        x = cam_w + 16
        y = 18

        # Header e instruções (agora só texto, sem mexer nos rects dos botões)
        self.screen.blit(self.bigfont.render("UWB — Testes", True, BLACK), (x, y))
        y += 32
        self.screen.blit(self.font.render("ESC: voltar ao menu", True, BLACK), (x, y))
        y += 22

        self.screen.blit(self.font.render("Mapa:", True, BLACK), (x, y)); y += 22
        self.screen.blit(self.font.render("LMB: adiciona âncora", True, BLACK), (x, y)); y += 20
        self.screen.blit(self.font.render("RMB: remove âncora (perto)", True, BLACK), (x, y)); y += 20
        self.screen.blit(self.font.render("SHIFT+LMB: mover TAG", True, BLACK), (x, y)); y += 24

        tx, ty = self.tag_pos
        self.screen.blit(self.font.render(f"Tag: x={tx:.2f}, y={ty:.2f}", True, BLACK), (x, y)); y += 20
        self.screen.blit(self.font.render(f"Âncoras: {len(self.anchors)}", True, BLACK), (x, y)); y += 20

        # label das ferramentas (posição alinhada com layout_hud)
        tools_label_y = self.textbox_ax.rect.y - 22
        self.screen.blit(self.font.render("Adicionar por coordenadas:", True, BLACK), (x, tools_label_y))

        # desenhar UI ELEMENTS
        self.textbox_ax.draw(self.screen)
        self.textbox_ay.draw(self.screen)
        self.btn_add_anchor_xy.draw(self.screen)
        self.btn_clear_anchors.draw(self.screen)

        dt_label_y = self.textbox_dt.rect.y - 18
        self.screen.blit(self.font.render("Intervalo entre medições UWB:", True, BLACK), (x, dt_label_y))
        self.textbox_dt.draw(self.screen)
        self.btn_apply_dt.draw(self.screen)

        # lista rolável
        self._draw_anchor_list()

        if self.show_ranges:
            self._draw_ranges_panel()

        if self.anchor_editor_open and self.anchor_editor_rect:
            self._draw_anchor_editor() 
        


                

    def close(self) -> None:
        
        pass

    def layout_hud(self):
        """Define posições dos elementos HUD/UI na sidebar direita."""
        cam_w = self.cam.viewport[0]
        sidebar_x = cam_w + 16

        y = 18

        # Header
        y += 32          # título
        y += 18          # "ESC: voltar..."
        y += 16          # espaço

        # Instruções (3 linhas)
        y += 22 * 4      # "Mapa:" + 3 linhas
        y += 18          # espaço

        # Estado (2 linhas)
        y += 20 * 2
        y += 14

        # ===== Ferramentas / Inputs =====
        y += 22  # "Adicionar por coordenadas:"
        y += 10

        # textboxes
        self.textbox_ax.rect.topleft = (sidebar_x, y)
        self.textbox_ay.rect.topleft = (sidebar_x + self.textbox_ax.rect.w + 10, y)
        y += self.textbox_ax.rect.h + 8

        # botões
        self.btn_add_anchor_xy.rect.topleft = (sidebar_x, y)
        y += self.btn_add_anchor_xy.rect.h + 8

        self.btn_clear_anchors.rect.topleft = (sidebar_x, y)
        y += self.btn_clear_anchors.rect.h + 24

        # dt entre medições
        self.textbox_dt.rect.topleft = (sidebar_x, y)
        self.btn_apply_dt.rect.topleft = (sidebar_x + self.textbox_dt.rect.w + 10, y)
        y += self.textbox_dt.rect.h + 14

        # marca onde termina a parte de ferramentas
        self._hud_y_after_tools = y

        # ===== área fixa da lista rolável =====
        y_title = y
        y_list = y_title + 22

        list_h = self.anchor_visible * self.anchor_line_h + 10
        list_w = self.SIDE_W - 32

        self.anchor_list_rect = pg.Rect(sidebar_x, y_list, list_w, list_h)

        # ===== área fixa da lista de ranges =====
        y_ranges_title = self.anchor_list_rect.bottom + 14
        y_ranges_list  = y_ranges_title + 22

        ranges_h = self.ranges_visible * self.ranges_line_h + 10
        ranges_w = list_w  # mesma largura das âncoras

        self.ranges_list_rect = pg.Rect(sidebar_x, y_ranges_list, ranges_w, ranges_h)

    ########################
    ##  Helpers internos  ##
    ########################

    def _find_nearest_anchor(self, wx: float, wy: float) -> tuple[int | None, float]:
        """Retorna (idx, dist) da âncora mais próxima do ponto (wx,wy)."""
        
        if not self.anchors:
            return None, float("inf")
        
        best_i = None
        best_d2 = float("inf")

        for i, (ax, ay) in enumerate(self.anchors):
            d2 = (ax - wx) ** 2 + (ay - wy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                
        return best_i, math.sqrt(best_d2)

    def _scroll_anchor_list(self, delta: int) -> None:
        """delta > 0 desce, delta < 0 sobe."""
        n = len(self.anchors)
        if n <= self.anchor_visible:
            self.anchor_scroll = 0
            return

        max_scroll = n - self.anchor_visible
        self.anchor_scroll = max(0, min(max_scroll, self.anchor_scroll + delta))


    def _draw_anchor_list(self) -> None:
        """Desenha a lista rolável dentro de self.anchor_list_rect."""
        if not self.anchor_list_rect:
            return

        r = self.anchor_list_rect
        x = r.x
        y = r.y

        # título acima do retângulo
        title_y = y - 22
        self.screen.blit(self.font.render("Âncoras:", True, BLACK), (x, title_y))

        # caixa
        pg.draw.rect(self.screen, (255, 255, 255), r)
        pg.draw.rect(self.screen, (200, 200, 200), r, 1)

        # decide quais itens mostrar
        start = self.anchor_scroll
        end = min(len(self.anchors), start + self.anchor_visible)
        visible = self.anchors[start:end]

        pad = 6
        yy = y + pad

        for i, (ax, ay) in enumerate(visible, start=start):
            # destaca se for a âncora selecionada
            row_rect = pg.Rect(x + 1, yy - 2, r.w - 2, self.anchor_line_h)
            if i == self.selected_anchor:
                pg.draw.rect(self.screen, (225, 235, 255), row_rect)

            txt = self.font.render(f"{i:02d}: x={ax:.2f}, y={ay:.2f}", True, BLACK)
            self.screen.blit(txt, (x + pad, yy))
            yy += self.anchor_line_h


        # mini “scrollbar” visual (opcional, mas ajuda muito)
        n = len(self.anchors)
        if n > self.anchor_visible:
            bar_w = 6
            bar_x = r.right - bar_w - 2
            bar_y = r.y + 2
            bar_h = r.height - 4

            # trilho
            pg.draw.rect(self.screen, (235, 235, 235), (bar_x, bar_y, bar_w, bar_h))

            # handle
            frac = self.anchor_visible / n
            handle_h = max(10, int(bar_h * frac))
            max_scroll = n - self.anchor_visible
            t = 0.0 if max_scroll == 0 else (self.anchor_scroll / max_scroll)
            handle_y = bar_y + int((bar_h - handle_h) * t)

            pg.draw.rect(self.screen, (180, 180, 180), (bar_x, handle_y, bar_w, handle_h))

    def _anchor_list_index_at(self, pos: tuple[int, int]) -> int | None:
        """Retorna o índice global da âncora (0..N-1) sob o mouse na lista rolável."""
        if not self.anchor_list_rect:
            return None
        r = self.anchor_list_rect
        if not r.collidepoint(pos):
            return None

        pad = 6
        x, y = pos
        rel_y = y - (r.y + pad)
        if rel_y < 0:
            return None

        row = int(rel_y // self.anchor_line_h)
        idx = self.anchor_scroll + row
        if 0 <= idx < len(self.anchors):
            if idx < self.anchor_scroll + self.anchor_visible:
                return idx
        return None

    def _open_anchor_editor(self, anchor_id: int) -> None:
        """Abre editor modal para editar NodeParams da âncora anchor_id."""
        if anchor_id < 0 or anchor_id >= len(self.anchors):
            return

        self.selected_anchor = anchor_id
        self.anchor_editor_open = True
        self.anchor_editor_id = anchor_id

        # define rect do editor (modal) sobre a área da lista de âncoras
        if self.anchor_list_rect:
            r = self.anchor_list_rect
            self.anchor_editor_rect = pg.Rect(r.x, r.y, r.w, 170)
        else:
            cam_w = self.cam.viewport[0]
            self.anchor_editor_rect = pg.Rect(cam_w + 16, 200, self.SIDE_W - 32, 170)

        p = self.anchor_params.get(anchor_id, NodeParams())

        self.textbox_a_ppm.set_text(f"{float(p.clock.drift_ppm):.3f}")
        self.textbox_a_tx.set_text(f"{float(p.ant.tx_ns):.3f}")
        self.textbox_a_rx.set_text(f"{float(p.ant.rx_ns):.3f}")
        self.textbox_a_bias.set_text(f"{float(p.range_bias_m):.4f}")

        ex = self.anchor_editor_rect.x + 10
        ey = self.anchor_editor_rect.y + 30

        self.textbox_a_ppm.rect.topleft = (ex + 70, ey); ey += 32
        self.textbox_a_tx.rect.topleft = (ex + 70, ey); ey += 32
        self.textbox_a_rx.rect.topleft = (ex + 70, ey); ey += 32
        self.textbox_a_bias.rect.topleft = (ex + 70, ey); ey += 40

        self.btn_anchor_apply.rect.topleft = (ex, ey)
        self.btn_anchor_close.rect.topleft = (ex + 100, ey)

        # deixa o campo ppm já pronto pra digitar
        for tb in (self.textbox_a_ppm, self.textbox_a_tx, self.textbox_a_rx, self.textbox_a_bias):
            tb.active = False
        self.textbox_a_ppm.active = True

    def _close_anchor_editor(self) -> None:
        '''Fecha o editor modal de âncora e descarta mudanças não aplicadas.'''
        self.anchor_editor_open = False
        self.anchor_editor_id = None
        self.anchor_editor_rect = None
        for tb in (self.textbox_a_ppm, self.textbox_a_tx, self.textbox_a_rx, self.textbox_a_bias):
            tb.active = False

    def _apply_anchor_editor(self) -> None:
        """Aplica valores do editor em self.anchor_params[anchor_id]."""
        if self.anchor_editor_id is None:
            return
        aid = self.anchor_editor_id
        if aid < 0 or aid >= len(self.anchors):
            return

        p = self.anchor_params.get(aid, NodeParams())

        def _to_float(s: str) -> float:
            return float(s.replace(",", ".").strip())

        try:
            p.clock.drift_ppm = _to_float(self.textbox_a_ppm.text)
            p.ant.tx_ns = _to_float(self.textbox_a_tx.text)
            p.ant.rx_ns = _to_float(self.textbox_a_rx.text)
            p.range_bias_m = _to_float(self.textbox_a_bias.text)
            self.anchor_params[aid] = p
        except ValueError:
            print("[UWB] Valores inválidos no editor da âncora.")

    def _draw_anchor_editor(self) -> None:
        ''' Desenha o editor modal de parâmetros da âncora selecionada, se estiver aberto.'''
        if not (self.anchor_editor_open and self.anchor_editor_rect):
            return

        r = self.anchor_editor_rect

        pg.draw.rect(self.screen, (252, 252, 252), r)
        pg.draw.rect(self.screen, (170, 170, 170), r, 1)

        aid = self.anchor_editor_id if self.anchor_editor_id is not None else -1
        title = self.font.render(f"Editar Âncora {aid:02d}", True, BLACK)
        self.screen.blit(title, (r.x + 10, r.y + 8))

        lx = r.x + 10
        ly = r.y + 34
        self.screen.blit(self.font.render("ppm:", True, BLACK), (lx, ly)); ly += 32
        self.screen.blit(self.font.render("tx(ns):", True, BLACK), (lx, ly)); ly += 32
        self.screen.blit(self.font.render("rx(ns):", True, BLACK), (lx, ly)); ly += 32
        self.screen.blit(self.font.render("bias(m):", True, BLACK), (lx, ly))

        self.textbox_a_ppm.draw(self.screen)
        self.textbox_a_tx.draw(self.screen)
        self.textbox_a_rx.draw(self.screen)
        self.textbox_a_bias.draw(self.screen)

        self.btn_anchor_apply.draw(self.screen)
        self.btn_anchor_close.draw(self.screen)

    def _compute_ranges_tick(self) -> None:
        '''Computa as medições de distância UWB para todas as âncoras na posição atual da TAG.'''
        tag = np.array(self.tag_pos, dtype=float)

        out = []
        for i, (ax, ay) in enumerate(self.anchors):
            a = np.array([ax, ay], dtype=float)

            # parâmetros por nó (âncora i e tag)
            p_anc = self.anchor_params.get(i, NodeParams())
            p_tag = self.tag_params

            clock_anchor = ClockModel(ppm=float(p_anc.clock.drift_ppm))
            clock_tag = ClockModel(ppm=float(p_tag.clock.drift_ppm))

            delay_anchor = AntennaDelayModel(tx_s=float(p_anc.ant.tx_s()), rx_s=float(p_anc.ant.rx_s()))
            delay_tag = AntennaDelayModel(tx_s=float(p_tag.ant.tx_s()), rx_s=float(p_tag.ant.rx_s()))

            # bias por âncora, campo global do canal e restauramos ao final
            _prev_bias = float(getattr(self.ranging, "global_bias", 0.0))
            self.ranging.global_bias = float(p_anc.range_bias_m)

            res = self.twr.simulate(
                    a_xy=a, tag_xy=tag, channel=self.ranging,
                    clock_anchor=clock_anchor,
                    clock_tag=clock_tag,
                    delay_anchor=delay_anchor,
                    delay_tag=delay_tag,
                )

            self.ranging.global_bias = _prev_bias

            out.append({
                "i": i,
                "r_true": res.r_true_m,
                "r_meas": res.r_est_m,   # estimativa via DS-TWR
                "nlos": res.is_nlos,
                "dropped": res.dropped,
                # debug opcional:
                "tof_true": res.tof_true_s,
                "tof_est": res.tof_est_s,
            })

        # salva também um frame padronizado (logger)
        if self.logger.enabled:
            protocol_name = type(self.twr).__name__  # ex: DS_TWR_Protocol
            # snapshot de config 
            cfg_snapshot = {
                "dt": self.ranging_cfg.dt,
                "noise_enabled": self.ranging_cfg.noise_enabled,
                "sigma_los": self.ranging_cfg.sigma_los,
                "sigma_nlos": self.ranging_cfg.sigma_nlos,
                "nlos_prob": self.ranging_cfg.nlos_prob,
                "dropout_prob": self.ranging_cfg.dropout_prob,
                "quantize_step": self.ranging_cfg.quantize_step,
                "protocol": protocol_name,
            }

            ranges = []
            for item in out:
                ranges.append(RangeSample(
                    anchor_id=item["i"],
                    r_true_m=float(item["r_true"]),
                    r_est_m=None if item["r_meas"] is None else float(item["r_meas"]),
                    is_nlos=bool(item["nlos"]),
                    dropped=bool(item.get("dropped", item["r_meas"] is None)),
                    tof_true_s=item.get("tof_true"),
                    tof_est_s=item.get("tof_est"),
                ))

            frame = UwbFrame(
                t_sim_s=float(self.sim_time_s),
                tag_xy=(float(self.tag_pos[0]), float(self.tag_pos[1])),
                anchors_xy=[(float(ax), float(ay)) for (ax, ay) in self.anchors],
                protocol=protocol_name,
                cfg=cfg_snapshot,
                ranges=ranges,
            )
            self.logger.add(frame)

        self.last_ranges = out
        self._scroll_ranges_list(0)

    def _apply_dt_from_box(self) -> None:
        '''Lê o valor do textbox_dt e aplica na configuração do modelo de ranging.'''
        try:
            dt_s = float(self.textbox_dt.text.replace(",", "."))
            # limites pra evitar travar/ficar lento demais
            dt_s = max(0.01, min(5.0, dt_s))
            self.ranging_cfg.dt = dt_s
            self.textbox_dt.set_text(f"{dt_s:.2f}")
            # reinicia acumulador para não “disparar” múltiplos ticks de uma vez
            self._tick_acc = 0.0
            print(f"[UWB] dt entre medições = {dt_s:.2f}s")
        except ValueError:
            print("[UWB] dt inválido.")
            # opcional: volta ao valor atual
            self.textbox_dt.set_text(f"{self.ranging_cfg.dt:.2f}")

    def _draw_map_overlay(self) -> None:
        """Painel pequeno no canto superior direito do MAPA com status dos toggles."""
        cam_w = self.cam.viewport[0]

        noise_txt = "ON" if self.ranging_cfg.noise_enabled else "OFF"
        nlos_on = self.ranging_cfg.nlos_prob > 0
        drop_on = self.ranging_cfg.dropout_prob > 0
        q = self.ranging_cfg.quantize_step
        q_txt = "OFF" if q is None else f"{q:.3f}m"
        proto_txt = self.twr_cfg.mode.value

        # overlay baseado em NodeParams
        tag_ppm = float(self.tag_params.clock.drift_ppm)

        # âncora selecionada (fallback: 0)
        sel = self.selected_anchor
        if sel is None:
            sel = 0 if len(self.anchors) > 0 else None

        tag_tx_ns = float(self.tag_params.ant.tx_ns)
        tag_rx_ns = float(self.tag_params.ant.rx_ns)

        if sel is not None and sel in self.anchor_params:
            ap = self.anchor_params[sel]
            anc_ppm = float(ap.clock.drift_ppm)
            anc_tx_ns = float(ap.ant.tx_ns)
            anc_rx_ns = float(ap.ant.rx_ns)
        else:
            anc_ppm = 0.0
            anc_tx_ns = 0.0
            anc_rx_ns = 0.0

        lines = [
            f"dt: {self.ranging_cfg.dt:.2f}s",
            f"Noise [N]: {noise_txt}",
            f"NLOS  [L]: {'ON' if nlos_on else 'OFF'}",
            f"Drop  [P]: {'ON' if drop_on else 'OFF'}",
            f"Quant [Q]: {q_txt}",
            f"H: ranges {'ON' if self.show_ranges else 'OFF'}",
            f"T: protocol {proto_txt}",
            f"Tag ppm [1/2]: {tag_ppm:+.1f} ppm",
            f"Anc ppm [3/4]: {anc_ppm:+.1f} ppm",
            f"Tag delay tx/rx [5/6]: {tag_tx_ns:.1f}/{tag_rx_ns:.1f} ns",
            f"Anc delay tx/rx [7/8]: {anc_tx_ns:.1f}/{anc_rx_ns:.1f} ns",
        ]

        pad = 8
        line_h = 18

        # tamanho do painel baseado no maior texto
        w = max(self.font.size(s)[0] for s in lines) + 2 * pad
        h = len(lines) * line_h + 2 * pad

        # canto superior direito do MAPA (com margem)
        x = cam_w - w - 12
        y = 12

        # fundo semi-transparente
        panel = pg.Surface((w, h), pg.SRCALPHA)
        panel.fill((255, 255, 255, 210))
        self.screen.blit(panel, (x, y))
        pg.draw.rect(self.screen, (40, 40, 40), (x, y, w, h), 1)

        yy = y + pad
        for s in lines:
            self.screen.blit(self.font.render(s, True, (20, 20, 20)), (x + pad, yy))
            yy += line_h

    def _scroll_ranges_list(self, delta: int) -> None:
        """delta > 0 desce, delta < 0 sobe."""
        n = len(self.last_ranges)
        if n <= self.ranges_visible:
            self.ranges_scroll = 0
            return

        max_scroll = n - self.ranges_visible
        self.ranges_scroll = max(0, min(max_scroll, self.ranges_scroll + delta))


    def _draw_ranges_panel(self) -> None:
        """Desenha painel rolável de ranges dentro de self.ranges_list_rect com clip."""
        if not self.ranges_list_rect:
            return

        r = self.ranges_list_rect
        x = r.x
        y = r.y

        # título acima do retângulo
        title_y = y - 22
        self.screen.blit(self.font.render("Ranges (último tick):", True, BLACK), (x, title_y))

        # caixa
        pg.draw.rect(self.screen, (255, 255, 255), r)
        pg.draw.rect(self.screen, (200, 200, 200), r, 1)

        # recorte (CLIP) para não “vazar” texto
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(r)

        start = self.ranges_scroll
        end = min(len(self.last_ranges), start + self.ranges_visible)
        visible = self.last_ranges[start:end]

        pad = 6
        yy = y + pad

        for item in visible:
            r_meas = item["r_meas"]
            meas_txt = "drop" if r_meas is None else f"{r_meas:.3f}m"
            flag = "NLOS" if item["nlos"] else "LOS"
            line = f"{item['i']:02d}: true={item['r_true']:.3f}  meas={meas_txt}  {flag}"
            self.screen.blit(self.font.render(line, True, BLACK), (x + pad, yy))
            yy += self.ranges_line_h

        self.screen.set_clip(prev_clip)

        # mini scrollbar
        n = len(self.last_ranges)
        if n > self.ranges_visible:
            bar_w = 6
            bar_x = r.right - bar_w - 2
            bar_y = r.y + 2
            bar_h = r.height - 4

            pg.draw.rect(self.screen, (235, 235, 235), (bar_x, bar_y, bar_w, bar_h))

            frac = self.ranges_visible / n
            handle_h = max(10, int(bar_h * frac))
            max_scroll = n - self.ranges_visible
            t = 0.0 if max_scroll == 0 else (self.ranges_scroll / max_scroll)
            handle_y = bar_y + int((bar_h - handle_h) * t)

            pg.draw.rect(self.screen, (180, 180, 180), (bar_x, handle_y, bar_w, handle_h))

    def _set_protocol(self, mode: TWRMode) -> None:
        '''
        Alterna entre os protocolos DS-TWR e SS-TWR.
        '''
        self.twr_cfg.mode = mode
        self.twr = self.twr_ds if mode == TWRMode.DS_TWR else self.twr_ss
        print(f"[UWB] Protocol = {mode.value}")

    def _active_protocols(self):
        # para manter DS/SS sincronizados (mesmos clocks/delays)
        return [self.twr_ds, self.twr_ss]

    def _add_ppm(self, target: str, delta_ppm: float) -> None:
        """
        Atualiza ppm (drift) de acordo com o alvo:
        - target == "tag": altera self.tag_params.clock.drift_ppm
        - target == "anchor": altera self.anchor_params[selected_anchor].clock.drift_ppm
        """
        if target == "tag":
            self.tag_params.clock.drift_ppm = float(self.tag_params.clock.drift_ppm) + float(delta_ppm)
            return

        if target == "anchor":
            aid = self.selected_anchor
            if aid is None:
                aid = 0 if len(self.anchors) > 0 else None
            if aid is None:
                return
            p = self.anchor_params.get(aid, NodeParams())
            p.clock.drift_ppm = float(p.clock.drift_ppm) + float(delta_ppm)
            self.anchor_params[aid] = p
            return

    def _add_delay_ns(self, target: str, delta_ns: float, which: str = "tx") -> None:
        """
        Ajusta delay em ns para tag ou âncora selecionada.
        which: "tx" ou "rx"
        """
        if which not in ("tx", "rx"):
            return

        if target == "tag":
            if which == "tx":
                self.tag_params.ant.tx_ns = float(self.tag_params.ant.tx_ns) + float(delta_ns)
            else:
                self.tag_params.ant.rx_ns = float(self.tag_params.ant.rx_ns) + float(delta_ns)
            return

        if target == "anchor":
            aid = self.selected_anchor
            if aid is None:
                aid = 0 if len(self.anchors) > 0 else None
            if aid is None:
                return
            p = self.anchor_params.get(aid, NodeParams())
            if which == "tx":
                p.ant.tx_ns = float(p.ant.tx_ns) + float(delta_ns)
            else:
                p.ant.rx_ns = float(p.ant.rx_ns) + float(delta_ns)
            self.anchor_params[aid] = p
            return


