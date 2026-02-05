# src/ui/uwb_test_screen.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pygame as pg
import math
import numpy as np

from src.ui.drawing import draw_axes, draw_grid
from src.ui.ui_elements import TextBoxDropdown
from src.ui.botton import Button
from src.uwb.ranging_model import RangingConfig, UwbRangingModel

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
        self.ranging = UwbRangingModel(self.ranging_cfg, seed=123)

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



    def handle_events(self, events) -> UwbActions:
        actions = UwbActions()

        self.layout_hud()

        for event in events:
            if event.type == pg.QUIT:
                actions.quite_app = True
                return actions
            
            cam_w = self.cam.viewport[0]

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
                    self.show_ranges = not self.show_ranges
            
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
                    if self.textbox_ax.handle_event(event):  # foco
                        continue
                    if self.textbox_ay.handle_event(event):
                        continue

                    if self.btn_add_anchor_xy.hit((mx, my)):
                        try:
                            x = float(self.textbox_ax.text.replace(",", "."))
                            y = float(self.textbox_ay.text.replace(",", "."))
                            self.anchors.append((x, y))
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
                    continue

                # RMB: remove ÂNCORA mais próxima (se estiver perto)
                if event.button == 3:
                    idx, d = self._find_nearest_anchor(wx, wy)
                    if idx is not None and d <= self.remove_radius_m:
                        self.anchors.pop(idx)
                    continue

        return actions

    def update(self, dt: float) -> None:
        self.textbox_ax.update(dt)
        self.textbox_ay.update(dt)
        self.textbox_dt.update(dt)

        # tick do modelo de ranging UWB
        self._tick_acc += dt
        while self._tick_acc >= self.ranging_cfg.dt:
            self._tick_acc -= self.ranging_cfg.dt
            self._compute_ranges_tick()

        pass

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
            txt = self.font.render(f"{i:02d}: ({ax:.2f}, {ay:.2f})", True, BLACK)
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

    def _compute_ranges_tick(self) -> None:
        '''Computa as medições de distância UWB para todas as âncoras na posição atual da TAG.'''
        tag = np.array(self.tag_pos, dtype=float)

        out = []
        for i, (ax, ay) in enumerate(self.anchors):
            a = np.array([ax, ay], dtype=float)
            res = self.ranging.measure_range(a, tag)
            out.append({
                "i": i,
                "r_true": res.r_true,
                "r_meas": res.r_meas,
                "nlos": res.is_nlos,
                "bias": res.bias,
                "noise": res.noise,
            })

        self.last_ranges = out
        self._scroll_ranges_list(0)  # ajusta scroll se necessário

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

        lines = [
            f"dt: {self.ranging_cfg.dt:.2f}s",
            f"Noise [N]: {noise_txt}",
            f"NLOS  [L]: {'ON' if nlos_on else 'OFF'}",
            f"Drop  [P]: {'ON' if drop_on else 'OFF'}",
            f"Quant [Q]: {q_txt}",
            f"H: ranges {'ON' if self.show_ranges else 'OFF'}",
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
