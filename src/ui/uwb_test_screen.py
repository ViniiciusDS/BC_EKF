# src/ui/uwb_test_screen.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pygame as pg
import math

from src.ui.drawing import draw_axes, draw_grid
from src.ui.ui_elements import TextBoxDropdown
from src.ui.botton import Button

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
                if mx < cam_w:
                    factor = 1.15 if event.y > 0 else 1/1.15
                    self.cam.zoom_at((mx, my), factor)
                    continue

            # compatibilidade wheel antigo (button 4/5)
            if event.type == pg.MOUSEBUTTONDOWN and event.button in (4, 5):
                mx, my = event.pos
                if mx < cam_w:
                    self.cam.zoom_at((mx, my), 1.15 if event.button == 4 else 1/1.15)
                    continue

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    actions.go_to_menu = True
                    return actions

                consumed = False
                if self.textbox_ax.handle_event(event):
                    consumed = True
                if (not consumed) and self.textbox_ay.handle_event(event):
                    consumed = True
                if consumed:
                    continue
            
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
                        self.anchors_scroll = 0
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

            # Scroll da lista de âncoras (mouse wheel)
            if event.type == pg.MOUSEWHEEL:
                mx, my = pg.mouse.get_pos()
                if self.anchor_list_rect and self.anchor_list_rect.collidepoint((mx, my)):
                    self._scroll_anchor_list(-event.y)  # wheel up = +1 no event.y
                    continue

        return actions

    def update(self, dt: float) -> None:
        self.textbox_ax.update(dt)
        self.textbox_ay.update(dt)
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

        # lista rolável
        self._draw_anchor_list()
                

    def close(self) -> None:
        # Etapa 1 não tem processos/arquivos pra fechar
        pass

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
        y += self.btn_clear_anchors.rect.h + 14

        # marca onde termina a parte de ferramentas
        self._hud_y_after_tools = y

        # ===== área fixa da lista rolável =====
        y_title = y
        y_list = y_title + 22

        list_h = self.anchor_visible * self.anchor_line_h + 10
        list_w = self.SIDE_W - 32

        self.anchor_list_rect = pg.Rect(sidebar_x, y_list, list_w, list_h)

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
