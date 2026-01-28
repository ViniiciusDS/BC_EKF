# src/ui/map_editor.py

import os
from typing import Optional, Tuple

import numpy as np
import pygame as pg

from src.environment import Environment, Obstacle
from src.ui.ui_elements import TextBoxDropdown
from src.utils import list_map_files, map_file_path
from src.config import MAPS_DIR


BLACK = (0, 0, 0)


class MapEditorScreen:
    """
    Gerencia o estado e a UI do editor de mapas.

    Responsabilidades:
      - Criar/remover paredes no Environment.
      - Gerenciar material da parede.
      - Gerenciar o nome do arquivo de mapa (TextBoxDropdown).
      - Salvar/carregar mapas.
      - Mensagens popup de feedback.
      - Desenhar HUD lateral do editor e preview da parede.

    A câmera (cam) e o desenho do mapa (grid, environment, eixos) 
    são responsabilidade do main_interactive.
    """

    def __init__(
        self,
        env: Environment,
        font: pg.font.Font,
        bigfont: pg.font.Font,
        side_width: int,
    ) -> None:
        self.env = env
        self.font = font
        self.bigfont = bigfont
        self.side_width = side_width

        # --- Estado do desenho de paredes ---
        self.first_pt: Optional[Tuple[float, float]] = None
        self.preview_pt: Optional[Tuple[float, float]] = None
        self.material: str = "metal"

        # --- Nome do mapa + dropdown ---
        existing_maps = list_map_files(MAPS_DIR)
        self.filename: str = existing_maps[0] if existing_maps else "mapa1"

        # A posição da caixa será ajustada em draw_sidebar()
        self.name_box = TextBoxDropdown(
            rect=(0, 0, 200, 28),
            font=self.font,
            options=existing_maps,
            placeholder="Nome do mapa",
        )
        self.name_box.set_text(self.filename)

        # --- Botões (os retângulos serão reposicionados em draw_sidebar) ---
        self.btn_save_as_rect = pg.Rect(0, 0, 160, 32)
        self.btn_load_rect = pg.Rect(0, 0, 160, 32)

        # --- Mensagem popup ---
        self.msg: Optional[str] = None
        self.msg_timer: float = 0.0

    # ------------------------------------------------------------------
    #  API pública usada pelo main_interactive
    # ------------------------------------------------------------------

    def reset_preview(self) -> None:
        """Limpa qualquer segmento em construção."""
        self.first_pt = None
        self.preview_pt = None

    def handle_event(self, event: pg.event.Event, cam, viewport_width: int) -> None:
        """
        Processa eventos específicos do editor, exceto:
          - ESC (sair do editor)
          - zoom (scroll)
          - pan (botão do meio)
        que continuam tratados no main_interactive.
        """
        # Primeiro deixa o TextBoxDropdown tentar consumir o evento
        if self.name_box and self.name_box.handle_event(event):
            self.filename = self.name_box.text.strip() or self.filename
            return

        # Teclado
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_m:
                self._cycle_material()
            return

        # Mouse
        if event.type == pg.MOUSEBUTTONDOWN:
            mx, my = event.pos

            # Clique na área do mapa
            if mx < viewport_width:
                wx, wy = cam.screen_to_world(mx, my)

                # LMB: desenhar paredes (2 cliques = 1 parede)
                if event.button == 1:
                    self._handle_left_click(wx, wy)
                # RMB: remover parede mais próxima
                elif event.button == 3:
                    self._handle_right_click(wx, wy)

            # Clique na HUD lateral
            else:
                if self._handle_hud_click(mx, my):
                    return

        elif event.type == pg.MOUSEMOTION:
            mx, my = event.pos
            # atualiza preview da parede se já temos o primeiro ponto
            if self.first_pt is not None and mx < viewport_width:
                self.preview_pt = cam.screen_to_world(mx, my)

    def update(self, dt: float) -> None:
        """Atualiza animações e timers (ex.: dropdown, popup)."""
        if self.name_box:
            self.name_box.update(dt)

        if self.msg_timer > 0:
            self.msg_timer -= dt
            if self.msg_timer <= 0:
                self.msg = None
                self.msg_timer = 0.0

    def draw_preview(self, screen: pg.Surface, cam) -> None:
        """Desenha a linha “fantasma” da parede em construção."""
        if self.first_pt is not None and self.preview_pt is not None:
            p0s = cam.world_to_screen(*self.first_pt)
            p1s = cam.world_to_screen(*self.preview_pt)
            pg.draw.line(screen, (0, 160, 0), p0s, p1s, 2)
            pg.draw.circle(screen, (0, 200, 0), p0s, 4)
            pg.draw.circle(screen, (0, 200, 0), p1s, 4)

    def draw_sidebar(self, screen: pg.Surface, cam, font, bigfont, side_color=(245, 245, 245)) -> None:
        """
        Desenha o HUD lateral do editor (texto, textbox, botões, popup).

        """
        # Área lateral
        pg.draw.rect(screen, side_color, (cam.viewport[0], 0, self.side_width, cam.viewport[1]))
        sidebar_x = cam.viewport[0] + 16
        y = 18
        LINE_H = 22

        def draw_text(surface, text, x, y, font, color=BLACK):
            img = font.render(text, True, color)
            surface.blit(img, (x, y))

        # Cabeçalho e instruções
        draw_text(screen, "Editor de mapa (beta)", sidebar_x, y, bigfont); y += 34
        draw_text(screen, "ESC: voltar ao menu", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "Scroll: zoom  |  Botão do meio: pan", sidebar_x, y, font); y += LINE_H
        y += 8
        draw_text(screen, "Desenho de paredes:", sidebar_x, y, bigfont); y += LINE_H
        draw_text(screen, "LMB: 1º clique = início parede", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "LMB: 2º clique = fim parede", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "RMB: remover parede mais próxima", sidebar_x, y, font); y += LINE_H

        y += 8
        draw_text(screen, f"Material atual: {self.material}", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "M: trocar material", sidebar_x, y, font); y += LINE_H

        # Nome do mapa + textbox
        draw_text(screen, "Nome do mapa:", sidebar_x, y, font); y += 25

        if self.name_box:
            self.name_box.rect.topleft = (sidebar_x, y)
            self.name_box.draw(screen)

            y = self.name_box.rect.bottom + 10

            if self.name_box.dropdown_open and self.name_box.options_filtered:
                num = min(self.name_box._max_visible, len(self.name_box.options_filtered))
                y += num * self.name_box._item_height + 10
        else:
            y += 60

        # Botões Salvar / Carregar
        self.btn_save_as_rect.topleft = (sidebar_x, y)
        self._draw_button(screen, self.btn_save_as_rect, "Salvar como", font)
        y += 40

        self.btn_load_rect.topleft = (sidebar_x, y)
        self._draw_button(screen, self.btn_load_rect, "Carregar (nome)", font)
        y += 40

        # Status
        y += 10
        if self.first_pt is not None:
            draw_text(screen, "Status: definindo fim da parede...", sidebar_x, y, font); y += LINE_H
        else:
            draw_text(screen, "Status: pronto para novo segmento", sidebar_x, y, font); y += LINE_H

        # Popup flutuante
        if self.msg:
            msg_img = font.render(self.msg, True, BLACK)
            pad = 10
            box_w = msg_img.get_width() + 2 * pad
            box_h = msg_img.get_height() + 2 * pad

            box_x = cam.viewport[0] + (self.side_width - box_w) // 2
            box_y = cam.viewport[1] - box_h - 20

            pg.draw.rect(screen, (255, 255, 210), (box_x, box_y, box_w, box_h), border_radius=6)
            pg.draw.rect(screen, BLACK, (box_x, box_y, box_w, box_h), 1, border_radius=6)
            screen.blit(msg_img, (box_x + pad, box_y + pad))

    # ------------------------------------------------------------------
    #  Lógica interna
    # ------------------------------------------------------------------

    def _handle_left_click(self, wx: float, wy: float) -> None:
        """Seleciona first_pt ou cria obstáculo."""
        if self.first_pt is None:
            self.first_pt = (wx, wy)
            self.preview_pt = (wx, wy)
        else:
            p0 = np.array(self.first_pt, dtype=float)
            p1 = np.array([wx, wy], dtype=float)
            if np.linalg.norm(p1 - p0) > 1e-3:
                self.env.add(Obstacle(p0, p1, material=self.material))
            self.first_pt = None
            self.preview_pt = None

    def _handle_right_click(self, wx: float, wy: float) -> None:
        """Remove a parede mais próxima dentro de um raio."""
        if self.env is None or not self.env.obstacles:
            return

        p_click = np.array([wx, wy], dtype=float)
        min_dist = float("inf")
        min_idx = None

        from src.utils import point_segment_distance  # evitar import circular no topo

        for idx, obs in enumerate(self.env.obstacles):
            d = point_segment_distance(p_click, obs.p0, obs.p1)
            if d < min_dist:
                min_dist = d
                min_idx = idx

        if min_idx is not None and min_dist < 0.4:
            self.env.obstacles.pop(min_idx)

    def _handle_hud_click(self, mx: int, my: int) -> bool:
        """Trata clique em botões e textbox da HUD lateral."""
        # TextBox já tentou handle_event antes, aqui lida só com botões
        if self.btn_save_as_rect.collidepoint(mx, my):
            self._save_map()
            return True

        if self.btn_load_rect.collidepoint(mx, my):
            self._load_map()
            return True

        return False

    def _cycle_material(self) -> None:
        mats = ["metal", "wall", "glass", "human"]
        try:
            idx = mats.index(self.material)
        except ValueError:
            idx = 0
        self.material = mats[(idx + 1) % len(mats)]
        print("Material atual do editor:", self.material)

    def _save_map(self) -> None:
        name = self.name_box.text.strip() if self.name_box else self.filename
        if not name:
            self._set_msg("Nome inválido")
            return

        self.filename = name
        path = map_file_path(MAPS_DIR, self.filename)
        try:
            self.env.save_json(path)
            self._set_msg(f"Mapa salvo como: {os.path.basename(path)}")
            # atualiza lista de mapas
            self.name_box.options_all = list_map_files(MAPS_DIR)
            self.name_box.update_filter()
        except Exception as e:
            self._set_msg(f"Erro ao salvar: {e}")

    def _load_map(self) -> None:
        name = self.name_box.text.strip() if self.name_box else self.filename
        if not name:
            self._set_msg("Nome inválido")
            return

        self.filename = name
        path = map_file_path(MAPS_DIR, self.filename)
        if not os.path.exists(path):
            self._set_msg("Arquivo não encontrado!")
            return

        try:
            self.env = Environment.load_json(path)
            self._set_msg("Mapa carregado!")
        except Exception as e:
            self._set_msg(f"Erro: {e}")

    def _set_msg(self, msg: str, duration: float = 2.0) -> None:
        self.msg = msg
        self.msg_timer = duration

    def _draw_button(self, screen: pg.Surface, rect: pg.Rect, label: str, font: pg.font.Font) -> None:
        pg.draw.rect(screen, (220, 220, 220), rect, border_radius=4)
        pg.draw.rect(screen, BLACK, rect, 1, border_radius=4)
        txt = font.render(label, True, BLACK)
        screen.blit(txt, (rect.x + 10, rect.y + (rect.height - txt.get_height()) // 2))