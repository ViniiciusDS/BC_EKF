# src/ui/map_editor.py

import os
from typing import Optional, Tuple

import numpy as np
import pygame as pg

from src.environment.environment import Environment, Obstacle
from src.environment.noise_zones import make_rect_noise_zone, point_in_noise_zone
from src.environment.noise_profiles import list_noise_profiles, noise_profile_label
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

        # --- Estado do desenho de zonas de ruído ---
        self.edit_mode: str = "wall"   # "wall" | "noise_zone"
        self.zone_start_pt: Optional[Tuple[float, float]] = None
        self.zone_preview_pt: Optional[Tuple[float, float]] = None
        self.zone_profiles = list_noise_profiles()
        self.zone_profile: str = "medium_noise"
        self.zone_hover_pt: Optional[Tuple[float, float]] = None  # para destacar zona sob o mouse

        # --- Nome do mapa + dropdown ---
        existing_maps = list_map_files(MAPS_DIR)
        self.filename: str = existing_maps[0] if existing_maps else "mapa1"

        # --- Dropdown de perfil de ruído ---
        self.zone_profile_box = TextBoxDropdown(
            rect=(0, 0, 200, 28),
            font=self.font,
            options=self.zone_profiles,
            placeholder="Perfil da zona",
        )
        self.zone_profile_box.set_text(self.zone_profile)
        self.zone_profile_box.options_filtered = list(self.zone_profiles)
        self.zone_profile_box.dropdown_open = False

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

        self.btn_toggle_zone_rect = pg.Rect(0, 0, 160, 32)
        self.btn_clear_preview_rect = pg.Rect(0, 0, 160, 32)

        # --- Mensagem popup ---
        self.msg: Optional[str] = None
        self.msg_timer: float = 0.0

    # ------------------------------------------------------------------
    #  API pública usada pelo main_interactive
    # ------------------------------------------------------------------

    def reset_preview(self) -> None:
        """Limpa qualquer desenho em construção."""
        self.first_pt = None
        self.preview_pt = None
        self.zone_start_pt = None
        self.zone_preview_pt = None
        self.zone_hover_pt = None
        if self.zone_profile_box:
            self.zone_profile_box.dropdown_open = False
            self.zone_profile_box.options_filtered = list(self.zone_profiles)

    def handle_event(self, event: pg.event.Event, cam, viewport_width: int) -> None:
        """
        Processa eventos específicos do editor, exceto:
          - ESC (sair do editor)
          - zoom (scroll)
          - pan (botão do meio)
        que continuam tratados no main_interactive.
        """
        # Primeiro deixa os TextBoxDropdown tentarem consumir o evento
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

            # dropdown de perfil da zona (com clique)
            if self._handle_zone_profile_dropdown_click(mx, my):
                return

            # Clique na área do mapa
            if mx < viewport_width:
                wx, wy = cam.screen_to_world(mx, my)

                if self.edit_mode == "wall":
                    # LMB: desenhar paredes (2 cliques = 1 parede)
                    if event.button == 1:
                        self._handle_left_click(wx, wy)
                    # RMB: remover parede mais próxima
                    elif event.button == 3:
                        self._handle_right_click(wx, wy)

                elif self.edit_mode == "noise_zone":
                    # LMB: clique e arraste para zona
                    if event.button == 1:
                        self._handle_zone_left_down(wx, wy)
                    # RMB: remover zona sob o clique
                    elif event.button == 3:
                        self._handle_zone_right_click(wx, wy)

            # Clique na HUD lateral
            else:
                if self._handle_hud_click(mx, my):
                    return

        elif event.type == pg.MOUSEMOTION:
            mx, my = event.pos

            if mx < viewport_width:
                wx, wy = cam.screen_to_world(mx, my)
                self.zone_hover_pt = (wx, wy)

                # preview de parede
                if self.edit_mode == "wall" and self.first_pt is not None:
                    self.preview_pt = (wx, wy)

                # preview de zona
                if self.edit_mode == "noise_zone" and self.zone_start_pt is not None:
                    self.zone_preview_pt = (wx, wy)
            else:
                self.zone_hover_pt = None

        elif event.type == pg.MOUSEBUTTONUP:
            mx, my = event.pos
            if self.edit_mode == "noise_zone" and event.button == 1 and mx < viewport_width:
                wx, wy = cam.screen_to_world(mx, my)
                self._handle_zone_left_up(wx, wy)

    def update(self, dt: float) -> None:
        """Atualiza animações e timers (ex.: dropdown, popup)."""
        if self.name_box:
            self.name_box.update(dt)

        if self.zone_profile_box:
            self.zone_profile_box.update(dt)

        if self.msg_timer > 0:
            self.msg_timer -= dt
            if self.msg_timer <= 0:
                self.msg = None
                self.msg_timer = 0.0

    def draw_preview(self, screen: pg.Surface, cam) -> None:
        """Desenha a parede ou zona em construção."""
        # Preview de parede
        if self.edit_mode == "wall" and self.first_pt is not None and self.preview_pt is not None:
            p0s = cam.world_to_screen(*self.first_pt)
            p1s = cam.world_to_screen(*self.preview_pt)
            pg.draw.line(screen, (0, 160, 0), p0s, p1s, 2)
            pg.draw.circle(screen, (0, 200, 0), p0s, 4)
            pg.draw.circle(screen, (0, 200, 0), p1s, 4)

        # Preview de zona
        if self.edit_mode == "noise_zone" and self.zone_start_pt is not None and self.zone_preview_pt is not None:
            p0 = cam.world_to_screen(*self.zone_start_pt)
            p1 = cam.world_to_screen(*self.zone_preview_pt)

            left = min(p0[0], p1[0])
            top = min(p0[1], p1[1])
            width = abs(p1[0] - p0[0])
            height = abs(p1[1] - p0[1])

            if width > 0 and height > 0:
                overlay = pg.Surface((width, height), pg.SRCALPHA)
                overlay.fill((255, 180, 0, 60))
                screen.blit(overlay, (left, top))
                pg.draw.rect(screen, (255, 140, 0), (left, top, width, height), 2)
    
    def get_zone_highlight_point(self):
        '''Retorna as coordenadas do mundo para destacar a zona sob o mouse, ou None se não estivermos
          editando zonas ou o mouse não estiver sobre uma zona.'''
        return self.zone_hover_pt if self.edit_mode == "noise_zone" else None

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

        draw_text(screen, f"Modo atual: {'Paredes' if self.edit_mode == 'wall' else 'Zonas'}", sidebar_x, y, bigfont); y += LINE_H

        if self.edit_mode == "wall":
            draw_text(screen, "Desenho de paredes:", sidebar_x, y, bigfont); y += LINE_H
            draw_text(screen, "LMB: 1º clique = início parede", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "LMB: 2º clique = fim parede", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "RMB: remover parede mais próxima", sidebar_x, y, font); y += LINE_H

            y += 8
            draw_text(screen, f"Material atual: {self.material}", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "M: trocar material", sidebar_x, y, font); y += LINE_H
        else:
            draw_text(screen, "Desenho de zonas:", sidebar_x, y, bigfont); y += LINE_H
            draw_text(screen, "LMB: clique e arraste retângulo", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "RMB: remover zona sob o clique", sidebar_x, y, font); y += LINE_H
            draw_text(screen, f"Perfil atual: {noise_profile_label(self.zone_profile)}", sidebar_x, y, font); y += LINE_H

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

        # Controles de zona
        self.btn_toggle_zone_rect.topleft = (sidebar_x, y)
        toggle_label = "Editar zonas" if self.edit_mode == "wall" else "Editar paredes"
        self._draw_button(screen, self.btn_toggle_zone_rect, toggle_label, font)
        y += 40

        self.btn_clear_preview_rect.topleft = (sidebar_x, y)
        self._draw_button(screen, self.btn_clear_preview_rect, "Cancelar desenho", font)
        y += 40

        draw_text(screen, "Perfil da zona:", sidebar_x, y, font); y += 25
        self.zone_profile_box.rect.topleft = (sidebar_x, y)
        self.zone_profile_box.options_filtered = list(self.zone_profiles)
        self.zone_profile_box.draw(screen)
        y = self.zone_profile_box.rect.bottom + 10

        if self.zone_profile_box.dropdown_open:
            num = min(self.zone_profile_box._max_visible, len(self.zone_profiles))
            y += num * self.zone_profile_box._item_height + 10

        # Botões Salvar / Carregar
        self.btn_save_as_rect.topleft = (sidebar_x, y)
        self._draw_button(screen, self.btn_save_as_rect, "Salvar como", font)
        y += 40

        self.btn_load_rect.topleft = (sidebar_x, y)
        self._draw_button(screen, self.btn_load_rect, "Carregar (nome)", font)
        y += 40

        # Status
        y += 10
        if self.edit_mode == "wall":
            if self.first_pt is not None:
                draw_text(screen, "Status: definindo fim da parede...", sidebar_x, y, font); y += LINE_H
            else:
                draw_text(screen, "Status: pronto para novo segmento", sidebar_x, y, font); y += LINE_H
        else:
            if self.zone_start_pt is not None:
                draw_text(screen, "Status: arrastando zona de ruído...", sidebar_x, y, font); y += LINE_H
            else:
                draw_text(screen, "Status: pronto para nova zona", sidebar_x, y, font); y += LINE_H

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

    def _handle_zone_left_down(self, wx: float, wy: float) -> None:
        '''Inicia o desenho de uma zona de ruído. O retângulo é definido por zone_start_pt e zone_preview_pt.'''
        # evita reiniciar uma zona se já estivermos no meio do desenho
        if self.zone_start_pt is not None:
            return

        self.zone_start_pt = (wx, wy)
        self.zone_preview_pt = (wx, wy)

    def _handle_zone_left_up(self, wx: float, wy: float) -> None:
        '''Finaliza o desenho da zona de ruído, adicionando-a ao ambiente se a área for grande o suficiente.'''
        if self.zone_start_pt is None:
            return

        x0, y0 = self.zone_start_pt
        x1, y1 = wx, wy

        w = x1 - x0
        h = y1 - y0

        if abs(w) > 1e-3 and abs(h) > 1e-3:
            zone = make_rect_noise_zone(
                x0,
                y0,
                w,
                h,
                self.zone_profile,
            )
            self.env.add_noise_zone(zone)
            self._set_msg(f"Zona adicionada: {noise_profile_label(self.zone_profile)}")
        else:
            self._set_msg("Zona ignorada: área muito pequena")

        self.zone_start_pt = None
        self.zone_preview_pt = None

    def _handle_zone_right_click(self, wx: float, wy: float) -> None:
        zones = getattr(self.env, "noise_zones", None)
        if not zones:
            return

        # remove a pior/última zona encontrada sob o clique
        for idx in range(len(zones) - 1, -1, -1):
            if point_in_noise_zone(wx, wy, zones[idx]):
                zones.pop(idx)
                break

    def _handle_hud_click(self, mx: int, my: int) -> bool:
        """Trata clique em botões e textbox da HUD lateral."""
        if self.btn_save_as_rect.collidepoint(mx, my):
            self._save_map()
            return True

        if self.btn_load_rect.collidepoint(mx, my):
            self._load_map()
            return True

        if self.btn_toggle_zone_rect.collidepoint(mx, my):
            self.reset_preview()
            self.edit_mode = "noise_zone" if self.edit_mode == "wall" else "wall"
            if self.zone_profile_box:
                self.zone_profile_box.dropdown_open = False
            return True

        if self.btn_clear_preview_rect.collidepoint(mx, my):
            self.reset_preview()
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
            self.reset_preview()
            if self.zone_profile_box:
                self.zone_profile_box.dropdown_open = False
            self._set_msg("Mapa carregado!")
        except Exception as e:
            self._set_msg(f"Erro: {e}")

    def _set_msg(self, msg: str, duration: float = 2.0) -> None:
        self.msg = msg
        self.msg_timer = duration

    def _draw_button(self, screen: pg.Surface, rect: pg.Rect, label: str, font: pg.font.Font) -> None:
        '''Desenha um botão simples com texto. O clique é tratado em _handle_hud_click().'''
        pg.draw.rect(screen, (220, 220, 220), rect, border_radius=4)
        pg.draw.rect(screen, BLACK, rect, 1, border_radius=4)
        txt = font.render(label, True, BLACK)
        screen.blit(txt, (rect.x + 10, rect.y + (rect.height - txt.get_height()) // 2))

    def _handle_zone_profile_dropdown_click(self, mx: int, my: int) -> bool:
        """
        Faz o zone_profile_box funcionar como dropdown de seleção direta,
        sem depender de digitação.
        """
        box = self.zone_profile_box
        if box is None:
            return False

        # clique no campo principal -> abre/fecha lista
        if box.rect.collidepoint(mx, my):
            box.dropdown_open = not box.dropdown_open
            box.options_filtered = list(self.zone_profiles)
            return True

        # clique em item da lista aberta
        if box.dropdown_open and box.options_filtered:
            num = min(box._max_visible, len(box.options_filtered))
            drop_rect = pg.Rect(
                box.rect.left,
                box.rect.bottom,
                box.rect.width,
                num * box._item_height,
            )

            if drop_rect.collidepoint(mx, my):
                idx = (my - drop_rect.top) // box._item_height
                idx = int(idx)

                if 0 <= idx < len(box.options_filtered[:num]):
                    choice = box.options_filtered[idx]
                    self.zone_profile = choice
                    box.set_text(choice)
                    box.dropdown_open = False
                    box.options_filtered = list(self.zone_profiles)
                    return True

            # clicou fora da lista aberta -> fecha
            box.dropdown_open = False

        return False