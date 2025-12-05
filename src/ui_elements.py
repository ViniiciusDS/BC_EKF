# src/ui_elements.py
import pygame as pg
import time

# common colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BORDER = (0, 0, 0)
BG_INACTIVE = (250, 250, 250)
BG_ACTIVE = (255, 255, 255)
BG_DROPDOWN = (245, 245, 245)


class TextBoxDropdown:
    """
    Caixa de texto com dropdown de opções (auto-complete simples).

    Uso:
      name_box = TextBoxDropdown(rect=(x, y, w, h), font=font, options=list_map_files(...))

    No loop:
      if name_box.handle_event(event):
          editor_filename = name_box.text

      name_box.update(dt)
      name_box.draw(screen)
    """
    def __init__(self, rect, font, options, placeholder="", max_len=32):
        self.rect = pg.Rect(rect)
        self.font = font
        self.options_all = list(options)          # lista completa (ex.: arquivos)
        self.options_filtered = list(options)     # lista filtrada com base em self.text
        self.text = ""
        self.cursor_pos = 0
        self.active = False                       # se está com foco
        self.dropdown_open = False                # se lista está aberta
        self.max_visible = 6                      # máx. itens visíveis ao mesmo tempo
        self.line_h = self.rect.h                # altura de cada linha no dropdown

        # cursor piscando
        self.blink_timer = 0.0
        self.show_cursor = True

        self.max_len = max_len                         # máx. caracteres no texto

    # -------------------------------------------------
    # Utilidades
    # -------------------------------------------------
    def set_text(self, text: str):
        self.text = text or ""
        self.cursor_pos = len(self.text)
        self.update_filter()

    def update_filter(self):
        t = self.text.lower().strip()
        if not t:
            self.options_filtered = list(self.options_all)
        else:
            self.options_filtered = [
                opt for opt in self.options_all
                if t in opt.lower()
            ]
    
    # -------------------------------------------------
    # Eventos, atualização e desenho
    # -------------------------------------------------
    def handle_event(self, event):
        """
        Retorna True se o texto foi alterado (via teclado ou clique).
        """
        changed = False

        # ===== Clique do mouse =====
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # clique dentro da caixa → foca e abre dropdown
            if self.rect.collidepoint(mx, my):
                self.active = True
                self.dropdown_open = True
                return False    # não mudou o texto
            
            # clique no dropdown
            if self.dropdown_open:
                drop_rect = self._dropdown_rect()
                if drop_rect.collidepoint(mx, my):
                    idx = (my - drop_rect.y) // self.line_h
                    opts = self.options_filtered[: self.max_visible]
                    if 0 <= idx < len(opts):
                        self.text = opts[idx]
                        self.cursor_pos = len(self.text)
                        self.dropdown_open = False
                        self.active = False
                        self.update_filter()
                        return True  # texto mudou
                # clique fora → desfoca e fecha dropdown
                else:
                    self.active = False
                    self.dropdown_open = False
                    return False    # não mudou o texto
            # Clicou em outro lugar
            self.active = False
            return False    # não mudou o texto

        # ===== Teclado =====
        if event.type == pg.KEYDOWN and self.active:
            if event.key == pg.K_RETURN:
                # Fecha dropdown e mantém o texto
                self.active = False
                self.dropdown_open = False
                return True  # texto confirmado
            if event.key == pg.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
                    changed = True

            elif event.key == pg.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
                    changed = True

            elif event.key == pg.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)

            elif event.key == pg.K_RIGHT:
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)

            elif event.key == pg.K_HOME:
                self.cursor_pos = 0

            elif event.key == pg.K_END:
                self.cursor_pos = len(self.text)

            else:
                if event.unicode and event.unicode.isprintable() and len(self.text) < self.max_len:
                    self.text = (
                        self.text[:self.cursor_pos] +
                        event.unicode +
                        self.text[self.cursor_pos:]
                    )
                    self.cursor_pos += 1
                    changed = True

            if changed:
                self.dropdown_open = True
                self.update_filter()

        return changed

    # -------------------------------------------------
    # Atualização e desenho
    # -------------------------------------------------
    def update(self, dt):
        # ===== cursor piscando =====
        self.blink_timer += dt
        if self.blink_timer >= 0.5:
            self.blink_timer = 0.0
            self.show_cursor = not self.show_cursor
    def _dropdown_rect(self):
        # Retorna o retângulo do dropdown baseado no número de itens filtrados  
        num = min(self.max_visible, len(self.options_filtered))
        return pg.Rect(self.rect.x, self.rect.bottom, self.rect.w, self.line_h * num)

    def draw(self, surface):
        # ===== Caixa =====
        bg = BG_ACTIVE if self.active else BG_INACTIVE
        pg.draw.rect(surface, bg, self.rect)
        pg.draw.rect(surface, BORDER, self.rect, 1)

        # texto da caixa
        text_surface = self.font.render(self.text, True, BLACK)
        surface.blit(text_surface, (self.rect.x + 4, self.rect.y + 4))

        # ===== cursor =====
        if self.active and self.show_cursor:
            prefix = self.text[:self.cursor_pos]
            cursor_x = self.rect.x + 4 + self.font.size(prefix)[0]
            cursor_y0 = self.rect.y + 4
            cursor_y1 = self.rect.y + self.rect.h - 4
            pg.draw.line(surface, BLACK, (cursor_x, cursor_y0), (cursor_x, cursor_y1), 1)

        # ===== dropdown =====
        if self.dropdown_open and self.options_filtered:
            drop_rect = self._dropdown_rect()
            
            # fundo
            pg.draw.rect(surface, BG_DROPDOWN, drop_rect)
            pg.draw.rect(surface, BORDER, drop_rect, 1)

            visible = self.options_filtered[:self.max_visible]
            y = drop_rect.y
            # itens individuais
            for opt in visible:
                line_rect = pg.Rect(drop_rect.x, y, drop_rect.w, self.line_h)

                # fundo da opção
                pg.draw.rect(surface, BG_DROPDOWN, line_rect)

                # borda individual da opção
                pg.draw.rect(surface, BORDER, line_rect, 1)

                # texto
                txt = self.font.render(opt, True, BLACK)
                surface.blit(txt, (line_rect.x + 4, line_rect.y + 4))

                y += self.line_h

    # ==== DETECTA CLIQUES EM ITENS DO DROPDOWN ==== 
    def dropdown_click(self, pos):
        if not self.show_dropdown:
            return None
        
        item_h = self.rect.height
        for i, opt in enumerate(self.options_filtered):
            item_rect = pg.Rect(self.rect.x, self.rect.bottom + i*item_h, self.rect.width, item_h)
            if item_rect.collidepoint(pos):
                self.text = opt
                self.cursor_pos = len(self.text)
                self.show_dropdown = False
                return opt
        return None
