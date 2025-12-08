# src/ui_elements.py
import pygame as pg
from typing import List, Sequence, Tuple, Union

# Cores comuns
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BORDER = (0, 0, 0)
BG_INACTIVE = (250, 250, 250)
BG_ACTIVE = (255, 255, 255)
BG_DROPDOWN = (245, 245, 245)
PLACEHOLDER_COLOR = (150, 150, 150)


RectArg = Union[pg.Rect, Tuple[int, int, int, int]]


class TextBoxDropdown:
    """
    Caixa de texto com dropdown de opções (auto-complete simples).

    - Clique na caixa: ativa edição, abre dropdown (se houver opções).
    - Digitação filtra as opções (case-insensitive, por substring).
    - ENTER confirma o texto.
    - Clique em uma opção do dropdown seleciona e confirma.
    - Fora da caixa/dropdown: desfoca e fecha dropdown.

    Atributos usados pelo código externo:
        - text: str
        - cursor_pos: int
        - options_all: List[str]
        - options_filtered: List[str]

    Métodos usados pelo código externo:
        - set_text(str)
        - update_filter()
        - handle_event(event) -> bool    (True = texto foi alterado/confirmado)
        - update(dt)
        - draw(surface)
    """

    def __init__(
        self,
        rect: RectArg,
        font: pg.font.Font,
        options: Sequence[str],
        placeholder: str = "",
        max_len: int = 32,
    ) -> None:
        # Garante pg.Rect
        self.rect: pg.Rect = rect if isinstance(rect, pg.Rect) else pg.Rect(rect)
        self.font = font

        # Texto
        self.text: str = ""
        self.placeholder: str = placeholder
        self.cursor_pos: int = 0
        self.max_len: int = max_len

        # Estado de foco e dropdown
        self.active: bool = False          # caixa está em edição?
        self.dropdown_open: bool = False   # dropdown visível?

        # Opções
        self.options_all: List[str] = list(options)
        self.options_filtered: List[str] = list(options)

        # Cursor piscante
        self._cursor_visible: bool = True
        self._cursor_timer: float = 0.0
        self._cursor_blink_period: float = 0.5  # segundos

        # Layout do dropdown
        self._item_height: int = self.rect.height
        self._max_visible: int = 6  # máximo de itens visíveis no dropdown

    # ======================================================================
    # API PÚBLICA
    # ======================================================================

    def set_text(self, text: str) -> None:
        """Define o texto manualmente e ajusta o cursor + filtro."""
        self.text = text[: self.max_len]
        self.cursor_pos = len(self.text)
        self.update_filter()

    def update_filter(self) -> None:
        """Atualiza a lista filtrada de opções com base no texto atual."""
        t = self.text.strip().lower()
        if not t:
            self.options_filtered = list(self.options_all)
        else:
            self.options_filtered = [
                opt for opt in self.options_all if t in opt.lower()
            ]

        # Se não houver opções filtradas, ainda podemos deixar o dropdown
        # fechado para não abrir uma caixa vazia automaticamente
        if not self.options_filtered:
            self.dropdown_open = False

    def handle_event(self, event: pg.event.Event) -> bool:
        """
        Processa eventos de mouse/teclado.

        Retorna:
            True  -> texto foi alterado e/ou confirmado (ENTER, clique).
            False -> texto não mudou.
        """
        # =========================
        # Mouse
        # =========================
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            # Clique dentro da caixa de texto
            if self.rect.collidepoint(mouse_pos):
                self.active = True
                # (Re)abre dropdown se houver opções
                if self.options_filtered:
                    self.dropdown_open = True
                return False  # Apenas foco, sem mudar texto ainda

            # Se dropdown está aberto, checa clique nos itens
            if self.dropdown_open:
                clicked_option = self._handle_dropdown_click(mouse_pos)
                if clicked_option is not None:
                    # Texto mudou por clique no dropdown
                    return True

            # Clique em qualquer outro lugar: fecha tudo
            self.active = False
            self.dropdown_open = False
            return False

        # =========================
        # Teclado
        # =========================
        if event.type == pg.KEYDOWN and self.active:
            # ENTER confirma texto atual
            if event.key == pg.K_RETURN:
                self.active = False
                self.dropdown_open = False
                return True  # confirmação

            # ESC cancela foco
            if event.key == pg.K_ESCAPE:
                self.active = False
                self.dropdown_open = False
                return False

            # Movimentação de cursor
            if event.key == pg.K_LEFT:
                if self.cursor_pos > 0:
                    self.cursor_pos -= 1
                return False
            if event.key == pg.K_RIGHT:
                if self.cursor_pos < len(self.text):
                    self.cursor_pos += 1
                return False
            if event.key == pg.K_HOME:
                self.cursor_pos = 0
                return False
            if event.key == pg.K_END:
                self.cursor_pos = len(self.text)
                return False

            # Backspace / Delete
            changed = False
            if event.key == pg.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[: self.cursor_pos - 1] + self.text[self.cursor_pos :]
                    self.cursor_pos -= 1
                    changed = True
            elif event.key == pg.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[: self.cursor_pos] + self.text[self.cursor_pos + 1 :]
                    changed = True

            # Caracteres "imprimíveis"
            elif event.unicode and event.unicode.isprintable():
                if len(self.text) < self.max_len:
                    self.text = (
                        self.text[: self.cursor_pos]
                        + event.unicode
                        + self.text[self.cursor_pos :]
                    )
                    self.cursor_pos += 1
                    changed = True

            if changed:
                self.update_filter()
                # Ao digitar, abri o dropdown se houver correspondências
                if self.options_filtered:
                    self.dropdown_open = True
                return True

        return False

    def update(self, dt: float) -> None:
        """Atualiza animação do cursor (piscar)."""
        if self.active:
            self._cursor_timer += dt
            if self._cursor_timer >= self._cursor_blink_period:
                self._cursor_timer = 0.0
                self._cursor_visible = not self._cursor_visible
        else:
            # Se não está ativo, cursor apagado
            self._cursor_visible = False
            self._cursor_timer = 0.0

    def draw(self, surface: pg.Surface) -> None:
        """Desenha a caixa de texto e, se aberto, o dropdown."""
        # Fundo da caixa
        bg_color = BG_ACTIVE if self.active else BG_INACTIVE
        pg.draw.rect(surface, bg_color, self.rect)
        pg.draw.rect(surface, BORDER, self.rect, 1)

        # Texto ou placeholder
        if self.text:
            text_surf = self.font.render(self.text, True, BLACK)
            text_color = BLACK
        else:
            text_surf = self.font.render(self.placeholder, True, PLACEHOLDER_COLOR)
            text_color = PLACEHOLDER_COLOR

        surface.blit(text_surf, (self.rect.x + 5, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

        # Cursor
        if self.active and self._cursor_visible:
            # Posição x do cursor baseada no texto até cursor_pos
            sub_text = self.text[: self.cursor_pos]
            cursor_x = self.rect.x + 5 + self.font.size(sub_text)[0]
            cursor_y1 = self.rect.y + 4
            cursor_y2 = self.rect.y + self.rect.height - 4
            pg.draw.line(surface, text_color, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)

        # Dropdown
        if self.dropdown_open and self.options_filtered:
            self._draw_dropdown(surface)

    # ======================================================================
    # MÉTODOS INTERNOS
    # ======================================================================

    def _draw_dropdown(self, surface: pg.Surface) -> None:
        """Desenha a lista de opções logo abaixo da caixa."""
        # Limita quantidade de itens visíveis
        items = self.options_filtered[: self._max_visible]
        if not items:
            return

        x = self.rect.x
        y = self.rect.bottom
        w = self.rect.width
        h_item = self._item_height
        total_h = h_item * len(items)

        # Fundo do dropdown
        dropdown_rect = pg.Rect(x, y, w, total_h)
        pg.draw.rect(surface, BG_DROPDOWN, dropdown_rect)
        pg.draw.rect(surface, BORDER, dropdown_rect, 1)

        # Itens
        for i, opt in enumerate(items):
            item_rect = pg.Rect(x, y + i * h_item, w, h_item)
            # hover 
            mx, my = pg.mouse.get_pos()
            is_hover = item_rect.collidepoint(mx, my)

            if is_hover:
                pg.draw.rect(surface, (220, 220, 220), item_rect)

            opt_surf = self.font.render(opt, True, BLACK)
            surface.blit(
                opt_surf,
                (item_rect.x + 5, item_rect.y + (item_rect.height - opt_surf.get_height()) // 2),
            )

    def _handle_dropdown_click(self, pos: Tuple[int, int]):
        """
        Trata clique em algum item do dropdown.

        Retorna:
            opção escolhida (str) ou None.
        """
        if not self.dropdown_open or not self.options_filtered:
            return None

        items = self.options_filtered[: self._max_visible]
        h_item = self._item_height

        for i, opt in enumerate(items):
            item_rect = pg.Rect(
                self.rect.x,
                self.rect.bottom + i * h_item,
                self.rect.width,
                h_item,
            )
            if item_rect.collidepoint(pos):
                self.text = opt
                self.cursor_pos = len(self.text)
                self.dropdown_open = False
                self.active = False
                self.update_filter()
                return opt

        # Clique fora dos itens (mas dentro da área do dropdown)
        self.dropdown_open = False
        return None

    # ======================================================================
    # Compatibilidade com código antigo
    # ======================================================================

    def dropdown_click(self, pos: Tuple[int, int]):
        """
        [LEGACY] Método de compatibilidade. Hoje o clique no dropdown
        é tratado diretamente em handle_event() para MOUSEBUTTONDOWN.

        Mantido apenas para não quebrar código antigo, mas não é usado
        no main_interactive atual.
        """
        return self._handle_dropdown_click(pos)