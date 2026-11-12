"""Botão simples do PyGame com texto limitado ao retângulo visual.

A área de clique e o retângulo desenhado são sempre idênticos.
Em telas com DPI/fonte diferentes, um rótulo longo é reduzido apenas
se necessário; nunca deve escapar do contorno do botão.
"""
import pygame as pg

BLACK = (20, 20, 20)
LBL = (40, 40, 40)


def fit_button_label_size(text_width, text_height, rect_width, rect_height,
                          padding_x=12, padding_y=6):
    """Calcula dimensões proporcionais de texto contidas no botão.

    Args:
        text_width, text_height: dimensões do texto renderizado (pixels).
        rect_width, rect_height: dimensões internas do botão (pixels).
        padding_x, padding_y: folga mínima total nas bordas por eixo.
    Returns:
        (width, height): tamanho-alvo; nunca excede o espaço disponível.
    """
    if min(text_width, text_height, rect_width, rect_height) <= 0:
        raise ValueError("Dimensões de texto/botão devem ser positivas")
    max_w = max(1, int(rect_width) - int(padding_x) * 2)
    max_h = max(1, int(rect_height) - int(padding_y) * 2)
    scale = min(1.0, max_w / text_width, max_h / text_height)
    return (
        min(max_w, max(1, int(text_width * scale))),
        min(max_h, max(1, int(text_height * scale))),
    )


class Button:
    def __init__(self, rect, text, font, bg=(245, 245, 245), fg=LBL, border=BLACK):
        self.rect = pg.Rect(rect)
        self.text = text
        self.font = font
        self.bg = bg
        self.fg = fg
        self.border = border

    def draw(self, surface):
        """Desenha rótulo centralizado, limitado ao retângulo clicável."""
        pg.draw.rect(surface, self.bg, self.rect, border_radius=6)
        pg.draw.rect(surface, self.border, self.rect, 1, border_radius=6)
        img = self.font.render(self.text, True, self.fg)
        target = fit_button_label_size(
            img.get_width(), img.get_height(), self.rect.w, self.rect.h
        )
        if target != img.get_size():
            # Caso raro: fonte/escala Windows maior do que o previsto.
            # Reduzir SOMENTE o texto, preservando a área de clique.
            img = pg.transform.smoothscale(img, target)
        surface.blit(img, (
            self.rect.x + (self.rect.w - img.get_width()) // 2,
            self.rect.y + (self.rect.h - img.get_height()) // 2,
        ))

    def hit(self, pos):
        return self.rect.collidepoint(pos)
