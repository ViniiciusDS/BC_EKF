"""Botão de interface PyGame com rótulo nativo e recorte de segurança.

Não redimensiona superfícies de texto: pg.font.render já rasteriza
com antialiasing na resolução correta. A área visual e de clique
continuam idênticas, mesmo quando uma fonte do Windows é maior.
"""
import pygame as pg

BLACK = (20, 20, 20)
LBL = (40, 40, 40)


class Button:
    def __init__(self, rect, text, font, bg=(245, 245, 245), fg=LBL, border=BLACK):
        self.rect = pg.Rect(rect)
        self.text = text
        self.font = font
        self.bg = bg
        self.fg = fg
        self.border = border

    def draw(self, surface):
        """Centraliza texto nítido no botão sem zoom ou reamostragem.

        Uma fonte muito grande pode ultrapassar o espaço disponível.
        Nesse caso recortamos apenas os pixels excedentes no retângulo
        do botão, sem deformar a fonte nem vazar sobre outros controles.
        """
        pg.draw.rect(surface, self.bg, self.rect, border_radius=6)
        pg.draw.rect(surface, self.border, self.rect, 1, border_radius=6)

        # Mantém a mesma qualidade da fonte dos outros textos da tela:
        # nunca usar transform.scale/smoothscale em caracteres.
        img = self.font.render(self.text, True, self.fg)
        position = img.get_rect(center=self.rect.center)

        previous_clip = surface.get_clip()
        try:
            # Respeita também um possível recorte prévio da janela.
            inner_rect = self.rect.inflate(-4, -2)
            surface.set_clip(previous_clip.clip(inner_rect))
            surface.blit(img, position)
        finally:
            # O recorte não pode afetar os próximos elementos desenhados.
            surface.set_clip(previous_clip)

    def hit(self, pos):
        return self.rect.collidepoint(pos)
