# src/ui/botton.py
import pygame as pg

BLACK = (20, 20, 20)
LBL = (40, 40, 40)

class Button:
    def __init__(self, rect, text, font, bg=(245,245,245), fg=LBL, border=BLACK):
        self.rect = pg.Rect(rect)
        self.text = text
        self.font = font
        self.bg = bg
        self.fg = fg
        self.border = border

    def draw(self, surface):
        pg.draw.rect(surface, self.bg, self.rect, border_radius=6)
        pg.draw.rect(surface, self.border, self.rect, 1, border_radius=6)
        img = self.font.render(self.text, True, self.fg)
        surface.blit(img, (self.rect.x + (self.rect.w - img.get_width())//2,
                           self.rect.y + (self.rect.h - img.get_height())//2))

    def hit(self, pos):
        return self.rect.collidepoint(pos)