from __future__ import annotations
import pygame as pg

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_BG = (245, 245, 248)
GRAY_BORDER = (80, 80, 90)
TEXT = (25, 25, 25)

ALGO_COLORS = {
    "trilaterate3d": (255, 0, 0),      # vermelho
    "lms": (140, 0, 220),              # roxo
    "gauss_newton": (0, 0, 0),         # preto
    "lmsp": (0, 90, 255),              # azul
    "bc_ekf": (255, 165, 0),           # laranja
}

ALGO_LABELS = {
    "trilaterate3d": "Trilateração",
    "lms": "LMS",
    "gauss_newton": "Gauss-Newton",
    "lmsp": "LMS-Ponderado",
    "bc_ekf": "BC-EKF",
}


def _draw_robot_icon(screen, x, y):
    '''Desenha o robô no mesmo estilo da simulação: tags front/rear, POI e heading.'''
    tag_color = (210, 40, 40)
    poi_color = (40, 90, 210)

    front = (x + 9, y)
    rear = (x - 9, y)
    poi = (x, y)

    pg.draw.line(screen, BLACK, rear, front, 2)

    for pt in (front, rear):
        pg.draw.circle(screen, tag_color, pt, 8)
        pg.draw.circle(screen, BLACK, pt, 2, 1)

    _draw_star4_icon(screen, poi, 6, poi_color, BLACK)

    tip = (x + 20, y)
    pg.draw.line(screen, BLACK, poi, tip, 2)
    pg.draw.line(screen, BLACK, tip, (tip[0] - 5, tip[1] - 4), 2)
    pg.draw.line(screen, BLACK, tip, (tip[0] - 5, tip[1] + 4), 2)


def _draw_star4_icon(screen, center, radius, fill_color, outline_color):
    '''Desenha uma estrela simples de 4 pontas para representar o POI do robô.'''
    cx, cy = center
    inner = max(2, radius // 3)

    pts = [
        (cx + radius, cy),
        (cx + inner, cy - inner),
        (cx, cy - radius),
        (cx - inner, cy - inner),
        (cx - radius, cy),
        (cx - inner, cy + inner),
        (cx, cy + radius),
        (cx + inner, cy + inner),
    ]

    pg.draw.polygon(screen, fill_color, pts)
    pg.draw.polygon(screen, outline_color, pts, 1)


def _draw_anchor_icon(screen, x, y):
    '''Desenha um ícone de círculo para representar uma âncora.'''
    pg.draw.circle(screen, (220, 70, 70), (x, y), 5)
    pg.draw.circle(screen, BLACK, (x, y), 5, 1)


def _draw_wall_icon(screen, x, y):
    '''Desenha um ícone de linha grossa para representar uma parede.'''
    pg.draw.line(screen, BLACK, (x - 16, y), (x + 16, y), 4)


def _draw_planned_path_icon(screen, x, y):
    '''Desenha um ícone de linha tracejada para representar o trajeto planejado do robô.'''
    pts = [(x - 18, y), (x - 8, y), (x + 2, y), (x + 12, y), (x + 20, y)]
    for i in range(len(pts) - 1):
        if i % 2 == 0:
            pg.draw.line(screen, BLACK, pts[i], pts[i + 1], 2)


def _draw_real_path_icon(screen, x, y):
    '''Desenha um ícone de linha contínua para representar o trajeto real do robô.'''
    pts = [(x - 18, y), (x - 10, y - 2), (x, y + 1), (x + 10, y - 1), (x + 18, y)]
    pg.draw.lines(screen, BLACK, False, pts, 2)


def _draw_algo_icon(screen, x, y, color):
    '''Desenha um ícone de linha com pontos para representar um algoritmo de localização.'''
    pts = [
        (x - 18, y),
        (x - 10, y - 1),
        (x - 2, y + 1),
        (x + 7, y - 2),
        (x + 18, y),
    ]
    pg.draw.lines(screen, color, False, pts, 2)
    for px, py in pts:
        pg.draw.circle(screen, color, (px, py), 2)
        pg.draw.circle(screen, BLACK, (px, py), 2, 1)


def draw_legend_overlay(screen: pg.Surface, font: pg.font.Font, bigfont: pg.font.Font, selected: dict | None = None):
    '''Desenha a sobreposição da legenda no centro da tela. Retorna o retângulo do botão de fechar para detectar cliques.'''
    sw, sh = screen.get_width(), screen.get_height()

    overlay = pg.Surface((sw, sh), pg.SRCALPHA)
    overlay.fill((0, 0, 0, 110))
    screen.blit(overlay, (0, 0))

    w, h = 380, 460
    x = (sw - w) // 2
    y = (sh - h) // 2
    rect = pg.Rect(x, y, w, h)

    pg.draw.rect(screen, GRAY_BG, rect, border_radius=10)
    pg.draw.rect(screen, GRAY_BORDER, rect, 2, border_radius=10)

    title = bigfont.render("Legenda", True, TEXT)
    screen.blit(title, (x + 20, y + 16))

    # botão fechar
    close_rect = pg.Rect(rect.right - 42, rect.y + 12, 26, 26)
    pg.draw.rect(screen, (230, 230, 235), close_rect, border_radius=4)
    pg.draw.rect(screen, GRAY_BORDER, close_rect, 1, border_radius=4)
    close_txt = font.render("X", True, TEXT)
    screen.blit(close_txt, (close_rect.x + 7, close_rect.y + 3))

    items = [
        ("Robô", _draw_robot_icon, None),
        ("Âncoras", _draw_anchor_icon, None),
        ("Parede", _draw_wall_icon, None),
        ("Trajeto planejado", _draw_planned_path_icon, None),
        ("Trajeto real", _draw_real_path_icon, None),
    ]

    yy = y + 64
    for label, func, extra in items:
        func(screen, x + 55, yy + 8)
        txt = font.render(label, True, TEXT)
        screen.blit(txt, (x + 90, yy))
        yy += 32

    sep_y = yy + 5
    pg.draw.line(screen, (180, 180, 190), (x + 20, sep_y), (x + w - 20, sep_y), 1)

    yy = sep_y + 18
    subt = font.render("Algoritmos", True, TEXT)
    screen.blit(subt, (x + 20, yy))
    yy += 30

    for key in ["trilaterate3d", "lms", "gauss_newton", "lmsp", "bc_ekf"]:
        if selected is not None and key in selected and not selected[key]:
            continue

        color = ALGO_COLORS[key]
        _draw_algo_icon(screen, x + 55, yy + 8, color)
        txt = font.render(ALGO_LABELS[key], True, color)
        screen.blit(txt, (x + 90, yy))
        yy += 32

    return close_rect