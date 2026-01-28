# src/ui/drawing.py

import math
import pygame as pg
# ==========================
# UI helpers
# ==========================
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRID = (210, 210, 210)
GRID5 = (170, 170, 170)
RED = (220, 60, 60)
BLUE = (55, 120, 220)
ORANGE = (250, 160, 60)
PURPLE = (185, 60, 200)
GRAY = (230, 230, 230)
LBL = (40, 40, 40)
GREEN = (50, 170, 80)



# ==========================
# Desenho mapa infinito
# ==========================
    
def draw_robot(surface, cam, x, y, theta, color=BLACK):
    L = 0.5
    W = 0.32
    p_front = (x + L * math.cos(theta), y + L * math.sin(theta))
    p_l = (x + W * math.cos(theta + 2.5), y + W * math.sin(theta + 2.5))
    p_r = (x + W * math.cos(theta - 2.5), y + W * math.sin(theta - 2.5))
    pts = [cam.world_to_screen(*p) for p in (p_front, p_l, p_r)]
    pg.draw.polygon(surface, color, pts)
    # nariz
    a = cam.world_to_screen(x, y)
    b = cam.world_to_screen(x + (L + 0.25) * math.cos(theta), y + (L + 0.25) * math.sin(theta))
    pg.draw.line(surface, WHITE, a, b, 2)


def draw_grid(surface, cam):
    w, h = cam.viewport
    # calcula os limites de mundo visíveis
    x0, y0 = cam.screen_to_world(0, h)
    x1, y1 = cam.screen_to_world(w, 0)
    # linhas verticais a cada 1m
    x_start = math.floor(min(x0, x1))
    x_end = math.ceil(max(x0, x1))
    for xm in range(x_start, x_end + 1):
        col = GRID if (xm % 5) else GRID5
        sx0, sy0 = cam.world_to_screen(xm, y0)
        sx1, sy1 = cam.world_to_screen(xm, y1)
        pg.draw.line(surface, col, (sx0, sy0), (sx1, sy1), 1)
    # horizontais
    y_start = math.floor(min(y0, y1))
    y_end = math.ceil(max(y0, y1))
    for ym in range(y_start, y_end + 1):
        col = GRID if (ym % 5) else GRID5
        sx0, sy0 = cam.world_to_screen(x0, ym)
        sx1, sy1 = cam.world_to_screen(x1, ym)
        pg.draw.line(surface, col, (sx0, sy0), (sx1, sy1), 1)


def draw_axes(surface, cam, font):
    """
    Desenha marcações e rótulos dos eixos X e Y no mapa, com passo adaptativo
    para evitar texto sobreposto quando o zoom está muito afastado.
    """
    w, h = cam.viewport
    if cam.scale <= 0:
        return

    # Queremos ~50 px entre labels
    min_px = 50
    raw_step = min_px / cam.scale  # em metros

    # "snapping" para valores agradáveis
    nice_steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    step = nice_steps[-1]
    for s in nice_steps:
        if s >= raw_step:
            step = s
            break

    # --------- EIXO X ----------
    x0_world, _ = cam.screen_to_world(0, h)
    x1_world, _ = cam.screen_to_world(w, h)
    x_min = min(x0_world, x1_world)
    x_max = max(x0_world, x1_world)

    start_x = math.floor(x_min / step) * step
    end_x   = math.ceil(x_max / step) * step
    n_x = int(round((end_x - start_x) / step)) + 1

    for i in range(n_x):
        xm = start_x + i * step
        sx, sy = cam.world_to_screen(xm, 0)
        if 0 <= sx <= w:
            text = f"{xm:g}"  # formatação sem zeros desnecessários
            img  = font.render(text, True, (120, 120, 120))
            surface.blit(img, (sx - img.get_width() // 2, h - img.get_height() - 2))

    # --------- EIXO Y ----------
    _, y0_world = cam.screen_to_world(0, h)
    _, y1_world = cam.screen_to_world(0, 0)
    y_min = min(y0_world, y1_world)
    y_max = max(y0_world, y1_world)

    start_y = math.floor(y_min / step) * step
    end_y   = math.ceil(y_max / step) * step
    n_y = int(round((end_y - start_y) / step)) + 1

    for i in range(n_y):
        ym = start_y + i * step
        sx, sy = cam.world_to_screen(0, ym)
        if 0 <= sy <= h:
            text = f"{ym:g}"
            img  = font.render(text, True, (120, 120, 120))
            surface.blit(img, (4, sy - img.get_height() // 2))

def draw_anchors(surface, cam, anchors3xN):
    if anchors3xN is None or anchors3xN.shape[1] == 0:
        return
    for ax, ay in zip(anchors3xN[0], anchors3xN[1]):
        sx, sy = cam.world_to_screen(ax, ay)
        pg.draw.circle(surface, RED, (sx, sy), 6)
        pg.draw.circle(surface, BLACK, (sx, sy), 6, 1)

def draw_path(surface, cam, path_xy, color, width=2, dashed=False):
    """
    Desenha uma polyline no plano do mundo.
    - Se dashed=False, usa pg.draw.lines direto.
    - Se dashed=True, desenha traço-pausa ao longo de CADA segmento consecutivo,
      garantindo que rotas fechadas (ex.: quadrado) apareçam corretamente.
    """
    if path_xy is None or len(path_xy) < 2:
        return

    pts = [cam.world_to_screen(x, y) for x, y in path_xy]

    if not dashed:
        pg.draw.lines(surface, color, False, pts, width)
        return

    # --- Desenho pontilhado contínuo ao longo de cada segmento (p[i] -> p[i+1]) ---
    # parâmetros do tracejado em pixels
    dash_len = 12
    gap_len = 6
    period = dash_len + gap_len

    for i in range(len(pts) - 1):
        (x0, y0) = pts[i]
        (x1, y1) = pts[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue

        # vetor unitário na direção do segmento
        ux = dx / seg_len
        uy = dy / seg_len

        # avança ao longo do segmento alternando traço e espaço
        dist = 0.0
        while dist < seg_len:
            # início do traço
            sx = x0 + ux * dist
            sy = y0 + uy * dist
            # fim do traço (clamp no fim do segmento)
            dist_end = min(dist + dash_len, seg_len)
            ex = x0 + ux * dist_end
            ey = y0 + uy * dist_end
            # desenha o "dash"
            pg.draw.line(surface, color, (int(sx), int(sy)), (int(ex), int(ey)), width)
            # pula o "gap"
            dist = dist_end + gap_len

def draw_text(surface, txt, x, y, font, color=LBL):
    img = font.render(txt, True, color)
    surface.blit(img, (x, y))