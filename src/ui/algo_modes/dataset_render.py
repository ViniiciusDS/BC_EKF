from __future__ import annotations

import numpy as np
import pygame as pg

from src.ui.drawing import draw_grid, draw_axes
from src.environment.environment import draw_environment
from src.ui.legend_overlay import draw_legend_overlay
from src.ui.algo_modes.shared import (
    ALGO_ORDER,
    ALGO_COLORS,
    draw_analyzer_panel,
)


BLACK = (0, 0, 0)
GRAY_D = (90, 90, 90)

def _safe_anchor_xy_array(data):
    """
    Normaliza âncoras para matriz N x 2.

    Aceita:
    - N x 2
    - N x 3
    - 2 x N
    - 3 x N
    """
    if data is None:
        return None

    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2:
        return None

    # Caso venha transposto: 2xN ou 3xN
    if arr.shape[0] in (2, 3) and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    if arr.shape[1] < 2:
        return None

    arr = arr[:, :2]

    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[valid]

    return arr if len(arr) > 0 else None


def draw_dataset_anchors(mode):
    """
    Desenha âncoras com estilo consistente, independentemente do formato interno.
    """
    anchors = _safe_anchor_xy_array(getattr(mode, "_dataset_anchors", None))

    if anchors is None:
        return

    screen = mode.host.screen
    cam = mode.host.cam

    for x, y in anchors:
        sx, sy = cam.world_to_screen(float(x), float(y))
        p = (int(sx), int(sy))

        # estilo parecido com o anterior: vermelho com borda preta
        pg.draw.circle(screen, (0, 0, 0), p, 7)
        pg.draw.circle(screen, (210, 55, 55), p, 6)
        
def _safe_xy_array(data):
    if data is None:
        return None

    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return None

    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[valid]

    if len(arr) == 0:
        return None

    return arr[:, :2]


def _draw_polyline_world(screen, cam, points, color, width=2, dashed=False, dash_px=14, gap_px=8):
    pts = _safe_xy_array(points)
    if pts is None or len(pts) < 2:
        return

    screen_pts = [cam.world_to_screen(float(x), float(y)) for x, y in pts]

    if not dashed:
        pg.draw.lines(screen, color, False, screen_pts, width)
        return

    for p0, p1 in zip(screen_pts[:-1], screen_pts[1:]):
        x0, y0 = p0
        x1, y1 = p1

        dx = x1 - x0
        dy = y1 - y0
        length = float(np.hypot(dx, dy))

        if length <= 1e-9:
            continue

        ux = dx / length
        uy = dy / length

        s = 0.0
        while s < length:
            e = min(s + dash_px, length)
            a = (x0 + ux * s, y0 + uy * s)
            b = (x0 + ux * e, y0 + uy * e)
            pg.draw.line(screen, color, a, b, width)
            s += dash_px + gap_px


def _draw_estimated_track(mode, algo, result):
    """
    Desenha a trajetória estimada de um algoritmo.

    Aceita tanto:
    - result como dict: {"posicoes": array}, {"positions": array}, {"trajectory": array}
    - result diretamente como array Nx2/Nx3

    A função também ignora pontos inválidos para evitar crash no pygame.
    """
    import numpy as np
    import pygame as pg

    if result is None:
        return

    screen = mode.host.screen
    cam = mode.host.cam

    # -------------------------------------------------
    # 1) Extrair pontos sem usar "or" com numpy array
    # -------------------------------------------------
    pts = None

    if isinstance(result, dict):
        if "posicoes" in result and result["posicoes"] is not None:
            pts = result["posicoes"]
        elif "positions" in result and result["positions"] is not None:
            pts = result["positions"]
        elif "trajectory" in result and result["trajectory"] is not None:
            pts = result["trajectory"]
        elif "traj" in result and result["traj"] is not None:
            pts = result["traj"]
    else:
        pts = result

    if pts is None:
        return

    # -------------------------------------------------
    # 2) Converter para array numérico
    # -------------------------------------------------
    try:
        pts = np.asarray(pts, dtype=float)
    except Exception:
        return

    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return

    # Pygame precisa de pares x,y.
    pts = pts[:, :2]

    # Remove NaN/Inf.
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]

    if pts.shape[0] < 2:
        return

    # -------------------------------------------------
    # 3) Quebrar a trajetória em segmentos válidos
    # -------------------------------------------------
    color = getattr(mode, "algo_colors", {}).get(algo, None)

    if color is None:
        try:
            from src.ui.algo_modes.shared import ALGO_COLORS
            color = ALGO_COLORS.get(algo, (0, 0, 0))
        except Exception:
            color = (0, 0, 0)

    line_width = 2

    try:
        # -------------------------------------------------
        # Converte pontos para coordenadas de tela
        # -------------------------------------------------
        screen_pts = []
        valid_pts = []

        for x, y in pts:
            screen_pt = cam.world_to_screen(float(x), float(y))
            
            # Valida o ponto de tela
            if isinstance(screen_pt, (tuple, list)) and len(screen_pt) >= 2:
                if np.isfinite(screen_pt[0]) and np.isfinite(screen_pt[1]):
                    screen_pts.append((int(screen_pt[0]), int(screen_pt[1])))
                    valid_pts.append([x, y])

        if len(screen_pts) < 2:
            return

        valid_pts = np.array(valid_pts, dtype=float)

        # -------------------------------------------------
        # 4) Desenhar segmentos, evitando conectar outliers
        # -------------------------------------------------
        max_world_jump = 1.5  # metros; ajuste se necessário

        segments = []
        current = []

        for i, (sx, sy) in enumerate(screen_pts):
            if i == 0:
                current = [(sx, sy)]
                continue

            jump = float(np.linalg.norm(valid_pts[i] - valid_pts[i - 1]))

            if jump > max_world_jump:
                if len(current) >= 2:
                    segments.append(current)
                current = [(sx, sy)]
            else:
                current.append((sx, sy))

        if len(current) >= 2:
            segments.append(current)

        for seg in segments:
            if len(seg) >= 2:
                pg.draw.lines(screen, color, False, seg, line_width)

        # Desenha os pontos por cima.
        for sx, sy in screen_pts:
            pg.draw.circle(screen, color, (sx, sy), 3)
            pg.draw.circle(screen, (0, 0, 0), (sx, sy), 3, 1)

    except Exception:
        return


def draw_reference_routes(mode):
    """
    Desenha rotas de referência/odometria do dataset.

    Convenção:
    - rota planejada/arquivo JSON: cinza tracejado;
    - odometria reconstruída/dataset_route: preto contínuo fino.
    """
    screen = mode.host.screen
    cam = mode.host.cam

    # Rota de referência vinda do JSON/PDF.
    ref = (
        getattr(mode, "_reference_route_display", None)
        if getattr(mode, "_reference_route_display", None) is not None
        else getattr(mode, "_route_waypoints", None)
    )

    if ref is not None:
        _draw_polyline_world(
            screen,
            cam,
            ref,
            color=(95, 95, 95),
            width=2,
            dashed=True,
            dash_px=14,
            gap_px=8,
        )

    # Rota odométrica/reconstruída. Evita duplicar se for igual à referência.
    odom = (
        getattr(mode, "_dataset_route", None)
        if getattr(mode, "dataset_source_type", "") == "real_encoder_uwb"
        else None
    )

    if odom is not None:
        arr = np.asarray(odom, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
            _draw_polyline_world(
                screen,
                cam,
                arr[:, :2],
                color=(0, 0, 0),
                width=2,
                dashed=False,
            )


def draw_dataset_world(mode):
    """
    Desenha o mundo principal: grade, eixos, mapa, âncoras, rotas e resultados.
    """
    screen = mode.host.screen
    cam = mode.host.cam

    # Limpa a área do mapa antes de redesenhar.
    # Sem isso, a tela anterior/menu fica aparecendo por baixo da grade.
    world_w = cam.viewport[0] if hasattr(cam, "viewport") else screen.get_width()
    world_rect = pg.Rect(0, 0, world_w, screen.get_height())
    pg.draw.rect(screen, (255, 255, 255), world_rect)

    draw_grid(screen, cam)
    draw_axes(screen, cam, mode.host.font)

    if getattr(mode, "_map_env", None) is not None:
        draw_environment(screen, cam, mode._map_env)

    draw_reference_routes(mode)

    draw_dataset_anchors(mode)

    if getattr(mode, "_batch_results", None) is not None:
        for algo in ALGO_ORDER:
            res = mode._batch_results.get(algo)
            _draw_estimated_track(mode, algo, res)


def draw_dataset_sidebar_controls(mode):
    """
    Desenha botões inferiores do painel lateral do Dataset Mode.
    Não desenha botões principais do host; apenas controles próprios do modo.
    """
    screen = mode.host.screen
    sidebar_x = mode.host.cam.viewport[0] + 16
    screen_h = screen.get_height()

    panel_x = mode.host.cam.viewport[0]
    panel_rect = pg.Rect(panel_x, 0, screen.get_width() - panel_x, screen_h)
    pg.draw.rect(screen, (245, 247, 252), panel_rect)

    legend_y = screen_h - 44
    toggle_y = legend_y - 40
    metric_y = toggle_y - 40

    if hasattr(mode, "btn_metric_mode"):
        mode.btn_metric_mode.rect.topleft = (sidebar_x, metric_y)
        mode.btn_metric_mode.rect.size = (190, 32)
        mode.btn_metric_mode.text = f"Métrica: {mode._metric_mode_label()}"
        mode.btn_metric_mode.draw(screen)

    mode.btn_toggle_analyzer.rect.topleft = (sidebar_x, toggle_y)
    mode.btn_toggle_analyzer.rect.size = (190, 32)
    mode.btn_toggle_analyzer.text = (
        "Ocultar Analyzer" if mode.show_analyzer else "Mostrar Analyzer"
    )
    mode.btn_toggle_analyzer.draw(screen)

    mode.btn_toggle_legend.rect.topleft = (sidebar_x, legend_y)
    mode.btn_toggle_legend.rect.size = (190, 32)
    mode.btn_toggle_legend.text = (
        "Ocultar Legenda" if mode.show_legend_overlay else "Mostrar Legenda"
    )
    mode.btn_toggle_legend.draw(screen)


def draw_dataset_status(mode):
    """
    Desenha textos pequenos de status do dataset no painel lateral.
    """
    screen = mode.host.screen
    font = mode.host.font

    sidebar_x = mode.host.cam.viewport[0] + 16
    y = 470

    label = getattr(mode, "_dataset_label", "")
    if label:
        txt = font.render(str(label), True, (150, 150, 150))
        screen.blit(txt, (sidebar_x, y))
        y += 28

    if getattr(mode, "_batch_dists", None) is not None:
        n = int(mode._batch_dists.shape[0])
        txt = font.render(f"Amostras: {n}", True, (150, 150, 150))
        screen.blit(txt, (sidebar_x, y))


def draw_dataset_analyzer(mode):
    if not getattr(mode, "show_analyzer", True):
        return

    if getattr(mode, "_dataset_stats", None) is None:
        return

    title = "Dataset Analyzer"

    if hasattr(mode, "_metric_mode_label"):
        try:
            title = f"Dataset Analyzer - {mode._metric_mode_label()}"
        except Exception:
            title = "Dataset Analyzer"

    draw_analyzer_panel(
        screen=mode.host.screen,
        font=mode.host.font,
        bigfont=mode.host.bigfont,
        title=title,
        stats=mode._dataset_stats,
        selected=mode.selected,
        x=10,
        y=40,
        w=500,
        h=380,
    )


def draw_dataset_legend(mode):
    if getattr(mode, "show_legend_overlay", False):
        mode._legend_close_rect = draw_legend_overlay(
            mode.host.screen,
            mode.host.font,
            mode.host.bigfont,
            selected=mode.selected,
        )
    else:
        mode._legend_close_rect = None


def draw_dataset_mode(mode):
    """
    Função principal de renderização do DatasetMode.
    O modal ainda fica no dataset_mode.py por enquanto.
    """
    draw_dataset_world(mode)
    draw_dataset_status(mode)
    draw_dataset_sidebar_controls(mode)

    if getattr(mode, "_batch_results", None) is not None:
        draw_dataset_analyzer(mode)

    draw_dataset_legend(mode)