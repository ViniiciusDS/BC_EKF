from __future__ import annotations

from typing import Mapping
import pygame as pg
from typing import Callable, Optional
import os
import json
import numpy as np

from src.uwb.algoritmos_step import NOMES_UI
from src.analysis.algo_metrics import build_ranking_summary
from src.environment.environment import Environment

MODE_DATASET = "dataset"
MODE_STEP = "step"
MODE_MONTE_CARLO = "monte_carlo"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_D = (90, 90, 90)
GRAY_L = (235, 235, 240)
RED = (200, 40, 40)
BLUE = (50, 100, 220)
GREEN = (0, 180, 0)
ORANGE = (255, 150, 0)

ALGO_ORDER = ["trilaterate3d", "lms", "gauss_newton", "lmsp", "bc_ekf"]
ALGO_COLORS = {
    "trilaterate3d": (255, 20, 20),
    "lms": (138, 0, 196),
    "gauss_newton": (0, 0, 0),
    "lmsp": (0, 100, 255),
    "bc_ekf": (255, 150, 0),
}


def load_anchors_from_json(
    filepath: str,
    format_converter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[np.ndarray, str]:
    """
    Carrega âncoras de um arquivo JSON com key 'anchors_xy'.
    
    Args:
        filepath: Caminho completo do arquivo JSON
        format_converter: Função opcional para converter o array (ex: transpose para 3xN)
    
    Returns:
        Tupla (anchors_array, filename_label)
    
    Levanta:
        FileNotFoundError: Se arquivo não existe
        ValueError: Se formato de âncoras for inválido
        json.JSONDecodeError: Se JSON for inválido
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    anchors_xy = np.array(data.get("anchors_xy", []), dtype=float)
    
    if anchors_xy.size == 0:
        raise ValueError("Arquivo de âncoras vazio")
    
    # Converter 2D para 3D se necessário (adicionar altura padrão = 1.0)
    if anchors_xy.ndim == 2 and anchors_xy.shape[1] == 2:
        anchors_nx3 = np.zeros((anchors_xy.shape[0], 3), dtype=float)
        anchors_nx3[:, 0] = anchors_xy[:, 0]
        anchors_nx3[:, 1] = anchors_xy[:, 1]
        anchors_nx3[:, 2] = 1.0
        anchors_xy = anchors_nx3
    elif not (anchors_xy.ndim == 2 and anchors_xy.shape[1] == 3):
        raise ValueError("Formato inválido de âncoras (esperado Nx2 ou Nx3)")
    
    # Aplicar formato converter se fornecido (ex: transpose para 3xN)
    if format_converter is not None:
        anchors_xy = format_converter(anchors_xy)
    
    filename = os.path.basename(filepath)
    return anchors_xy, filename


def load_route_from_json(filepath: str) -> tuple[np.ndarray, str]:
    """
    Carrega waypoints de rota de um arquivo JSON com key 'waypoints'.
    
    Args:
        filepath: Caminho completo do arquivo JSON
    
    Returns:
        Tupla (waypoints_array, filename_label)
    
    Levanta:
        FileNotFoundError: Se arquivo não existe
        ValueError: Se waypoints forem inválidos
        json.JSONDecodeError: Se JSON for inválido
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    waypoints = np.array(data.get("waypoints", []), dtype=float)
    
    if waypoints.size == 0:
        raise ValueError("Rota vazia ou sem waypoints")
    
    if len(waypoints) < 2:
        raise ValueError("Rota deve ter pelo menos 2 pontos")
    
    filename = os.path.basename(filepath)
    return waypoints, filename


def load_map_from_json(filepath: str) -> tuple[Environment, str]:
    """
    Carrega mapa de um arquivo JSON usando Environment.load_json().
    
    Args:
        filepath: Caminho completo do arquivo JSON
    
    Returns:
        Tupla (environment_object, filename_label)
    
    Levanta:
        FileNotFoundError: Se arquivo não existe
        Exception: Se erro ao carregar mapa
    """
    env = Environment.load_json(filepath)
    filename = os.path.basename(filepath)
    return env, filename
def default_selected() -> dict[str, bool]:
    return {k: True for k in ALGO_ORDER}


def ordered_active_algos(
    stats: Mapping[str, dict] | None,
    selected: Mapping[str, bool] | None = None,
) -> list[str]:
    if not stats:
        return []

    return [
        algo
        for algo in ALGO_ORDER
        if algo in stats and (selected is None or selected.get(algo, False))
    ]

def ranking_positions(
    stats: Mapping[str, dict] | None,
    selected: Mapping[str, bool] | None = None,
) -> dict[str, int]:
    ranked = build_ranking_summary(stats, selected=selected, top_k=99)
    out: dict[str, int] = {}
    for row in ranked:
        out[row["algo"]] = int(row["rank"])
    return out

def draw_analyzer_panel(
    *,
    screen,
    font,
    bigfont,
    title: str,
    stats: Mapping[str, dict] | None,
    selected: Mapping[str, bool] | None = None,
    x: int = 18,
    y: int = 18,
    w: int = 430,
    h: int = 305,
    panel_fill=(250, 250, 252, 238),
    box_fill=(242, 242, 245),
    border=(120, 120, 130),
    text=(30, 30, 30),
    header=(70, 70, 70),
):
    '''Desenha um painel de análise de algoritmos, mostrando as estatísticas fornecidas e um boxplot comparativo.'''
    if not stats:
        return

    panel = pg.Surface((w, h), pg.SRCALPHA)
    panel.fill(panel_fill)
    screen.blit(panel, (x, y))
    pg.draw.rect(screen, border, (x, y, w, h), 1)

    screen.blit(bigfont.render(title, True, (20, 20, 20)), (x + 10, y + 8))

    header_y = y + 40
    col_name = x + 12
    col_rmse = x + 155
    col_mae = x + 220
    col_max = x + 285

    screen.blit(font.render("Algoritmo", True, header), (col_name, header_y))
    screen.blit(font.render("RMSE", True, header), (col_rmse, header_y))
    screen.blit(font.render("MAE", True, header), (col_mae, header_y))
    screen.blit(font.render("Max", True, header), (col_max, header_y))

    ordered = ordered_active_algos(stats, selected)

    yy = header_y + 24
    for algo in ordered:
        st = stats[algo]
        color = ALGO_COLORS.get(algo, BLACK)
        label = NOMES_UI.get(algo, algo).split(") ", 1)[-1]

        pg.draw.rect(screen, color, (col_name, yy + 5, 10, 10))
        screen.blit(font.render(label[:16], True, color), (col_name + 18, yy))
        screen.blit(font.render(f"{st['rmse']:.3f}", True, text), (col_rmse, yy))
        screen.blit(font.render(f"{st['mae']:.3f}", True, text), (col_mae, yy))
        screen.blit(font.render(f"{st['max_err']:.3f}", True, text), (col_max, yy))
        yy += 22

    box_x = x + 12
    box_y = y + 150
    box_w = w - 24
    box_h = 135

    draw_boxplot_panel(
        screen,
        font,
        stats,
        ordered,
        box_x,
        box_y,
        box_w,
        box_h,
        selected=selected,
        box_fill=box_fill,
    )

def draw_boxplot_panel(
    screen,
    font,
    stats,
    ordered_algos,
    x,
    y,
    w,
    h,
    *,
    selected=None,
    box_fill=(242, 242, 245),
):
    '''Desenha um boxplot horizontal para os algoritmos listados, usando as estatísticas fornecidas.'''
    if not ordered_algos:
        return

    pg.draw.rect(screen, box_fill, (x, y, w, h))
    pg.draw.rect(screen, (150, 150, 150), (x, y, w, h), 1)
    screen.blit(font.render("Boxplot de erro", True, (50, 50, 50)), (x + 6, y + 4))

    rank_pos = ranking_positions(stats, selected=selected)

    max_err = max(max(1e-9, stats[algo].get("max", 0.0)) for algo in ordered_algos)

    label_w = 128
    plot_x0 = x + label_w
    plot_x1 = x + w - 10
    plot_w = plot_x1 - plot_x0

    plot_y = y + 24
    row_h = max(18, (h - 30) // max(1, len(ordered_algos)))

    for i, algo in enumerate(ordered_algos):
        st = stats[algo]
        color = ALGO_COLORS.get(algo, BLACK)
        yy = plot_y + i * row_h + row_h // 2

        def sx(v):
            return int(plot_x0 + (v / max_err) * plot_w)

        s_min = sx(st["min"])
        s_q1 = sx(st["q1"])
        s_med = sx(st["median"])
        s_q3 = sx(st["q3"])
        s_max = sx(st["max"])

        label = NOMES_UI.get(algo, algo).split(") ", 1)[-1]
        rank = rank_pos.get(algo)

        if rank is not None:
            label_txt = f"{rank}º {label}"
        else:
            label_txt = label

        screen.blit(font.render(label_txt[:12], True, color), (x + 6, yy - 9))

        pg.draw.line(screen, color, (s_min, yy), (s_max, yy), 2)
        pg.draw.rect(screen, color, (s_q1, yy - 5, max(2, s_q3 - s_q1), 10), 1)
        pg.draw.line(screen, color, (s_med, yy - 6), (s_med, yy + 6), 2)
        pg.draw.line(screen, color, (s_min, yy - 4), (s_min, yy + 4), 2)
        pg.draw.line(screen, color, (s_max, yy - 4), (s_max, yy + 4), 2)