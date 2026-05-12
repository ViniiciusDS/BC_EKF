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
    Carrega âncoras de um arquivo JSON.

    Formatos aceitos:
    - anchors_xy: [[x, y], ...]
    - anchors_xy: [[x, y, z], ...]

    Se vier apenas x,y, usa anchor_height_m do JSON.
    Se anchor_height_m não existir, usa z=1.0 como fallback.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Mantém compatibilidade com arquivos antigos e também aceita "anchors"
    raw_anchors = data.get("anchors_xy", data.get("anchors", []))
    anchors_xy = np.array(raw_anchors, dtype=float)

    if anchors_xy.size == 0:
        raise ValueError("Arquivo de âncoras vazio")

    # altura padrão das âncoras, caso o arquivo venha apenas com x,y
    anchor_height_m = float(
        data.get(
            "anchor_height_m",
            data.get("anchor_z_m", data.get("z_anchor", 1.0))
        )
    )

    # Converter 2D para 3D se necessário
    if anchors_xy.ndim == 2 and anchors_xy.shape[1] == 2:
        anchors_nx3 = np.zeros((anchors_xy.shape[0], 3), dtype=float)
        anchors_nx3[:, 0] = anchors_xy[:, 0]
        anchors_nx3[:, 1] = anchors_xy[:, 1]
        anchors_nx3[:, 2] = anchor_height_m
        anchors_xy = anchors_nx3

    elif anchors_xy.ndim == 2 and anchors_xy.shape[1] == 3:
        # já está em x,y,z
        pass

    else:
        raise ValueError("Formato inválido de âncoras (esperado Nx2 ou Nx3)")

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
    x: int = 10,
    y: int = 10,
    w: int = 500,
    h: int = 380,
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

    screen.blit(bigfont.render(title, True, (20, 20, 20)), (x + 12, y + 10))

    # colunas com espaçamento maior para acomodar nomes longos
    col_algo = x + 18
    col_rmse = x + 190
    col_mae = x + 270
    col_max = x + 345
    y0 = y + 48

    screen.blit(font.render("Algoritmo", True, header), (col_algo, y0))
    screen.blit(font.render("RMSE", True, header), (col_rmse, y0))
    screen.blit(font.render("MAE", True, header), (col_mae, y0))
    screen.blit(font.render("Max", True, header), (col_max, y0))

    ordered = ordered_active_algos(stats, selected)
    row_y = y0 + 28
    row_h = 28

    # Garante altura mínima para caber tabela + boxplot sem vazar do painel.
    min_box_h = 32 + 18 * max(1, len(ordered))
    required_h = (row_y - y) + (row_h * len(ordered)) + 8 + min_box_h + 12
    if h < required_h:
        h = required_h
        panel = pg.Surface((w, h), pg.SRCALPHA)
        panel.fill(panel_fill)
        screen.blit(panel, (x, y))
        pg.draw.rect(screen, border, (x, y, w, h), 1)
        screen.blit(bigfont.render(title, True, (20, 20, 20)), (x + 12, y + 10))
        screen.blit(font.render("Algoritmo", True, header), (col_algo, y0))
        screen.blit(font.render("RMSE", True, header), (col_rmse, y0))
        screen.blit(font.render("MAE", True, header), (col_mae, y0))
        screen.blit(font.render("Max", True, header), (col_max, y0))

    for algo in ordered:
        st = stats[algo]
        color = ALGO_COLORS.get(algo, BLACK)
        label = NOMES_UI.get(algo, algo).split(") ", 1)[-1]

        rmse = st.get("rmse", None)
        mae = st.get("mae", None)
        maxe = st.get("max", st.get("max_err", None))

        pg.draw.rect(screen, color, (col_algo - 10, row_y + 6, 8, 8))
        screen.blit(font.render(str(label), True, color), (col_algo + 6, row_y))
        screen.blit(
            font.render(f"{rmse:.3f}" if rmse is not None else "-", True, text),
            (col_rmse, row_y),
        )
        screen.blit(
            font.render(f"{mae:.3f}" if mae is not None else "-", True, text),
            (col_mae, row_y),
        )
        screen.blit(
            font.render(f"{maxe:.3f}" if maxe is not None else "-", True, text),
            (col_max, row_y),
        )
        row_y += row_h

    # boxplot abaixo da tabela, com altura dinâmica
    box_x = x + 12
    box_y = row_y + 8
    box_w = w - 24
    box_h = h - (box_y - y) - 12

    if box_h > 40:
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


# =========================================================
# FILE / JSON HELPERS
# =========================================================

import os
import json
from pathlib import Path
import numpy as np


def list_files_by_extension(dirname: str, exts: tuple[str, ...]) -> list[str]:
    """
    Lista arquivos de uma pasta por extensão.

    Uso genérico para modais, telas e modos diferentes.
    """
    if not dirname or not os.path.isdir(dirname):
        return []

    exts = tuple(e.lower() for e in exts)
    files = []

    for name in os.listdir(dirname):
        full = os.path.join(dirname, name)

        if os.path.isfile(full) and name.lower().endswith(exts):
            files.append(name)

    return sorted(files, key=str.lower)


def parse_anchor_uwb_ids(raw_ids):
    """
    Converte IDs de âncoras vindos do JSON para inteiros.

    Aceita:
    - [6, 3, 9]
    - ["6", "3", "9"]
    - ["Da6", "Da3", "Da9"]
    """
    if raw_ids is None:
        return None

    parsed = []

    for x in raw_ids:
        sx = str(x).strip()

        if sx.lower().startswith("da"):
            sx = sx[2:]

        parsed.append(int(sx))

    return parsed


def load_anchor_uwb_ids_from_json(path: str):
    """
    Lê IDs UWB reais das âncoras, se existirem no JSON.

    Campos aceitos:
    - anchor_ids_uwb
    - uwb_ids
    - anchor_ids
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        raw_ids = (
            raw.get("anchor_ids_uwb", None)
            or raw.get("uwb_ids", None)
            or raw.get("anchor_ids", None)
        )

        return parse_anchor_uwb_ids(raw_ids)

    except Exception:
        return None


def load_route_waypoints_from_json(path: str):
    """
    Loader genérico de rota.

    Formato esperado:
    {
        "waypoints": [[x0, y0], [x1, y1], ...]
    }

    Retorna:
        np.ndarray com shape (N, 2)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    waypoints = data.get("waypoints", None)

    if waypoints is None:
        raise ValueError("Arquivo de rota inválido: campo 'waypoints' não encontrado.")

    pts = np.asarray(waypoints, dtype=float)

    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        raise ValueError("Arquivo de rota inválido: waypoints insuficientes.")

    return pts[:, :2].copy()