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

# =========================================================
# ALGORITHM VARIANTS / SIDEBAR SELECTION
# =========================================================

ALGO_VARIANTS = {
    # O botão "trilaterate3d" vira um seletor de variantes.
    # key=None significa desligado.
    "trilaterate3d": [
        {"key": None, "label": "○ trilat. off", "short": "off"},
        {"key": "trilaterate3d", "label": "◉ trilat. 3D", "short": "3D"},
        {"key": "trilat_geo_sang2019", "label": "◉ trilat. geo", "short": "geo"},
        {"key": "trilat_geo_triplet_sang2019", "label": "◉ trilat. trinca", "short": "trinca"},
    ],
    "lms": [
    {"key": None, "label": "○ LS off", "short": "off"},
    {"key": "lms", "label": "◉ LS atual", "short": "atual"},
    {"key": "ls_sang2019", "label": "◉ LS Sang19", "short": "Sang19"},
    {"key": "ls_li2023", "label": "◉ LS Li23", "short": "Li23"},
    {"key": "ls_gn_li2023", "label": "◉ LS+GN Li23", "short": "LS+GN"},
    ],
    "lmsp": [
    {"key": None, "label": "○ LMS-P off", "short": "off"},
    {"key": "lmsp", "label": "◉ LMS-P atual", "short": "atual"},
    {"key": "wls_sigma", "label": "◉ WLS sigma", "short": "sigma"},
    {"key": "wls_rkf_fan2022", "label": "◉ WLS-RKF", "short": "Fan22"},
    {"key": "rwls_wang2017", "label": "◉ RWLS Wang17", "short": "Wang17"},
    ],
    "gauss_newton": [
    {"key": None, "label": "○ GN off", "short": "off"},
    {"key": "gauss_newton", "label": "◉ GN atual", "short": "atual"},
    {"key": "gn_wang2020", "label": "◉ GN Wang20", "short": "Wang20"},
    {"key": "gn_wang2020_damped", "label": "◉ GN damped", "short": "damped"},
    {"key": "gn_wang2020_mahal", "label": "◉ GN Mahalan.", "short": "Mahalan."},
    ],
}


ALGO_RESULT_ALIAS = {
    # As variantes aparecem no Analyzer/render como o grupo original.
    "trilat_geo_sang2019": "trilaterate3d",
    "trilat_geo_triplet_sang2019": "trilaterate3d",
    "ls_sang2019": "lms",
    "ls_li2023": "lms",
    "ls_gn_li2023": "lms",
    "wls_sigma": "lmsp",
    "wls_rkf_fan2022": "lmsp",
    "rwls_wang2017": "lmsp",
    "gn_wang2020": "gauss_newton",
    "gn_wang2020_damped": "gauss_newton",
    "gn_wang2020_mahal": "gauss_newton",
}


def default_algorithm_variant_state():
    """
    Estado inicial das variantes.

    Por padrão, mantém o comportamento antigo:
    - trilaterate3d começa usando a implementação antiga.
    - os demais algoritmos seguem usando apenas selected True/False.
    """
    return {
        "trilaterate3d": "trilaterate3d",
        "lms": "lms",
        "lmsp": "lmsp",
        "gauss_newton": "gauss_newton",
    }


def get_algorithm_variants(base_key: str):
    return ALGO_VARIANTS.get(base_key, [])


def algorithm_result_alias(concrete_key: str) -> str:
    return ALGO_RESULT_ALIAS.get(concrete_key, concrete_key)


def algorithm_active_key(base_key: str, selected: dict, variant_state: dict | None = None):
    """
    Retorna o nome concreto do algoritmo que deve ser enviado ao run_batch().

    Para algoritmos sem variantes:
        selected[base_key] True  -> base_key
        selected[base_key] False -> None

    Para algoritmos com variantes:
        retorna variant_state[base_key], ou None se estiver desligado.
    """
    variant_state = variant_state or {}

    if base_key in ALGO_VARIANTS:
        return variant_state.get(base_key, base_key if selected.get(base_key, False) else None)

    return base_key if selected.get(base_key, False) else None


def cycle_algorithm_variant(base_key: str, selected: dict, variant_state: dict | None = None):
    """
    Avança o estado de um botão.

    Se o algoritmo tiver variantes:
        off -> v1 -> v2 -> ... -> off

    Se não tiver variantes:
        toggle True/False antigo.
    """
    variant_state = variant_state or {}

    variants = get_algorithm_variants(base_key)

    if not variants:
        selected[base_key] = not selected.get(base_key, False)
        return selected.get(base_key, False), variant_state.get(base_key)

    current = variant_state.get(base_key, None)

    keys = [v["key"] for v in variants]

    try:
        idx = keys.index(current)
    except ValueError:
        idx = 0

    next_key = keys[(idx + 1) % len(keys)]
    variant_state[base_key] = next_key

    selected[base_key] = next_key is not None

    return selected[base_key], next_key


def algorithm_button_label(base_key: str, selected: dict, variant_state: dict | None = None):
    """
    Texto que aparece no botão lateral.
    """
    variant_state = variant_state or {}

    variants = get_algorithm_variants(base_key)

    if not variants:
        return f"◉ {base_key}" if selected.get(base_key, False) else f"○ {base_key}"

    current = variant_state.get(base_key, None)

    for v in variants:
        if v["key"] == current:
            return v["label"]

    return variants[0]["label"]

def resolve_file_with_extensions(dirname: str, filename: str, exts: tuple[str, ...]) -> str:
    """
    Resolve um arquivo mesmo quando o nome veio sem extensão.

    Exemplo:
        dirname = resultados/datasets
        filename = dataset_bc_20260601_144123_C1_C2
        exts = (".txt", ".csv")

    Tenta:
        dataset_bc_20260601_144123_C1_C2
        dataset_bc_20260601_144123_C1_C2.txt
        dataset_bc_20260601_144123_C1_C2.csv

    Também tenta casamento case-insensitive.
    """
    import os

    filename = str(filename).strip()

    if not filename:
        return os.path.join(dirname, filename)

    # Se já veio caminho absoluto ou relativo completo, testa direto.
    direct = filename
    if not os.path.isabs(direct):
        direct = os.path.join(dirname, filename)

    if os.path.isfile(direct):
        return direct

    root, ext = os.path.splitext(filename)

    # Se tem extensão conhecida mas não existe, retorna direto.
    # Se tem extensão parcial/desconhecida, tenta resolver por prefixo.
    known_exts = tuple(e.lower() for e in exts)

    if ext and ext.lower() in known_exts:
        return direct

    # Tenta completar extensão.
    for e in exts:
        candidate = os.path.join(dirname, filename + e)
        if os.path.isfile(candidate):
            return candidate

    # Tenta busca case-insensitive.
    if os.path.isdir(dirname):
        target_names = [filename + e for e in exts]

        lower_targets = {t.lower(): t for t in target_names}

        for real_name in os.listdir(dirname):
            if real_name.lower() in lower_targets:
                return os.path.join(dirname, real_name)

    # Tenta busca por prefixo quando o texto veio truncado pelo campo.
    if os.path.isdir(dirname):
        prefix = filename.lower()

        matches = []

        for real_name in os.listdir(dirname):
            full = os.path.join(dirname, real_name)

            if not os.path.isfile(full):
                continue

            low = real_name.lower()

            if not low.endswith(known_exts):
                continue

            if low.startswith(prefix):
                matches.append(real_name)

        if len(matches) == 1:
            return os.path.join(dirname, matches[0])

        # Caso o prefixo contenha uma extensão parcial, tenta com o root.
        if ext:
            root_prefix = root.lower()
            matches = []

            for real_name in os.listdir(dirname):
                full = os.path.join(dirname, real_name)

                if not os.path.isfile(full):
                    continue

                low = real_name.lower()

                if not low.endswith(known_exts):
                    continue

                if low.startswith(root_prefix):
                    matches.append(real_name)

            if len(matches) == 1:
                return os.path.join(dirname, matches[0])
            
    # fallback
    return direct