from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.ui.algo_modes.dataset_bc_prep import expand_bc_uwb_rows_if_needed


@dataclass
class RealPipelineResult:
    ok: bool
    message: str = ""

    encoder_samples: Optional[list[dict]] = None
    uwb_rows: Optional[list[dict]] = None

    real_dataset: Optional[dict] = None
    real_aligned_rows: Optional[list[dict]] = None

    batch_dists: Optional[np.ndarray] = None
    batch_devs: Optional[np.ndarray] = None

    real_range_matrix: Optional[np.ndarray] = None
    real_sigma_matrix: Optional[np.ndarray] = None
    real_timestamps: Optional[list[float]] = None
    real_anchor_ids: Optional[list[int]] = None


def _norm_col_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("ç", "c")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )


def _read_table_auto(path: str) -> pd.DataFrame:
    """
    Lê CSV/TXT com separador automático.
    Aceita vírgula, ponto e vírgula, tab ou espaços.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass

    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception as e:
        raise ValueError(f"Falha ao ler arquivo: {path}: {e}")


def _coerce_float_series(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def _detect_time_scale(raw_time: np.ndarray) -> float:
    """
    Detecta se timestamp está em ms ou s.

    Heurística:
    - se o span for muito grande, assume ms;
    - caso contrário, assume segundos.
    """
    t = np.asarray(raw_time, dtype=float)
    t = t[np.isfinite(t)]

    if t.size < 2:
        return 1.0

    span = float(np.nanmax(t) - np.nanmin(t))

    # logs do ESP normalmente vêm em millis: dezenas/centenas de milhares
    if span > 1000.0:
        return 0.001

    return 1.0


def normalize_encoder_file(
    path: str,
    *,
    cfg: Any,
) -> list[dict]:
    """
    Normaliza arquivo de encoder real para lista de amostras:

    [
        {
            "t": segundos desde início,
            "left_dist": m acumulado ou incremental,
            "right_dist": m acumulado ou incremental,
            "left_vel": m/s,
            "right_vel": m/s,
            "left_ticks": ...,
            "right_ticks": ...
        },
        ...
    ]

    Suporta dois formatos principais:

    1) Formato padrão:
       timestamp, left_ticks, right_ticks

    2) Formato do ESP usado no experimento:
       contador direita, contador esquerda,
       distância direita, distância esquerda,
       velocidade direita, velocidade esquerda,
       tempo millis

    O modo com colunas de distância é controlado por:
       REAL_ENCODER_USE_DISTANCE_COLUMNS
       REAL_ENCODER_DISTANCE_UNIT_SCALE
    """
    df = _read_table_auto(path)

    if df.empty:
        raise ValueError("Arquivo de encoder vazio")

    original_cols = list(df.columns)
    norm_map = {_norm_col_name(c): c for c in original_cols}
    norm_cols = list(norm_map.keys())

    use_dist_cols = bool(getattr(cfg, "REAL_ENCODER_USE_DISTANCE_COLUMNS", True))
    dist_scale = float(getattr(cfg, "REAL_ENCODER_DISTANCE_UNIT_SCALE", 0.01))

    swap_lr = bool(getattr(cfg, "REAL_ENCODER_SWAP_LR", False))
    invert_left = bool(getattr(cfg, "REAL_ENCODER_INVERT_LEFT", False))
    invert_right = bool(getattr(cfg, "REAL_ENCODER_INVERT_RIGHT", False))

    wheel_radius = float(getattr(cfg, "WHEEL_RADIUS", 0.035))
    ticks_per_rev = float(getattr(cfg, "ENCODER_TICKS_PER_REV", 1075.0))
    meters_per_tick = 2.0 * np.pi * wheel_radius / ticks_per_rev

    # -------------------------------
    # Caso 1: formato padrão por nomes
    # -------------------------------
    required_standard = {"timestamp", "left_ticks", "right_ticks"}

    if required_standard.issubset(set(norm_cols)):
        t_raw = _coerce_float_series(df[norm_map["timestamp"]]).to_numpy()
        l_ticks = _coerce_float_series(df[norm_map["left_ticks"]]).to_numpy()
        r_ticks = _coerce_float_series(df[norm_map["right_ticks"]]).to_numpy()

        t_scale = _detect_time_scale(t_raw)
        t = (t_raw - np.nanmin(t_raw)) * t_scale

        left_dist = (l_ticks - l_ticks[0]) * meters_per_tick
        right_dist = (r_ticks - r_ticks[0]) * meters_per_tick

    # -----------------------------------------------------
    # Caso 2: formato ESP por ordem de colunas do main.cpp
    # -----------------------------------------------------
    else:
        if df.shape[1] < 7:
            raise ValueError(
                "Arquivo de encoder inválido. Colunas obrigatórias não encontradas: "
                "timestamp, left_ticks, right_ticks"
            )

        # Ordem informada:
        # 0 contador direita
        # 1 contador esquerda
        # 2 distância direita
        # 3 distância esquerda
        # 4 velocidade direita
        # 5 velocidade esquerda
        # 6 tempo millis
        c0 = _coerce_float_series(df.iloc[:, 0]).to_numpy()
        c1 = _coerce_float_series(df.iloc[:, 1]).to_numpy()
        d_right_raw = _coerce_float_series(df.iloc[:, 2]).to_numpy()
        d_left_raw = _coerce_float_series(df.iloc[:, 3]).to_numpy()
        v_right_raw = _coerce_float_series(df.iloc[:, 4]).to_numpy()
        v_left_raw = _coerce_float_series(df.iloc[:, 5]).to_numpy()
        t_raw = _coerce_float_series(df.iloc[:, 6]).to_numpy()

        t_scale = _detect_time_scale(t_raw)
        t = (t_raw - np.nanmin(t_raw)) * t_scale

        if use_dist_cols:
            right_dist = d_right_raw * dist_scale
            left_dist = d_left_raw * dist_scale
        else:
            right_dist = (c0 - c0[0]) * meters_per_tick
            left_dist = (c1 - c1[0]) * meters_per_tick

        r_ticks = c0
        l_ticks = c1

    # -------------------------------
    # Tratamento de orientação
    # -------------------------------
    if swap_lr:
        left_dist, right_dist = right_dist, left_dist
        l_ticks, r_ticks = r_ticks, l_ticks

    if invert_left:
        left_dist = -left_dist

    if invert_right:
        right_dist = -right_dist

    valid = (
        np.isfinite(t)
        & np.isfinite(left_dist)
        & np.isfinite(right_dist)
    )

    t = t[valid]
    left_dist = left_dist[valid]
    right_dist = right_dist[valid]
    l_ticks = l_ticks[valid] if len(l_ticks) == len(valid) else np.full_like(t, np.nan)
    r_ticks = r_ticks[valid] if len(r_ticks) == len(valid) else np.full_like(t, np.nan)

    if len(t) == 0:
        raise ValueError("Arquivo de encoder sem amostras válidas")

    # ordena por tempo
    order = np.argsort(t)
    t = t[order]
    left_dist = left_dist[order]
    right_dist = right_dist[order]
    l_ticks = l_ticks[order]
    r_ticks = r_ticks[order]

    # velocidades por diferença finita
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.nan

    left_vel = np.diff(left_dist, prepend=left_dist[0]) / dt
    right_vel = np.diff(right_dist, prepend=right_dist[0]) / dt

    left_vel[~np.isfinite(left_vel)] = 0.0
    right_vel[~np.isfinite(right_vel)] = 0.0

    samples = []

    for i in range(len(t)):
        samples.append({
            "t": float(t[i]),
            "left_dist": float(left_dist[i]),
            "right_dist": float(right_dist[i]),
            "left_vel": float(left_vel[i]),
            "right_vel": float(right_vel[i]),
            "left_ticks": float(l_ticks[i]) if np.isfinite(l_ticks[i]) else np.nan,
            "right_ticks": float(r_ticks[i]) if np.isfinite(r_ticks[i]) else np.nan,
        })

    return samples


def _find_first_existing_col(norm_map: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in norm_map:
            return norm_map[c]
    return None


def normalize_uwb_file(
    path: str,
    *,
    anchor_id_map: dict[int, int] | None = None,
    default_sigma: float = 0.05,
) -> list[dict]:
    """
    Normaliza arquivo UWB real.

    Saída em formato simples ou BC.

    Formato simples:
        timestamp, anchor_id, range, sigma

    Formato BC:
        timestamp, anchor_id, range_front, sigma_front, range_rear, sigma_rear

    Se vier em formato largo:
        timestamp, Da6_t1, Da6_t2, Da3_t1, Da3_t2, ...
    converte para linhas BC por timestamp/âncora.
    """
    df = _read_table_auto(path)

    if df.empty:
        raise ValueError("Arquivo UWB vazio")

    original_cols = list(df.columns)
    norm_map = {_norm_col_name(c): c for c in original_cols}
    norm_cols = list(norm_map.keys())

    time_col = _find_first_existing_col(
        norm_map,
        ["timestamp", "timestamp_s", "time", "t", "millis", "tempo", "tempo_ms"]
    )

    if time_col is None:
        # fallback: primeira coluna
        time_col = original_cols[0]

    t_raw = _coerce_float_series(df[time_col]).to_numpy()
    scale = _detect_time_scale(t_raw)
    t = (t_raw - np.nanmin(t_raw)) * scale

    rows = []

    # ------------------------------------------------
    # Caso 1: formato longo com anchor_id + range
    # ------------------------------------------------
    anchor_col = _find_first_existing_col(
        norm_map,
        ["anchor_id", "anchor", "id", "da", "anchorid"]
    )
    range_col = _find_first_existing_col(
        norm_map,
        ["range", "dist", "distance", "distancia", "range_m"]
    )
    sigma_col = _find_first_existing_col(
        norm_map,
        ["sigma", "std", "stddev", "desvio", "desvio_padrao"]
    )

    if anchor_col is not None and range_col is not None:
        anchor_raw = df[anchor_col].to_numpy()
        ranges = _coerce_float_series(df[range_col]).to_numpy()

        if sigma_col is not None:
            sigmas = _coerce_float_series(df[sigma_col]).to_numpy()
        else:
            sigmas = np.full_like(ranges, float(default_sigma), dtype=float)

        for i in range(len(df)):
            try:
                raw_id = int(anchor_raw[i])
                mapped_id = anchor_id_map.get(raw_id, raw_id) if anchor_id_map else raw_id
                r = float(ranges[i])
                s = float(sigmas[i])
            except Exception:
                continue

            if not np.isfinite(t[i]) or not np.isfinite(r):
                continue

            if not np.isfinite(s) or s <= 0:
                s = float(default_sigma)

            rows.append({
                "timestamp": float(t[i]),
                "anchor_id": int(mapped_id),
                "range": r,
                "sigma": s,
            })

        return rows

    # ------------------------------------------------
    # Caso 2: formato largo do ESP
    # Exemplo:
    # timestamp,Da6_t1,Da6_t2,Da3_t1,Da3_t2,...
    # ------------------------------------------------
    # Agrupa colunas por padrão Da<ID>_t1/t2.
    groups = {}

    for col in original_cols:
        if col == time_col:
            continue

        name = str(col).strip()
        low = name.lower()

        # aceita Da6_t1, da6_t2, D6_t1 etc.
        anchor_id = None
        tag = None

        import re
        m = re.search(r"d?a?(\d+).*t([12])", low)

        if m:
            anchor_id = int(m.group(1))
            tag = int(m.group(2))

        if anchor_id is None or tag is None:
            continue

        mapped_id = anchor_id_map.get(anchor_id, anchor_id) if anchor_id_map else anchor_id

        groups.setdefault(mapped_id, {})
        groups[mapped_id][tag] = col

    if not groups:
        raise ValueError("Formato UWB não reconhecido")

    for i in range(len(df)):
        if not np.isfinite(t[i]):
            continue

        for mapped_id, cols in groups.items():
            # Caso tenha duas tags
            if 1 in cols and 2 in cols:
                try:
                    r1 = float(df.loc[df.index[i], cols[1]])
                    r2 = float(df.loc[df.index[i], cols[2]])
                except Exception:
                    continue

                if np.isfinite(r1) and np.isfinite(r2):
                    rows.append({
                        "timestamp": float(t[i]),
                        "anchor_id": int(mapped_id),
                        "range_front": r1,
                        "sigma_front": float(default_sigma),
                        "range_rear": r2,
                        "sigma_rear": float(default_sigma),
                    })

            # Caso só tenha uma tag
            elif 1 in cols:
                try:
                    r = float(df.loc[df.index[i], cols[1]])
                except Exception:
                    continue

                if np.isfinite(r):
                    rows.append({
                        "timestamp": float(t[i]),
                        "anchor_id": int(mapped_id),
                        "range": r,
                        "sigma": float(default_sigma),
                    })

    return rows


def _rows_are_bc(rows: list[dict]) -> bool:
    if not rows:
        return False

    sample = rows[0]
    return (
        "range_front" in sample
        and "range_rear" in sample
        and "sigma_front" in sample
        and "sigma_rear" in sample
    )


def build_range_sigma_matrices(
    uwb_rows: list[dict],
    *,
    n_anchors: int,
    default_sigma: float = 0.05,
    min_valid_anchors: int = 2,
):
    """
    Constrói matrizes M x N de ranges e sigmas para métodos instantâneos.

    Se o UWB for BC, usa a tag frontal como representação dos métodos
    instantâneos.

    Remove timestamps com menos de min_valid_anchors medições válidas,
    evitando milhares de warnings nos métodos estáticos.
    """
    if not uwb_rows:
        return None, None, [], []

    timestamps = sorted({float(r["timestamp"]) for r in uwb_rows})
    anchor_ids = sorted({int(r["anchor_id"]) for r in uwb_rows})

    # Só filtra para 0..N-1 se os IDs já parecem estar remapeados.
    if n_anchors is not None and n_anchors > 0:
        if all(0 <= a < n_anchors for a in anchor_ids):
            anchor_ids = [a for a in anchor_ids if 0 <= a < n_anchors]

    if not timestamps or not anchor_ids:
        return None, None, [], []

    ts_to_i = {t: i for i, t in enumerate(timestamps)}
    aid_to_j = {a: j for j, a in enumerate(anchor_ids)}

    R = np.full((len(timestamps), len(anchor_ids)), np.nan, dtype=float)
    S = np.full_like(R, float(default_sigma), dtype=float)

    for row in uwb_rows:
        try:
            t = float(row["timestamp"])
            aid = int(row["anchor_id"])
        except Exception:
            continue

        if t not in ts_to_i or aid not in aid_to_j:
            continue

        i = ts_to_i[t]
        j = aid_to_j[aid]

        if "range_front" in row:
            r = row.get("range_front")
            s = row.get("sigma_front", default_sigma)
        else:
            r = row.get("range")
            s = row.get("sigma", default_sigma)

        try:
            r = float(r)
            s = float(s)
        except Exception:
            continue

        if not np.isfinite(r):
            continue

        if not np.isfinite(s) or s <= 0:
            s = float(default_sigma)

        R[i, j] = r
        S[i, j] = s

    valid_counts = np.sum(np.isfinite(R), axis=1)

    # Preferência: manter linhas completas, quando existirem.
    complete = valid_counts == R.shape[1]

    if np.any(complete):
        mask = complete
    else:
        # Fallback: pelo menos min_valid_anchors.
        min_valid_anchors = max(1, int(min_valid_anchors))
        mask = valid_counts >= min_valid_anchors

    if not np.any(mask):
        return None, None, [], []

    R = R[mask]
    S = S[mask]
    timestamps = np.asarray(timestamps, dtype=float)[mask].tolist()

    return R, S, timestamps, anchor_ids


def load_real_encoder_uwb_dataset(
    *,
    encoder_path: str,
    uwb_path: str,
    dataset_anchors,
    anchor_uwb_ids: list[int] | None,
    cfg: Any,
) -> RealPipelineResult:
    """
    Pipeline principal do dataset real:
    - lê e normaliza encoder;
    - lê e normaliza UWB;
    - remapeia IDs reais das âncoras para índices internos;
    - gera matriz de ranges/sigmas para métodos instantâneos.
    """
    try:
        encoder_samples = normalize_encoder_file(encoder_path, cfg=cfg)
    except Exception as e:
        return RealPipelineResult(False, f"Erro ao carregar encoder real: {e}")

    anchors = np.asarray(dataset_anchors, dtype=float) if dataset_anchors is not None else None
    n_anchors = int(anchors.shape[0]) if anchors is not None and anchors.ndim == 2 else 0

    anchor_id_map = None

    if anchor_uwb_ids:
        anchor_id_map = {int(raw): int(i) for i, raw in enumerate(anchor_uwb_ids)}

        default_sigma = float(getattr(cfg, "UWB_NOISE_STD", 0.05))

    # Mapa preferencial: IDs reais do arquivo de âncoras -> índice interno 0..N-1
    anchor_id_map = None
    if anchor_uwb_ids:
        anchor_id_map = {int(raw): int(i) for i, raw in enumerate(anchor_uwb_ids)}

    try:
        uwb_rows = normalize_uwb_file(
            uwb_path,
            anchor_id_map=anchor_id_map,
            default_sigma=default_sigma,
        )
    except Exception as e:
        return RealPipelineResult(False, f"Erro ao carregar UWB real: {e}")

    if not uwb_rows:
        return RealPipelineResult(False, "UWB real sem amostras válidas")

    # Diagnóstico seguro: sempre inicializa observed_ids
    observed_ids = sorted({
        int(r["anchor_id"])
        for r in uwb_rows
        if isinstance(r, dict) and "anchor_id" in r
    })

    # Fallback: se os IDs UWB reais não foram informados, mas o arquivo possui
    # exatamente n_anchors IDs distintos, remapeia automaticamente para 0..N-1.
    #
    # Exemplo:
    # observed_ids = [3, 6, 7, 8, 9]
    # auto_map     = {3:0, 6:1, 7:2, 8:3, 9:4}
    #
    # Observação: o ideal é usar anchor_uwb_ids vindo do arquivo de âncoras,
    # porque isso preserva a correspondência física correta.
    if anchor_id_map is None and n_anchors > 0:
        if len(observed_ids) == n_anchors and any(a >= n_anchors for a in observed_ids):
            auto_map = {raw: i for i, raw in enumerate(observed_ids)}

            for r in uwb_rows:
                try:
                    r["anchor_id"] = auto_map[int(r["anchor_id"])]
                except Exception:
                    pass

            observed_ids = sorted({
                int(r["anchor_id"])
                for r in uwb_rows
                if isinstance(r, dict) and "anchor_id" in r
            })

    # Opcional, temporário para confirmar:
    print(
        "[REAL_UWB_IDS]",
        "anchor_uwb_ids=", anchor_uwb_ids,
        "observed_ids=", observed_ids,
        "n_anchors=", n_anchors,
    )

    if not uwb_rows:
        return RealPipelineResult(False, "UWB real sem amostras válidas")

    # Para matriz dos métodos instantâneos, a função já usa front quando BC.
    R, S, timestamps, anchor_ids = build_range_sigma_matrices(
        uwb_rows,
        n_anchors=n_anchors,
        default_sigma=default_sigma,
        min_valid_anchors=2,
    )

    if R is None or S is None:
        return RealPipelineResult(False, "Falha ao montar matriz UWB real")

    real_dataset = {
        "encoder_path": encoder_path,
        "uwb_path": uwb_path,
        "n_encoder": len(encoder_samples),
        "n_uwb_rows": len(uwb_rows),
        "is_bc": _rows_are_bc(uwb_rows),
    }

    return RealPipelineResult(
        True,
        encoder_samples=encoder_samples,
        uwb_rows=uwb_rows,
        real_dataset=real_dataset,
        real_aligned_rows=uwb_rows,
        batch_dists=R,
        batch_devs=S,
        real_range_matrix=R,
        real_sigma_matrix=S,
        real_timestamps=timestamps,
        real_anchor_ids=anchor_ids,
    )