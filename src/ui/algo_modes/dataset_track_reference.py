"""Aproximação visual da pista a partir de uma volta de odometria.

Não é ground truth independente. Não exporta dados para métricas e não
altera as amostras; serve exclusivamente como sobreposição no mapa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ui.algo_modes.dataset_lap_view import LapSlice


@dataclass(frozen=True)
class ApproxTrack:
    xy: np.ndarray                     # Polyline fechada em metros (N+1, 2)
    source_lap: int                    # Volta usada como aproximação
    span_x_m: float                    # Extensão no sistema odométrico
    span_y_m: float
    perimeter_m: float                 # Comprimento após suavização
    raw_closure_gap_m: float           # Desvio início-fim antes do fechamento
    provenance: str = "odometry-derived; NOT ground truth"


def build_track_from_first_complete_lap(
    poses_xytheta: np.ndarray,
    laps: tuple[LapSlice, ...],
    *,
    sample_count: int = 256,
    smoothing_radius: int = 3,
) -> ApproxTrack | None:
    """Reamostra e suaviza uma volta completa, mantendo coordenadas originais.

    Não registra/alinha outras voltas à referência, evitando ocultar drift.
    Retorna None se o trecho estiver ausente, degenerado ou distante demais
    de um circuito fechado para justificar a aproximação visual.
    """
    poses = np.asarray(poses_xytheta, dtype=float)
    if poses.ndim != 2 or poses.shape[1] < 2:
        raise ValueError("poses precisa ser N x 2 ou N x 3")
    if sample_count < 32 or smoothing_radius < 0:
        raise ValueError("sample_count >= 32 e smoothing_radius >= 0")

    lap = next((l for l in laps if l.complete and l.sample_count >= 16), None)
    if lap is None:
        return None
    raw = poses[lap.start:lap.stop, :2]
    if not np.all(np.isfinite(raw)):
        return None

    # Descartar duplicatas consecutivas, sem alterar o histórico original.
    keep = np.r_[True, np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-9]
    raw = raw[keep]
    if raw.shape[0] < 16:
        return None

    open_segments = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    travelled = float(np.sum(open_segments))
    closure = float(np.linalg.norm(raw[-1] - raw[0]))
    if travelled < .5 or closure > max(.75, .2 * travelled):
        return None

    closed = np.vstack((raw, raw[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    length_along = np.r_[0., np.cumsum(lengths)]
    perimeter = float(length_along[-1])
    if perimeter <= .5:
        return None

    # Mesmo espaçamento espacial para não concentrar amostras nas curvas lentas.
    dist = np.linspace(0, perimeter, sample_count, endpoint=False)
    xy = np.column_stack([
        np.interp(dist, length_along, closed[:, axis]) for axis in (0, 1)
    ])
    if smoothing_radius:
        radius = min(smoothing_radius, max(0, (sample_count - 1) // 4))
        offsets = np.arange(-radius, radius + 1)
        sigma = max(radius / 1.8, 1.)
        weights = np.exp(-.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()
        xy = sum(w * np.roll(xy, shift=-int(offset), axis=0)
                 for w, offset in zip(weights, offsets))

    closed_smooth = np.vstack((xy, xy[0]))
    return ApproxTrack(
        xy=closed_smooth,
        source_lap=lap.number,
        span_x_m=float(np.ptp(xy[:, 0])),
        span_y_m=float(np.ptp(xy[:, 1])),
        perimeter_m=float(np.linalg.norm(np.diff(closed_smooth, axis=0), axis=1).sum()),
        raw_closure_gap_m=closure,
    )
