"""Detecção conservadora de voltas por cruzamento do portão inicial.

O portão é uma linha perpendicular à direção de partida. O detector
trabalha com trajetórias XY *já calculadas* (odometria ou UWB/LMS), sem
modificar posições, timestamps, dados UWB ou o processamento dos solvers.
Não faz associação de voltas usando o contador físico da coluna 4.
"""
from __future__ import annotations

import math

import numpy as np

from src.ui.algo_modes.dataset_lap_view import LapSlice


def split_start_gate_laps(
    positions_xy,
    timestamps_s,
    *,
    start_xy,
    direction_xy,
    sample_indices=None,
    total_samples=None,
    half_width_m=0.4,
    hysteresis_m=0.12,
    min_lap_s=30.0,
    min_lap_distance_m=4.5,
    departure_m=0.8,
    max_step_m=0.8,
):
    """Segmenta voltas apenas após cruzamentos válidos no sentido de partida.

    Args:
        positions_xy: amostras Nx2+ no referencial da pista.
        timestamps_s: tempos crescentes da sessão, N amostras.
        start_xy: centro físico do portão (metros).
        direction_xy: vetor apontando para frente ao sair do início.
        sample_indices: índices de cada amostra em PCNT bruto; para UWB,
            são os índices da amostra PCNT que originou cada posição.
        total_samples: tamanho original da sessão PCNT (obrigatório se
            sample_indices cobre apenas os snapshots UWB).
        half_width_m: metade da largura lateral do portão.
        hysteresis_m: distância anterior necessária para armar o portão.
        min_lap_s: intervalo mínimo entre cruzamentos (e desde t0).
        min_lap_distance_m: percurso incremental mínimo entre cruzamentos.
        departure_m: robô deve se afastar do início antes de retornar.
        max_step_m: rejeita saltos superiores a este valor no cruzamento.

    Returns:
        (LapSlice... , diagnostics): fatias em índices PCNT. A primeira
        fatia só é marcada completa quando a gravação começa perto do
        portão, e a última permanece parcial. Não estima tempo exato
        de cruzamento nem inventa voltas quando não há evidência.
    """
    xy = np.asarray(positions_xy, dtype=float)
    t = np.asarray(timestamps_s, dtype=float)
    origin = np.asarray(start_xy, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)

    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 2:
        raise ValueError("Trajetória deve ter ao menos duas posições XY")
    if t.shape != (len(xy),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps devem ser finitos e estritamente crescentes")
    if origin.shape != (2,) or direction.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(direction).all():
        raise ValueError("Portão e direção requerem pares XY finitos")
    dn = float(np.linalg.norm(direction))
    if dn <= 1e-9:
        raise ValueError("Direção do portão não pode ser zero")

    params = (half_width_m, hysteresis_m, min_lap_s, min_lap_distance_m, departure_m, max_step_m)
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in params):
        raise ValueError("Tolerâncias do portão devem ser números positivos finitos")
    if hysteresis_m >= departure_m:
        raise ValueError("departure_m deve ser maior que hysteresis_m")

    if sample_indices is None:
        idx = np.arange(len(xy), dtype=int)
    else:
        idx = np.asarray(sample_indices)
        if idx.shape != (len(xy),) or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("sample_indices precisa de índices inteiros por posição")
        if idx[0] < 0 or np.any(np.diff(idx) <= 0):
            raise ValueError("sample_indices deve ser crescente e não negativo")
    if total_samples is None:
        total_samples = len(xy)
    if type(total_samples) not in (int, np.int64, np.int32) or total_samples < 2 or idx[-1] >= total_samples:
        raise ValueError("total_samples deve abranger os índices PCNT")

    n = direction / dn
    tangent = np.array([-n[1], n[0]], dtype=float)
    signed = (xy[:, :2] - origin) @ n
    lateral = (xy[:, :2] - origin) @ tangent
    radius = np.linalg.norm(xy[:, :2] - origin, axis=1)

    # Se o primeiro snapshot começa distante do portão, o trecho até o
    # primeiro retorno é parcial (não completar volta fictícia).
    first_near = bool(np.isfinite(radius[0]) and radius[0] <= max(half_width_m, departure_m))
    armed = False
    departed = False
    distance_since_gate = 0.0
    last_gate_time = float(t[0])
    crossings = []
    crossing_times = []
    rejected_jumps = 0

    for i in range(1, len(xy)):
        p0, p1 = xy[i-1, :2], xy[i, :2]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            # Um intervalo sem posição válida invalida o percurso acumulado.
            armed = False
            departed = False
            distance_since_gate = 0.0
            continue
        step = float(np.linalg.norm(p1 - p0))
        if step > max_step_m:
            # Não ligar posições UWB desconexas. Após um salto, exigir
            # novo afastamento e distância mínima antes de contar volta.
            armed = False
            departed = False
            distance_since_gate = 0.0
            rejected_jumps += 1
            continue
        distance_since_gate += step
        if radius[i] >= departure_m:
            departed = True
        if signed[i] <= -hysteresis_m:
            armed = True

        # Travessia positiva (direção do robô ao partir); rejeita retorno.
        if not (armed and departed and signed[i-1] <= 0 < signed[i]):
            continue
        denom = signed[i] - signed[i-1]
        if denom <= 1e-12:
            continue
        fraction = -signed[i-1] / denom
        crossing_lateral = float(lateral[i-1] + fraction * (lateral[i] - lateral[i-1]))
        if abs(crossing_lateral) > half_width_m:
            continue
        crossing_time = float(t[i-1] + fraction * (t[i] - t[i-1]))
        if (crossing_time - last_gate_time < min_lap_s
                or distance_since_gate < min_lap_distance_m):
            continue
        raw_index = int(idx[i])
        if crossings and raw_index <= crossings[-1]:
            continue

        crossings.append(raw_index)
        crossing_times.append(crossing_time)
        last_gate_time = crossing_time
        distance_since_gate = 0.0
        departed = False
        armed = False

    boundaries = [0, *crossings, int(total_samples)]
    slices = []
    for j, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        if stop <= start:
            continue
        slices.append(LapSlice(
            number=len(slices)+1,
            start=max(0, start - (1 if j else 0)),
            stop=stop,
            firmware_count=-1,
            complete=(j < len(crossings) and (j > 0 or first_near)),
        ))
    diagnostics = {
        "source_samples": len(xy),
        "pcnt_samples": int(total_samples),
        "crossing_indices_pcnt": crossings,
        "crossing_times_s": crossing_times,
        "complete_laps": sum(item.complete for item in slices),
        "first_segment_near_gate": first_near,
        "rejected_large_steps": rejected_jumps,
        "gate_xy_m": origin.tolist(),
        "gate_forward_xy": n.tolist(),
    }
    return tuple(slices), diagnostics
