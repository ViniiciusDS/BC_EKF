"""Segmentação conservadora de voltas físicas, somente para visualização.

Dois modelos de largada:
* crossing: portão transversal, cruzado em +direção (pista contínua).
* corner_return: chegada à região de largada pelo trecho de aproximação,
  apropriada quando a pista vira 90 graus no início (LAB7A: +Y -> +X).

A suavização é aplicada SOMENTE à cópia de pontos usada para procurar
cruzamentos. Não altera estimativas UWB, PCNT, algoritmos ou Analyzer.
"""
from __future__ import annotations

import math
import numpy as np
from src.ui.algo_modes.dataset_lap_view import LapSlice


def _median_xy_for_gate(positions, window):
    """Mediana local por coordenada para evitar eventos por outlier isolado.

    Mantém NaN na amostra originalmente inválida e não interpola lacunas.
    """
    if type(window) is not int or window < 1 or window > 21 or window % 2 == 0:
        raise ValueError("smoothing_window deve ser ímpar entre 1 e 21")
    out = np.asarray(positions, dtype=float)[:, :2].copy()
    if window == 1:
        return out
    raw = out.copy()
    valid = np.all(np.isfinite(raw), axis=1)
    half = window // 2
    for i in range(len(raw)):
        if not valid[i]:
            continue
        lo, hi = max(0, i-half), min(len(raw), i+half+1)
        neighbors = raw[lo:hi][valid[lo:hi]]
        if len(neighbors):
            out[i] = np.median(neighbors, axis=0)
    return out



def first_complete_lap_index(laps):
    """Índice de navegação (1-based) da primeira volta confirmada.

    Preserva os trechos parciais no conjunto original. A navegação do
    Dataset Mode abre uma volta completa, mas permite voltar à fatia
    inicial com os botões Anterior/Próxima.
    """
    return next(
        (index for index, lap in enumerate(laps, start=1) if lap.complete),
        1,
    )


def split_start_gate_laps(
    positions_xy, timestamps_s, *, start_xy, direction_xy,
    sample_indices=None, total_samples=None,
    half_width_m=.4, hysteresis_m=.12, min_lap_s=30.,
    min_lap_distance_m=4.5, departure_m=.8, max_step_m=.8,
    gate_mode="crossing", return_direction_xy=None, return_radius_m=.5,
    smoothing_window=1, approach_lookback_s=2.5,
    approach_min_m=.06, max_gap_s=8.,
):
    """Detecta voltas sem inventar evento a partir de um ponto isolado.

    Em 'corner_return', exige: partida perto do início, afastamento
    >= departure_m, percurso >= min_lap_distance_m, intervalo mínimo,
    reentrada no círculo de retorno e deslocamento apontando para o
    vetor de aproximação física (p.ex., +Y no LAB7A). Essa reentrada
    não depende de cruzar x<0 antes de virar em +X.

    O algoritmo recusa grandes saltos UWB e preserva a última amostra
    confiável; em lacunas longas reinicia as evidências de deslocamento.
    Todos os índices das fatias se referem ao registro PCNT original.
    """
    xy_input = np.asarray(positions_xy, dtype=float)
    t = np.asarray(timestamps_s, dtype=float)
    origin = np.asarray(start_xy, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    if xy_input.ndim != 2 or xy_input.shape[1] < 2 or len(xy_input) < 2:
        raise ValueError("Trajetória deve conter ao menos duas posições XY")
    if t.shape != (len(xy_input),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps devem ser finitos e estritamente crescentes")
    if origin.shape != (2,) or direction.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(direction).all():
        raise ValueError("Portão e direção exigem dois valores XY finitos")
    if gate_mode not in ("crossing", "corner_return"):
        raise ValueError("gate_mode deve ser crossing ou corner_return")
    d_norm = float(np.linalg.norm(direction))
    if d_norm <= 1e-9:
        raise ValueError("Vetor de saída do portão não pode ser zero")

    params = {
        "half_width_m": half_width_m, "hysteresis_m": hysteresis_m,
        "min_lap_s": min_lap_s, "min_lap_distance_m": min_lap_distance_m,
        "departure_m": departure_m, "max_step_m": max_step_m,
        "return_radius_m": return_radius_m,
        "approach_lookback_s": approach_lookback_s,
        "approach_min_m": approach_min_m, "max_gap_s": max_gap_s,
    }
    for name, value in params.items():
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} deve ser positivo e finito")
    if hysteresis_m >= departure_m:
        raise ValueError("departure_m deve ser maior que hysteresis_m")
    if gate_mode == "corner_return" and return_radius_m >= departure_m:
        raise ValueError("return_radius_m deve ser menor que departure_m")
    if gate_mode == "corner_return":
        incoming = np.asarray(return_direction_xy, dtype=float)
        if incoming.shape != (2,) or not np.isfinite(incoming).all() or np.linalg.norm(incoming) <= 1e-9:
            raise ValueError("return_direction_xy exige direção de aproximação válida")
        incoming = incoming / np.linalg.norm(incoming)

    if sample_indices is None:
        idx = np.arange(len(xy_input), dtype=int)
    else:
        idx = np.asarray(sample_indices)
        if (idx.shape != (len(xy_input),)
                or not np.issubdtype(idx.dtype, np.integer)
                or idx[0] < 0 or np.any(np.diff(idx) <= 0)):
            raise ValueError("sample_indices deve conter inteiros crescentes")
    if total_samples is None:
        total_samples = len(xy_input)
    if (not isinstance(total_samples, (int, np.integer))
            or isinstance(total_samples, (bool, np.bool_))
            or total_samples < 2 or idx[-1] >= total_samples):
        raise ValueError("total_samples não abrange índices PCNT")

    xy = _median_xy_for_gate(xy_input, smoothing_window)
    n = direction / d_norm
    tangent = np.array([-n[1], n[0]], dtype=float)
    signed = (xy - origin) @ n
    lateral = (xy - origin) @ tangent
    radius = np.linalg.norm(xy - origin, axis=1)

    first_near = bool(
        np.isfinite(radius[0]) and radius[0] <= (
            return_radius_m if gate_mode == "corner_return"
            else max(half_width_m, departure_m)
        )
    )
    armed = False
    departed = False
    traveled = 0.0
    last_gate_time = float(t[0])
    last_good = 0 if np.isfinite(xy[0]).all() else None
    crossings, crossing_times = [], []
    rejected_jumps = 0
    rejected_long_gaps = 0
    rejected_arrival_direction = 0
    potential_arrivals = 0

    for i in range(1, len(xy)):
        if not np.isfinite(xy[i]).all():
            continue
        if last_good is None:
            last_good = i
            continue
        p = last_good
        elapsed_gap = float(t[i] - t[p])
        if elapsed_gap > max_gap_s:
            # Não inventar percurso durante ausência prolongada de posição.
            rejected_long_gaps += 1
            armed = False
            departed = False
            traveled = 0.0
            last_good = i
            continue
        step = float(np.linalg.norm(xy[i] - xy[p]))
        if step > max_step_m:
            # Outlier isolado: esperar recuperação sem acumular teleporte.
            # Se a posição real seguir distante, o max_gap_s fará reset.
            rejected_jumps += 1
            continue

        traveled += step
        if radius[i] >= departure_m:
            departed = True
        if signed[i] <= -hysteresis_m:
            armed = True

        event_time = None
        if gate_mode == "crossing":
            # Cruzamento da linha ortogonal à direção de partida.
            if armed and departed and signed[p] <= 0 < signed[i]:
                fraction = -signed[p] / (signed[i] - signed[p])
                cross_lateral = lateral[p] + fraction * (lateral[i] - lateral[p])
                if abs(cross_lateral) <= half_width_m:
                    event_time = float(t[p] + fraction * (t[i] - t[p]))
        else:
            # Volta fechada num CANTO: entrada no raio do ponto de
            # largada vinda do lado físico de retorno (+Y no LAB7A).
            entered = (radius[p] >= return_radius_m > radius[i])
            if entered and departed:
                potential_arrivals += 1
                # Deslocamento acumulado por ~2.5 s, não direção
                # instantânea de uma amostra UWB potencialmente ruidosa.
                lookback_idx = int(np.searchsorted(
                    t[:i+1], t[i] - approach_lookback_s, side="left"
                ))
                # Não atravessar uma lacuna maior que max_gap_s.
                if lookback_idx < i and t[i] - t[lookback_idx] <= max_gap_s:
                    motion = xy[i] - xy[lookback_idx]
                    m_norm = float(np.linalg.norm(motion))
                    if (np.isfinite(motion).all()
                            and m_norm >= approach_min_m
                            and float(np.dot(motion, incoming)) >= .5 * m_norm):
                        event_time = float(t[i])
                    else:
                        rejected_arrival_direction += 1
                else:
                    rejected_arrival_direction += 1

        if (event_time is not None and
                event_time - last_gate_time >= min_lap_s and
                traveled >= min_lap_distance_m):
            raw_index = int(idx[i])
            if not crossings or raw_index > crossings[-1]:
                crossings.append(raw_index)
                crossing_times.append(event_time)
                last_gate_time = event_time
                traveled = 0.0
                departed = False
                armed = False
        last_good = i

    boundaries = [0, *crossings, int(total_samples)]
    slices = [
        LapSlice(
            number=j+1, start=max(0, start-(1 if j else 0)),
            stop=stop, firmware_count=-1,
            complete=(j < len(crossings) and (j > 0 or first_near)),
        )
        for j, (start, stop) in enumerate(zip(boundaries, boundaries[1:]))
        if stop > start
    ]
    # Evidência direta para separar erro de referencial de regras
    # restritivas do portão. Todos estes valores são DESCRITIVOS:
    # nunca deslocam automaticamente o portão para combinar com UWB.
    finite = np.flatnonzero(np.isfinite(radius))
    closest = int(finite[np.argmin(radius[finite])]) if len(finite) else None
    first_valid = int(finite[0]) if len(finite) else None
    gate_radius = return_radius_m if gate_mode == "corner_return" else half_width_m
    window_start = float(t[0])
    per_window = []
    for win in range(int(math.ceil((float(t[-1]) - window_start) / 80.0))):
        lo, hi = window_start + 80*win, window_start + 80*(win+1)
        candidates = np.flatnonzero((t >= lo) & (t < hi) & np.isfinite(radius))
        if len(candidates):
            near = int(candidates[np.argmin(radius[candidates])])
            per_window.append({
                "window": win+1,
                "start_s": round(lo-window_start, 2),
                "end_s": round(min(hi, float(t[-1]))-window_start, 2),
                "nearest_distance_m": round(float(radius[near]), 3),
                "nearest_xy_m": [round(float(v), 3) for v in xy[near]],
                "nearest_time_s": round(float(t[near]), 3),
            })

    diagnostics = {
        "source_samples": len(xy), "pcnt_samples": int(total_samples),
        "valid_xy_samples": int(len(finite)),
        "samples_inside_gate_radius": int(np.sum(
            np.isfinite(radius) & (radius <= gate_radius)
        )),
        "closest_gate_distance_m": (
            round(float(radius[closest]), 3) if closest is not None else None
        ),
        "closest_gate_xy_m": (
            [round(float(v), 3) for v in xy[closest]]
            if closest is not None else None
        ),
        "closest_gate_time_s": (
            round(float(t[closest]), 3) if closest is not None else None
        ),
        "first_valid_xy_m": (
            [round(float(v), 3) for v in xy[first_valid]]
            if first_valid is not None else None
        ),
        "first_valid_time_s": (
            round(float(t[first_valid]), 3)
            if first_valid is not None else None
        ),
        "nearest_gate_by_80s_window": per_window,
        "gate_mode": gate_mode, "crossing_indices_pcnt": crossings,
        "crossing_times_s": crossing_times,
        "complete_laps": sum(p.complete for p in slices),
        "first_segment_near_gate": first_near,
        "rejected_large_steps": rejected_jumps,
        "rejected_long_gaps": rejected_long_gaps,
        "potential_arrivals": potential_arrivals,
        "rejected_arrival_direction": rejected_arrival_direction,
        "gate_xy_m": origin.tolist(), "gate_forward_xy": n.tolist(),
        "return_radius_m": return_radius_m if gate_mode == "corner_return" else None,
    }
    return tuple(slices), diagnostics
