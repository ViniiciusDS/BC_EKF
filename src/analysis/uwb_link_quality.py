"""Auditoria NÃO destrutiva de qualidade dos enlaces UWB do ESP32.

Trabalha diretamente com os 5+3*N campos de cada linha PCNT. Não
interpola ranges, não combina leituras de instantes diferentes e não
altera filtros, estimadores, perfis ou arquivos de entrada.

LOAO = leave-one-anchor-out: estima XY com as OUTRAS âncoras na mesma
linha PCNT, supondo altura Z conhecida da tag. O resíduo da excluída
é uma medida de consistência, NÃO erro de localização ou prova de NLOS.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.odometry.pcntlog_adapter import load_pcnt_sessions
from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    read_pcnt_anchor_slot_mapping,
    validate_robot2026_anchor_geometry,
)


def _positive_or_zero(value, name):
    """Valida parâmetro não negativo, numérico e finito."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} deve ser número finito não negativo")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} deve ser número finito não negativo")
    return result


def _summarize(values):
    """Estatísticas descritivas sem imputação e sem NaN em JSON."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return None
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {
        "count": int(a.size),
        "mean_m": float(np.mean(a)),
        "rms_m": float(np.sqrt(np.mean(a*a))),
        "p50_m": float(np.median(a)),
        "p95_m": float(np.percentile(a, 95)),
        "max_m": float(np.max(a)),
    }


def _estimate_xy_excluding(anchors, fresh, target, tag_z, *, max_condition=1e6):
    """LS de diferenças de quadrados com no mínimo três âncoras auxiliares.

    'fresh' associa o índice da âncora ao range NOVO da MESMA linha.
    Não usa odometria, resultado de outro solver nem range repetido.
    A solução é diagnóstica, sem correção robusta nem inferência NLOS.
    """
    others = [key for key in sorted(fresh) if key != target]
    if len(others) < 3:
        return None
    xyz = anchors[others]
    measured = np.asarray([fresh[j] for j in others], dtype=float)
    dz = xyz[:, 2] - tag_z
    horizontal_sq = measured*measured - dz*dz
    if np.any(horizontal_sq < 0):
        return None
    xy = xyz[:, :2]
    matrix = 2.0 * (xy[1:] - xy[0])
    squared_xy = np.sum(xy*xy, axis=1)
    rhs = ((squared_xy[1:] - squared_xy[0])
           - (horizontal_sq[1:] - horizontal_sq[0]))
    if np.linalg.matrix_rank(matrix, tol=1e-10) < 2:
        return None
    if not np.isfinite(np.linalg.cond(matrix)) or np.linalg.cond(matrix) > max_condition:
        return None
    estimate, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return estimate if np.isfinite(estimate).all() else None


def build_uwb_link_quality_report(
    log_path, anchor_json, *, session_id=1, tag_height_m,
    height_tolerance_m=0.0, loa_outlier_threshold_m=0.30,
):
    """Gera resumo completo por slot SPI e âncora física mapeada.

    tag_height_m é obrigatória e precisa ser medida no MESMO referencial
    de Z das âncoras. height_tolerance_m tolera incerteza na cota/tag:
    uma medida só é sinalizada impossível quando
    range + tolerance < abs(z_ancora - z_tag).

    A regra do firmware de "nova leitura" é replicada: triplas
    idênticas consecutivas do slot são potencialmente reutilizadas.
    Nunca se assume que duas triplas iguais representam duas medições
    fisicamente independentes.
    """
    tag_height_m = _positive_or_zero(tag_height_m, "tag_height_m")
    height_tolerance_m = _positive_or_zero(height_tolerance_m, "height_tolerance_m")
    loa_outlier_threshold_m = _positive_or_zero(
        loa_outlier_threshold_m, "loa_outlier_threshold_m"
    )
    if loa_outlier_threshold_m <= 0:
        raise ValueError("loa_outlier_threshold_m deve ser positivo")

    path, config_path = Path(log_path), Path(anchor_json)
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError("JSON de âncoras deve ser objeto")
    values = config.get("anchors_xy", config.get("anchors"))
    if not isinstance(values, list) or any(
        not isinstance(p, (list, tuple)) or len(p) != 3 for p in values
    ):
        raise ValueError("É necessário anchors_xy Nx3 com altura Z de cada âncora")
    try:
        anchors = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Coordenadas das âncoras devem ser finitas") from exc
    if (anchors.ndim != 2 or anchors.shape[1] != 3
            or len(anchors) < 3 or not np.isfinite(anchors).all()):
        raise ValueError("É necessário anchors_xy Nx3 finito com altura Z")
    validate_robot2026_anchor_geometry(anchors)
    mapping = read_pcnt_anchor_slot_mapping(config_path, len(anchors))

    sessions = load_pcnt_sessions(path)
    if type(session_id) is not int or not 1 <= session_id <= len(sessions):
        raise ValueError(f"Sessão inexistente; disponíveis: 1..{len(sessions)}")
    rows = sessions[session_id-1]
    if not rows[0].uwb:
        raise ValueError("CSV contém somente PCNT; sem triplas UWB para diagnosticar")
    if len(rows[0].uwb) != len(mapping):
        raise ValueError(
            f"Log possui {len(rows[0].uwb)} slots; perfil mapeia {len(mapping)}"
        )

    slots = []
    for slot, anchor_index in enumerate(mapping):
        anchor = anchors[anchor_index] if anchor_index is not None else None
        slots.append({
            "slot_spi": slot,
            "anchor_index_0_based": anchor_index,
            "mapped": anchor_index is not None,
            "anchor_xyz_m": anchor.tolist() if anchor is not None else None,
            "height_separation_m": (
                float(abs(anchor[2]-tag_height_m)) if anchor is not None else None
            ),
            "new_valid": 0, "repeated_valid": 0, "invalid": 0,
            "physically_impossible_new": 0,
            "loa_attempted": 0, "loa_solved": 0,
            "loa_residual_above_threshold": 0,
        })
    previous_valid = [None] * len(mapping)
    loa_by_slot = [[] for _ in mapping]
    ranges_by_slot = [[] for _ in mapping]
    sigma_by_slot = [[] for _ in mapping]
    examples = []
    fresh_at_least_three = fresh_at_least_four = 0
    loa_rows = 0
    impossible_new_total = 0

    for raw_index, row in enumerate(rows):
        fresh = {}
        fresh_slot = {}
        for slot, triple in enumerate(row.uwb):
            avg, var, std = triple
            state = slots[slot]
            if (not all(math.isfinite(v) for v in triple)
                    or avg <= 0 or var < 0 or std < 0):
                state["invalid"] += 1
                continue
            if previous_valid[slot] == triple:
                state["repeated_valid"] += 1
                continue
            previous_valid[slot] = triple
            state["new_valid"] += 1
            anchor_idx = mapping[slot]
            if anchor_idx is None:
                continue
            rng = avg/100.0
            sigma = std/100.0
            ranges_by_slot[slot].append(rng)
            sigma_by_slot[slot].append(sigma)
            fresh_slot[anchor_idx] = slot
            # A geometria registra cada medição nova, mesmo impossível.
            # Não altera o frame entregue ao solver normal.
            min_range = abs(float(anchors[anchor_idx, 2]) - tag_height_m)
            if rng + height_tolerance_m < min_range:
                state["physically_impossible_new"] += 1
                impossible_new_total += 1
                if len(examples) < 20:
                    examples.append({
                        "pcnt_row_index_0_based": raw_index,
                        "timestamp_s_relative_to_session": float(
                            (row.timestamp_ms - rows[0].timestamp_ms)/1000.0
                        ),
                        "slot_spi": slot,
                        "anchor_index_0_based": anchor_idx,
                        "measured_range_m": rng,
                        "vertical_minimum_m": min_range,
                    })
            else:
                # Para LOAO, precisa existir raio horizontal real.
                fresh[anchor_idx] = rng

        if len(fresh_slot) >= 3:
            fresh_at_least_three += 1
        if len(fresh_slot) >= 4:
            fresh_at_least_four += 1

        if len(fresh) < 4:
            continue
        # Se ao menos 4 âncoras passaram na plausibilidade física,
        # cada uma pode ser excluída usando outras três ou mais.
        tested = False
        for anchor_idx in fresh:
            slot = fresh_slot[anchor_idx]
            slots[slot]["loa_attempted"] += 1
            estimate = _estimate_xy_excluding(anchors, fresh, anchor_idx, tag_height_m)
            if estimate is None:
                continue
            difference = anchors[anchor_idx] - np.array([
                estimate[0], estimate[1], tag_height_m
            ], dtype=float)
            predicted = float(np.linalg.norm(difference))
            residual = float(fresh[anchor_idx] - predicted)
            if not math.isfinite(residual):
                continue
            tested = True
            slots[slot]["loa_solved"] += 1
            loa_by_slot[slot].append(residual)
            if abs(residual) > loa_outlier_threshold_m:
                slots[slot]["loa_residual_above_threshold"] += 1
        if tested:
            loa_rows += 1

    for slot, state in enumerate(slots):
        state["range_new_summary_m"] = _summarize(ranges_by_slot[slot])
        state["reported_sigma_new_summary_m"] = _summarize(sigma_by_slot[slot])
        state["loa_signed_residual_summary_m"] = _summarize(loa_by_slot[slot])
        state["loa_abs_residual_summary_m"] = _summarize(
            [abs(value) for value in loa_by_slot[slot]]
        )
        # 'new_valid' inclui triplas numericamente válidas mesmo com
        # range fisicamente impossível; este módulo NÃO filtra o solver.
        state["fresh_fraction_all_rows"] = state["new_valid"]/len(rows)
        state["loa_outlier_fraction_of_solved"] = (
            state["loa_residual_above_threshold"]/state["loa_solved"]
            if state["loa_solved"] else None
        )
    mapped = [item for item in slots if item["mapped"]]
    return {
        "schema": "bc_ekf.uwb_link_quality.v1",
        "log_filename": path.name,
        "anchor_config_filename": config_path.name,
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "anchor_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "session_id": session_id,
        "sessions_detected": len(sessions),
        "pcnt_rows": len(rows),
        "slot_count": len(mapping),
        "mapped_slots": len(mapped),
        "tag_height_m": tag_height_m,
        "height_tolerance_m": height_tolerance_m,
        "loa_outlier_threshold_m": loa_outlier_threshold_m,
        "fresh_rows_with_3plus_mapped_ranges": fresh_at_least_three,
        "fresh_rows_with_4plus_mapped_ranges": fresh_at_least_four,
        "loa_rows_with_at_least_one_solution": loa_rows,
        "physically_impossible_new_total": impossible_new_total,
        "impossible_new_examples_first_20": examples,
        "per_slot": slots,
        "notes": [
            "Diagnóstico somente leitura: nenhuma medição foi removida do Dataset Mode",
            "Slot SPI não é identidade física garantida; conferir mapeamento no firmware",
            "Repetições são triplas exatamente idênticas no mesmo slot, não eventos de ranging independentes",
            "Triplas fisicamente impossíveis são contadas apenas quando NOVAS; limites dependem da altura medida da tag",
            "LOAO usa outras leituras NOVAS da mesma linha PCNT e altura da tag fixa em solução LS 2D",
            "Resíduos LOAO medem inconsistência, não ground truth nem prova de NLOS ou falha da âncora",
            "LOAO usa a coordenada informada e pode falhar por geometria mal condicionada ou por erros de outras âncoras",
            "Timestamps PCNT são de registro, não necessariamente instante físico do ranging",
            "Os números de dist.mat do outro experimento não devem ser associados a esta sessão sem arquivos e mapeamento confirmados",
        ],
    }
