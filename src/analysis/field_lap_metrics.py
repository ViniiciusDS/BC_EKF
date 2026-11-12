"""Métricas descritivas por volta do LAB7A sem ground truth sincronizado.

Este módulo consome resultados já calculados pelo Dataset Mode. Nunca
reexecuta estimadores, altera fatias de tempo ou modifica o CSV de campo.
A distância à pista nominal é GEOMÉTRICA; a diferença UWB x odometria
é uma discordância entre sensores, não erro de posição verdadeiro.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.analysis.algo_metrics import distances_to_polyline


def _distance_summary(distances):
    """Resumo de distâncias em metros, excluindo exclusivamente NaN/Inf."""
    values = np.asarray(distances, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "rms_m": float(np.sqrt(np.mean(np.square(values)))),
        "mean_m": float(np.mean(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(np.max(values)),
        "count": int(len(values)),
    }


def _closure(points):
    """Distância entre primeiros/últimos pontos válidos de um trecho."""
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return None
    valid = xy[np.isfinite(xy[:, :2]).all(axis=1), :2]
    return float(np.linalg.norm(valid[-1] - valid[0])) if len(valid) >= 2 else None


def _validate_xy(data, name, expected=None):
    result = np.asarray(data, dtype=float)
    if (result.ndim != 2 or result.shape[1] < 2
            or (expected is not None and len(result) != expected)):
        raise ValueError(f"{name}: posições XY incompatíveis")
    return result


def build_field_lap_metrics(
    laps, pcnt_timestamps_s, odometry_xytheta, uwb_sample_indices,
    algorithm_results, *, reference_route=None, gate_diagnostics=None,
):
    """Produz métricas e amostragens *por volta*, sem confundir janelas com voltas.

    Args:
        laps: LapSlice da segmentação física, em índices PCNT; o desenho
            repete uma amostra nas fronteiras e este método não a repete.
        pcnt_timestamps_s: vetor de timestamps da sessão PCNT.
        odometry_xytheta: poses Nx3 da mesma sessão PCNT.
        uwb_sample_indices: índices crescentes do PCNT de cada snapshot.
        algorithm_results: dicionário de batch com `posicoes` por estimador,
            preservando alinhamento aos índices UWB.
        reference_route: polilinha nominal XY espacial (não GT temporal).
        gate_diagnostics: contagens da detecção física; informativo.

    Returns:
        Relatório JSON-serializável. Parcial/completa é propriedade da
        segmentação; nenhum trecho incompleto é classificado como volta
        física completa por inferência.
    """
    t = np.asarray(pcnt_timestamps_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps PCNT devem ser estritamente crescentes")
    odom = _validate_xy(odometry_xytheta, "odometria", len(t))
    idx = np.asarray(uwb_sample_indices)
    if (idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer)
            or (len(idx) and (idx[0] < 0 or idx[-1] >= len(t)
                               or np.any(np.diff(idx) <= 0)))):
        raise ValueError("índices UWB devem ser inteiros e crescentes dentro do PCNT")
    if not isinstance(algorithm_results, dict) or not algorithm_results:
        raise ValueError("Resultados de batch não disponíveis")

    reference = None
    if reference_route is not None:
        reference = _validate_xy(reference_route, "rota nominal")[:, :2]
        if len(reference) < 2 or not np.isfinite(reference).all():
            raise ValueError("rota nominal precisa de >= 2 pontos XY finitos")

    algorithms = {}
    for key, result in algorithm_results.items():
        if not isinstance(result, dict) or result.get("posicoes") is None:
            continue
        positions = _validate_xy(result["posicoes"], f"algoritmo {key}", len(idx))
        algorithms[key] = (
            positions[:, :2],
            str(result.get("algo_key_actual", key)),
        )
    if not algorithms:
        raise ValueError("Nenhum resultado de algoritmo com posições compatíveis")
    if not laps or not any(lap.complete for lap in laps):
        raise ValueError("Nenhuma volta física completa foi detectada")

    report_laps = []
    prev_stop = 0
    for lap in laps:
        # LapSlice.start inclui um ponto de sobreposição para o DESENHO.
        # Não duplicar essa observação nas métricas/contagens.
        start = max(int(lap.start), prev_stop)
        stop = int(lap.stop)
        if start < 0 or stop > len(t) or stop <= start:
            raise ValueError("Fatias de voltas fora dos índices da sessão")
        if report_laps and start != prev_stop:
            raise ValueError("Fatias de volta não cobrem a sessão continuamente")
        prev_stop = stop

        inside = (idx >= start) & (idx < stop)
        indexes = np.flatnonzero(inside)
        odom_seg = odom[start:stop, :2]
        # Comprimento odométrico é soma dos deslocamentos incrementais
        # calculados, e não distância de fechamento.
        segment_length = (
            float(np.linalg.norm(np.diff(odom_seg, axis=0), axis=1).sum())
            if len(odom_seg) >= 2 and np.isfinite(odom_seg).all() else None
        )
        odom_closure = _closure(odom_seg)
        item = {
            "segment_number": int(lap.number),
            "complete": bool(lap.complete),
            "pcnt_start_inclusive": start,
            "pcnt_stop_exclusive": stop,
            "pcnt_samples": stop - start,
            "uwb_snapshots": int(len(indexes)),
            "start_s": float(t[start]),
            "end_s": float(t[stop - 1]),
            "duration_s": float(t[stop - 1] - t[start]),
            "odometry": {
                "path_length_m": segment_length,
                "closure_gap_m": odom_closure,
                "closure_is_not_ground_truth_error": True,
            },
            "algorithms": {},
        }
        odo_at_uwb = odom[idx[indexes], :2]
        for name, (positions, concrete_key) in algorithms.items():
            subset = positions[indexes]
            finite = np.isfinite(subset).all(axis=1)
            kept = subset[finite]
            disagree = np.linalg.norm(kept - odo_at_uwb[finite], axis=1)
            if reference is not None and len(kept):
                nominal = distances_to_polyline(kept, reference)
            else:
                nominal = np.asarray([], dtype=float)
            item["algorithms"][name] = {
                "implementation": concrete_key,
                "valid_samples": int(np.sum(finite)),
                "invalid_samples": int(len(subset) - np.sum(finite)),
                "valid_fraction": (
                    float(np.mean(finite)) if len(subset) else None
                ),
                "nominal_route_distance": _distance_summary(nominal),
                "odometry_disagreement": _distance_summary(disagree),
                "estimated_closure_gap_m": _closure(kept),
            }
        report_laps.append(item)

    if prev_stop != len(t):
        raise ValueError("Fatias físicas não alcançam o final do registro PCNT")
    gate = gate_diagnostics or {}
    return {
        "schema": "bc_ekf.field_lap_metrics.v1",
        "reference_type": "nominal_spatial_polyline_not_ground_truth",
        "segmentation_type": "physical_start_gate",
        "pcnt_samples": int(len(t)),
        "uwb_snapshots": int(len(idx)),
        "complete_laps": int(sum(lap["complete"] for lap in report_laps)),
        "partial_segments": int(sum(not lap["complete"] for lap in report_laps)),
        "gate_returns": len(gate.get("crossing_times_s", [])),
        "algorithm_keys": list(algorithms),
        "segments": report_laps,
        "notes": [
            "Distância RMS à rota nominal não é erro verdadeiro de localização",
            "Discordância UWB x odometria não é ground truth",
            "Gap de fechamento mede a distância entre posições estimadas no trecho, não deriva real medida",
            "Primeiro e último segmentos podem ser parciais e são identificados explicitamente",
            "As amostras PCNT compartilhadas apenas para desenho não são contadas duas vezes",
            "Resultados não são recalculados; usam exatamente o batch já executado",
        ],
    }


def save_field_lap_metrics(report, output_dir, stem):
    """Salva JSON e CSV plano sem substituir relatórios de campo anteriores.

    Retorna ambos os caminhos. Um sufixo incremental permite repetir
    relatórios do mesmo log após testar configurações ou estimadores.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (not isinstance(stem, str) or not stem
            or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                   for ch in stem)):
        raise ValueError("Nome base do relatório inválido")
    base = None
    for number in range(1, 10000):
        candidate = directory / f"{stem}_voltas_{number:03d}"
        if not candidate.with_suffix(".json").exists() and not candidate.with_suffix(".csv").exists():
            base = candidate
            break
    if base is None:
        raise ValueError("Limite de nomes de relatório esgotado")
    json_path, csv_path = base.with_suffix(".json"), base.with_suffix(".csv")

    columns = [
        "segment_number", "complete", "start_s", "end_s", "duration_s",
        "pcnt_samples", "uwb_snapshots", "odom_length_m", "odom_closure_m",
        "algorithm", "implementation", "valid_samples", "valid_fraction",
        "nominal_rms_m", "nominal_p95_m",
        "odom_disagreement_rms_m", "estimated_closure_gap_m",
    ]
    created_json = False
    created_csv = False
    try:
        with json_path.open("x", encoding="utf-8") as stream:
            created_json = True
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            created_csv = True
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for segment in report["segments"]:
                for name, metrics in segment["algorithms"].items():
                    ref = metrics["nominal_route_distance"] or {}
                    comp = metrics["odometry_disagreement"] or {}
                    writer.writerow({
                        "segment_number": segment["segment_number"],
                        "complete": segment["complete"],
                        "start_s": segment["start_s"],
                        "end_s": segment["end_s"],
                        "duration_s": segment["duration_s"],
                        "pcnt_samples": segment["pcnt_samples"],
                        "uwb_snapshots": segment["uwb_snapshots"],
                        "odom_length_m": segment["odometry"]["path_length_m"],
                        "odom_closure_m": segment["odometry"]["closure_gap_m"],
                        "algorithm": name,
                        "implementation": metrics["implementation"],
                        "valid_samples": metrics["valid_samples"],
                        "valid_fraction": metrics["valid_fraction"],
                        "nominal_rms_m": ref.get("rms_m"),
                        "nominal_p95_m": ref.get("p95_m"),
                        "odom_disagreement_rms_m": comp.get("rms_m"),
                        "estimated_closure_gap_m": metrics["estimated_closure_gap_m"],
                    })
    except Exception:
        # Exclusivamente arquivos criados por ESTA chamada. Uma falha
        # de open('x') jamais pode apagar arquivo de outra execução.
        if created_csv:
            csv_path.unlink(missing_ok=True)
        if created_json:
            json_path.unlink(missing_ok=True)
        raise
    return json_path, csv_path
