"""Diagnóstico não destrutivo de logs ESP32 PCNT+UWB para ensaios de campo.

Sem ground truth, as estatísticas descrevem a aquisição e não a precisão.
Não altera timestamps, frames UWB, odometria nem saídas dos algoritmos.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.odometry.pcntlog_adapter import load_pcnt_sessions, convert_session
from src.ui.algo_modes.dataset_uwb2026_pipeline import read_robot2026_profile_from_anchor_json
from src.ui.algo_modes.dataset_odometry_pipeline import ROBOT2026_PROFILE
from src.ui.algo_modes.dataset_uwb2026_pipeline import inspect_robot2026_capture


def summarize_encoder_motion(rows, profile):
    """Resumo cinemático dos encoders, sem inferir orientação física.

    Integra com o mesmo perfil do Dataset Mode. Velocidades aqui são
    estimativas odométricas, não velocidade verdadeira medida.
    """
    samples = convert_session(rows, profile)
    speeds = [abs(p["v_m_s"]) for p in samples[1:]]
    turns = [abs(p["omega_rad_s"]) for p in samples[1:]]
    dt = [(b.timestamp_ms - a.timestamp_ms) / 1000.0
          for a, b in zip(rows, rows[1:])]
    wheel_l = [p["delta_left_ticks"] for p in samples[1:]]
    wheel_r = [p["delta_right_ticks"] for p in samples[1:]]
    def median(values):
        ordered = sorted(values)
        if not ordered:
            return None
        middle = len(ordered) // 2
        return (ordered[middle] if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2)
    return {
        "sample_count": len(samples),
        "duration_s": samples[-1]["timestamp_s"],
        "estimated_travel_m": round(sum(abs(p["ds_m"]) for p in samples), 5),
        "net_displacement_m": round(math.hypot(
            samples[-1]["x_m"] - samples[0]["x_m"],
            samples[-1]["y_m"] - samples[0]["y_m"]), 5),
        "max_abs_linear_speed_m_s": round(max(speeds, default=0.0), 5),
        "max_abs_angular_speed_rad_s": round(max(turns, default=0.0), 5),
        "median_period_s": median(dt),
        "negative_delta_left_fraction": (
            sum(d < 0 for d in wheel_l) / len(wheel_l) if wheel_l else 0.0
        ),
        "negative_delta_right_fraction": (
            sum(d < 0 for d in wheel_r) / len(wheel_r) if wheel_r else 0.0
        ),
        "profile": {
            "wheel_radius_m": profile.wheel_radius_m,
            "wheel_base_m": profile.wheel_base_m,
            "ticks_per_wheel_rev": profile.ticks_per_wheel_rev,
            "swap_encoder_columns": profile.swap_encoder_columns,
            "invert_left": profile.invert_left,
            "invert_right": profile.invert_right,
        },
        "interpretation": (
            "Descritivo: confirmar ordem, sinais e escalas com movimento "
            "físico reto e giro de sentido conhecido; não é calibração automática."
        ),
    }


def build_field_diagnostic(log_path, anchors_path, session_id=1, *, profile=None):
    """Inspeciona a sessão bruta e retorna um relatório serializável em JSON.

    As triplas repetidas são contabilizadas por slot. Não são consideradas
    snapshots independentes, nem reconstruídas a partir de leituras antigas.
    """
    path = Path(log_path)
    sessions = load_pcnt_sessions(path)
    if not 1 <= int(session_id) <= len(sessions):
        raise ValueError(f"Sessão {session_id} inexistente")
    rows = sessions[int(session_id) - 1]
    summary = inspect_robot2026_capture(
        path, session_id=int(session_id), anchor_path=anchors_path
    )
    selected_profile = profile or (read_robot2026_profile_from_anchor_json(anchors_path)
                                   if anchors_path else ROBOT2026_PROFILE)
    encoder = summarize_encoder_motion(rows, selected_profile)
    per_slot = []
    if rows[0].uwb:
        for slot in range(len(rows[0].uwb)):
            prev = None
            valid = repeated = invalid = 0
            for row in rows:
                triple = row.uwb[slot]
                a, v, s = triple
                if not all(math.isfinite(x) for x in triple) or a <= 0 or v < 0 or s < 0:
                    invalid += 1
                    continue
                if triple == prev:
                    repeated += 1
                else:
                    valid += 1
                prev = triple
            per_slot.append({
                "slot": slot, "new_valid": valid,
                "repeated_valid": repeated, "invalid": invalid,
            })
    warnings = list(summary.get("warnings", []))
    # Avisos são sinalizações, não causas comprovadas.
    if summary.get("status") == "ready_for_static_batch":
        samples = max(1, int(summary.get("pcnt_samples", 0)))
        if summary.get("uwb_snapshots", 0) / samples < 0.2:
            warnings.append("Menos de 20% das linhas contêm snapshot UWB completo e novo")
        if summary.get("uwb_snapshots", 0) < 30:
            warnings.append("Poucos snapshots completos para análise estatística")
    summary["warnings"] = list(dict.fromkeys(warnings))
    return {
        "schema": "bc_ekf.field_diagnostic.v1",
        "synthetic_or_real": "not_inferred",
        "ground_truth_available": False,
        "log_filename": path.name,
        "anchor_config_filename": Path(anchors_path).name if anchors_path else None,
        "session_id": int(session_id),
        "summary": summary,
        "slot_quality": per_slot,
        "encoder_quality": encoder,
        "notes": [
            "Timestamps são de registro ESP32, não necessariamente instantes físicos TWR",
            "Contagem de eventos não identifica necessariamente voltas reais",
            "Pose PCNT e rota nominal não são ground truth sincronizado",
            "Diagnóstico somente de leitura: CSV original preservado",
        ],
    }


def save_field_diagnostic(report, output_path):
    """Exporta JSON de diagnóstico sem sobrepor arquivos de entrada."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return destination
