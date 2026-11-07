"""Diagnóstico não destrutivo de logs ESP32 PCNT+UWB para ensaios de campo.

Sem ground truth, as estatísticas descrevem a aquisição e não a precisão.
Não altera timestamps, frames UWB, odometria nem saídas dos algoritmos.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.odometry.pcntlog_adapter import load_pcnt_sessions
from src.ui.algo_modes.dataset_uwb2026_pipeline import inspect_robot2026_capture


def build_field_diagnostic(log_path, anchors_path, session_id=1):
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
