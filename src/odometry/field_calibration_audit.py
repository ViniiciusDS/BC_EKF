"""Auditoria de deriva odométrica usando voltas reconhecidas pelo PR #12/13.

Os índices de volta vêm do UWB-LS, e as posições odométricas são
reconstruídas dos mesmos contadores PCNT. Esses eventos são REFERÊNCIA
DE RETORNO APROXIMADA, não ground truth contínuo. Nenhum parâmetro é
ajustado por otimização nem escrito no perfil original.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path

from src.odometry.pcntlog_adapter import Robot2026Profile, convert_session, load_pcnt_sessions


def _sha256(path):
    """Hash dos bytes exatos do arquivo, sem reformatar JSON/CSV."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: esperado número finito")
    return float(value)


def _validate_lap_source(report, log_path, anchor_path, session_count):
    """Exige proveniência F7 e fatias sem lacuna, em índices PCNT."""
    if report.get("schema") != "bc_ekf.field_lap_metrics.v1":
        raise ValueError("Relatório deve ser exportação F7 do PR #13")
    if report.get("segmentation_type") != "physical_start_gate":
        raise ValueError("Apenas voltas detectadas pelo portão físico são aceitas")
    if report.get("source_file") != Path(log_path).name:
        raise ValueError("O nome do CSV não corresponde à origem do relatório F7")
    if report.get("source_sha256") != _sha256(log_path):
        raise ValueError("O SHA-256 do CSV bruto difere do relatório F7")
    if report.get("anchor_config_file") != Path(anchor_path).name:
        raise ValueError("O nome do JSON de âncoras difere do relatório F7")
    if report.get("anchor_config_sha256") != _sha256(anchor_path):
        raise ValueError("O SHA-256 do perfil de âncoras difere do relatório F7")
    session_id = report.get("session_id")
    if type(session_id) is not int or not 1 <= session_id <= session_count:
        raise ValueError("Sessão do relatório F7 inexistente no log")
    if type(report.get("pcnt_samples")) is not int:
        raise ValueError("Contagem PCNT F7 inválida")
    segments = report.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Relatório F7 sem segmentos")
    previous = 0
    complete = []
    for i, seg in enumerate(segments, start=1):
        if not isinstance(seg, dict) or type(seg.get("complete")) is not bool:
            raise ValueError("Segmento F7 inválido")
        start, stop = seg.get("pcnt_start_inclusive"), seg.get("pcnt_stop_exclusive")
        if type(start) is not int or type(stop) is not int:
            raise ValueError("Limites PCNT do segmento devem ser inteiros")
        if start != previous or stop <= start or stop > report["pcnt_samples"]:
            raise ValueError("Fatias F7 devem cobrir PCNT sem sobreposição ou lacuna")
        if seg.get("segment_number") != i or seg.get("pcnt_samples") != stop-start:
            raise ValueError("Identificadores ou quantidade de PCNT F7 incoerentes")
        if seg["complete"]:
            if stop >= report["pcnt_samples"]:
                raise ValueError("Volta completa deve terminar num retorno registrado")
            complete.append((seg, start, stop))
        previous = stop
    if previous != report["pcnt_samples"]:
        raise ValueError("Fatias F7 não cobrem todos os dados PCNT")
    if (not complete or len(complete) != report.get("complete_laps")
            or sum(not s["complete"] for s in segments) != report.get("partial_segments")):
        raise ValueError("Contagem de voltas F7 incoerente")
    gate_times = report.get("gate_crossing_times_s")
    if (not isinstance(gate_times, list) or
            len(gate_times) != report.get("gate_returns") or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in gate_times)):
        raise ValueError("Eventos de retorno UWB ausentes ou inválidos no relatório F7")
    return session_id, complete


def _trend(series):
    """OLS descritivo por número de volta; NÃO é incerteza estatística."""
    n = len(series)
    if n < 2:
        return None
    x = tuple(float(i+1) for i in range(n))
    mx = sum(x)/n
    my = sum(series)/n
    denominator = sum((v-mx)**2 for v in x)
    slope = sum((a-mx)*(b-my) for a,b in zip(x,series))/denominator
    intercept = my - slope*mx
    total = sum((v-my)**2 for v in series)
    residual = sum((v-(intercept+slope*u))**2 for u,v in zip(x,series))
    return {
        "slope_m_per_lap": slope,
        "intercept_m": intercept,
        "r_squared_descriptive": 1-residual/total if total > 1e-18 else None,
        "first_m": series[0],
        "last_m": series[-1],
        "ratio_last_first": (series[-1]/series[0] if series[0] > 1e-12 else None),
        "sample_count": n,
    }


def _uwb_odom_trend(complete, algorithm):
    """Extrai discordância já calculada pelo F7, sem recalcular UWB."""
    values = []
    for seg, _, _ in complete:
        alg = seg.get("algorithms", {}).get(algorithm)
        metric = alg.get("odometry_disagreement") if isinstance(alg, dict) else None
        if not isinstance(metric, dict) or metric.get("rms_m") is None:
            raise ValueError(f"Algoritmo {algorithm} ausente ou sem discordância na volta {seg['segment_number']}")
        values.append(_finite_number(metric["rms_m"], "RMS UWB × odometria"))
    return {"algorithm": algorithm, "per_complete_lap_rms_m": values,
            "linear_trend": _trend(values)}


def _scenario(rows, profile, complete, *, name, expected_lap_turn_deg):
    poses = convert_session(rows, profile)
    first_index = complete[0][1]
    x0, y0 = poses[first_index]["x_m"], poses[first_index]["y_m"]
    data = []
    for seg, start, stop in complete:
        # Os índices F7 são [start,stop); já o detector marca o
        # RETORNO no índice stop. Usar stop como endpoint do giro
        # garante portão a portão, sem incluir amostra falsa no F7.
        a, b = poses[start], poses[stop]
        gate_displacement = math.hypot(b["x_m"]-a["x_m"], b["y_m"]-a["y_m"])
        from_initial_gate = math.hypot(b["x_m"]-x0, b["y_m"]-y0)
        yaw = math.degrees(b["theta_rad"]-a["theta_rad"])
        # Comprimento odométrico reconstruído entre as duas passagens,
        # não distância aproximada da pista e nem ground truth.
        distance = sum(abs(p["ds_m"]) for p in poses[start+1:stop+1])
        data.append({
            "segment_number": int(seg["segment_number"]),
            "gate_start_index_pcnt": start,
            "gate_end_index_pcnt": stop,
            "lap_duration_s": b["timestamp_s"]-a["timestamp_s"],
            "odom_path_m": distance,
            "gate_to_gate_closure_m": gate_displacement,
            "cumulative_offset_from_first_gate_m": from_initial_gate,
            "odom_net_heading_deg_unwrapped": yaw,
            "heading_residual_deg_assumed": (
                yaw-expected_lap_turn_deg
                if expected_lap_turn_deg is not None else None
            ),
            "start_xy_m": [a["x_m"],a["y_m"]],
            "end_xy_m": [b["x_m"],b["y_m"]],
        })
    return {
        "name": name,
        "profile_used": {
            "left_scale": profile.left_scale,
            "right_scale": profile.right_scale,
            "wheel_base_m": profile.wheel_base_m,
            "swap_encoder_columns": profile.swap_encoder_columns,
            "invert_left": profile.invert_left,
            "invert_right": profile.invert_right,
            "initial_theta_deg": math.degrees(profile.initial_theta_rad),
        },
        "complete_laps": data,
        "mean_gate_to_gate_closure_m": sum(d["gate_to_gate_closure_m"] for d in data)/len(data),
        "final_cumulative_offset_from_first_gate_m": data[-1]["cumulative_offset_from_first_gate_m"],
        "heading_residuals_deg_assumed": [d["heading_residual_deg_assumed"] for d in data],
    }


def audit_field_odometry(
    lap_report, log_path, anchor_path, profile: Robot2026Profile,
    *, algorithm="lms", expected_lap_turn_deg=None, scale_candidate=None,
):
    """Audita odometria com proveniência forte; nunca escolhe perfil vencedor.

    O relatório F7 foi gerado a partir de uma geometria UWB validada
    pelo operador. Mesmo assim, cruzamentos são aproximações espaciais,
    não medições independentes do centro físico do robô.
    """
    if not isinstance(lap_report, dict):
        raise ValueError("lap_report deve ser objeto JSON")
    sessions = load_pcnt_sessions(log_path)
    session_id, complete = _validate_lap_source(
        lap_report, log_path, anchor_path, len(sessions)
    )
    rows = sessions[session_id-1]
    if len(rows) != lap_report["pcnt_samples"]:
        raise ValueError("Sessão PCNT não possui a contagem F7 registrada")
    profile.validate()
    if expected_lap_turn_deg is not None:
        expected_lap_turn_deg = _finite_number(
            expected_lap_turn_deg, "giro esperado por volta"
        )
        if abs(expected_lap_turn_deg) < 1e-9:
            raise ValueError("Giro esperado não pode ser zero")
    variants = [("perfil_de_campo", profile)]
    if scale_candidate is not None:
        if not isinstance(scale_candidate, (tuple,list)) or len(scale_candidate)!=2:
            raise ValueError("Par candidato exige duas escalas positivas")
        left = _finite_number(scale_candidate[0], "escala esquerda")
        right = _finite_number(scale_candidate[1], "escala direita")
        if left <= 0 or right <= 0:
            raise ValueError("Escalas candidatas devem ser positivas")
        variants.append(("escalas_experimentais", replace(
            profile, left_scale=left, right_scale=right
        )))
    return {
        "schema": "bc_ekf.field_odometry_audit.v1",
        "session_id": session_id,
        "pcnt_samples": len(rows),
        "source_sha256": _sha256(log_path),
        "anchor_config_sha256": _sha256(anchor_path),
        "gate_reference_kind": "uwb_lms_estimated_return_indices_not_ground_truth",
        "complete_lap_count": len(complete),
        "expected_lap_turn_deg_assumption": expected_lap_turn_deg,
        "uwb_odom_disagreement_from_f7": _uwb_odom_trend(complete, algorithm),
        "scenarios": [
            _scenario(rows, variant, complete, name=name,
                      expected_lap_turn_deg=expected_lap_turn_deg)
            for name, variant in variants
        ],
        "selected_calibration": None,
        "applied_to_original_profile": False,
        "limitations": [
            "Retornos UWB-LMS aproximam o mesmo lugar físico, mas não são ground truth sincronizado",
            "Fechamento odométrico não é erro verdadeiro: centro do gate e cruzamentos têm tolerância",
            "Giro esperado por volta, se fornecido, é hipótese topológica e não ângulo medido externamente",
            "Tendência de divergência UWB × odometria não determina qual sensor está errado",
            "Escalas experimentais são simulação; nenhum ajuste é escolhido ou aplicado automaticamente",
            "Calibração de entre-eixos exige giro isolado com ângulo físico conhecido e sinais verificados",
        ],
    }
