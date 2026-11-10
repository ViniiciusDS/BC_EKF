"""Diagnóstico exploratório de odometria PCNT, sem alterar logs ou perfis.

Compara poses iniciais, sinais e escalas. Não escolhe calibração pelo formato
da pista nominal, que não é ground truth temporal sincronizado.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from src.odometry.pcntlog_adapter import Robot2026Profile, convert_session, load_pcnt_sessions


def load_straight_calibration_candidates(paths, profile: Robot2026Profile):
    """Lê relatórios PR #10 coerentes com o perfil e calcula médias descritivas."""
    if not paths:
        return None
    pairs, sources = [], []
    for source in paths:
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "bc_ekf.assisted_encoder_calibration.v1":
            raise ValueError(f"{path.name}: formato de calibração desconhecido")
        if data.get("reference", {}).get("kind") != "straight" or data.get("warnings"):
            raise ValueError(f"{path.name}: relatório com avisos ou não é de reta")
        used = data.get("profile_used", {})
        for field in (
            "wheel_radius_m", "wheel_base_m", "ticks_per_wheel_rev",
            "left_scale", "right_scale", "swap_encoder_columns",
            "invert_left", "invert_right",
        ):
            actual, expected = used.get(field), getattr(profile, field)
            if isinstance(expected, bool):
                valid = type(actual) is bool and actual == expected
            else:
                valid = (type(actual) in (float, int) and math.isfinite(actual)
                         and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12))
            if not valid:
                raise ValueError(f"{path.name}: perfil incompatível em {field}")
        raw = data.get("suggestions_not_applied", {})
        left, right = raw.get("left_scale_candidate"), raw.get("right_scale_candidate")
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
               for v in (left, right)):
            raise ValueError(f"{path.name}: escalas ausentes ou inválidas")
        pairs.append((float(left), float(right)))
        sources.append(path.name)
    return {
        "left_scale": sum(p[0] for p in pairs) / len(pairs),
        "right_scale": sum(p[1] for p in pairs) / len(pairs),
        "sources": sources,
        "count": len(pairs),
        "method": "arithmetic_mean_of_independent_straight_runs_unvalidated",
    }


def load_nominal_route(path):
    """Lê waypoints XY em metros; nunca se trata de referência temporal."""
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = data.get("waypoints")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("Rota precisa de pelo menos dois waypoints")
    route = []
    for p in points:
        if (not isinstance(p, (list, tuple)) or len(p) < 2
                or any(type(v) not in (float, int) or not math.isfinite(v)
                       for v in p[:2])):
            raise ValueError("Rota com ponto XY inválido")
        route.append((float(p[0]), float(p[1])))
    if not any(math.dist(a, b) > 1e-12 for a, b in zip(route, route[1:])):
        raise ValueError("Rota degenerada")
    return route


def route_distance_rms(points_xy, route):
    """Distância RMS geométrica à polilinha, NÃO erro verdadeiro de posição."""
    segments = []
    for a, b in zip(route, route[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm2 = dx*dx + dy*dy
        if norm2 > 1e-24:
            segments.append((a, dx, dy, norm2))
    if not segments:
        raise ValueError("Rota sem segmentos válidos")
    squared, count = 0.0, 0
    for x, y in points_xy:
        best = math.inf
        for a, dx, dy, norm2 in segments:
            t = max(0.0, min(1.0, ((x-a[0])*dx + (y-a[1])*dy)/norm2))
            dist2 = (x-(a[0]+t*dx))**2 + (y-(a[1]+t*dy))**2
            best = min(best, dist2)
        squared += best
        count += 1
    return math.sqrt(squared / count) if count else None


def candidate_profiles(base: Robot2026Profile, *, measured_pose=None, scales=None):
    """Gera variantes isoladas, sem alterar o perfil passado."""
    base.validate()
    cases = [("original", base)]
    if measured_pose is not None:
        if len(measured_pose) != 3 or not all(
            type(v) in (int, float) and math.isfinite(v) for v in measured_pose
        ):
            raise ValueError("Pose medida exige x, y e heading em graus finitos")
        x, y, degrees = measured_pose
        cases.append(("pose_medida", replace(
            base, initial_x_m=float(x), initial_y_m=float(y),
            initial_theta_rad=math.radians(degrees),
        )))
    cases.extend([
        ("heading_mais_90", replace(base, initial_theta_rad=base.initial_theta_rad + math.pi/2)),
        ("heading_menos_90", replace(base, initial_theta_rad=base.initial_theta_rad - math.pi/2)),
        ("trocar_colunas", replace(base, swap_encoder_columns=not base.swap_encoder_columns)),
        # Hipótese física distinta da troca isolada: o log pode ter
        # colunas WR/WL trocadas na identificação. Ao trocar também os
        # sinais associados, ambas as rodas permanecem avançando,
        # enquanto o sinal da rotação é espelhado (dr-dl -> dl-dr).
        # NÃO implica que seja o mapeamento correto: verificar curva
        # fisicamente observada antes de aplicar ao perfil do robô.
        ("trocar_colunas_e_sinais", replace(
            base,
            swap_encoder_columns=not base.swap_encoder_columns,
            invert_left=base.invert_right,
            invert_right=base.invert_left,
            left_scale=base.right_scale,
            right_scale=base.left_scale,
        )),
        ("inverter_esquerda", replace(base, invert_left=not base.invert_left)),
        ("inverter_direita", replace(base, invert_right=not base.invert_right)),
        ("inverter_ambas", replace(
            base, invert_left=not base.invert_left, invert_right=not base.invert_right)),
    ])
    if scales is not None:
        if len(scales) != 2 or any(
            type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in scales
        ):
            raise ValueError("Escalas exigem dois números positivos finitos")
        cases.append(("escalas_reta", replace(
            base, left_scale=float(scales[0]), right_scale=float(scales[1]),
        )))
        # Relatórios PR10 identificam escalas pelos canais do perfil
        # ORIGINAL. Ao testar a hipótese de canais fisicamente trocados,
        # as escalas precisam acompanhar esses canais, não ficar presas
        # aos rótulos esquerda/direita possivelmente incorretos.
        cases.append(("trocar_colunas_e_sinais_com_escalas_reta", replace(
            base,
            swap_encoder_columns=not base.swap_encoder_columns,
            invert_left=base.invert_right,
            invert_right=base.invert_left,
            left_scale=float(scales[1]),
            right_scale=float(scales[0]),
        )))
    return cases


def compare_odometry_hypotheses(
    log_path, profile: Robot2026Profile, *, session_id=1,
    nominal_route=None, measured_pose=None, scale_candidates=None,
    observed_turn=None,
):
    """Produz relatório e linhas CSV; todo cenário usa a mesma sessão bruta."""
    profile.validate()
    sessions = load_pcnt_sessions(log_path)
    if not 1 <= int(session_id) <= len(sessions):
        raise ValueError(f"Sessão {session_id} inexistente")
    rows = sessions[int(session_id) - 1]
    if len(rows) < 2:
        raise ValueError("Sessão precisa de pelo menos duas amostras")
    if observed_turn is not None:
        if (not isinstance(observed_turn, (tuple, list))
                or len(observed_turn) != 3):
            raise ValueError("Giro observado exige (início_s, fim_s, cw/ccw)")
        t_start, t_end, expected = observed_turn
        if (type(t_start) not in (int, float) or type(t_end) not in (int, float)
                or not math.isfinite(t_start) or not math.isfinite(t_end)
                or not 0 <= t_start < t_end
                or expected not in ("cw", "ccw")):
            raise ValueError("Intervalo do giro ou direção observada inválidos")
        session_duration = (rows[-1].timestamp_ms - rows[0].timestamp_ms) / 1000.0
        if t_end > session_duration + 1e-9:
            raise ValueError("Fim do giro observado excede duração da sessão")
    variants = candidate_profiles(profile, measured_pose=measured_pose, scales=scale_candidates)
    cases, trace = [], []
    for label, variant in variants:
        poses = convert_session(rows, variant)
        begin, end = poses[0], poses[-1]
        dx, dy = end["x_m"]-begin["x_m"], end["y_m"]-begin["y_m"]
        theta0 = begin["theta_rad"]
        c, s = math.cos(theta0), math.sin(theta0)
        # Observação física independente: compara o sentido de uma curva
        # delimitada pelo operador. Nunca modifica/calibra parâmetros.
        # Faixa pequena (<=5 graus) é indeterminada, não um acerto.
        turn_check = None
        if observed_turn is not None:
            t_start, t_end, expected = observed_turn
            subset = [p for p in poses if t_start <= p["timestamp_s"] <= t_end]
            if len(subset) < 2:
                raise ValueError("Janela do giro deve conter ao menos duas amostras")
            delta_deg = math.degrees(
                subset[-1]["theta_rad"] - subset[0]["theta_rad"]
            )
            predicted = (
                "ccw" if delta_deg > 5.0
                else "cw" if delta_deg < -5.0 else "indeterminate"
            )
            turn_check = {
                "expected_direction": expected,
                "estimated_direction": predicted,
                "direction_consistent": predicted == expected,
                "heading_change_deg": delta_deg,
                "selected_samples": len(subset),
                "selected_start_s": subset[0]["timestamp_s"],
                "selected_end_s": subset[-1]["timestamp_s"],
            }
        cases.append({
            "hypothesis": label,
            "profile": {
                "initial_x_m": variant.initial_x_m,
                "initial_y_m": variant.initial_y_m,
                "initial_theta_deg": math.degrees(variant.initial_theta_rad),
                "swap_encoder_columns": variant.swap_encoder_columns,
                "invert_left": variant.invert_left,
                "invert_right": variant.invert_right,
                "left_scale": variant.left_scale, "right_scale": variant.right_scale,
            },
            "samples": len(poses),
            "duration_s": end["timestamp_s"],
            "left_wheel_total_m": sum(p["left_distance_m"] for p in poses),
            "right_wheel_total_m": sum(p["right_distance_m"] for p in poses),
            "integrated_abs_forward_m": sum(abs(p["ds_m"]) for p in poses),
            "start_xy_m": [begin["x_m"], begin["y_m"]],
            "end_xy_m": [end["x_m"], end["y_m"]],
            "end_heading_deg_unwrapped": math.degrees(end["theta_rad"]),
            "heading_change_deg": math.degrees(end["theta_rad"] - theta0),
            "observed_turn_check": turn_check,
            "relative_end_robot_frame_m": [c*dx+s*dy, -s*dx+c*dy],
            "nominal_route_rms_distance_m": (
                route_distance_rms(
                    [(p["x_m"], p["y_m"]) for p in poses], nominal_route)
                if nominal_route is not None else None
            ),
        })
        for p in poses:
            trace.append({
                "hypothesis": label, "timestamp_s": p["timestamp_s"],
                "x_m": p["x_m"], "y_m": p["y_m"],
                "theta_deg": math.degrees(p["theta_rad"]),
            })
    raw_hash = hashlib.sha256(Path(log_path).read_bytes()).hexdigest()
    report = {
        "schema": "bc_ekf.field_odometry_hypotheses.v1",
        "log_filename": Path(log_path).name,
        "raw_log_sha256": raw_hash,
        "session_id": int(session_id),
        "session_count": len(sessions),
        "nominal_route_supplied": nominal_route is not None,
        "external_turn_observation": (
            {"start_s": observed_turn[0], "end_s": observed_turn[1],
             "direction": observed_turn[2]}
            if observed_turn is not None else None
        ),
        "cases": cases,
        "best_case": None,
        "calibration_applied": False,
        "notes": [
            "Comparação exploratória: nenhuma hipótese é automaticamente escolhida",
            "Pose inicial altera alinhamento no mapa, não forma local da trajetória",
            "Distância RMS à rota nominal não é RMSE verdadeiro de localização",
            "Rota nominal pode conter aproximações ou erros de medição",
            "Escalas de reta são candidatas sujeitas a confirmação física",
            "Sinais WR/WL precisam ser validados em movimento conhecido",
            "Checagem de giro observado só compara sentido, não acurácia angular nem identidade elétrica de encoder",
        ],
    }
    return report, trace
