"""Diagnóstico assistido e NÃO destrutivo dos encoders do robô ESP32.

A referência de reta/giro é medida externamente e digitada pelo operador.
Não altera o perfil, arquivos, comandos nem algoritmos do BC-EKF.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from src.odometry.pcntlog_adapter import (
    Robot2026Profile, load_pcnt_sessions, convert_session,
)


@dataclass(frozen=True)
class CalibrationReference:
    """Um único movimento com grandeza física informada pelo operador.

    kind='straight': distance_m estritamente positiva na direção para frente.
    kind='turn': angle_deg assinado (+ anti-horário, - horário).
    Limites em segundos relativos ao início da sessão, inclusive.
    """
    kind: str
    value: float
    start_s: float | None = None
    end_s: float | None = None

    def validate(self):
        if self.kind not in ("straight", "turn"):
            raise ValueError("Movimento deve ser straight ou turn")
        if not math.isfinite(self.value) or self.value == 0:
            raise ValueError("Referência deve ser finita e diferente de zero")
        if self.kind == "straight" and self.value <= 0:
            raise ValueError("Para reta, informe uma distância positiva para frente")
        for name, item in (("start_s", self.start_s), ("end_s", self.end_s)):
            if item is not None and (not math.isfinite(item) or item < 0):
                raise ValueError(f"{name} deve ser não negativo e finito")
        if self.start_s is not None and self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("end_s deve ser maior que start_s")


def analyze_encoder_calibration(log_path, profile: Robot2026Profile,
                                reference: CalibrationReference, session_id=1):
    """Calcula resumo e *sugestões*, nunca aplica ajustes automaticamente.

    Os limites devem abranger o movimento isolado; uma sessão que contém
    múltiplas manobras não permite calibração confiável sem recorte.
    """
    reference.validate()
    profile.validate()
    sessions = load_pcnt_sessions(log_path)
    if not 1 <= int(session_id) <= len(sessions):
        raise ValueError("Sessão inexistente")
    session = sessions[int(session_id) - 1]
    t0 = session[0].timestamp_ms
    selected = [
        row for row in session
        if (reference.start_s is None or (row.timestamp_ms - t0) / 1000 >= reference.start_s)
        and (reference.end_s is None or (row.timestamp_ms - t0) / 1000 <= reference.end_s)
    ]
    if len(selected) < 2:
        raise ValueError("Intervalo precisa conter ao menos duas amostras")
    # Reutiliza EXATAMENTE as convenções cinemáticas do projeto.
    poses = convert_session(selected, profile)
    dl = sum(p["left_distance_m"] for p in poses)
    dr = sum(p["right_distance_m"] for p in poses)
    ds = (dl + dr) / 2
    dtheta = (dr - dl) / profile.wheel_base_m
    warnings = []
    suggestions = {}
    tol = 1e-9
    if reference.kind == "straight":
        # Distância reta positiva observada externamente; não resolve
        # raios individuais quando as rodas discordam.
        if dl <= tol or dr <= tol:
            warnings.append("Uma ou ambas as rodas não avançaram conforme a convenção do perfil; verificar WR/WL, swap e inversões")
        else:
            # Por roda: fator multiplicativo do left_scale/right_scale,
            # válido somente se movimento realmente foi reto para frente.
            suggestions["left_scale_candidate"] = profile.left_scale * reference.value / dl
            suggestions["right_scale_candidate"] = profile.right_scale * reference.value / dr
        if abs(dl - dr) > max(0.02, 0.1 * abs(ds)):
            warnings.append("Diferença superior ao limite indicativo entre rodas em ensaio reto; investigar patinagem, sinais e geometria")
        observed = ds
        target = reference.value
        unit = "m"
    else:
        target = math.radians(reference.value)
        observed = dtheta
        unit = "rad"
        if dl * dr >= 0:
            warnings.append("As rodas não apresentam incrementos opostos no giro; verificar sentido e tipo de manobra")
        # Base: theta=(dr-dl)/base. Só há solução física se o sinal condiz.
        base_candidate = (dr - dl) / target
        if base_candidate > tol and dl * dr < 0:
            suggestions["wheel_base_m_candidate"] = base_candidate
        else:
            warnings.append("Giro incompatível com a convenção angular; não sugerir base positiva")
        # Um único giro não identifica raio de roda e base simultaneamente.
    if abs(target) > tol:
        relative_error = (observed - target) / abs(target)
    else:
        relative_error = None
    return {
        "schema": "bc_ekf.assisted_encoder_calibration.v1",
        "log_filename": Path(log_path).name,
        "session_id": int(session_id),
        "reference": {
            "kind": reference.kind,
            "value": reference.value,
            "unit": "m" if reference.kind == "straight" else "deg",
            "start_s": reference.start_s, "end_s": reference.end_s,
        },
        "samples": len(selected),
        "duration_s": round((selected[-1].timestamp_ms - selected[0].timestamp_ms) / 1000, 6),
        "observed": {
            "left_wheel_m": dl,
            "right_wheel_m": dr,
            "center_displacement_integrated_m": ds,
            "heading_change_deg": math.degrees(dtheta),
            "comparison_value": observed,
            "comparison_unit": unit,
            "signed_relative_error": relative_error,
        },
        "profile_used": {
            "wheel_radius_m": profile.wheel_radius_m,
            "wheel_base_m": profile.wheel_base_m,
            "ticks_per_wheel_rev": profile.ticks_per_wheel_rev,
            "swap_encoder_columns": profile.swap_encoder_columns,
            "invert_left": profile.invert_left,
            "invert_right": profile.invert_right,
            "left_scale": profile.left_scale,
            "right_scale": profile.right_scale,
        },
        "suggestions_not_applied": suggestions,
        "warnings": warnings,
        "notes": [
            "Valores sugeridos são hipóteses: validar com repetições independentes e medição física",
            "Requer movimento isolado com distância/ângulo externo medido",
            "Não calibra automaticamente CPR, raio individual e entre-eixos de forma conjunta",
            "Nenhum CSV ou JSON de configuração foi modificado",
        ],
    }
