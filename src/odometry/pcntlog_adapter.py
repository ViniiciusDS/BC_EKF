"""Adaptador de logs do robô ESP32 (PCNT + UWB) para o BC_EKF.

Suporta `pcntLog.csv` sem cabeçalho, com 5 ou 20 colunas.
Não altera os dados brutos nem junta sessões após reset do timestamp.
A odometria usa integração no ângulo médio, como reconstruc.m.

Os campos UWB são transmitidos pelo firmware em centímetros/cm²; os
arquivos normalizados usam metros/m². Cada slot 0..4 deve ser associado
explicitamente à geometria real das âncoras antes de rodar localização.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Robot2026Profile:
    wheel_radius_m: float = 0.03425
    wheel_base_m: float = 0.229
    ticks_per_wheel_rev: float = 2280.0
    invert_left: bool = False
    invert_right: bool = True
    # Logs novos podem trazer WR,WL em vez de WL,WR.
    swap_encoder_columns: bool = False
    left_scale: float = 1.0
    right_scale: float = 1.0
    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_theta_rad: float = 0.0

    def validate(self) -> None:
        if self.wheel_radius_m <= 0 or self.wheel_base_m <= 0:
            raise ValueError("Raio e entre-eixos precisam ser positivos")
        if self.ticks_per_wheel_rev <= 0 or self.left_scale <= 0 or self.right_scale <= 0:
            raise ValueError("CPR e escalas das rodas precisam ser positivos")


@dataclass(frozen=True)
class RawPcntRow:
    timestamp_ms: int
    left_count: int
    right_count: int
    lap_count: int
    track_status: int
    uwb: tuple[tuple[float, float, float], ...] = ()


def load_pcnt_sessions(path: str | Path) -> list[list[RawPcntRow]]:
    """Lê o CSV bruto e separa sessões quando o timestamp não cresce.

    Observação: não ordena linhas; a ordem de aquisição é essencial.
    """
    sessions: list[list[RawPcntRow]] = []
    current: list[RawPcntRow] = []
    prev_ms: int | None = None
    expected_cols: int | None = None
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, cols in enumerate(csv.reader(f, skipinitialspace=True), 1):
            if not cols or not any(col.strip() for col in cols):
                continue
            # Cinco colunas PCNT fixas + 3 campos para cada slot UWB.
            # Compatível com logs antigos de 5/20 e futuros 23/26/29...
            if len(cols) != 5 and (len(cols) < 8 or (len(cols)-5) % 3 != 0):
                raise ValueError(
                    f"Linha {line_no}: esperado 5 + 3*N colunas (N slots UWB), "
                    f"recebeu {len(cols)}"
                )
            if expected_cols is None:
                expected_cols = len(cols)
            elif len(cols) != expected_cols:
                raise ValueError(f"Linha {line_no}: formato alternou de {expected_cols} para {len(cols)} colunas")
            try:
                t, left, right, laps, status = (int(float(s)) for s in cols[:5])
                triples = tuple(
                    tuple(float(s) for s in cols[5 + 3 * i : 8 + 3 * i])
                    for i in range((len(cols) - 5) // 3)
                ) if len(cols) > 5 else ()
            except ValueError as exc:
                raise ValueError(f"Linha {line_no}: dados não numéricos") from exc
            sample = RawPcntRow(t, left, right, laps, status, triples)
            if prev_ms is not None and t <= prev_ms:
                if current:
                    sessions.append(current)
                current = []
            current.append(sample)
            prev_ms = t
    if current:
        sessions.append(current)
    if not sessions:
        raise ValueError("Arquivo sem amostras")
    return sessions


def convert_session(session: Iterable[RawPcntRow], cfg: Robot2026Profile) -> list[dict]:
    """Converte uma sessão em poses, velocidades e contagens normalizadas.

    Cada amostra do CSV bruto gera uma amostra de saída (sem decimação).
    Preserva theta contínuo, além de theta_wrapped para visualização.
    """
    cfg.validate()
    rows = list(session)
    if not rows:
        raise ValueError("Sessão vazia")
    mpt = 2 * math.pi * cfg.wheel_radius_m / cfg.ticks_per_wheel_rev
    sgn_l = -1 if cfg.invert_left else 1
    sgn_r = -1 if cfg.invert_right else 1
    zero = rows[0]
    x, y, theta = cfg.initial_x_m, cfg.initial_y_m, cfg.initial_theta_rad
    last_t_ms = zero.timestamp_ms
    # Configurável por ensaio: não muda a convenção dos logs antigos.
    zero_l, zero_r = (
        (zero.right_count, zero.left_count) if cfg.swap_encoder_columns
        else (zero.left_count, zero.right_count)
    )
    last_l, last_r = sgn_l * zero_l, sgn_r * zero_r
    out = []
    for i, row in enumerate(rows):
        raw_l, raw_r = (
            (row.right_count, row.left_count) if cfg.swap_encoder_columns
            else (row.left_count, row.right_count)
        )
        cur_l = sgn_l * raw_l
        cur_r = sgn_r * raw_r
        dt_s = (row.timestamp_ms - last_t_ms) / 1000.0 if i else 0.0
        if i and dt_s <= 0:
            raise ValueError("Timestamps não crescentes dentro da mesma sessão")
        delta_l = cur_l - last_l if i else 0
        delta_r = cur_r - last_r if i else 0
        dl = delta_l * mpt * cfg.left_scale
        dr = delta_r * mpt * cfg.right_scale
        ds = 0.5 * (dl + dr)
        dtheta = (dr - dl) / cfg.wheel_base_m
        theta_mid = theta + 0.5 * dtheta
        x += ds * math.cos(theta_mid)
        y += ds * math.sin(theta_mid)
        theta += dtheta
        out.append({
            "timestamp_ms": row.timestamp_ms - zero.timestamp_ms,
            "timestamp_s": (row.timestamp_ms - zero.timestamp_ms) / 1000.0,
            "left_ticks": cur_l - sgn_l * zero_l,
            "right_ticks": cur_r - sgn_r * zero_r,
            "delta_left_ticks": delta_l,
            "delta_right_ticks": delta_r,
            "left_distance_m": dl,
            "right_distance_m": dr,
            "ds_m": ds,
            "dtheta_rad": dtheta,
            "x_m": x,
            "y_m": y,
            "theta_rad": theta,
            "theta_wrapped_rad": math.atan2(math.sin(theta), math.cos(theta)),
            "v_m_s": ds / dt_s if dt_s else 0.0,
            "omega_rad_s": dtheta / dt_s if dt_s else 0.0,
            "lap_count": row.lap_count,
            "track_status": row.track_status,
        })
        last_t_ms, last_l, last_r = row.timestamp_ms, cur_l, cur_r
    return out


def convert_uwb_session(session: Iterable[RawPcntRow]) -> list[dict]:
    """Extrai ranges de N slots UWB, convertendo cm e cm² em SI.

    Repetição é sinalizada, não descartada. Um timestamp de registro no SD
    não equivale necessariamente ao instante físico da medição UWB.
    """
    rows = list(session)
    if not rows or not rows[0].uwb:
        return []
    zero_t = rows[0].timestamp_ms
    prev_triplets: dict[int, tuple[float, float, float]] = {}
    out = []
    for row in rows:
        for slot, (avg, var, std) in enumerate(row.uwb):
            if not all(map(math.isfinite, (avg, var, std))):
                continue
            # Valores sentinela negativos do firmware, distâncias inválidas.
            if avg <= 0 or var < 0 or std < 0:
                continue
            repeated = prev_triplets.get(slot) == (avg, var, std)
            prev_triplets[slot] = (avg, var, std)
            out.append({
                "timestamp": row.timestamp_ms - zero_t,  # ms, para o pipeline existente
                "anchor_id": slot,  # índice SPI; revisar identidade física no JSON
                "range": avg / 100.0,
                "sigma": std / 100.0,
                "variance_m2": var / 10000.0,
                "possibly_reused": int(repeated),
            })
    return out


def _write_csv(path: Path, rows: list[dict], headers: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_pcnt_sessions(
    input_path: str | Path,
    output_dir: str | Path,
    profile: Robot2026Profile | None = None,
) -> list[dict]:
    """Exporta CSV de encoder, odometria e UWB; retorna manifesto por sessão."""
    cfg = profile or Robot2026Profile()
    sessions = load_pcnt_sessions(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for session_id, session in enumerate(sessions, 1):
        stem = f"robot2026_session_{session_id:02d}"
        odo = convert_session(session, cfg)
        uwb = convert_uwb_session(session)
        encoder = [
            {"timestamp": p["timestamp_ms"], "left_ticks": p["left_ticks"], "right_ticks": p["right_ticks"]}
            for p in odo
        ]
        _write_csv(output / f"{stem}_encoder.csv", encoder, ["timestamp", "left_ticks", "right_ticks"])
        _write_csv(output / f"{stem}_odometry.csv", odo, list(odo[0]))
        uwb_file = None
        if session[0].uwb:
            _write_csv(output / f"{stem}_uwb.csv", uwb,
                       ["timestamp", "anchor_id", "range", "sigma", "variance_m2", "possibly_reused"])
            uwb_file = f"{stem}_uwb.csv"
        manifest.append({
            "session_id": session_id,
            "source_rows": len(session),
            "start_time_ms_raw": session[0].timestamp_ms,
            "end_time_ms_raw": session[-1].timestamp_ms,
            "duration_s": (session[-1].timestamp_ms - session[0].timestamp_ms) / 1000,
            "laps_end": session[-1].lap_count,
            "has_uwb": bool(session[0].uwb),
            "valid_uwb_rows": len(uwb),
            "possibly_reused_uwb_rows": sum(r["possibly_reused"] for r in uwb),
            "encoder_file": f"{stem}_encoder.csv",
            "odometry_file": f"{stem}_odometry.csv",
            "uwb_file": uwb_file,
            "final_pose": {"x_m": odo[-1]["x_m"], "y_m": odo[-1]["y_m"], "theta_rad": odo[-1]["theta_rad"]},
        })
    (output / "pcntlog_manifest.json").write_text(
        json.dumps({"profile": cfg.__dict__ if hasattr(cfg, "__dict__") else {
            field: getattr(cfg, field) for field in cfg.__dataclass_fields__},
            "sessions": manifest,
            "important": [
                "Nenhuma amostra foi decimada; sessões separadas por reset do timestamp.",
                "Contagem direita invertida; os CSV de encoder já têm sinais corrigidos.",
                "timestamps dos CSV encoder/UWB estão em milissegundos relativos à sessão.",
                "ID UWB nos CSV é o slot SPI 0..4, não necessariamente o ID físico da âncora.",
                "Mesmas médias UWB consecutivas podem ser frames reutilizados.",
                "PcntLog antigo com 5 colunas não contém dados UWB.",
                "Odometria é estimativa; não deve ser tratada como ground truth.",
            ]}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
