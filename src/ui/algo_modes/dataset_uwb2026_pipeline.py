"""Importação do log ESP32 PCNT+UWB 2026 para uma tag.

Bruto sem cabeçalho, 20 colunas: timestamp, L, R, volta, status,
cinco triplas (média cm, variância cm², desvio cm).
O timestamp é do registro do ESP32, não do instante físico do ranging.
Nenhum range é interpolado; slots SPI exigem mapeamento explícito.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np

from src.odometry.pcntlog_adapter import (
    Robot2026Profile, load_pcnt_sessions, convert_uwb_session,
)
from src.ui.algo_modes.dataset_odometry_pipeline import (
    Odometry2026Data, ROBOT2026_PROFILE, load_robot2026_odometry,
)


@dataclass(frozen=True)
class SingleTagUwb2026:
    odometry: Odometry2026Data
    distances: np.ndarray
    deviations: np.ndarray
    sample_indices: np.ndarray
    timestamps_s: np.ndarray
    valid_readings: int
    repeated_readings: int
    invalid_readings: int
    slot_to_anchor_index: tuple[int | None, ...]

    @property
    def sample_count(self) -> int:
        return len(self.sample_indices)


def read_robot2026_profile_from_anchor_json(
    anchor_json: str | Path,
) -> Robot2026Profile:
    """Lê configuração específica do ensaio sem alterar os defaults do projeto.

    Campo opcional "robot_2026" no JSON de âncoras. A orientação pode ser
    especificada em graus, mas o modelo interno usa radianos.
    """
    with Path(anchor_json).open("r", encoding="utf-8") as stream:
        root = json.load(stream)
    values = root.get("robot_2026", {})
    if not isinstance(values, dict):
        raise ValueError("robot_2026 deve ser um objeto JSON")
    allowed = {
        "wheel_radius_m", "wheel_base_m", "ticks_per_wheel_rev",
        "invert_left", "invert_right", "swap_encoder_columns",
        "left_scale", "right_scale",
        "initial_x_m", "initial_y_m", "initial_theta_deg", "initial_theta_rad",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("Campos desconhecidos em robot_2026: " + ", ".join(unknown))
    if "initial_theta_deg" in values and "initial_theta_rad" in values:
        raise ValueError("Use initial_theta_deg OU initial_theta_rad, não ambos")

    params = {}
    for key, value in values.items():
        if key == "initial_theta_deg":
            if isinstance(value, bool):
                raise ValueError("initial_theta_deg deve ser numérico")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("initial_theta_deg deve ser finito")
            params["initial_theta_rad"] = math.radians(number)
        elif key in ("invert_left", "invert_right", "swap_encoder_columns"):
            if type(value) is not bool:
                raise ValueError(f"{key} deve ser booleano")
            params[key] = value
        else:
            if isinstance(value, bool):
                raise ValueError(f"{key} deve ser numérico")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{key} deve ser finito")
            params[key] = number

    profile = replace(ROBOT2026_PROFILE, **params)
    profile.validate()
    return profile


def read_pcnt_anchor_slot_mapping(anchor_json: str | Path,
                                  number_of_anchors: int) -> tuple[int | None, ...]:
    """Exige mapa de slot SPI 0..4 para a posição no array de âncoras."""
    with Path(anchor_json).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("meta", {}).get("example_only") is True:
        raise ValueError(
            "Configuração é apenas EXEMPLO: substitua as coordenadas medidas "
            "e altere meta.example_only para false antes do ensaio"
        )
    mapping = data.get("pcnt_slot_to_anchor_index")
    if not isinstance(mapping, list) or len(mapping) < 3:
        raise ValueError(
            "JSON precisa de pcnt_slot_to_anchor_index: índices inteiros "
            "(ou null), um por slot SPI; ao menos 3 slots"
        )
    used: set[int] = set()
    output = []
    for slot, idx in enumerate(mapping):
        if idx is None:
            output.append(None)
            continue
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError(f"Slot {slot}: índice deve ser inteiro ou null")
        if not 0 <= idx < number_of_anchors:
            raise ValueError(f"Slot {slot}: índice fora do layout")
        if idx in used:
            raise ValueError(f"Índice {idx} atribuído a mais de um slot")
        output.append(idx)
        used.add(idx)
    if len(used) < 3:
        raise ValueError("São necessárias três âncoras mapeadas para localização 2D")
    return tuple(output)


def load_robot2026_single_tag_uwb(
    pcnt_path: str | Path,
    *,
    session_id: int,
    anchor_count: int,
    slot_to_anchor_index: tuple[int | None, ...],
    min_valid_anchors: int = 3,
    min_sigma_m: float = 0.02,
    profile: Robot2026Profile | None = None,
) -> SingleTagUwb2026:
    """Preserva PCNT integral, usa no batch só frames com 3+ ranges novos.

    Triplas UWB idênticas às anteriores do mesmo slot são tratadas como
    potencialmente reutilizadas; não se pressupõe independência.
    """
    if min_valid_anchors < 3 or min_sigma_m <= 0:
        raise ValueError("Min. 3 âncoras e sigma mínimo positivo necessários")
    if anchor_count < 3 or len(slot_to_anchor_index) < 3:
        raise ValueError("Layout/slots inválidos")
    assigned = [v for v in slot_to_anchor_index if v is not None]
    if any(not isinstance(v, int) or not 0 <= v < anchor_count for v in assigned):
        raise ValueError("Índice de slot fora do layout")
    if len(set(assigned)) != len(assigned):
        raise ValueError("Âncora mapeada repetidamente")
    if min_valid_anchors > len(assigned):
        raise ValueError("Mínimo de ranges supera número de âncoras mapeadas")

    path = Path(pcnt_path)
    sessions = load_pcnt_sessions(path)
    if not 1 <= session_id <= len(sessions):
        raise ValueError(f"Sessão {session_id} inexistente")
    raw = sessions[session_id - 1]
    if not raw or not raw[0].uwb:
        raise ValueError(
            "Arquivo PCNT possui somente 5 colunas (sem UWB). "
            "Para esta fonte, selecione um log bruto com 5+3*N colunas."
        )
    if len(raw[0].uwb) != len(slot_to_anchor_index):
        raise ValueError(
            f"Log tem {len(raw[0].uwb)} slots UWB, mas JSON mapeia "
            f"{len(slot_to_anchor_index)}. Ajuste pcnt_slot_to_anchor_index."
        )
    odo = load_robot2026_odometry(
        path, session_id=session_id, profile=profile or ROBOT2026_PROFILE
    )
    n = len(raw)
    R = np.full((n, anchor_count), np.nan, dtype=float)
    S = np.full_like(R, np.nan)
    stamps = {row.timestamp_ms - raw[0].timestamp_ms: i for i, row in enumerate(raw)}
    rows = convert_uwb_session(raw)
    repeated, valid = 0, 0
    for item in rows:
        slot = int(item["anchor_id"])
        anchor = slot_to_anchor_index[slot]
        if anchor is None:
            continue
        if item["possibly_reused"]:
            repeated += 1
            continue
        distance = float(item["range"])
        sigma = float(item["sigma"])
        if not np.isfinite(distance) or not 0 < distance <= 100:
            continue
        i = stamps[int(item["timestamp"])]
        R[i, anchor] = distance
        S[i, anchor] = max(min_sigma_m, sigma)
        valid += 1

    invalid = n * len(assigned) - valid - repeated
    indices = np.flatnonzero(np.sum(np.isfinite(R), axis=1) >= min_valid_anchors)
    if indices.size == 0:
        raise ValueError(
            "Nenhuma linha possui 3 ranges UWB NOVOS e válidos. "
            "Confira atualização dos slots SPI e sincronização; "
            "não combinar leituras antigas como se fossem simultâneas."
        )
    return SingleTagUwb2026(
        odometry=odo,
        distances=R[indices],
        deviations=S[indices],
        sample_indices=indices,
        timestamps_s=odo.timestamps_s[indices],
        valid_readings=valid,
        repeated_readings=repeated,
        invalid_readings=invalid,
        slot_to_anchor_index=slot_to_anchor_index,
    )




def read_robot2026_view_settings(anchor_json: str | Path) -> dict:
    """Configura visualização, sem alterar PCNT/UWB ou inferir ground truth.

    Modos: lap_counter (legado), time_windows (80s padrão) e start_gate.
    O modo start_gate exige ponto e direção do portão medidos ou declarados
    explicitamente no referencial das âncoras. A fonte de cruzamentos é
    odometria ou UWB/LMS após batch, sempre indicada ao operador.
    """
    with Path(anchor_json).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    cfg = data.get("view_2026", {})
    if not isinstance(cfg, dict):
        raise ValueError("view_2026 deve ser um objeto")
    allowed = {
        "segmentation", "window_duration_s", "gate_source",
        "gate_xy_m", "gate_direction_xy", "gate_half_width_m",
        "gate_hysteresis_m", "gate_min_lap_s", "gate_min_lap_distance_m",
        "gate_departure_m", "gate_max_step_m",
        "gate_mode", "gate_return_direction_xy", "gate_return_radius_m",
        "gate_smoothing_window", "gate_approach_lookback_s",
        "gate_approach_min_m", "gate_max_gap_s",
    }
    extra = set(cfg) - allowed
    if extra:
        raise ValueError("Campos não reconhecidos em view_2026: " + ", ".join(sorted(extra)))
    kind = cfg.get("segmentation", "lap_counter")
    if kind not in ("lap_counter", "time_windows", "start_gate"):
        raise ValueError("view_2026.segmentation deve ser lap_counter, time_windows ou start_gate")

    def positive(name, default):
        value = cfg.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"view_2026.{name} deve ser positivo")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"view_2026.{name} deve ser positivo") from exc
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"view_2026.{name} deve ser positivo")
        return number

    duration = positive("window_duration_s", 80.0)
    settings = {"segmentation": kind, "window_duration_s": duration}
    if kind == "start_gate":
        source = cfg.get("gate_source", "odometry")
        if source not in ("odometry", "uwb_lms"):
            raise ValueError("view_2026.gate_source deve ser odometry ou uwb_lms")

        def vector(name):
            value = cfg.get(name)
            if not isinstance(value, list) or len(value) != 2 or any(
                isinstance(v, bool) for v in value
            ):
                raise ValueError(f"view_2026.{name} exige [x, y] em metros")
            try:
                result = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"view_2026.{name} exige números finitos") from exc
            if not np.isfinite(result).all():
                raise ValueError(f"view_2026.{name} exige números finitos")
            return result.tolist()

        xy = vector("gate_xy_m")
        direction = vector("gate_direction_xy")
        if float(np.linalg.norm(direction)) < 1e-9:
            raise ValueError("view_2026.gate_direction_xy não pode ser zero")
        gate_mode = cfg.get("gate_mode", "crossing")
        if gate_mode not in ("crossing", "corner_return"):
            raise ValueError("view_2026.gate_mode deve ser crossing ou corner_return")
        smooth = cfg.get("gate_smoothing_window", 1)
        if type(smooth) is not int or not (1 <= smooth <= 21) or smooth % 2 == 0:
            raise ValueError("view_2026.gate_smoothing_window exige ímpar de 1 a 21")
        incoming = (
            vector("gate_return_direction_xy")
            if gate_mode == "corner_return" else None
        )
        if incoming is not None and float(np.linalg.norm(incoming)) < 1e-9:
            raise ValueError("gate_return_direction_xy não pode ser zero")
        settings.update({
            "gate_source": source,
            "gate_mode": gate_mode,
            "gate_return_direction_xy": incoming,
            "gate_smoothing_window": smooth,
            "gate_return_radius_m": positive("gate_return_radius_m", .5),
            "gate_approach_lookback_s": positive("gate_approach_lookback_s", 2.5),
            "gate_approach_min_m": positive("gate_approach_min_m", .06),
            "gate_max_gap_s": positive("gate_max_gap_s", 8.),
            "gate_xy_m": xy,
            "gate_direction_xy": direction,
            "gate_half_width_m": positive("gate_half_width_m", .4),
            "gate_hysteresis_m": positive("gate_hysteresis_m", .12),
            "gate_min_lap_s": positive("gate_min_lap_s", 30.),
            "gate_min_lap_distance_m": positive("gate_min_lap_distance_m", 4.5),
            "gate_departure_m": positive("gate_departure_m", .8),
            "gate_max_step_m": positive("gate_max_step_m", .8),
        })
        if settings["gate_hysteresis_m"] >= settings["gate_departure_m"]:
            raise ValueError("gate_departure_m deve exceder gate_hysteresis_m")
        if gate_mode == "corner_return" and settings["gate_return_radius_m"] >= settings["gate_departure_m"]:
            raise ValueError("gate_return_radius_m deve ser menor que gate_departure_m")
    return settings


def read_uwb2026_processing_settings(anchor_json: str | Path) -> dict:
    """Configuração reproduzível de filtros sem modificar os dados brutos.

    O padrão preserva as regras PR4/PR5. Para quatro âncoras e
    trilateração 3D, pode-se exigir quatro medições novas.
    """
    with Path(anchor_json).open("r", encoding="utf-8") as file:
        root = json.load(file)
    config = root.get("uwb_2026", {})
    if not isinstance(config, dict):
        raise ValueError("uwb_2026 deve ser um objeto")
    allowed = {"min_valid_anchors", "min_sigma_m"}
    if set(config) - allowed:
        raise ValueError(
            "Campos inválidos em uwb_2026: " +
            ", ".join(sorted(set(config) - allowed))
        )
    n = config.get("min_valid_anchors", 3)
    sigma = config.get("min_sigma_m", .02)
    if type(n) is not int or not 3 <= n <= 32:
        raise ValueError("uwb_2026.min_valid_anchors deve ser inteiro entre 3 e 32")
    if isinstance(sigma, bool) or not np.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("uwb_2026.min_sigma_m deve ser positivo")
    return {"min_valid_anchors": n, "min_sigma_m": float(sigma)}


def validate_robot2026_anchor_geometry(positions: np.ndarray) -> None:
    """Confere âncoras no referencial XY antes de iniciar os solvers."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError("anchors_xy deve ser uma matriz Nx2 ou Nx3")
    if len(positions) < 3 or not np.isfinite(positions).all():
        raise ValueError("São necessárias 3+ âncoras com coordenadas finitas")
    xy = positions[:, :2]
    if np.linalg.matrix_rank(xy - xy.mean(axis=0), tol=1e-8) < 2:
        raise ValueError("Âncoras colineares: distribuição XY inválida para 2D")


def inspect_robot2026_capture(
    pcnt_path: str | Path,
    *,
    session_id: int = 1,
    anchor_path: str | Path | None = None,
) -> dict:
    """Diagnóstico não destrutivo para a bateria de testes.

    Resultados 'blocked' não descartam amostras nem criam dados faltantes.
    """
    path = Path(pcnt_path)
    sessions = load_pcnt_sessions(path)
    if not 1 <= int(session_id) <= len(sessions):
        raise ValueError(f"Sessão {session_id} inexistente (1..{len(sessions)})")
    rows = sessions[int(session_id) - 1]
    dt = np.diff([r.timestamp_ms for r in rows])
    out = {
        "file": str(path),
        "session": int(session_id),
        "sessions_detected": len(sessions),
        "pcnt_samples": len(rows),
        "duration_s": round((rows[-1].timestamp_ms - rows[0].timestamp_ms) / 1000, 3),
        "median_log_period_ms": float(np.median(dt)) if dt.size else None,
        "laps_reported": int(rows[-1].lap_count - rows[0].lap_count),
        "columns": 5 + 3 * len(rows[0].uwb),
        "uwb_slot_count": len(rows[0].uwb),
        "status": "odometry_only" if not rows[0].uwb else "blocked",
        "warnings": [],
    }
    if not rows[0].uwb:
        out["warnings"].append(
            "Registro PCNT de 5 colunas: não contém dados UWB; odometria disponível"
        )
        return out
    if anchor_path is None:
        out["warnings"].append(
            "É necessário selecionar o JSON com posições medidas e mapeamento SPI"
        )
        return out

    try:
        with Path(anchor_path).open("r", encoding="utf-8") as stream:
            setup = json.load(stream)
        pos = np.asarray(setup.get("anchors_xy", setup.get("anchors", [])), dtype=float)
        validate_robot2026_anchor_geometry(pos)
        mapping = read_pcnt_anchor_slot_mapping(anchor_path, len(pos))
        profile = read_robot2026_profile_from_anchor_json(anchor_path)
        settings = read_uwb2026_processing_settings(anchor_path)
        data = load_robot2026_single_tag_uwb(
            path, session_id=int(session_id),
            anchor_count=len(pos), slot_to_anchor_index=mapping,
            profile=profile, **settings,
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        out["warnings"].append(str(exc))
        return out

    # Timestamps no log ESP32 de mudanças nas triplas, não timestamps TWR.
    # Permite diagnosticar a cadência OBSERVADA sem afirmar instante físico.
    acquisition = []
    for slot in range(len(mapping)):
        valid_new = [
            int(item["timestamp"])
            for item in convert_uwb_session(rows)
            if int(item["anchor_id"]) == slot and not item["possibly_reused"]
        ]
        deltas = np.diff(valid_new)
        acquisition.append({
            "slot": slot,
            "anchor_index": mapping[slot],
            "valid_new_readings": len(valid_new),
            "median_log_update_ms": float(np.median(deltas)) if deltas.size else None,
            "p95_log_update_ms": float(np.percentile(deltas, 95)) if deltas.size else None,
        })
    out.update({
        "status": "ready_for_static_batch",
        "anchor_count": int(len(pos)),
        "spi_slot_to_anchor_index": list(mapping),
        "effective_cpr": profile.ticks_per_wheel_rev,
        "raw_encoder_order_swapped": profile.swap_encoder_columns,
        "new_ranges_required": settings["min_valid_anchors"],
        "start_pose_xytheta": [
            profile.initial_x_m, profile.initial_y_m, profile.initial_theta_rad
        ],
        "uwb_slot_record_update_stats": acquisition,
        "uwb_timing_note": "Diferenças entre timestamps de gravação no ESP32; NÃO são tempos de TWR físicos",
        "uwb_snapshots": data.sample_count,
        "uwb_valid_readings": data.valid_readings,
        "uwb_repeated_readings_excluded": data.repeated_readings,
        "uwb_invalid_readings": data.invalid_readings,
        "uwb_missing_fraction": round(
            1.0 - data.sample_count / max(1, data.odometry.sample_count), 4
        ),
    })
    if data.sample_count < 10:
        out["warnings"].append(
            "Menos de 10 snapshots UWB completos: apenas verificação inicial"
        )
    if data.repeated_readings:
        out["warnings"].append(
            "Triplas UWB idênticas foram excluídas; timestamps indicam gravação, não ranging"
        )
    return out
