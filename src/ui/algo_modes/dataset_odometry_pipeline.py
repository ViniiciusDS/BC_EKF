"""Importa odometria do ESP32 PCNT no Dataset Mode, sem Pygame ou UWB.

Aceita o pcntLog.csv original (5/20 colunas) e CSVs por sessão do conversor.
Reconstrói sempre pelas contagens, sem usar poses previamente exportadas.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.odometry.pcntlog_adapter import (
    RawPcntRow,
    Robot2026Profile,
    convert_session,
    load_pcnt_sessions,
)

# Perfil experimental do hardware; não altera o config dos ensaios antigos.
ROBOT2026_PROFILE = Robot2026Profile(
    wheel_radius_m=0.03425,
    wheel_base_m=0.229,
    ticks_per_wheel_rev=4560.0,
    invert_left=False,
    invert_right=True,
)

RAW_PREFIX = "Bruto: "
PROCESSED_PREFIX = "Processado: "


@dataclass(frozen=True)
class Odometry2026Data:
    path: Path
    session_id: int
    total_sessions: int
    timestamps_s: np.ndarray
    poses_xytheta: np.ndarray
    v_m_s: np.ndarray
    omega_rad_s: np.ndarray
    laps: np.ndarray
    profile: Robot2026Profile

    @property
    def sample_count(self) -> int:
        return len(self.timestamps_s)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0]) if self.sample_count else 0.0


def list_robot2026_sources(raw_dir: str | Path, processed_dir: str | Path) -> list[str]:
    """Retorna opções de arquivos efetivamente importáveis, com identificação da origem."""
    result: list[str] = []
    raw, processed = Path(raw_dir), Path(processed_dir)
    if raw.is_dir():
        result.extend(
            RAW_PREFIX + path.name
            for path in sorted(raw.iterdir(), key=lambda p: p.name.lower())
            if path.is_file() and path.suffix.lower() in {".csv", ".txt"}
            and "pcnt" in path.name.lower()
        )
    if processed.is_dir():
        result.extend(
            PROCESSED_PREFIX + path.name
            for path in sorted(processed.iterdir(), key=lambda p: p.name.lower())
            if path.is_file()
            and path.name.lower().endswith(("_encoder.csv", "_odometry.csv"))
        )
    return result


def resolve_robot2026_source(
    selection: str, raw_dir: str | Path, processed_dir: str | Path
) -> Path:
    """Resolve uma opção da lista sem permitir paths externos às pastas previstas."""
    selection = str(selection).strip()
    if selection.startswith(RAW_PREFIX):
        folder, filename = Path(raw_dir), selection[len(RAW_PREFIX):]
    elif selection.startswith(PROCESSED_PREFIX):
        folder, filename = Path(processed_dir), selection[len(PROCESSED_PREFIX):]
    else:
        raise ValueError("Selecione uma opção 'Bruto:' ou 'Processado:'")
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Nome de arquivo inválido")
    result = folder / filename
    if not result.is_file():
        raise FileNotFoundError(f"Arquivo de odometria não encontrado: {result}")
    return result


def _read_converted_encoder(path: Path, invert_right: bool) -> list[RawPcntRow]:
    """Reconstrói entrada bruta a partir de ticks já corrigidos pelo conversor.

    CSVs exportados já têm sinal direito corrigido: desfazemos a inversão
    ao construir RawPcntRow para convert_session não corrigir uma segunda vez.
    """
    rows: list[RawPcntRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        if not {"left_ticks", "right_ticks"} <= columns:
            raise ValueError("CSV convertido exige left_ticks e right_ticks")
        if not ({"timestamp_ms", "timestamp_s", "timestamp"} & columns):
            raise ValueError("CSV convertido exige timestamp_ms, timestamp_s ou timestamp")
        for lineno, item in enumerate(reader, 2):
            try:
                if "timestamp_ms" in columns:
                    t = round(float(item["timestamp_ms"]))
                elif "timestamp_s" in columns:
                    t = round(float(item["timestamp_s"]) * 1000.0)
                else:
                    t = round(float(item["timestamp"]))
                left = int(float(item["left_ticks"]))
                right = int(float(item["right_ticks"]))
                laps = int(float(item.get("lap_count") or 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Linha {lineno}: dados de encoder inválidos") from exc
            rows.append(RawPcntRow(t, left, -right if invert_right else right, laps, 0))
    if not rows:
        raise ValueError("CSV convertido vazio")
    if any(b.timestamp_ms <= a.timestamp_ms for a, b in zip(rows, rows[1:])):
        raise ValueError("Timestamp não crescente: selecione uma única sessão")
    return rows


def load_robot2026_odometry(
    path: str | Path, session_id: int = 1, profile: Robot2026Profile | None = None
) -> Odometry2026Data:
    """Reintegra todas as amostras do PCNT com o perfil 2026 de 4560 CPR."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = profile or ROBOT2026_PROFILE
    cfg.validate()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first = next(csv.reader(f), [])
    raw = False
    if len(first) in (5, 20):
        try:
            raw = all(np.isfinite(float(value.strip())) for value in first[:5])
        except (ValueError, TypeError):
            pass

    if raw:
        sessions = load_pcnt_sessions(path)
        if not (1 <= session_id <= len(sessions)):
            raise ValueError(
                f"Sessão {session_id} inexistente; disponíveis 1..{len(sessions)}"
            )
        rows = sessions[session_id - 1]
        used_id, total = session_id, len(sessions)
    else:
        rows = _read_converted_encoder(path, invert_right=cfg.invert_right)
        found = re.search(r"session_(\d+)", path.stem, flags=re.I)
        used_id, total = (int(found.group(1)) if found else 1), 1

    samples = convert_session(rows, cfg)
    stamps = np.array([r["timestamp_s"] for r in samples], dtype=float)
    poses = np.array([[r["x_m"], r["y_m"], r["theta_rad"]] for r in samples], dtype=float)
    v = np.array([r["v_m_s"] for r in samples], dtype=float)
    w = np.array([r["omega_rad_s"] for r in samples], dtype=float)
    laps = np.array([r["lap_count"] for r in samples], dtype=int)
    return Odometry2026Data(path, used_id, total, stamps, poses, v, w, laps, cfg)
