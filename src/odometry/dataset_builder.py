from __future__ import annotations

from typing import Iterable, Sequence

from .models import EncoderSample, OdometrySample, DifferentialDriveConfig
from .differential_drive import integrate_trajectory
from .sync import interpolate_odometry_pose


def normalize_uwb_row(row: dict) -> dict:
    """
    Normaliza uma linha UWB para um formato mínimo esperado.
    """
    time_keys = ("timestamp_s", "timestamp", "time", "t")
    anchor_keys = ("anchor_id", "anchor", "id_anchor")
    range_keys = ("range", "distance", "dist")
    sigma_keys = ("sigma", "std", "deviation")

    def first_value(keys):
        for k in keys:
            if k in row:
                return row[k]
        return None

    t = first_value(time_keys)
    anchor_id = first_value(anchor_keys)
    r = first_value(range_keys)
    sigma = first_value(sigma_keys)

    if t is None:
        raise ValueError(f"Linha UWB sem timestamp: {row}")
    if anchor_id is None:
        raise ValueError(f"Linha UWB sem anchor_id: {row}")
    if r is None:
        raise ValueError(f"Linha UWB sem range: {row}")

    out = {
        "timestamp_s": float(t),
        "anchor_id": int(float(anchor_id)),
        "range": float(r),
    }

    if sigma is not None:
        out["sigma"] = float(sigma)

    # preserva chaves extras
    for k, v in row.items():
        if k not in out and k not in ("timestamp", "time", "t"):
            out[k] = v

    return out


def normalize_uwb_rows(rows: Iterable[dict]) -> list[dict]:
    out = [normalize_uwb_row(r) for r in rows]
    out.sort(key=lambda r: (r["timestamp_s"], r["anchor_id"]))
    return out


def build_aligned_dataset_rows(
    odom_samples: Sequence[OdometrySample],
    uwb_rows: Iterable[dict],
    *,
    clamp: bool = True,
) -> list[dict]:
    """
    Para cada medição UWB, injeta a pose odométrica interpolada no mesmo timestamp.
    """
    uwb_rows = normalize_uwb_rows(uwb_rows)
    out: list[dict] = []

    for row in uwb_rows:
        pose = interpolate_odometry_pose(
            odom_samples,
            row["timestamp_s"],
            clamp=clamp,
        )

        merged = dict(row)
        merged["odom_x"] = float(pose.x)
        merged["odom_y"] = float(pose.y)
        merged["odom_theta"] = float(pose.theta)
        merged["odom_source_index_left"] = int(pose.source_index_left)
        merged["odom_source_index_right"] = int(pose.source_index_right)
        merged["odom_alpha"] = float(pose.alpha)
        out.append(merged)

    return out


def build_dataset_from_encoder_and_uwb(
    encoder_samples: Sequence[EncoderSample],
    uwb_rows: Iterable[dict],
    drive_cfg: DifferentialDriveConfig,
    *,
    clamp: bool = True,
):
    """
    Pipeline completo:
    encoder -> odometria -> alinhamento com UWB
    """
    odom_samples = integrate_trajectory(encoder_samples, drive_cfg)
    aligned_rows = build_aligned_dataset_rows(
        odom_samples,
        uwb_rows,
        clamp=clamp,
    )

    return {
        "encoder_samples": list(encoder_samples),
        "odometry_samples": odom_samples,
        "uwb_rows": normalize_uwb_rows(uwb_rows),
        "aligned_rows": aligned_rows,
    }


def grouped_rows_by_timestamp(aligned_rows: Sequence[dict]) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = {}
    for row in aligned_rows:
        t = float(row["timestamp_s"])
        grouped.setdefault(t, []).append(row)
    return grouped