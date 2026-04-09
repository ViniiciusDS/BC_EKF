from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import OdometrySample, Pose2D


@dataclass(slots=True)
class TimeAlignedPose:
    timestamp_s: float
    x: float
    y: float
    theta: float
    source_index_left: int
    source_index_right: int
    alpha: float


def normalize_angle(theta: float) -> float:
    while theta >= math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def shortest_angular_difference(theta_a: float, theta_b: float) -> float:
    """
    Retorna a menor diferença angular para ir de theta_a até theta_b.
    """
    return normalize_angle(theta_b - theta_a)


def interpolate_angle(theta_a: float, theta_b: float, alpha: float) -> float:
    """
    Interpola dois ângulos respeitando wrap em [-pi, pi).
    """
    dtheta = shortest_angular_difference(theta_a, theta_b)
    return normalize_angle(theta_a + alpha * dtheta)


def _odometry_timestamps(odom_samples: Sequence[OdometrySample]) -> list[float]:
    return [float(s.timestamp_s) for s in odom_samples]


def interpolate_odometry_pose(
    odom_samples: Sequence[OdometrySample],
    timestamp_s: float,
    *,
    clamp: bool = True,
) -> TimeAlignedPose:
    """
    Interpola a pose odométrica no timestamp desejado.

    Se clamp=True:
      - antes do início -> usa primeira pose
      - depois do fim   -> usa última pose
    Caso contrário, lança ValueError fora da faixa.
    """
    if not odom_samples:
        raise ValueError("Lista de odometria vazia")

    times = _odometry_timestamps(odom_samples)
    t = float(timestamp_s)

    if t <= times[0]:
        if not clamp and t < times[0]:
            raise ValueError("timestamp antes do início da odometria")
        s = odom_samples[0]
        return TimeAlignedPose(
            timestamp_s=t,
            x=s.x,
            y=s.y,
            theta=s.theta,
            source_index_left=0,
            source_index_right=0,
            alpha=0.0,
        )

    if t >= times[-1]:
        if not clamp and t > times[-1]:
            raise ValueError("timestamp depois do fim da odometria")
        s = odom_samples[-1]
        n = len(odom_samples) - 1
        return TimeAlignedPose(
            timestamp_s=t,
            x=s.x,
            y=s.y,
            theta=s.theta,
            source_index_left=n,
            source_index_right=n,
            alpha=0.0,
        )

    idx_right = bisect.bisect_left(times, t)
    idx_left = idx_right - 1

    s0 = odom_samples[idx_left]
    s1 = odom_samples[idx_right]

    t0 = float(s0.timestamp_s)
    t1 = float(s1.timestamp_s)

    if t1 <= t0:
        # fallback defensivo
        return TimeAlignedPose(
            timestamp_s=t,
            x=s0.x,
            y=s0.y,
            theta=s0.theta,
            source_index_left=idx_left,
            source_index_right=idx_left,
            alpha=0.0,
        )

    alpha = (t - t0) / (t1 - t0)

    x = (1.0 - alpha) * s0.x + alpha * s1.x
    y = (1.0 - alpha) * s0.y + alpha * s1.y
    theta = interpolate_angle(s0.theta, s1.theta, alpha)

    return TimeAlignedPose(
        timestamp_s=t,
        x=x,
        y=y,
        theta=theta,
        source_index_left=idx_left,
        source_index_right=idx_right,
        alpha=alpha,
    )


def sample_odometry_at_timestamps(
    odom_samples: Sequence[OdometrySample],
    timestamps_s: Iterable[float],
    *,
    clamp: bool = True,
) -> list[TimeAlignedPose]:
    """
    Interpola a odometria para uma sequência de timestamps.
    """
    return [
        interpolate_odometry_pose(odom_samples, t, clamp=clamp)
        for t in timestamps_s
    ]


def extract_unique_timestamps(
    rows: Iterable[dict],
    *,
    time_keys: tuple[str, ...] = ("timestamp", "timestamp_s", "time", "t"),
    sort_output: bool = True,
) -> list[float]:
    """
    Extrai timestamps únicos de uma lista de dicts.
    Útil para medições UWB já carregadas em memória.
    """
    out = []
    seen = set()

    for row in rows:
        t_value = None
        for key in time_keys:
            if key in row:
                t_value = row[key]
                break

        if t_value is None:
            continue

        t = float(t_value)
        if t not in seen:
            seen.add(t)
            out.append(t)

    if sort_output:
        out.sort()

    return out


def build_time_aligned_odometry(
    odom_samples: Sequence[OdometrySample],
    uwb_rows: Iterable[dict],
    *,
    time_keys: tuple[str, ...] = ("timestamp", "timestamp_s", "time", "t"),
    clamp: bool = True,
) -> list[TimeAlignedPose]:
    """
    Constrói a odometria alinhada com os timestamps presentes nas linhas do UWB.
    """
    timestamps = extract_unique_timestamps(uwb_rows, time_keys=time_keys, sort_output=True)
    return sample_odometry_at_timestamps(odom_samples, timestamps, clamp=clamp)


def poses_to_dict_rows(poses: Sequence[TimeAlignedPose]) -> list[dict]:
    """
    Converte poses alinhadas para uma estrutura simples serializável.
    """
    rows = []
    for p in poses:
        rows.append(
            {
                "timestamp_s": float(p.timestamp_s),
                "x": float(p.x),
                "y": float(p.y),
                "theta": float(p.theta),
                "source_index_left": int(p.source_index_left),
                "source_index_right": int(p.source_index_right),
                "alpha": float(p.alpha),
            }
        )
    return rows