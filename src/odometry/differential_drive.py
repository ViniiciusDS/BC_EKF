from __future__ import annotations

import math
from typing import Iterable

from .models import (
    DifferentialDriveConfig,
    EncoderDelta,
    EncoderSample,
    OdometrySample,
    Pose2D,
)


def normalize_angle(theta: float) -> float:
    """
    Normaliza ângulo para o intervalo [-pi, pi).
    """
    while theta >= math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def ticks_to_distance_m(delta_ticks: int, cfg: DifferentialDriveConfig) -> float:
    """
    Converte delta de ticks para deslocamento linear da roda.
    """
    ticks_per_rev = cfg.encoder.ticks_per_wheel_rev
    if ticks_per_rev <= 0:
        raise ValueError("ticks_per_wheel_rev deve ser > 0")

    meters_per_tick = (2.0 * math.pi * cfg.wheel_radius_m) / ticks_per_rev
    return float(delta_ticks) * meters_per_tick


def apply_tick_inversion(
    delta_left_ticks: int,
    delta_right_ticks: int,
    cfg: DifferentialDriveConfig,
) -> tuple[int, int]:
    """
    Inverte sinais conforme montagem física do robô.
    """
    dl = -delta_left_ticks if cfg.invert_left else delta_left_ticks
    dr = -delta_right_ticks if cfg.invert_right else delta_right_ticks
    return dl, dr


def integrate_step(
    pose: Pose2D,
    delta_left_ticks: int,
    delta_right_ticks: int,
    cfg: DifferentialDriveConfig,
) -> tuple[Pose2D, float, float]:
    """
    Integra um passo de odometria diferencial.
    Retorna:
    - nova pose
    - ds
    - dtheta
    """
    dl_ticks, dr_ticks = apply_tick_inversion(delta_left_ticks, delta_right_ticks, cfg)

    d_left = ticks_to_distance_m(dl_ticks, cfg)
    d_right = ticks_to_distance_m(dr_ticks, cfg)

    ds = 0.5 * (d_left + d_right)
    dtheta = (d_right - d_left) / cfg.wheel_base_m

    theta_mid = pose.theta + 0.5 * dtheta

    x_new = pose.x + ds * math.cos(theta_mid)
    y_new = pose.y + ds * math.sin(theta_mid)
    theta_new = normalize_angle(pose.theta + dtheta)

    return Pose2D(x=x_new, y=y_new, theta=theta_new), ds, dtheta


def samples_to_deltas(samples: Iterable[EncoderSample]) -> list[EncoderDelta]:
    """
    Converte amostras acumuladas em deltas consecutivos.
    """
    samples = list(samples)
    if len(samples) < 2:
        return []

    deltas: list[EncoderDelta] = []

    prev = samples[0]
    for cur in samples[1:]:
        dt_s = float(cur.timestamp_s - prev.timestamp_s)
        if dt_s < 0:
            raise ValueError("timestamps do encoder devem estar em ordem crescente")

        deltas.append(
            EncoderDelta(
                timestamp_s=float(cur.timestamp_s),
                dt_s=dt_s,
                delta_left_ticks=int(cur.left_ticks - prev.left_ticks),
                delta_right_ticks=int(cur.right_ticks - prev.right_ticks),
            )
        )
        prev = cur

    return deltas


def integrate_trajectory(
    samples: Iterable[EncoderSample],
    cfg: DifferentialDriveConfig,
    initial_pose: Pose2D | None = None,
) -> list[OdometrySample]:
    """
    Integra uma trajetória completa a partir de amostras acumuladas do encoder.
    """
    samples = list(samples)
    if not samples:
        return []

    pose = initial_pose or Pose2D(x=0.0, y=0.0, theta=0.0)
    out: list[OdometrySample] = [
        OdometrySample(
            timestamp_s=float(samples[0].timestamp_s),
            x=pose.x,
            y=pose.y,
            theta=pose.theta,
            ds=0.0,
            dtheta=0.0,
        )
    ]

    for delta in samples_to_deltas(samples):
        pose, ds, dtheta = integrate_step(
            pose,
            delta.delta_left_ticks,
            delta.delta_right_ticks,
            cfg,
        )
        out.append(
            OdometrySample(
                timestamp_s=delta.timestamp_s,
                x=pose.x,
                y=pose.y,
                theta=pose.theta,
                ds=ds,
                dtheta=dtheta,
            )
        )

    return out