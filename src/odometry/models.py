from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EncoderConfig:
    """
    Configuração do encoder/motor.

    ticks_per_wheel_rev:
        número de ticks por volta completa da roda.
        Se você só conhece ticks do motor e a redução,
        já passe o valor convertido para a roda.
    """
    ticks_per_wheel_rev: float


@dataclass(slots=True)
class DifferentialDriveConfig:
    """
    Configuração geométrica do robô diferencial.
    """
    wheel_radius_m: float
    wheel_base_m: float
    encoder: EncoderConfig
    invert_left: bool = False
    invert_right: bool = False


@dataclass(slots=True)
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass(slots=True)
class EncoderSample:
    """
    Amostra bruta/acumulada dos encoders.
    timestamp_s:
        tempo em segundos
    left_ticks / right_ticks:
        contadores acumulados ou absolutos lidos no instante
    """
    timestamp_s: float
    left_ticks: int
    right_ticks: int


@dataclass(slots=True)
class EncoderDelta:
    """
    Incremento entre duas amostras.
    """
    timestamp_s: float
    dt_s: float
    delta_left_ticks: int
    delta_right_ticks: int


@dataclass(slots=True)
class OdometrySample:
    """
    Pose integrada do robô em um instante.
    """
    timestamp_s: float
    x: float
    y: float
    theta: float
    ds: float
    dtheta: float