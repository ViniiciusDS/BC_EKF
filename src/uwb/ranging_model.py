# src/uwb/ranging_model.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RangingConfig:
    dt: float = 0.10                 # atualização por tick
    noise_enabled: bool = True
    sigma_los: float = 0.03          # m (LOS ~ 3 cm)
    sigma_nlos: float = 0.15         # m (NLOS pior)
    nlos_prob: float = 0.10          # probabilidade de NLOS por medição
    nlos_bias_mean: float = 0.30     # m (bias positivo médio em NLOS)
    dropout_prob: float = 0.00       # prob. de falhar medição
    quantize_step: float | None = None  # ex: 0.01 para 1 cm; None desliga

@dataclass
class RangingResult:
    r_true: float
    r_meas: float | None
    is_nlos: bool
    noise: float
    bias: float

class UwbRangingModel:
    def __init__(self, cfg: RangingConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Bias por âncora (indexado por i) pode ser colocado depois;
        # aqui deixamos um bias global simples (você pode evoluir p/ vetor).
        self.global_bias = 0.0

    def measure_range(self, a_xy: np.ndarray, tag_xy: np.ndarray) -> RangingResult:
        r_true = float(np.linalg.norm(a_xy - tag_xy))

        # dropout
        if self.rng.random() < self.cfg.dropout_prob:
            return RangingResult(r_true=r_true, r_meas=None, is_nlos=False, noise=0.0, bias=0.0)

        is_nlos = (self.rng.random() < self.cfg.nlos_prob)

        # bias NLOS (positivo) - exponencial com média nlos_bias_mean
        bias = 0.0
        if is_nlos and self.cfg.nlos_bias_mean > 0:
            bias = float(self.rng.exponential(self.cfg.nlos_bias_mean))

        # ruído
        noise = 0.0
        if self.cfg.noise_enabled:
            sigma = self.cfg.sigma_nlos if is_nlos else self.cfg.sigma_los
            noise = float(self.rng.normal(0.0, sigma))

        r_meas = r_true + self.global_bias + bias + noise

        # quantização
        if self.cfg.quantize_step is not None and self.cfg.quantize_step > 0:
            q = self.cfg.quantize_step
            r_meas = float(np.round(r_meas / q) * q)

        return RangingResult(r_true=r_true, r_meas=r_meas, is_nlos=is_nlos, noise=noise, bias=bias)
