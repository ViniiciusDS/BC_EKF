# src/uwb/ranging_model.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

C0 = 299_792_458.0  # velocidade da luz (m/s)

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
    '''  Canal/observação (LOS/NLOS + bias + noise + dropout + quantização).
        Padronizado para ser usado como ChannelToFModel pelo protocolo TWR.
        sample_tof(a_xy, tag_xy, c) -> (tof_true_s, tof_meas_s or None, is_nlos, dropped, components)
    '''
    def __init__(self, cfg: RangingConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.global_bias = 0.0  # mantém em metros, evoluir depois para colocar por âncora/tag


    # ===== Amostra erro em metros =====
    def _sample_range_error_m(self) -> tuple[bool, bool, float, float, float]:
        ''' retorna:
            is_nlos, dropped, bias_m, noise_m, r_meas_err_m
        '''
        # dropout
        if self.rng.random() < self.cfg.dropout_prob:
            return False, True, 0.0, 0.0, 0.0
        
        is_nlos = (self.rng.random() < self.cfg.nlos_prob)

        # bias NLOS (positivo) - exponencial com média nlos_bias_mean
        bias_m = 0.0
        if is_nlos and self.cfg.nlos_bias_mean > 0:
            bias_m = float(self.rng.exponential(self.cfg.nlos_bias_mean))
        
        # ruído
        noise_m = 0.0
        if self.cfg.noise_enabled:
            sigma = self.cfg.sigma_nlos if is_nlos else self.cfg.sigma_los
            noise_m = float(self.rng.normal(0.0, sigma))
        
        total_err_m = self.global_bias + bias_m + noise_m
        return is_nlos, False, bias_m, noise_m, total_err_m
    
    # ===== Api usada para protocolos TOF =====
    def sample_tof(
                self,
                a_xy: np.ndarray,
                tag_xy: np.ndarray,
                c: float = C0,
            ) -> tuple[float, float | None, bool, bool, dict]:
        '''
        Interface padronizada para protocolos.
        Retorna:
          (tof_true_s, tof_meas_s or None, is_nlos, dropped, components)
        '''
        r_true_m = float(np.linalg.norm(a_xy - tag_xy))
        tof_true_s = r_true_m / c

        is_nlos, dropped, bias_m, noise_m, err_m = self._sample_range_error_m()
        if dropped:
            return tof_true_s, None, is_nlos, dropped, {"bias_m": bias_m, "noise_m": noise_m}
        
        # erro em tempo (s)
        err_s = err_m / c
        tof_meas_s = tof_true_s + err_s

        # quantização (se ativa) - dominío de distância
        # mas aplica em ToF via metros/c
        if self.cfg.quantize_step is not None and self.cfg.quantize_step > 0:
            q = self.cfg.quantize_step
            r_meas_m = r_true_m + err_m
            r_meas_m = float(np.round(r_meas_m / q) * q)
            tof_meas_s = r_meas_m / c
        
        comps = {
            "bias_m": bias_m,
            "noise_m": noise_m,
            "global_bias_m": self.global_bias,
            "err_m": err_m,
            "err_s": err_s,
            "quantize_step_m": self.cfg.quantize_step,
        }
        return tof_true_s, float(tof_meas_s), is_nlos, False, comps
    
    # ===== API antiga (será removida futuramente) =====
    def measure_range(self, a_xy: np.ndarray, tag_xy: np.ndarray) -> RangingResult:
        '''' API antiga, será removida futuramente. 
            Retorna RangingResult com r_true, r_meas, is_nlos, noise, bias.
        '''
        tof_true, tof_meas, is_nlos, dropped, comps = self.sample_tof(a_xy, tag_xy, c=C0)

        r_true = float(comps.get("r_true_m", np.linalg.norm(a_xy - tag_xy)))
        if dropped or tof_meas is None:
            return RangingResult(r_true=r_true, r_meas=None, is_nlos=False, noise=0.0, bias=0.0)

        r_meas = float(C0 * tof_meas)

        # para manter campos antigos:
        bias = float(comps.get("bias_m", 0.0))
        noise = float(comps.get("noise_m", 0.0))
        return RangingResult(r_true=r_true, r_meas=r_meas, is_nlos=is_nlos, noise=noise, bias=bias)
