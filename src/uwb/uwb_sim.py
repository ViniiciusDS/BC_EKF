# src/uwb/uwb_sim.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.uwb.ranging_model  import UwbRangingModel, RangingConfig
from src.uwb.twr_protocols   import (
    DS_TWR_Protocol, SS_TWR_Protocol, TWRConfig, TWRMode,
    ClockModel as TWRClock, AntennaDelayModel, TWRResult
)
from src.uwb.node_params import NodeParams, ns_to_s



###############################################################
# Configuração padrão (usada quando nada é passado explícito)
###############################################################
@dataclass
class UwbSimConfig:
    """
    Configuração padrão do pipeline TWR para o simulador.
    Pode ser criada ou vinda do HUD/ExperimentConfig.

    Parâmetros de canal (RangingConfig):
        sigma_los       : ruído em LOS (m)
        sigma_nlos      : ruído em NLOS (m)
        nlos_prob       : probabilidade de NLOS
        nlos_bias_mean  : bias médio exponencial em NLOS (m)
        dropout_prob    : probabilidade de perda total
        quantize_step   : passo de quantização (m), None = desligado

    Parâmetros de protocolo (TWRConfig):
        mode            : TWRMode.DS_TWR ou SS_TWR
        reply_delay_s   : delay de resposta em ambos os nós (s)

    Parâmetros padrão de nó (aplica a todos se não houver override):
        default_ppm     : drift do clock (ppm)
        default_tx_ns   : atraso TX da antena (ns)
        default_rx_ns   : atraso RX da antena (ns)
        default_bias_m  : bias aditivo de ranging (m)
    """
    # --- canal ---
    sigma_los:      float = 0.03
    sigma_nlos:     float = 0.15
    nlos_prob:      float = 0.10
    nlos_bias_mean: float = 0.30
    dropout_prob:   float = 0.00
    quantize_step:  Optional[float] = None

    # --- protocolo ---
    mode:           TWRMode = TWRMode.DS_TWR
    reply_delay_s:  float   = 0.0005   # 0.5 ms

    # --- parâmetros padrão de nó ---
    default_ppm:    float = 0.0
    default_tx_ns:  float = 0.0
    default_rx_ns:  float = 0.0
    default_bias_m: float = 0.0

    def to_ranging_cfg(self) -> RangingConfig:
        return RangingConfig(
            noise_enabled   = True,
            sigma_los       = self.sigma_los,
            sigma_nlos      = self.sigma_nlos,
            nlos_prob       = self.nlos_prob,
            nlos_bias_mean  = self.nlos_bias_mean,
            dropout_prob    = self.dropout_prob,
            quantize_step   = self.quantize_step,
        )

    def to_twr_cfg(self) -> TWRConfig:
        return TWRConfig(
            mode                  = self.mode,
            reply_delay_anchor_s  = self.reply_delay_s,
            reply_delay_tag_s     = self.reply_delay_s,
        )

    def default_node_params(self) -> NodeParams:
        p = NodeParams()
        p.clock.drift_ppm = self.default_ppm
        p.ant.tx_ns       = self.default_tx_ns
        p.ant.rx_ns       = self.default_rx_ns
        p.range_bias_m    = self.default_bias_m
        return p


#######################################################################
# Conversor: NodeParams (node_params.py) → tipos do twr_protocols.py
#######################################################################
def _node_to_twr_clock(p: NodeParams) -> TWRClock:
    """Converte NodeParams.clock → ClockModel do twr_protocols."""
    return TWRClock(offset_s=0.0, ppm=p.clock.drift_ppm)

def _node_to_twr_delay(p: NodeParams) -> AntennaDelayModel:
    """Converte NodeParams.ant → AntennaDelayModel do twr_protocols."""
    return AntennaDelayModel(
        tx_s = ns_to_s(p.ant.tx_ns),
        rx_s = ns_to_s(p.ant.rx_ns),
    )


#############################
# Pipeline principal
##############################
class UwbSimPipeline:
    """
    Encapsula o pipeline completo:
      UwbRangingModel (canal) + DS/SS_TWR_Protocol + NodeParams por âncora.

    Interface pública:
        measure(x_state, anchors, l, z_c) → z_k  (array 2*N)
        measure(x_state, anchors, l, z_c, return_meta=True) → (z_k, meta_list)
    """

    def __init__(
        self,
        ranging_cfg:   RangingConfig,
        twr_cfg:       TWRConfig,
        tag_params:    NodeParams,
        anchor_params: Dict[int, NodeParams],
        default_params: NodeParams,
        seed:          Optional[int] = None,
    ) -> None:
        self.seed = None

        # Inicializa o modelo de canal e o protocolo TWR
        self.ranging_model = UwbRangingModel(ranging_cfg, seed=seed)

        if twr_cfg.mode == TWRMode.DS_TWR:
            self.protocol = DS_TWR_Protocol(twr_cfg, seed=seed)
        else:
            self.protocol = SS_TWR_Protocol(twr_cfg, seed=seed)

        # Parâmetros de nó
        self.tag_params     = tag_params
        self.anchor_params  = anchor_params   # Dict[anchor_idx → NodeParams]
        self.default_params = default_params  # fallback para âncoras sem override

        # Seta Seed 
        self.seed = seed

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @seed.setter
    def seed(self, value: Optional[int]) -> None:
        self._seed = None if value is None else int(value)

        # reseed de verdade (se os objetos existirem)
        if hasattr(self, "ranging_model") and self.ranging_model is not None:
            import numpy as np
            self.ranging_model.rng = np.random.default_rng(self._seed)

        if hasattr(self, "protocol") and self.protocol is not None:
            import numpy as np
            self.protocol.rng = np.random.default_rng(self._seed)


    # Construtor conveniente 
    @classmethod
    def from_defaults(cls, seed: Optional[int] = None) -> "UwbSimPipeline":
        """Cria pipeline com configuração padrão (sem clock drift, sem delays)."""
        cfg = UwbSimConfig()
        return cls(
            ranging_cfg    = cfg.to_ranging_cfg(),
            twr_cfg        = cfg.to_twr_cfg(),
            tag_params     = cfg.default_node_params(),
            anchor_params  = {},
            default_params = cfg.default_node_params(),
            seed           = seed,
        )

    @classmethod
    def from_config(
        cls,
        sim_cfg:       UwbSimConfig,
        tag_params:    Optional[NodeParams] = None,
        anchor_params: Optional[Dict[int, NodeParams]] = None,
        seed:          Optional[int] = None,
    ) -> "UwbSimPipeline":
        """Cria pipeline a partir de um UwbSimConfig + overrides opcionais."""
        default = sim_cfg.default_node_params()
        return cls(
            ranging_cfg    = sim_cfg.to_ranging_cfg(),
            twr_cfg        = sim_cfg.to_twr_cfg(),
            tag_params     = tag_params    or default,
            anchor_params  = anchor_params or {},
            default_params = default,
            seed           = seed,
        )
    
    @classmethod
    def from_sim_cfg(cls, cfg, seed: int | None = None) -> "UwbSimPipeline":
        """
        Alias para compatibilidade.
        Alguns lugares chamam from_sim_cfg(), mas o nome 'oficial' aqui é from_config().
        """
        return cls.from_config(cfg, seed=seed)

    # Medição principal 
    def measure(
        self,
        x_state:     np.ndarray,        # [x, y, theta]
        anchors:     np.ndarray,        # 3 x N
        l:           float,             # metade do baseline
        z_c:         float,             # altura das tags
        return_meta: bool = False,
    ):
        """
        Gera o vetor de medições UWB z_k (shape 2*N) usando o pipeline TWR.

        Para cada âncora i, roda o protocolo TWR para a tag frontal e traseira
        separadamente, usando os NodeParams da âncora (ou default) e da tag.

        Retorna:
            - return_meta=False : z_k  (np.ndarray shape (2*N,))
            - return_meta=True  : (z_k, meta_list)  meta_list tem 2*N entradas
        """
        if anchors is None or anchors.shape[1] == 0:
            empty = np.empty((0,))
            return (empty, []) if return_meta else empty

        xk, yk, th = float(x_state[0]), float(x_state[1]), float(x_state[2])
        pf = np.array([xk + l * np.cos(th), yk + l * np.sin(th)])   # 2D (xy)
        pr = np.array([xk - l * np.cos(th), yk - l * np.sin(th)])   # 2D (xy)

        num_anchors = anchors.shape[1]
        z_k       = np.zeros(2 * num_anchors)
        meta_list = []

        # Converte tag params uma vez (mesmos para todas as âncoras)
        twr_clock_tag   = _node_to_twr_clock(self.tag_params)
        twr_delay_tag   = _node_to_twr_delay(self.tag_params)

        for i in range(num_anchors):
            a_xy = anchors[:2, i]   # só x,y para o protocolo 2D

            # Obtém params da âncora i (ou default)
            anc_p           = self.anchor_params.get(i, self.default_params)
            twr_clock_anc   = _node_to_twr_clock(anc_p)
            twr_delay_anc   = _node_to_twr_delay(anc_p)
            range_bias      = anc_p.range_bias_m

            # --- Tag frontal ---
            res_f: TWRResult = self.protocol.simulate(
                a_xy        = a_xy,
                tag_xy      = pf,
                channel     = self.ranging_model,
                clock_anchor = twr_clock_anc,
                clock_tag    = twr_clock_tag,
                delay_anchor = twr_delay_anc,
                delay_tag    = twr_delay_tag,
            )

            # --- Tag traseira ---
            res_r: TWRResult = self.protocol.simulate(
                a_xy        = a_xy,
                tag_xy      = pr,
                channel     = self.ranging_model,
                clock_anchor = twr_clock_anc,
                clock_tag    = twr_clock_tag,
                delay_anchor = twr_delay_anc,
                delay_tag    = twr_delay_tag,
            )

            # Fallback seguro: se dropout → usa distância euclidiana pura
            r_true_f = float(np.linalg.norm(
                np.array([xk + l*np.cos(th), yk + l*np.sin(th)]) - a_xy
            ))
            r_true_r = float(np.linalg.norm(
                np.array([xk - l*np.cos(th), yk - l*np.sin(th)]) - a_xy
            ))

            z_f = (res_f.r_est_m + range_bias) if (res_f.r_est_m is not None) else r_true_f
            z_r = (res_r.r_est_m + range_bias) if (res_r.r_est_m is not None) else r_true_r

            z_k[2*i]     = float(z_f)
            z_k[2*i + 1] = float(z_r)

            if return_meta:
                meta_list.append({
                    "anchor_idx": i,
                    "tag": "front",
                    "r_true": res_f.r_true_m,
                    "r_est":  res_f.r_est_m,
                    "is_nlos": res_f.is_nlos,
                    "dropped": res_f.dropped,
                    "ppm_anchor": anc_p.clock.drift_ppm,
                    "bias_m":    range_bias,
                })
                meta_list.append({
                    "anchor_idx": i,
                    "tag": "rear",
                    "r_true": res_r.r_true_m,
                    "r_est":  res_r.r_est_m,
                    "is_nlos": res_r.is_nlos,
                    "dropped": res_r.dropped,
                    "ppm_anchor": anc_p.clock.drift_ppm,
                    "bias_m":    range_bias,
                })

        return (z_k, meta_list) if return_meta else z_k
    
    def measure_ranges_and_sigmas(
        self,
        x_state: np.ndarray,          # [x,y,theta]
        anchors: np.ndarray,          # 3 x N
        l: float,                     # metade do baseline
        tag: str = "mid",             # "front" | "rear" | "mid"
        return_meta: bool = False,
        dropout_sigma: float = 10.0,  # sigma reportado quando dropped (pra dataset)
    ):
        """
        Mede range para UMA tag (front/rear/mid) e retorna:
          ranges_m: (N,)
          sigmas_m: (N,)
          meta (opcional): lista com N dicts

        O sigma_i é escolhido de forma compatível com o canal:
          - LOS  -> sigma_los
          - NLOS -> sigma_nlos
          - drop -> sigma grande (dropout_sigma)
        """
        if anchors is None or anchors.shape[1] == 0:
            empty = np.empty((0,), dtype=float)
            return (empty, empty, []) if return_meta else (empty, empty)

        xk, yk, th = float(x_state[0]), float(x_state[1]), float(x_state[2])

        if tag == "front":
            tag_xy = np.array([xk + l*np.cos(th), yk + l*np.sin(th)], dtype=float)
        elif tag == "rear":
            tag_xy = np.array([xk - l*np.cos(th), yk - l*np.sin(th)], dtype=float)
        else:  # "mid"
            tag_xy = np.array([xk, yk], dtype=float)

        num_anchors = anchors.shape[1]
        ranges = np.zeros((num_anchors,), dtype=float)
        sigmas = np.zeros((num_anchors,), dtype=float)
        meta_list = []

        # tag params (uma vez)
        twr_clock_tag = _node_to_twr_clock(self.tag_params)
        twr_delay_tag = _node_to_twr_delay(self.tag_params)

        # tenta capturar sigma_los/nlos do ranging_model 
        sigma_los = float(getattr(self.ranging_model, "sigma_los", 0.03))
        sigma_nlos = float(getattr(self.ranging_model, "sigma_nlos", 0.15))
        # se seu UwbRangingModel guarda config em .cfg, prefere ela:
        cfg = getattr(self.ranging_model, "cfg", None)
        if cfg is not None:
            sigma_los = float(getattr(cfg, "sigma_los", sigma_los))
            sigma_nlos = float(getattr(cfg, "sigma_nlos", sigma_nlos))

        for i in range(num_anchors):
            a_xy = anchors[:2, i]

            anc_p = self.anchor_params.get(i, self.default_params)
            twr_clock_anc = _node_to_twr_clock(anc_p)
            twr_delay_anc = _node_to_twr_delay(anc_p)
            range_bias = float(anc_p.range_bias_m)

            res: TWRResult = self.protocol.simulate(
                a_xy          = a_xy,
                tag_xy        = tag_xy,
                channel       = self.ranging_model,
                clock_anchor  = twr_clock_anc,
                clock_tag     = twr_clock_tag,
                delay_anchor  = twr_delay_anc,
                delay_tag     = twr_delay_tag,
            )

            r_true = float(np.linalg.norm(tag_xy - a_xy))

            dropped = bool(getattr(res, "dropped", False))
            is_nlos = bool(getattr(res, "is_nlos", False))

            # range medido 
            if (getattr(res, "r_est_m", None) is None) or dropped:
                z = r_true
            else:
                z = float(res.r_est_m)

            z = float(z) + range_bias
            ranges[i] = z

            # sigma_i (reportado)
            if dropped:
                sig = float(dropout_sigma)
            else:
                sig = float(sigma_nlos if is_nlos else sigma_los)

            sigmas[i] = sig

            if return_meta:
                meta_list.append({
                    "anchor_idx": i,
                    "tag": tag,
                    "r_true": float(getattr(res, "r_true_m", r_true)),
                    "r_est": getattr(res, "r_est_m", None),
                    "is_nlos": is_nlos,
                    "dropped": dropped,
                    "ppm_anchor": float(anc_p.clock.drift_ppm),
                    "bias_m": float(range_bias),
                    "sigma_i": float(sig),
                })

        return (ranges, sigmas, meta_list) if return_meta else (ranges, sigmas)

    # Atualização dinâmica de parâmetros 
    def update_anchor_params(self, anchor_idx: int, params: NodeParams) -> None:
        """Atualiza NodeParams de uma âncora específica em runtime."""
        self.anchor_params[anchor_idx] = params

    def update_tag_params(self, params: NodeParams) -> None:
        """Atualiza NodeParams da tag em runtime."""
        self.tag_params = params

    def set_all_anchor_params(self, params: NodeParams) -> None:
        """Define o mesmo NodeParams como default para todas as âncoras."""
        self.default_params = params
        self.anchor_params  = {}   # limpa overrides individuais