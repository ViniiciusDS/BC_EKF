# src/uwb/twr_protocols.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Dict, Any

import numpy as np

C0 = 299_792_458.0  # velocidade da luz (m/s)


class TWRMode(str, Enum):
    ''' Modos de TWR suportados. evoluir dps. '''
    SS_TWR = "SS_TWR"
    DS_TWR = "DS_TWR"


@dataclass
class ClockModel:
    """
    Modelo simples de clock local:
      local_time = offset_s + skew * real_time
    onde skew ≈ 1 + ppm*1e-6
    """
    offset_s: float = 0.0
    ppm: float = 0.0  # drift (ppm)

    @property
    def skew(self) -> float:
        return 1.0 + self.ppm * 1e-6

    def to_local(self, t_real_s: float) -> float:
        return self.offset_s + self.skew * t_real_s


@dataclass
class AntennaDelayModel:
    """
    Atrasos de hardware (segundos). No real isso inclui TX/RX antenna delay, etc.
    Modelo simétrico e extensível.
    """
    tx_s: float = 0.0
    rx_s: float = 0.0


@dataclass
class TimestampJitter:
    """
    Jitter de timestamp (segundos). Representa quantização/ruído no timestamping.
    """
    sigma_s: float = 0.0  # ex: 1e-9 ~ 1 ns

    def sample(self, rng: np.random.Generator) -> float:
        if self.sigma_s <= 0:
            return 0.0
        return float(rng.normal(0.0, self.sigma_s))


@dataclass
class TWRConfig:
    mode: TWRMode = TWRMode.DS_TWR
    c: float = C0

    # delays "controlados" (em segundos) entre RX e TX no respectivo nó
    reply_delay_anchor_s: float = 0.0005   # 0.5 ms
    reply_delay_tag_s: float = 0.0005      # 0.5 ms

    # jitter de timestamp
    ts_jitter: TimestampJitter = field(default_factory=lambda: TimestampJitter(sigma_s=0.0))


@dataclass
class TWRResult:
    r_true_m: float
    r_est_m: Optional[float]      # None se dropout/falha
    tof_true_s: float
    tof_est_s: Optional[float]    # None se dropout/falha

    is_nlos: bool
    dropped: bool

    # componentes p/ debug
    components: Dict[str, Any]


class ChannelToFModel(Protocol):
    """
    Interface padronizada do 'canal' para o protocolo.
    O protocolo pede: "me dá ToF verdadeiro + erro (NLOS/noise/dropout)".

    Isso permite trocar o canal depois.
    """
    def sample_tof(
        self,
        a_xy: np.ndarray,
        tag_xy: np.ndarray,
        c: float,
    ) -> tuple[float, Optional[float], bool, bool, Dict[str, Any]]:
        """
        Retorna:
          (tof_true_s, tof_meas_s or None, is_nlos, dropped, components)
        """


class TWRProtocol(Protocol):
    """
    Interface padronizada para protocolos.
    """
    def simulate(
        self,
        a_xy: np.ndarray,
        tag_xy: np.ndarray,
        channel: ChannelToFModel,
        clock_anchor: ClockModel | None = None,
        clock_tag: ClockModel | None = None,
        delay_anchor: AntennaDelayModel | None = None,
        delay_tag: AntennaDelayModel | None = None,
    ) -> TWRResult:
        ...


class DS_TWR_Protocol:
    """
    DS-TWR clássico (double-sided).
    Implementação com:
      - clock skew/offset nos dois nós
      - delays TX/RX (antenna delay) nos dois nós
      - reply delays controlados
      - jitter de timestamp
      - canal gerando tof_meas (inclui NLOS/noise/dropout)
    """
    def __init__(
        self,
        cfg: TWRConfig,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
    
    def simulate(
            self,
            a_xy: np.ndarray,
            tag_xy: np.ndarray, 
            channel: ChannelToFModel,
            clock_anchor: ClockModel | None = None,
            clock_tag: ClockModel | None = None,
            delay_anchor: AntennaDelayModel | None = None,
            delay_tag: AntennaDelayModel | None = None,
            ) -> TWRResult:
        ''' Simula uma execução do protocolo DS-TWR.'''
        c = self.cfg.c

        # canal entrega tof_true e tof_meas (ou None em dropout)
        tof_true, tof_meas, is_nlos, dropped, ch_comp = channel.sample_tof(a_xy, tag_xy, c=c)
        r_true = float(np.linalg.norm(a_xy - tag_xy))

        if dropped or (tof_meas is None):
            return TWRResult(
                r_true_m=r_true,
                r_est_m=None,
                tof_true_s=tof_true,
                tof_est_s=None,
                is_nlos=is_nlos,
                dropped=True,
                components={"channel": ch_comp, "reason": "dropout"},
            )

        # Para DS-TWR, tof_meas como "tempo de voo efetivo" (inclui NLOS/noise)
        tof_eff = tof_meas

        # timeline real (segundos)
        # Tag envia POLL em t1_real = 0
        t1_real = 0.0

        # Anchor recebe POLL: soma propagation + delays (TX do tag + RX do anchor)
        t2_real = t1_real + delay_tag.tx_s + tof_eff + delay_anchor.rx_s

        # Anchor envia RESP depois de reply_delay_anchor (no tempo real)
        t3_real = t2_real + self.cfg.reply_delay_anchor_s + delay_anchor.tx_s

        # Tag recebe RESP
        t4_real = t3_real + tof_eff + delay_tag.rx_s

        # Tag envia FINAL depois de reply_delay_tag
        t5_real = t4_real + self.cfg.reply_delay_tag_s + delay_tag.tx_s

        # Anchor recebe FINAL
        t6_real = t5_real + tof_eff + delay_anchor.rx_s

        # Evita quebrar se clocks/delays não forem passados
        clock_tag = clock_tag or ClockModel()
        clock_anchor = clock_anchor or ClockModel()
        delay_tag = delay_tag or AntennaDelayModel()
        delay_anchor = delay_anchor or AntennaDelayModel()

        # timestamps locais (com skew/offset) + jitter
        def ts_tag(t_real: float) -> float:
            return clock_tag.to_local(t_real) + self.cfg.ts_jitter.sample(self.rng)

        def ts_anchor(t_real: float) -> float:
            return clock_anchor.to_local(t_real) + self.cfg.ts_jitter.sample(self.rng)

        t1 = ts_tag(t1_real)
        t4 = ts_tag(t4_real)
        t5 = ts_tag(t5_real)

        t2 = ts_anchor(t2_real)
        t3 = ts_anchor(t3_real)
        t6 = ts_anchor(t6_real)

        # DS-TWR termos:
        # Ra = t4 - t1  (tag round-trip)
        # Rb = t6 - t3  (anchor round-trip)
        # Da = t3 - t2  (anchor reply delay)
        # Db = t5 - t4  (tag reply delay)
        Ra = t4 - t1
        Rb = t6 - t3
        Da = t3 - t2
        Db = t5 - t4

        denom = (Ra + Rb + Da + Db)
        if denom <= 0:
            return TWRResult(
                r_true_m=r_true,
                r_est_m=None,
                tof_true_s=tof_true,
                tof_est_s=None,
                is_nlos=is_nlos,
                dropped=True,
                components={"channel": ch_comp, "reason": "invalid_denom", "Ra": Ra, "Rb": Rb, "Da": Da, "Db": Db},
            )

        tof_est = (Ra * Rb - Da * Db) / denom
        r_est = float(c * tof_est)

        return TWRResult(
            r_true_m=r_true,
            r_est_m=r_est,
            tof_true_s=tof_true,
            tof_est_s=float(tof_est),
            is_nlos=is_nlos,
            dropped=False,
            components={
                "channel": ch_comp,
                "timestamps": {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6},
                "terms": {"Ra": Ra, "Rb": Rb, "Da": Da, "Db": Db},
                "clocks": {
                    "tag": {"offset_s": clock_tag.offset_s, "ppm": clock_tag.ppm},
                    "anchor": {"offset_s": clock_anchor.offset_s, "ppm": clock_anchor.ppm},
                },
                "delays": {
                    "tag": {"tx_s": delay_tag.tx_s, "rx_s": delay_tag.rx_s},
                    "anchor": {"tx_s": delay_anchor.tx_s, "rx_s": delay_anchor.rx_s},
                },
            },
        )

class SS_TWR_Protocol:
    """
    SS-TWR (single-sided).
    Mais sensível a clock skew (ppm) porque não cancela tão bem quanto DS-TWR.

    Modelo:
      Tag envia POLL -> Anchor responde após reply_delay_anchor.
      Tag estima ToF a partir do round-trip e do reply delay conhecido/medido.

    Estimativa típica:
      tof_est ≈ (Tround_tag - Treply_anchor) / 2
    Onde:
      Tround_tag  = t4 - t1 (no clock do TAG)
      Treply_anchor = t3 - t2 (no clock do ANCHOR)
    """
    def __init__(
        self,
        cfg: TWRConfig,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def simulate(self, 
                a_xy: np.ndarray, 
                tag_xy: np.ndarray, 
                channel: ChannelToFModel,
                clock_anchor: ClockModel | None = None,
                clock_tag: ClockModel | None = None,
                delay_anchor: AntennaDelayModel | None = None,
                delay_tag: AntennaDelayModel | None = None,
                 ) -> TWRResult:
        ''' Simula uma execução do protocolo SS-TWR.'''
        c = self.cfg.c

        # Evita Quebrar
        clock_tag = clock_tag or ClockModel()
        clock_anchor = clock_anchor or ClockModel()
        delay_tag = delay_tag or AntennaDelayModel()
        delay_anchor = delay_anchor or AntennaDelayModel()

        tof_true, tof_meas, is_nlos, dropped, ch_comp = channel.sample_tof(a_xy, tag_xy, c=c)
        r_true = float(np.linalg.norm(a_xy - tag_xy))

        if dropped or (tof_meas is None):
            return TWRResult(
                r_true_m=r_true,
                r_est_m=None,
                tof_true_s=tof_true,
                tof_est_s=None,
                is_nlos=is_nlos,
                dropped=True,
                components={"channel": ch_comp, "reason": "dropout"},
            )

        tof_eff = tof_meas

        # timeline real
        t1_real = 0.0
        t2_real = t1_real + delay_tag.tx_s + tof_eff + delay_anchor.rx_s
        t3_real = t2_real + self.cfg.reply_delay_anchor_s + delay_anchor.tx_s
        t4_real = t3_real + tof_eff + delay_tag.rx_s

        def ts_tag(t_real: float) -> float:
            return clock_tag.to_local(t_real) + self.cfg.ts_jitter.sample(self.rng)

        def ts_anchor(t_real: float) -> float:
            return clock_anchor.to_local(t_real) + self.cfg.ts_jitter.sample(self.rng)

        t1 = ts_tag(t1_real)
        t4 = ts_tag(t4_real)

        t2 = ts_anchor(t2_real)
        t3 = ts_anchor(t3_real)

        Tround = (t4 - t1)      # no clock do TAG
        Treply = (t3 - t2)      # no clock do ANCHOR

        # estimativa SS-TWR
        tof_est = 0.5 * (Tround - Treply)

        if tof_est <= 0:
            return TWRResult(
                r_true_m=r_true,
                r_est_m=None,
                tof_true_s=tof_true,
                tof_est_s=None,
                is_nlos=is_nlos,
                dropped=True,
                components={
                    "channel": ch_comp,
                    "reason": "invalid_tof",
                    "terms": {"Tround": Tround, "Treply": Treply},
                },
            )

        r_est = float(c * tof_est)

        return TWRResult(
            r_true_m=r_true,
            r_est_m=r_est,
            tof_true_s=tof_true,
            tof_est_s=float(tof_est),
            is_nlos=is_nlos,
            dropped=False,
            components={
                "channel": ch_comp,
                "timestamps": {"t1": t1, "t2": t2, "t3": t3, "t4": t4},
                "terms": {"Tround": Tround, "Treply": Treply},
                "clocks": {
                    "tag": {"offset_s": clock_tag.offset_s, "ppm": clock_tag.ppm},
                    "anchor": {"offset_s": clock_anchor.offset_s, "ppm": clock_anchor.ppm},
                },
                "delays": {
                    "tag": {"tx_s": delay_tag.tx_s, "rx_s": delay_tag.rx_s},
                    "anchor": {"tx_s": delay_anchor.tx_s, "rx_s": delay_anchor.rx_s},
                },
            },
        )
