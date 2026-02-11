# src/uwb/node_params.py
from __future__ import annotations
from dataclasses import dataclass, field


# Velocidade da luz (m/s)
C0 = 299_792_458.0

def ns_to_s(ns: float) -> float:
    '''Converte nanosegundos para segundos.'''
    return ns * 1e-9

def s_to_ns(s: float) -> float:
    '''Converte segundos para nanosegundos.'''
    return s * 1e9

@dataclass
class ClockModel:
    """Modelo simples de clock: erro de frequência em ppm."""
    drift_ppm: float = 0.0  # +ppm = clock mais rápido

    def rate(self) -> float:
        return 1.0 + self.drift_ppm * 1e-6

@dataclass
class AntennaDelays:
    """
    Delays internos (hardware) em ns.
    Em TWR isso entra como erro no ToF se não calibrado.
    """
    tx_ns: float = 0.0
    rx_ns: float = 0.0

    def tx_s(self) -> float:
        '''Retorna o delay de transmissão em segundos.'''
        return ns_to_s(self.tx_ns)

    def rx_s(self) -> float:
        ''' Retorna o delay de recepção em segundos.'''
        return ns_to_s(self.rx_ns)

@dataclass
class NodeParams:
    """Parâmetros comuns a Anchor/Tag."""
    clock: ClockModel = field(default_factory=ClockModel)
    ant: AntennaDelays = field(default_factory=AntennaDelays)
    range_bias_m: float = 0.0  # bias aditivo direto em metros (opcional)
