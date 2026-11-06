"""Seleção de voltas de PCNT apenas para visualização, sem alterar amostras.

O contador de voltas do ESP32 aumenta ao cruzar o marcador na pista.
Assim, o trecho com contador zero é a primeira volta gravada. O último
bloco é considerado parcial porque não tem evento de conclusão posterior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VIEW_MODES = ("single", "range", "all")


@dataclass(frozen=True)
class LapSlice:
    number: int
    start: int  # inclusivo
    stop: int   # exclusivo
    firmware_count: int
    complete: bool

    @property
    def sample_count(self) -> int:
        return self.stop - self.start


def split_laps(counter: np.ndarray) -> tuple[LapSlice, ...]:
    """Segmenta pelos incrementos de lap_count, preservando ordem de aquisição.

    Reutiliza um ponto na fronteira entre voltas para manter o traço contínuo;
    isso afeta somente o desenho, nunca o dataset original.
    """
    laps = np.asarray(counter)
    if laps.ndim != 1:
        raise ValueError("lap_count deve ter uma dimensão")
    if laps.size == 0:
        return ()
    if not np.all(np.isfinite(laps)):
        raise ValueError("lap_count contém NaN/Inf")
    crossings = np.flatnonzero(np.diff(laps) > 0) + 1
    starts = np.concatenate(([0], crossings))
    stops = np.concatenate((crossings, [len(laps)]))
    result = []
    for i, (start, stop) in enumerate(zip(starts, stops)):
        # Uma amostra isolada de pós-cruzamento não forma uma nova volta visível.
        if stop - start < 2:
            continue
        draw_start = max(0, int(start) - (1 if i else 0))
        result.append(LapSlice(
            number=len(result) + 1,
            start=draw_start,
            stop=int(stop),
            firmware_count=int(laps[start]),
            complete=(i < len(crossings)),
        ))
    return tuple(result)



def split_time_windows(
    timestamps_s: np.ndarray, duration_s: float = 80.0
) -> tuple[LapSlice, ...]:
    """Cria JANELAS temporais, não voltas físicas.

    Útil para logs em que 'lap_count' marca eventos de sensor e não
    fechamentos da trajetória. Não descarta amostras originais.
    """
    t = np.asarray(timestamps_s, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("timestamps_s precisa ser vetor finito")
    if duration_s <= 0 or not np.isfinite(duration_s):
        raise ValueError("duration_s deve ser positivo")
    if not len(t):
        return ()
    if np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps não crescentes")
    ids = np.floor((t - t[0]) / duration_s).astype(int)
    changes = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(t)]
    parts = []
    for i in range(len(changes) - 1):
        start, stop = int(changes[i]), int(changes[i + 1])
        if stop <= start:
            continue
        # Sobrepõe um ponto para preservar continuidade visual nas janelas.
        parts.append(LapSlice(
            number=i+1, start=max(0, start - (1 if i else 0)),
            stop=stop, firmware_count=-1, complete=(i < len(changes)-2),
        ))
    return tuple(parts)


def select_laps(
    lap_slices: tuple[LapSlice, ...], mode: str,
    start: int = 1, window: int = 5,
) -> tuple[LapSlice, ...]:
    """Devolve apenas as fatias de desenho para toda/uma/faixa de voltas."""
    if mode not in VIEW_MODES:
        raise ValueError(f"Modo de visualização inválido: {mode}")
    if not lap_slices:
        return ()
    if mode == "all":
        return lap_slices
    first = min(max(int(start), 1), len(lap_slices)) - 1
    if mode == "single":
        return lap_slices[first : first + 1]
    return lap_slices[first : first + max(1, int(window))]


def selection_label(mode: str, selected: tuple[LapSlice, ...], total: int) -> str:
    """Resumo compacto para a interface do Dataset Mode."""
    if not selected:
        return "Nenhuma volta disponível"
    if total == 1 and not selected[0].complete:
        return "Trecho sem volta marcada"
    if mode == "all":
        return f"Todas ({total - 1} + parcial)" if not selected[-1].complete else f"Todas ({total})"
    if mode == "single":
        if not selected[0].complete:
            return f"Trecho parcial {selected[0].number}/{total}"
        return f"Volta {selected[0].number}/{total}"
    suffix = " + parcial" if not selected[-1].complete else ""
    return f"Voltas {selected[0].number}-{selected[-1].number}/{total}{suffix}"
