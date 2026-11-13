"""Pré-processamento UWB opcional, genérico e independente da origem.

Contrato: ranges MxN (uma tag) ou MxTxN (T tags), N âncoras.
A entrada já deve estar normalizada em metros e com índices/tempos
associados pelo adaptador. Nunca interpolamos medições nem produzimos
ground truth. 'off' mantém o legado; 'flag' só diagnostica; 'mask'
marca medições sinalizadas com NaN, mantendo TODOS os índices originais.

Separar *qualidade da medição* de *inovação do filtro*: rejeição por
inovação depende da pose prevista e pertence ao estimador, não aqui.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

import numpy as np


# Flags independentes; um mesmo range pode carregar mais de um motivo.
MISSING_RANGE = 1
NONPOSITIVE_RANGE = 2
EXCESSIVE_RANGE = 4
BELOW_VERTICAL_MIN = 8
INVALID_REPORTED_SIGMA = 16
EXCESSIVE_REPORTED_SIGMA = 32
POSSIBLY_REPEATED = 64

FLAG_NAMES = {
    MISSING_RANGE: "missing_nonfinite_range",
    NONPOSITIVE_RANGE: "nonpositive_range",
    EXCESSIVE_RANGE: "above_max_range",
    BELOW_VERTICAL_MIN: "below_vertical_distance",
    INVALID_REPORTED_SIGMA: "invalid_reported_sigma",
    EXCESSIVE_REPORTED_SIGMA: "above_max_reported_sigma",
    POSSIBLY_REPEATED: "possibly_repeated_consecutively",
}


def _finite_nonnegative(value, key, *, strictly_positive=False):
    if (type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0 or (strictly_positive and value == 0)):
        raise ValueError(f"{key} deve ser numérico, finito e positivo"
                         if strictly_positive else
                         f"{key} deve ser numérico, finito e não negativo")
    return float(value)


@dataclass(frozen=True)
class UwbPreprocessConfig:
    """Política experimental declarativa, sem parâmetros LAB7A embutidos.

    mode='off': sem diagnóstico nem alteração; padrão legado.
    mode='flag': sinaliza problemas, sem mudar dados.
    mode='mask': sinaliza e substitui células assinaladas por NaN.
    """
    mode: str = "off"
    max_range_m: float | None = None
    tag_heights_m: float | tuple[float, ...] | None = None
    vertical_tolerance_m: float = 0.0
    max_reported_sigma_m: float | None = None
    detect_consecutive_repeats: bool = False

    def __post_init__(self):
        if self.mode not in ("off", "flag", "mask"):
            raise ValueError("mode deve ser off, flag ou mask")
        if self.max_range_m is not None:
            _finite_nonnegative(self.max_range_m, "max_range_m", strictly_positive=True)
        _finite_nonnegative(self.vertical_tolerance_m, "vertical_tolerance_m")
        if self.max_reported_sigma_m is not None:
            _finite_nonnegative(
                self.max_reported_sigma_m, "max_reported_sigma_m", strictly_positive=True
            )
        if type(self.detect_consecutive_repeats) is not bool:
            raise ValueError("detect_consecutive_repeats deve ser booleano")
        heights = self.tag_heights_m
        if heights is not None:
            if type(heights) in (int, float):
                _finite_nonnegative(heights, "tag_heights_m")
            elif (isinstance(heights, (list, tuple))
                  and len(heights) > 0):
                for height in heights:
                    _finite_nonnegative(height, "tag_heights_m")
            else:
                raise ValueError("tag_heights_m deve ser altura ou lista de alturas")

    @classmethod
    def from_dict(cls, data):
        """Rejeita parâmetros desconhecidos, sem fallback silencioso."""
        if not isinstance(data, dict):
            raise ValueError("uwb_preprocessing deve ser objeto JSON")
        allowed = {f.name for f in fields(cls)}
        extra = set(data) - allowed
        if extra:
            raise ValueError("Campos desconhecidos em uwb_preprocessing: "
                             + ", ".join(sorted(extra)))
        return cls(**data)


@dataclass(frozen=True)
class UwbPreprocessResult:
    """Arrays independentes da entrada, diagnóstico e alinhamento preservado."""
    distances: np.ndarray
    deviations: np.ndarray | None
    flags: np.ndarray
    accepted_mask: np.ndarray
    sample_indices: np.ndarray
    timestamps_s: np.ndarray | None
    summary: dict


def load_uwb_preprocess_config(anchor_json: str | Path):
    """Configuração opcional e genérica no JSON de geometria.

    Sem o campo 'uwb_preprocessing', retorna None (pipeline legado).
    Qualquer fonte de dados pode usar o mesmo JSON: real ou simulada.
    """
    with Path(anchor_json).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("JSON de âncoras deve ser objeto")
    if "uwb_preprocessing" not in data:
        return None
    return UwbPreprocessConfig.from_dict(data["uwb_preprocessing"])


def _tag_heights(config, n_tags):
    value = config.tag_heights_m
    if value is None:
        return None
    if type(value) in (int, float):
        return np.full(n_tags, float(value), dtype=float)
    if len(value) != n_tags:
        raise ValueError(
            f"tag_heights_m: esperado {n_tags} alturas, recebeu {len(value)}"
        )
    return np.asarray(value, dtype=float)


def _indices_and_timestamps(row_count, sample_indices, timestamps_s):
    if sample_indices is None:
        indices = np.arange(row_count, dtype=np.int64)
    else:
        src = np.asarray(sample_indices)
        if (src.shape != (row_count,) or src.dtype.kind not in ("i", "u")
                or np.any(src < 0)
                or (row_count > 1 and np.any(np.diff(src) <= 0))):
            raise ValueError("sample_indices deve ser vetor de inteiros crescentes")
        indices = src.astype(np.int64, copy=True)
    if timestamps_s is None:
        times = None
    else:
        times = np.asarray(timestamps_s, dtype=float)
        if (times.shape != (row_count,) or not np.all(np.isfinite(times))
                or (row_count > 1 and np.any(np.diff(times) <= 0))):
            raise ValueError("timestamps_s deve ser vetor finito e estritamente crescente")
        times = times.copy()
    return indices, times


def preprocess_uwb_batch(
    anchors_Nx3, distances, deviations=None, *,
    config: UwbPreprocessConfig | dict | None = None,
    sample_indices=None, timestamps_s=None,
) -> UwbPreprocessResult:
    """Prepara um conjunto MxN ou MxTxN, sem mutar ou reordenar inputs.

    M é o número de instantes/snapshots, T o número de tags e N o
    número de âncoras. Ao aplicar máscara, mantém M e os índices
    originais; um algoritmo pode falhar com poucos ranges válidos,
    mas isso deve aparecer como ausência de estimativa, nunca imputação.
    """
    cfg = (UwbPreprocessConfig() if config is None else
           UwbPreprocessConfig.from_dict(config) if isinstance(config, dict)
           else config)
    if not isinstance(cfg, UwbPreprocessConfig):
        raise ValueError("config deve ser UwbPreprocessConfig, dict ou None")
    anchors = np.asarray(anchors_Nx3, dtype=float)
    raw = np.asarray(distances, dtype=float)
    if (anchors.ndim != 2 or anchors.shape[1] not in (2, 3)
            or anchors.shape[0] == 0 or not np.all(np.isfinite(anchors))):
        raise ValueError("anchors_Nx3 deve ter formato Nx2/Nx3 finito")
    if raw.ndim not in (2, 3) or raw.shape[0] == 0:
        raise ValueError("distances deve ser MxN ou MxTxN não vazio")
    if raw.shape[-1] != anchors.shape[0]:
        raise ValueError("Quantidade de colunas UWB não coincide com âncoras")
    M, N = raw.shape[0], raw.shape[-1]
    n_tags = raw.shape[1] if raw.ndim == 3 else 1
    sigmas = None if deviations is None else np.asarray(deviations, dtype=float)
    if sigmas is not None and sigmas.shape != raw.shape:
        raise ValueError("deviations deve ter a mesma forma de distances")
    indices, times = _indices_and_timestamps(M, sample_indices, timestamps_s)
    if cfg.max_reported_sigma_m is not None and sigmas is None and cfg.mode != "off":
        raise ValueError("max_reported_sigma_m requer deviations disponíveis")
    heights = _tag_heights(cfg, n_tags)
    if heights is not None and anchors.shape[1] != 3:
        raise ValueError("Verificação vertical requer coordenadas de âncoras Nx3")

    # As cópias evitam mutar dados de entrada/arquivos. Preserva NaN,
    # sentinelas e precisão original no modo off ou apenas flag.
    output = raw.copy()
    sigma_output = sigmas.copy() if sigmas is not None else None
    flags = np.zeros(raw.shape, dtype=np.uint8)
    if cfg.mode != "off":
        finite = np.isfinite(raw)
        flags[~finite] |= MISSING_RANGE
        flags[finite & (raw <= 0)] |= NONPOSITIVE_RANGE
        good = finite & (raw > 0)
        if cfg.max_range_m is not None:
            flags[good & (raw > cfg.max_range_m)] |= EXCESSIVE_RANGE
        if heights is not None:
            delta_z = np.abs(heights[:, None] - anchors[None, :, 2])
            minimum = delta_z - cfg.vertical_tolerance_m
            if raw.ndim == 2:
                flags[good & (raw < minimum[0])] |= BELOW_VERTICAL_MIN
            else:
                flags[good & (raw < minimum[None, :, :])] |= BELOW_VERTICAL_MIN
        if sigmas is not None:
            sigma_bad = ~np.isfinite(sigmas) | (sigmas < 0)
            flags[good & sigma_bad] |= INVALID_REPORTED_SIGMA
            if cfg.max_reported_sigma_m is not None:
                flags[good & ~sigma_bad
                      & (sigmas > cfg.max_reported_sigma_m)] |= EXCESSIVE_REPORTED_SIGMA
        if cfg.detect_consecutive_repeats and M > 1:
            # Igualdade exata entre linhas adjacentes é apenas SUSPEITA
            # de reutilização; não demonstra ausência de ranging novo.
            repeat = good[1:] & good[:-1] & (raw[1:] == raw[:-1])
            if sigmas is not None:
                repeat &= (sigmas[1:] == sigmas[:-1])
            flags[1:][repeat] |= POSSIBLY_REPEATED
        if cfg.mode == "mask":
            excluded = flags != 0
            output[excluded] = np.nan
            if sigma_output is not None:
                sigma_output[excluded] = np.nan

    accepted = (flags == 0) & np.isfinite(output) & (output > 0)
    reasons = {
        name: int(np.count_nonzero((flags & bit) != 0))
        for bit, name in FLAG_NAMES.items()
    }
    flagged = flags != 0
    original_finite_positive = np.isfinite(raw) & (raw > 0)
    return UwbPreprocessResult(
        distances=output,
        deviations=sigma_output,
        flags=flags,
        accepted_mask=accepted,
        sample_indices=indices,
        timestamps_s=times,
        summary={
            "schema": "bc_ekf.uwb_preprocessing.v1",
            "config": asdict(cfg),
            "rows": int(M),
            "tags": int(n_tags),
            "anchors": int(N),
            "total_cells": int(raw.size),
            "flagged_cells": int(np.count_nonzero(flagged)),
            "flag_counts": reasons,
            "newly_masked_finite_positive_cells": int(
                np.count_nonzero(flagged & original_finite_positive)
            ) if cfg.mode == "mask" else 0,
            "rows_with_any_flag": int(np.count_nonzero(
                np.any(flagged.reshape(M, -1), axis=1)
            )),
            "per_anchor_flagged_cells": [
                int(np.count_nonzero(flagged[..., a])) for a in range(N)
            ],
            "note": (
                "Flags são indicadores, não causas comprovadas. 'mask' "
                "preserva a linha/tempo e apenas marca célula com NaN; "
                "nenhum range ausente é inventado."
            ),
        },
    )
