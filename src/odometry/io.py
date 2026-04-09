from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .models import EncoderSample


TIME_COLUMNS = ("timestamp", "timestamp_s", "time", "t")
LEFT_COLUMNS = ("left_ticks", "ticks_left", "encoder_left", "left")
RIGHT_COLUMNS = ("right_ticks", "ticks_right", "encoder_right", "right")


def _normalize_col_name(name: str) -> str:
    return name.strip().lower()


def _find_column(fieldnames: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_col_name(f): f for f in fieldnames}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    return None


def resolve_encoder_csv_columns(fieldnames: list[str]) -> dict[str, str]:
    """
    Resolve os nomes reais das colunas do arquivo para:
    - timestamp
    - left_ticks
    - right_ticks
    """
    if not fieldnames:
        raise ValueError("Arquivo sem cabeçalho")

    time_col = _find_column(fieldnames, TIME_COLUMNS)
    left_col = _find_column(fieldnames, LEFT_COLUMNS)
    right_col = _find_column(fieldnames, RIGHT_COLUMNS)

    missing = []
    if time_col is None:
        missing.append("timestamp")
    if left_col is None:
        missing.append("left_ticks")
    if right_col is None:
        missing.append("right_ticks")

    if missing:
        raise ValueError(
            "Arquivo de encoder inválido. Colunas obrigatórias não encontradas: "
            + ", ".join(missing)
        )

    return {
        "timestamp": time_col,
        "left_ticks": left_col,
        "right_ticks": right_col,
    }


def parse_encoder_row(row: dict, colmap: dict[str, str]) -> EncoderSample:
    """
    Converte uma linha do arquivo em EncoderSample.
    """
    try:
        timestamp_s = float(row[colmap["timestamp"]])
        left_ticks = int(float(row[colmap["left_ticks"]]))
        right_ticks = int(float(row[colmap["right_ticks"]]))
    except KeyError as exc:
        raise ValueError(f"Coluna ausente ao parsear linha: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Linha inválida no arquivo de encoder: {row}") from exc

    return EncoderSample(
        timestamp_s=timestamp_s,
        left_ticks=left_ticks,
        right_ticks=right_ticks,
    )


def _sniff_delimiter(sample_text: str) -> str:
    """
    Tenta descobrir o delimitador.
    Prioridade prática:
    vírgula, ponto e vírgula, tab, espaço.
    """
    candidates = [",", ";", "\t"]
    counts = {d: sample_text.count(d) for d in candidates}

    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best

    return " "


def load_encoder_delimited_file(path: str | Path) -> list[EncoderSample]:
    """
    Lê arquivo tabular (.csv ou .txt) com cabeçalho.
    Suporta delimitadores:
    - vírgula
    - ponto e vírgula
    - tab
    - espaço
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Arquivo vazio: {path}")

    delimiter = _sniff_delimiter(text[:2048])

    lines = text.splitlines()
    if delimiter == " ":
        # trata múltiplos espaços como separador
        rows = []
        header = lines[0].strip().split()
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.strip().split()
            if len(values) != len(header):
                raise ValueError(
                    f"Linha com número de colunas diferente do cabeçalho: {line}"
                )
            rows.append(dict(zip(header, values)))
        fieldnames = header
    else:
        reader = csv.DictReader(lines, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    colmap = resolve_encoder_csv_columns(list(fieldnames))
    samples = [parse_encoder_row(row, colmap) for row in rows]
    samples.sort(key=lambda s: s.timestamp_s)

    return samples


def load_encoder_csv(path: str | Path) -> list[EncoderSample]:
    """
    Mantido por compatibilidade.
    """
    return load_encoder_delimited_file(path)


def load_encoder_txt(path: str | Path) -> list[EncoderSample]:
    """
    Loader explícito para .txt com formato tabular e cabeçalho.
    """
    return load_encoder_delimited_file(path)


def validate_encoder_samples(samples: list[EncoderSample]) -> None:
    """
    Validação básica de consistência temporal.
    """
    if not samples:
        raise ValueError("Nenhuma amostra de encoder encontrada")

    for i in range(1, len(samples)):
        if samples[i].timestamp_s < samples[i - 1].timestamp_s:
            raise ValueError("Timestamps do encoder não estão em ordem crescente")


def load_and_validate_encoder_csv(path: str | Path) -> list[EncoderSample]:
    samples = load_encoder_csv(path)
    validate_encoder_samples(samples)
    return samples


def load_and_validate_encoder_txt(path: str | Path) -> list[EncoderSample]:
    samples = load_encoder_txt(path)
    validate_encoder_samples(samples)
    return samples


def load_encoder_file(path: str | Path) -> list[EncoderSample]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".csv", ".txt"):
        return load_encoder_delimited_file(path)

    raise ValueError(f"Formato de arquivo não suportado: {suffix}")


def load_and_validate_encoder_file(path: str | Path) -> list[EncoderSample]:
    samples = load_encoder_file(path)
    validate_encoder_samples(samples)
    return samples