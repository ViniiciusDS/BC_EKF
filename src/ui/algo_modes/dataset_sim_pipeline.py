from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.uwb.algoritmos_estaticos import carregar_ensaio_lab


@dataclass
class SimPipelineResult:
    ok: bool
    message: str = ""

    dataset_path: Optional[str] = None
    dataset_label: str = ""

    batch_dists: Optional[np.ndarray] = None
    batch_devs: Optional[np.ndarray] = None

    simulated_kind: str = "Front"
    is_bc: bool = False


def normalize_simulated_kind(kind: str) -> str:
    """
    Normaliza nomes equivalentes para o padrão usado no DatasetMode.
    """
    key = str(kind or "Front").strip().lower()

    kind_map = {
        "front": "Front",
        "top": "Front",
        "tag1": "Front",
        "t1": "Front",

        "rear": "Rear",
        "bot": "Rear",
        "bottom": "Rear",
        "tag2": "Rear",
        "t2": "Rear",

        "mid": "Mid",
        "middle": "Mid",
        "center": "Mid",
        "centro": "Mid",

        "bc": "BC",
        "bc-ekf": "BC",
        "bcekf": "BC",
    }

    return kind_map.get(key, "Front")


def load_sim_txt_dataset(path: str):
    """
    Carrega dataset simulado em TXT/CSV simples.

    Formato esperado:
        dist_0, sigma_0, dist_1, sigma_1, ...

    Retorna:
        dists: M x N
        devs:  M x N
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            if line.startswith("#"):
                continue

            if ";" in line:
                parts = [p.strip() for p in line.split(";") if p.strip()]
            elif "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
            else:
                parts = [p.strip() for p in line.split() if p.strip()]

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(vals)

    if not rows:
        raise ValueError(f"Nenhuma linha válida encontrada em {path}")

    data = np.array(rows, dtype=float)

    if data.ndim != 2:
        raise ValueError("Dataset inválido: matriz não é 2D")

    if data.shape[1] % 2 != 0:
        raise ValueError(
            f"Número de colunas inválido: {data.shape[1]} "
            f"(esperado par: dist,sigma,dist,sigma,...)"
        )

    dists = data[:, 0::2]
    devs = data[:, 1::2]

    return dists, devs


def load_jsonl_dataset(path: str):
    """
    Carrega dataset JSONL gerado pelo simulador.

    Usa o campo z_k.
    Por padrão, se vier com duas tags por âncora, mantém somente a tag frontal,
    preservando o comportamento anterior do _load_jsonl().
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if "z_k" in obj and obj["z_k"]:
                rows.append(obj["z_k"])

    if not rows:
        raise ValueError("JSONL sem medições z_k válidas")

    dists_2n = np.array(rows, dtype=float)

    if dists_2n.ndim != 2:
        raise ValueError("JSONL inválido: matriz z_k não é 2D")

    # Mantém compatibilidade com comportamento antigo:
    # z_k = [A0_front, A0_rear, A1_front, A1_rear, ...]
    # usa front.
    dists = dists_2n[:, 0::2]

    return dists, None


def load_simulated_dataset_file(path: str):
    """
    Carrega dataset simulado a partir de arquivo.

    Ordem:
    - .jsonl: loader próprio;
    - demais: tenta carregar_ensaio_lab();
    - fallback: loader TXT/CSV genérico.
    """
    if not path:
        raise ValueError("Caminho de dataset vazio")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".jsonl"):
        return load_jsonl_dataset(path)

    try:
        return carregar_ensaio_lab(path)
    except Exception:
        return load_sim_txt_dataset(path)


def normalize_simulated_dataset_by_kind(
    *,
    batch_dists,
    batch_devs,
    dataset_anchors,
    simulated_kind: str,
    cfg: Any,
) -> SimPipelineResult:
    """
    Normaliza datasets simulados conforme o tipo selecionado.

    Dataset com N colunas:
        já representa uma tag/posição única.

    Dataset com 2N colunas:
        [A0_front, A0_rear, A1_front, A1_rear, ...]

    Tipos:
        Front -> colunas pares
        Rear  -> colunas ímpares
        Mid   -> média entre Front e Rear
        BC    -> mantém 2N colunas para o preparador BC-EKF
    """
    kind = normalize_simulated_kind(simulated_kind)

    if batch_dists is None:
        return SimPipelineResult(False, "Dataset simulado não carregado")

    if dataset_anchors is None:
        return SimPipelineResult(False, "Âncoras não carregadas")

    dists = np.asarray(batch_dists, dtype=float)

    devs = (
        np.asarray(batch_devs, dtype=float)
        if batch_devs is not None
        else None
    )

    if dists.ndim != 2:
        return SimPipelineResult(False, "Dataset inválido: matriz de distâncias não é 2D")

    anchors = np.asarray(dataset_anchors, dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] < 2:
        return SimPipelineResult(False, "Layout de âncoras inválido")

    n_anchors = int(anchors.shape[0])
    n_cols = int(dists.shape[1])

    # Dataset já reduzido.
    if n_cols == n_anchors:
        return SimPipelineResult(
            True,
            batch_dists=dists,
            batch_devs=devs,
            simulated_kind=kind,
            is_bc=(kind == "BC"),
        )

    # Dataset com duas tags por âncora.
    if n_cols == 2 * n_anchors:
        if kind == "BC":
            return SimPipelineResult(
                True,
                batch_dists=dists,
                batch_devs=devs,
                simulated_kind=kind,
                is_bc=True,
            )

        front = dists[:, 0::2]
        rear = dists[:, 1::2]

        if devs is not None:
            dev_front = devs[:, 0::2]
            dev_rear = devs[:, 1::2]
        else:
            dev_front = None
            dev_rear = None

        if kind == "Front":
            out_dists = front
            out_devs = dev_front

        elif kind == "Rear":
            out_dists = rear
            out_devs = dev_rear

        elif kind == "Mid":
            out_dists = 0.5 * (front + rear)

            if dev_front is not None and dev_rear is not None:
                out_devs = 0.5 * np.sqrt(dev_front**2 + dev_rear**2)
            else:
                out_devs = None

        else:
            out_dists = front
            out_devs = dev_front
            kind = "Front"

        if out_devs is None:
            out_devs = np.full_like(
                out_dists,
                float(getattr(cfg, "UWB_NOISE_STD", 0.05)),
                dtype=float,
            )

        return SimPipelineResult(
            True,
            batch_dists=out_dists,
            batch_devs=out_devs,
            simulated_kind=kind,
            is_bc=False,
        )

    return SimPipelineResult(
        False,
        (
            f"Incompatibilidade: dataset possui {n_cols} colunas de âncora, "
            f"mas o layout possui {n_anchors} âncoras"
        ),
    )


def load_and_normalize_simulated_dataset(
    *,
    dataset_path: str,
    dataset_anchors,
    simulated_kind: str,
    cfg: Any,
) -> SimPipelineResult:
    """
    Pipeline principal do dataset simulado:
    - carrega arquivo;
    - normaliza conforme Front/Rear/Mid/BC.
    """
    try:
        dists, devs = load_simulated_dataset_file(dataset_path)
    except Exception as e:
        return SimPipelineResult(False, f"Erro ao carregar dataset simulado: {e}")

    result = normalize_simulated_dataset_by_kind(
        batch_dists=dists,
        batch_devs=devs,
        dataset_anchors=dataset_anchors,
        simulated_kind=simulated_kind,
        cfg=cfg,
    )

    if not result.ok:
        return result

    result.dataset_path = dataset_path
    result.dataset_label = os.path.basename(dataset_path)

    return result