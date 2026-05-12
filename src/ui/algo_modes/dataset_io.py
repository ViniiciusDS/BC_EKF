from __future__ import annotations

import os
import numpy as np

from src.ui.algo_modes.shared import (
    load_anchors_from_json,
    load_map_from_json,
    load_anchor_uwb_ids_from_json,
    load_route_waypoints_from_json,
)


def apply_anchors_to_dataset_mode(mode, anchors_path: str) -> bool:
    """
    Carrega âncoras e aplica no estado do DatasetMode.

    Mantém aqui a validação específica do DatasetMode:
    - dataset comum: N colunas == N âncoras;
    - dataset BC simulado: 2N colunas == N âncoras antes do preparo BC.
    """
    try:
        anchors, _ = load_anchors_from_json(anchors_path)
        anchors = np.asarray(anchors, dtype=float)

        if anchors.ndim != 2 or anchors.shape[1] < 2:
            raise ValueError("Arquivo de âncoras inválido.")

        mode._dataset_anchors = anchors
        mode._anchors_path = anchors_path
        mode._anchors_uwb_ids = load_anchor_uwb_ids_from_json(anchors_path)

        return validate_dataset_anchor_compatibility(mode)

    except ValueError as e:
        print(f"[DATASET] erro ao carregar âncoras: {e}")
        mode.host._set_msg(f"Erro ao carregar âncoras: {str(e)}")
        return False

    except Exception as e:
        print(f"[DATASET] erro ao carregar âncoras: {e}")
        mode.host._set_msg("Erro ao carregar âncoras")
        return False


def validate_dataset_anchor_compatibility(mode) -> bool:
    """
    Valida compatibilidade entre matriz de distâncias e layout de âncoras.

    Esta função é específica do DatasetMode porque depende do estado:
    - mode._batch_dists
    - mode._dataset_anchors
    - mode.dataset_source_type
    - mode.simulated_dataset_kind
    """
    if mode._batch_dists is None or mode._dataset_anchors is None:
        return True

    n_dataset = int(mode._batch_dists.shape[1])
    n_anchors = int(mode._dataset_anchors.shape[0])

    if mode.dataset_source_type == "simulated" and mode.simulated_dataset_kind == "BC":
        expected = 2 * n_anchors

        if n_dataset != expected:
            mode.host._set_msg(
                f"Dataset BC inválido: esperado front+rear "
                f"({expected} colunas), recebido {n_dataset}"
            )
            return False

        return True

    if n_dataset != n_anchors:
        mode.host._set_msg(
            f"Incompatibilidade: dataset possui {n_dataset} colunas de âncora, "
            f"mas o layout possui {n_anchors} âncoras"
        )
        return False

    return True


def apply_route_to_dataset_mode(mode, route_path: str) -> bool:
    """
    Carrega rota de referência e aplica no DatasetMode.
    """
    try:
        pts = load_route_waypoints_from_json(route_path)

        mode._route_waypoints = pts.copy()
        mode._reference_route_display = pts.copy()
        mode._reference_route_dense = pts.copy()
        mode._route_label = os.path.basename(route_path)

        return True

    except ValueError as e:
        print(f"[DATASET] erro ao carregar rota: {e}")
        mode.host._set_msg(f"Erro ao carregar rota: {str(e)}")
        return False

    except Exception as e:
        print(f"[DATASET] erro ao carregar rota: {e}")
        mode.host._set_msg("Erro ao carregar rota")
        return False


def apply_map_to_dataset_mode(mode, map_path: str) -> bool:
    """
    Carrega mapa e aplica no DatasetMode.
    """
    try:
        mode._map_env, mode._map_label = load_map_from_json(map_path)
        return True

    except Exception as e:
        print(f"[DATASET] erro ao carregar mapa: {e}")
        mode.host._set_msg("Erro ao carregar mapa")
        return False