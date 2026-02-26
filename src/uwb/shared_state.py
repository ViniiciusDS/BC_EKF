# src/uwb/shared_state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.uwb.uwb_sim import UwbSimPipeline, UwbSimConfig
from src.uwb.node_params import NodeParams
from src.uwb.twr_protocols import TWRMode


@dataclass
class SharedUwbState:
    # geometria
    tag_xy: Tuple[float, float] = (0.0, 0.0)
    anchors_xy: List[Tuple[float, float]] = None  # list[(x,y)]

    # parâmetros / configs
    seed: int = 123
    tag_params: NodeParams = None
    anchor_params: Dict[int, NodeParams] = None

    # pipeline compartilhado
    pipeline: UwbSimPipeline = None

    @staticmethod
    def make_default(seed: int = 123) -> "SharedUwbState":
        cfg = UwbSimConfig()
        default = cfg.default_node_params()
        st = SharedUwbState(
            tag_xy=(0.0, 0.0),
            anchors_xy=[],
            seed=int(seed),
            tag_params=default,
            anchor_params={},
            pipeline=UwbSimPipeline.from_config(cfg, seed=seed),
        )
        return st

    def anchors_np(self) -> np.ndarray:
        """(3, N) com z=0"""
        if not self.anchors_xy:
            return np.zeros((3, 0), dtype=float)
        a = np.array(self.anchors_xy, dtype=float)  # (N,2)
        z = np.zeros((a.shape[0], 1), dtype=float)
        anc = np.hstack([a, z]).T  # (3,N)
        return anc
    
    def anchors_np3(self):
        """(3, N) com z=0"""
        if not self.anchors_xy:
            return np.zeros((3,0), dtype=float)
        xy = np.array(self.anchors_xy, dtype=float).T  # 2xN
        z = np.zeros((1, xy.shape[1]), dtype=float)
        return np.vstack([xy, z])

    def reindex_anchor_params(self) -> None:
        """garante que anchor_params tenha chaves 0..N-1"""
        N = len(self.anchors_xy) if self.anchors_xy else 0
        new = {i: self.anchor_params.get(i, NodeParams()) for i in range(N)}
        self.anchor_params = new

    def sync_pipeline_from_state(self) -> None:
        """empurra params atuais para o pipeline"""
        if self.pipeline is None:
            return
        self.reindex_anchor_params()
        self.pipeline.seed = int(self.seed)
        self.pipeline.tag_params = self.tag_params
        self.pipeline.anchor_params = self.anchor_params

    def set_protocol(self, mode: TWRMode) -> None:
        """troca DS/SS sem recriar o estado externo"""
        # recria só o pipeline (mantém configs atuais do ranging_model, params, seed)
        if self.pipeline is None:
            return
        sim_cfg = UwbSimConfig()
        # tenta manter as configs atuais, se existirem
        try:
            # copia canal
            sim_cfg.sigma_los = float(self.pipeline.ranging_model.cfg.sigma_los)
            sim_cfg.sigma_nlos = float(self.pipeline.ranging_model.cfg.sigma_nlos)
            sim_cfg.nlos_prob = float(self.pipeline.ranging_model.cfg.nlos_prob)
            sim_cfg.nlos_bias_mean = float(self.pipeline.ranging_model.cfg.nlos_bias_mean)
            sim_cfg.dropout_prob = float(self.pipeline.ranging_model.cfg.dropout_prob)
            sim_cfg.quantize_step = self.pipeline.ranging_model.cfg.quantize_step
        except Exception:
            pass
        sim_cfg.mode = mode

        old = self.pipeline
        self.pipeline = UwbSimPipeline.from_sim_cfg(sim_cfg, seed=self.seed)
        # reaplica params
        self.pipeline.tag_params = self.tag_params
        self.pipeline.anchor_params = dict(self.anchor_params)
        self.pipeline.default_params = old.default_params
