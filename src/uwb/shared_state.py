# src/uwb/shared_state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

from src.uwb.uwb_sim import UwbSimPipeline, UwbSimConfig
from src.uwb.node_params import NodeParams
from src.uwb.twr_protocols import TWRMode
from src.uwb.node_params_serialization import node_params_to_dict, dict_to_node_params

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
            pipeline=UwbSimPipeline.from_config(cfg, seed=seed, env=None),
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
        old_env = getattr(old, "env", None)
        self.pipeline = UwbSimPipeline.from_sim_cfg(sim_cfg, seed=self.seed, env=old_env)
        # reaplica params
        self.pipeline.tag_params = self.tag_params
        self.pipeline.anchor_params = dict(self.anchor_params)
        self.pipeline.default_params = old.default_params

    def to_dict(self) -> dict:
        """Exporta estado completo para JSON."""
        return {
            "anchors_xy": list(self.anchors_xy),
            "tag_params": node_params_to_dict(self.tag_params),
            "anchor_params": {
                i: node_params_to_dict(p)
                for i, p in self.anchor_params.items()
            },
            "seed": self.seed,
            "meta": {
                "count": len(self.anchors_xy),
                "timestamp": datetime.now().isoformat(),
            }
        }
    
    @staticmethod
    def from_dict(data: dict) -> "SharedUwbState":
        """Carrega estado completo de JSON."""
        state = SharedUwbState.make_default(seed=data.get("seed", 123))
        state.anchors_xy = data.get("anchors_xy", [])
        
        if "tag_params" in data:
            state.tag_params = dict_to_node_params(data["tag_params"])
        
        if "anchor_params" in data:
            state.anchor_params = {
                int(k): dict_to_node_params(v)
                for k, v in data["anchor_params"].items()
            }
        
        state.sync_pipeline_from_state()
        return state