from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import json
import os
import glob

from src.uwb.node_params import NodeParams
from src.uwb.twr_protocols import TWRMode


@dataclass
class NodeParamsDTO:
    ppm: float = 0.0
    tx_ns: float = 0.0
    rx_ns: float = 0.0
    bias_m: float = 0.0

    @staticmethod
    def from_nodeparams(p: NodeParams) -> "NodeParamsDTO":
        '''Cria NodeParamsDTO a partir de NodeParams.'''
        return NodeParamsDTO(
            ppm=float(p.clock.drift_ppm),
            tx_ns=float(p.ant.tx_ns),
            rx_ns=float(p.ant.rx_ns),
            bias_m=float(p.range_bias_m),
        )

    def to_nodeparams(self) -> NodeParams:
        '''Cria NodeParams a partir deste DTO.'''
        p = NodeParams()
        p.clock.drift_ppm = float(self.ppm)
        p.ant.tx_ns = float(self.tx_ns)
        p.ant.rx_ns = float(self.rx_ns)
        p.range_bias_m = float(self.bias_m)
        return p


@dataclass
class ExperimentConfig:
    seed: int
    dt: float
    tag_xy: Tuple[float, float]
    anchors_xy: List[Tuple[float, float]]
    tag_params: NodeParamsDTO
    anchor_params: Dict[int, NodeParamsDTO]
    protocol: str  # "DS-TWR" or "SS-TWR" (usa .value do enum)
    ranging_cfg: Dict[str, Any]

    # --------- IO ---------

    def to_dict(self) -> Dict[str, Any]:
        '''Converte este ExperimentConfig para um dicionário serializável.'''
        return {
            "seed": int(self.seed),
            "dt": float(self.dt),
            "tag_xy": [float(self.tag_xy[0]), float(self.tag_xy[1])],
            "anchors_xy": [[float(x), float(y)] for (x, y) in self.anchors_xy],
            "tag_params": {
                "ppm": float(self.tag_params.ppm),
                "tx_ns": float(self.tag_params.tx_ns),
                "rx_ns": float(self.tag_params.rx_ns),
                "bias_m": float(self.tag_params.bias_m),
            },
            "anchor_params": {
                str(i): {
                    "ppm": float(p.ppm),
                    "tx_ns": float(p.tx_ns),
                    "rx_ns": float(p.rx_ns),
                    "bias_m": float(p.bias_m),
                }
                for i, p in self.anchor_params.items()
            },
            "protocol": str(self.protocol),
            "ranging_cfg": dict(self.ranging_cfg),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        '''Cria ExperimentConfig a partir de um dicionário (ex: carregado de JSON).'''
        tp = d.get("tag_params", {})
        ap = d.get("anchor_params", {})
        return ExperimentConfig(
            seed=int(d.get("seed", 123)),
            dt=float(d.get("dt", 0.10)),
            tag_xy=(float(d.get("tag_xy", [0.0, 0.0])[0]), float(d.get("tag_xy", [0.0, 0.0])[1])),
            anchors_xy=[(float(x), float(y)) for (x, y) in d.get("anchors_xy", [])],
            tag_params=NodeParamsDTO(
                ppm=float(tp.get("ppm", 0.0)),
                tx_ns=float(tp.get("tx_ns", 0.0)),
                rx_ns=float(tp.get("rx_ns", 0.0)),
                bias_m=float(tp.get("bias_m", 0.0)),
            ),
            anchor_params={
                int(k): NodeParamsDTO(
                    ppm=float(v.get("ppm", 0.0)),
                    tx_ns=float(v.get("tx_ns", 0.0)),
                    rx_ns=float(v.get("rx_ns", 0.0)),
                    bias_m=float(v.get("bias_m", 0.0)),
                )
                for k, v in ap.items()
            },
            protocol=str(d.get("protocol", TWRMode.DS_TWR.value)),
            ranging_cfg=dict(d.get("ranging_cfg", {})),
        )

    def save_json(self, path: str) -> str:
        '''Salva este ExperimentConfig em um arquivo JSON no caminho especificado. Retorna o caminho salvo.'''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load_json(path: str) -> "ExperimentConfig":
        '''Carrega um ExperimentConfig de um arquivo JSON no caminho especificado.'''
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return ExperimentConfig.from_dict(d)

    @staticmethod
    def find_latest(directory: str = "experiments", pattern: str = "exp_*.json") -> Optional[str]:
        '''Encontra o arquivo mais recente que corresponda ao padrão no diretório especificado.
          Retorna o caminho ou None se não encontrar nada.'''
        os.makedirs(directory, exist_ok=True)
        files = glob.glob(os.path.join(directory, pattern))
        if not files:
            return None
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]

    # --------- Bridge (UI apply/capture) ---------

    @staticmethod
    def capture_from_screen(screen: Any) -> "ExperimentConfig":
        """
        Cria ExperimentConfig a partir do estado atual da tela (duck typing).
        Espera atributos: seed, ranging_cfg, tag_pos, anchors, tag_params, anchor_params, twr_cfg.
        """
        seed = int(getattr(screen, "seed", 123))
        dt = float(screen.ranging_cfg.dt)
        tag_xy = (float(screen.tag_pos[0]), float(screen.tag_pos[1]))
        anchors_xy = [(float(ax), float(ay)) for (ax, ay) in screen.anchors]

        tag_dto = NodeParamsDTO.from_nodeparams(screen.tag_params)
        anc_dto: Dict[int, NodeParamsDTO] = {
            int(i): NodeParamsDTO.from_nodeparams(p) for i, p in screen.anchor_params.items()
        }

        rcfg = {
            "noise_enabled": bool(screen.ranging_cfg.noise_enabled),
            "sigma_los": float(screen.ranging_cfg.sigma_los),
            "sigma_nlos": float(screen.ranging_cfg.sigma_nlos),
            "nlos_prob": float(screen.ranging_cfg.nlos_prob),
            "dropout_prob": float(screen.ranging_cfg.dropout_prob),
            "quantize_step": None if screen.ranging_cfg.quantize_step is None else float(screen.ranging_cfg.quantize_step),
        }

        protocol = str(screen.twr_cfg.mode.value)

        return ExperimentConfig(
            seed=seed,
            dt=dt,
            tag_xy=tag_xy,
            anchors_xy=anchors_xy,
            tag_params=tag_dto,
            anchor_params=anc_dto,
            protocol=protocol,
            ranging_cfg=rcfg,
        )

    def apply_to_screen(self, screen: Any) -> None:
        """
        Aplica este experimento na tela (duck typing).
        Espera que a tela tenha: seed/textbox_seed, ranging_cfg/textbox_dt, tag_pos, anchors,
        tag_params, anchor_params, _set_protocol, _reset_run_same_seed, _reseed_everything.
        """
        # seed
        screen.seed = int(self.seed)
        if hasattr(screen, "textbox_seed"):
            screen.textbox_seed.set_text(str(screen.seed))

        # dt
        screen.ranging_cfg.dt = float(self.dt)
        if hasattr(screen, "textbox_dt"):
            screen.textbox_dt.set_text(f"{screen.ranging_cfg.dt:.2f}")

        # posições
        screen.tag_pos = (float(self.tag_xy[0]), float(self.tag_xy[1]))
        screen.anchors = [(float(x), float(y)) for (x, y) in self.anchors_xy]

        # params
        screen.tag_params = self.tag_params.to_nodeparams()

        # reindexa anchor_params conforme número de âncoras
        screen.anchor_params = {i: NodeParams() for i in range(len(screen.anchors))}
        for i, dto in self.anchor_params.items():
            if i in screen.anchor_params:
                screen.anchor_params[i] = dto.to_nodeparams()

        # ranging cfg
        rc = self.ranging_cfg
        if "noise_enabled" in rc:
            screen.ranging_cfg.noise_enabled = bool(rc["noise_enabled"])
        if "sigma_los" in rc:
            screen.ranging_cfg.sigma_los = float(rc["sigma_los"])
        if "sigma_nlos" in rc:
            screen.ranging_cfg.sigma_nlos = float(rc["sigma_nlos"])
        if "nlos_prob" in rc:
            screen.ranging_cfg.nlos_prob = float(rc["nlos_prob"])
        if "dropout_prob" in rc:
            screen.ranging_cfg.dropout_prob = float(rc["dropout_prob"])
        if "quantize_step" in rc:
            screen.ranging_cfg.quantize_step = rc["quantize_step"]

        # protocolo
        if self.protocol == TWRMode.SS_TWR.value:
            screen._set_protocol(TWRMode.SS_TWR)
        else:
            screen._set_protocol(TWRMode.DS_TWR)

        # reseed/reset (garante reprodutibilidade)
        if hasattr(screen, "_reseed_everything"):
            screen._reseed_everything(screen.seed)
        if hasattr(screen, "_reset_run_same_seed"):
            screen._reset_run_same_seed()
