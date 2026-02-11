# src/uwb/dataset.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import time
from pathlib import Path

@dataclass
class RangeSample:
    anchor_id: int
    r_true_m: float
    r_est_m: Optional[float]          # None se drop
    is_nlos: bool
    dropped: bool
    tof_true_s: Optional[float] = None
    tof_est_s: Optional[float] = None

@dataclass
class UwbFrame:
    t_sim_s: float                    # tempo simulado acumulado
    tag_xy: Tuple[float, float]
    anchors_xy: List[Tuple[float, float]]     # snapshot das âncoras
    protocol: str                     # "DS-TWR", "SS-TWR", etc.
    cfg: Dict[str, Any]               # snapshot config (ranging/protocol/etc)
    ranges: List[RangeSample]         # medições por âncora

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclass dentro de lista já vira dict via asdict
        return d


class UwbDatasetLogger:
    """
    Logger em memória + flush em arquivo (JSONL).
    JSONL = 1 frame por linha, fácil de stream/replay.
    """
    def __init__(self) -> None:
        self.frames: List[UwbFrame] = []
        self.enabled: bool = False
        self._t0_wall: float = time.time()

    def start(self) -> None:
        self.frames.clear()
        self.enabled = True
        self._t0_wall = time.time()

    def stop(self) -> None:
        self.enabled = False

    def add(self, frame: UwbFrame) -> None:
        if self.enabled:
            self.frames.append(frame)

    def save_jsonl(self, filepath: str) -> str:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for fr in self.frames:
                f.write(json.dumps(fr.to_dict(), ensure_ascii=False) + "\n")
        return str(path)

    @staticmethod
    def load_jsonl(filepath: str) -> List[UwbFrame]:
        path = Path(filepath)
        frames: List[UwbFrame] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ranges = [RangeSample(**rs) for rs in d["ranges"]]
                frames.append(
                    UwbFrame(
                        t_sim_s=d["t_sim_s"],
                        tag_xy=tuple(d["tag_xy"]),
                        anchors_xy=[tuple(xy) for xy in d["anchors_xy"]],
                        protocol=d["protocol"],
                        cfg=d["cfg"],
                        ranges=ranges,
                    )
                )
        return frames


class UwbReplay:
    """
    Replay sequencial dos frames gravados.
    """
    def __init__(self, frames: List[UwbFrame]) -> None:
        self.frames = frames
        self.i = 0
        self.playing = False

    def reset(self) -> None:
        self.i = 0

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def step(self) -> Optional[UwbFrame]:
        if self.i >= len(self.frames):
            return None
        fr = self.frames[self.i]
        self.i += 1
        return fr

    def next_if_playing(self) -> Optional[UwbFrame]:
        if not self.playing:
            return None
        return self.step()
