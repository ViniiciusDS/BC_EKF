# src/ui/algo_modes/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass
class AlgoActions:
    go_to_menu: bool = False
    quite_app: bool = False

class AlgoMode(Protocol):
    def handle_events(self, events) -> AlgoActions: ...
    def update(self, dt: float) -> None: ...
    def draw(self) -> None: ...
    def close(self) -> None: ...