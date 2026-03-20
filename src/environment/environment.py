# src/environment.py
# Representação do ambiente 2D + utilidades geométricas (LOS, reflexão)
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
import pygame as pg
import json
import os

from .noise_zones import normalize_noise_zones
from .noise_profiles import noise_profile_color, noise_profile_label


Vec2 = np.ndarray  # shape (2,)

@dataclass
class Obstacle:
    """
    Segmento 2D que representa, por exemplo, uma parede.
    Parâmetros físicos simples para a camada 1 (LOS + 1 reflexão).
    """
    p0: Vec2                 # endpoint 1 (x, y)
    p1: Vec2                 # endpoint 2 (x, y)
    material: str = "metal"  # rótulo apenas informativo
    R: float = 0.8           # refletividade efetiva (0..1)
    A: float = 0.2           # absorção efetiva (0..1); não usamos no passo 1
    sigma_diff: float = 0.05 # ruído extra em NLOS (m)
    nlos_bias: float = 0.25  # viés típico em NLOS (m)
    p_drop: float = 0.0      # prob. de dropout (desligado aqui)

    # Inicializador explícito para converter listas em np.array
    def __init__(self, p0, p1, material="wall"):
        self.p0 = np.array(p0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.material = material



class Environment:
    """
    Apenas uma coleção de obstáculos.
    """
    def __init__(self, obstacles: List[Obstacle] | None = None):
        self.obstacles: List[Obstacle] = obstacles or []
        # bounds
        self.bounds = None # (xmin, xmax, ymin, ymax) ou None

        self.noise_zones: list[dict] = []  # lista de zonas de ruído (dicionários normalizados)

    
    # Adiciona um obstáculo
    def add(self, obs: Obstacle) -> None:
        self.obstacles.append(obs)

    # clear
    def clear(self) -> None:
        self.obstacles.clear()
        self.noise_zones.clear()

    # --- serialization to dict/JSON ---
    def to_dict(self) -> dict:
        """Converte o ambiente para um dicionário serializável em JSON."""
        obs_list = []
        for obs in self.obstacles:
            obs_list.append({
                "p0": obs.p0.tolist(),
                "p1": obs.p1.tolist(),
                "material": obs.material,
                "noise_zones": list(self.noise_zones),
            })
        data = {"obstacles": obs_list}
        if self.bounds is not None:
            data["bounds"] = list(self.bounds)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Environment":
        """Cria um Environment a partir de um dicionário (carregado de JSON)."""
        env = cls()
        for o in data.get("obstacles", []):
            p0 = o.get("p0", [0.0, 0.0])
            p1 = o.get("p1", [0.0, 0.0])
            material = o.get("material", "wall")
            env.add(Obstacle(np.array(p0, dtype=float),
                                np.array(p1, dtype=float),
                                material=material))
        if "bounds" in data:
            env.bounds = tuple(data["bounds"])

        env.noise_zones = normalize_noise_zones(data.get("noise_zones", []))
        
        return env

    def save_json(self, filepath: str):
        """Salva o ambiente em um arquivo JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> "Environment":
        """Carrega um ambiente de um arquivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
    
    def set_noise_zones(self, zones):
        '''Define as zonas de ruído do ambiente, normalizando a entrada.'''
        self.noise_zones = normalize_noise_zones(zones)


    def add_noise_zone(self, zone: dict):
        '''Adiciona uma zona de ruído ao ambiente, normalizando a entrada.'''
        normalized = normalize_noise_zones([zone])
        if normalized:
            self.noise_zones.append(normalized[0])


    def clear_noise_zones(self):
        '''Remove todas as zonas de ruído do ambiente.'''
        self.noise_zones = []

# ===============================
# Utilidades geométricas básicas
# ===============================

def _ccw(a: Vec2, b: Vec2, c: Vec2) -> float:
    """Área assinada (2x) do triângulo ABC (sinal dá orientação)."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def segments_intersect(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> bool:
    """
    Interseção de segmentos fechados [a0,a1] e [b0,b1].
    Robusto o suficiente para uso em tempo real.
    """
    d1 = _ccw(a0, a1, b0)
    d2 = _ccw(a0, a1, b1)
    d3 = _ccw(b0, b1, a0)
    d4 = _ccw(b0, b1, a1)

    if (d1 == 0 and _on_segment(a0, a1, b0)) or \
       (d2 == 0 and _on_segment(a0, a1, b1)) or \
       (d3 == 0 and _on_segment(b0, b1, a0)) or \
       (d4 == 0 and _on_segment(b0, b1, a1)):
        return True

    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)

def _on_segment(a: Vec2, b: Vec2, p: Vec2) -> bool:
    """Retorna True se p está sobre o segmento [a,b]."""
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

def reflect_point_about_line(P: Vec2, L0: Vec2, L1: Vec2) -> Vec2:
    """
    Reflete o ponto P em relação à linha infinita que passa por L0-L1.
    (Usado para criar o 'ponto imagem' da âncora.)
    """
    v = L1 - L0
    if np.allclose(v, 0.0):
        return P.copy()
    v = v / np.linalg.norm(v)
    # projeta (P-L0) no vetor da linha
    proj = L0 + v * np.dot(P - L0, v)
    # espelha em relação ao ponto projetado
    return proj + (proj - P)

def los_blocked(tag_xy: Vec2, anchor_xy: Vec2, env: Environment | None) -> bool:
    """
    True se QUALQUER obstáculo intersecta o segmento tag->âncora.
    """
    if env is None:
        return False
    a0 = np.asarray(tag_xy, float)
    a1 = np.asarray(anchor_xy, float)
    for o in env.obstacles:
        if segments_intersect(a0, a1, o.p0, o.p1):
            return True
    return False

def draw_environment(surface, cam: Camera, env):
    if env is None:
        return

    draw_noise_zones(surface, cam, env)
    
    # mapeia material -> cor
    COLOR_BY_MATERIAL = {
        "metal":   (80,  80,  80),
        "wall":    (120, 80,  80),   # parede alvenaria
        "glass":   (80,  120, 160),
        "human":   (160, 120, 80),
    }

    for obs in env.obstacles:
        p0s = cam.world_to_screen(*obs.p0)
        p1s = cam.world_to_screen(*obs.p1)

        mat = getattr(obs, "material", "metal")
        col = COLOR_BY_MATERIAL.get(mat, (100, 100, 100))

        pg.draw.line(surface, col, p0s, p1s, 3)
        pg.draw.circle(surface, col, p0s, 4)
        pg.draw.circle(surface, col, p1s, 4)

    

def draw_noise_zones(surface, cam: Camera, env):
    '''Desenha as zonas de ruído do ambiente como retângulos semitransparentes, usando as cores definidas pelos perfis de ruído.'''
    if env is None:
        return

    zones = getattr(env, "noise_zones", None)
    if not zones:
        return

    for zone in zones:
        color = noise_profile_color(zone.get("profile"))
        if color is None:
            continue

        x = float(zone["x"])
        y = float(zone["y"])
        w = float(zone["w"])
        h = float(zone["h"])

        p0 = cam.world_to_screen(x, y)
        p1 = cam.world_to_screen(x + w, y + h)

        left = min(p0[0], p1[0])
        top = min(p0[1], p1[1])
        width = abs(p1[0] - p0[0])
        height = abs(p1[1] - p0[1])

        if width <= 0 or height <= 0:
            continue

        overlay = pg.Surface((width, height), pg.SRCALPHA)
        overlay.fill(color)
        surface.blit(overlay, (left, top))

        border_color = color[:3]
        pg.draw.rect(surface, border_color, (left, top, width, height), 2)

