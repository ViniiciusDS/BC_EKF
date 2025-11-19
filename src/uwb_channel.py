# src/uwb_channel.py
# Camada 1 do canal UWB: LOS + 1 reflexão especular (leve e determinístico)
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from .environment import Environment, reflect_point_about_line, los_blocked

def uwb_range_measure(
    tag_xy: np.ndarray,
    anchor_xy: np.ndarray,
    base_sigma: float,
    env: Environment | None,
    rng: np.random.Generator,
    params: Dict | None = None
) -> Tuple[float, Dict]:
    """
    Retorna (distância_simulada_m, meta) entre 'tag' e 'âncora' em 2D.

    Comportamento:
      - Se não há ambiente, retorna LOS com ruído gaussiano (código atual).
      - Se há ambiente:
          * Se LOS livre: d_los + N(0, base_sigma)
          * Se NLOS:
              - Calcula reflexão especular única (espelha a âncora em cada obstáculo);
              - Escolhe a reflexão 'melhor' (distância/ganho);
              - Aplica viés e ruído maiores (parâmetros do obstáculo);
              - (Dropout desabilitado por padrão para não quebrar EKF).

    Parâmetros:
      base_sigma : desvio padrão para LOS (m).
      params     : dict opcional com ajustes:
                   {'layer': 'B', 'disable_dropout': True}
    """
    params = params or {}
    disable_dropout = params.get("disable_dropout", True)

    tag_xy = np.asarray(tag_xy, float)
    anc_xy = np.asarray(anchor_xy, float)

    d_los = float(np.linalg.norm(tag_xy - anc_xy))
    meta = {'mode': 'LOS', 'used': 'direct', 'd_los': d_los}

    # Sem ambiente → modelo simples
    if env is None or len(env.obstacles) == 0:
        return d_los + rng.normal(0, base_sigma), meta

    # LOS livre?
    if not los_blocked(tag_xy, anc_xy, env):
        z = d_los + rng.normal(0, base_sigma)
        return z, meta

    # NLOS (uma reflexão especular)
    meta['mode'] = 'NLOS'
    best = None  # (custo, d_refl, obstáculo)

    for o in env.obstacles:
        A_mirror = reflect_point_about_line(anc_xy, o.p0, o.p1)
        d_refl = float(np.linalg.norm(tag_xy - A_mirror))
        # custo simples: menor é melhor; penaliza materiais pouco reflexivos
        gain = max(o.R, 1e-3)
        cost = d_refl / gain
        if (best is None) or (cost < best[0]):
            best = (cost, d_refl, o)

    if best is None:
        # fallback NLOS brando
        z = d_los + 0.30 + rng.normal(0, max(base_sigma, 0.15))
        meta['used'] = 'fallback'
        return z, meta

    _, d_refl, obs = best
    if (not disable_dropout) and (rng.random() < obs.p_drop):
        # se um dia quisermos dropout, retornamos np.nan
        # (o EKF atual não aceita NaN, então mantemos desabilitado por padrão)
        meta['used'] = f'dropout:{obs.material}'
        return np.nan, meta

    noise = np.hypot(base_sigma, obs.sigma_diff)  # LOS σ + difusão do material
    z = d_refl + obs.nlos_bias + rng.normal(0, noise)
    meta['used'] = f'reflect:{obs.material}'
    meta['d_refl'] = d_refl
    return z, meta
