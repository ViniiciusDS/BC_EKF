# src/uwb/dataset_export.py
from __future__ import annotations
import os, json
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

TagMode = Literal["front", "rear", "mid"]

@dataclass
class DatasetConfig:
    out_dir: str = "datasets"
    name: str = "uwb_dataset"
    tag_mode: TagMode = "mid"
    dt_sample: float = 0.05          # amostra a cada 50 ms
    max_samples: int = 5000          # segurança
    include_time: bool = False       # se True, prefixa t na linha

def _tag_xy_from_pose(xytheta: np.ndarray, baseline: float, mode: TagMode) -> np.ndarray:
    """xytheta = [x,y,theta]. Retorna [x,y] da tag escolhida."""
    x, y, th = float(xytheta[0]), float(xytheta[1]), float(xytheta[2])
    c, s = np.cos(th), np.sin(th)
    half = 0.5 * baseline
    if mode == "mid":
        return np.array([x, y], dtype=float)
    if mode == "front":
        return np.array([x + half*c, y + half*s], dtype=float)
    # rear
    return np.array([x - half*c, y - half*s], dtype=float)

def export_uwb_dataset_from_sim(
    sim,
    waypoints: np.ndarray,
    shared_uwb,
    cfg: DatasetConfig,
    *,
    v_cmd: float = 0.25,
    w_cmd: float = 0.8,
) -> str:
    """
    Roda a rota automaticamente e salva dataset no formato:
      r0 s0 r1 s1 ...
    Usa UWB bruto do pipeline (com ruído, bias etc.) e salva o sigma por âncora.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    out_txt = os.path.join(cfg.out_dir, cfg.name if cfg.name.endswith(".txt") else (cfg.name + ".txt"))
    out_meta = os.path.splitext(out_txt)[0] + "_meta.json"

    # sanity
    if waypoints is None or len(waypoints) < 2:
        raise ValueError("waypoints inválidos")

    # prepara loop
    t = 0.0
    next_sample_t = 0.0
    n_samples = 0

    # garante que anchors no sim estejam sincronizadas
    anchors = shared_uwb.anchors_np3()  # (3,N)
    sim.anchors = anchors
    nA = anchors.shape[1]
    if nA == 0:
        raise ValueError("Sem âncoras")

    baseline = float(getattr(sim, "baseline", getattr(sim, "L", 0.65)))
    l = float(getattr(sim, "l", 0.325))

    # vamos usar o controlador existente
    from src.control.waypoint_controller import waypoint_controller

    # escreve
    with open(out_txt, "w", encoding="utf-8") as f:
        while n_samples < cfg.max_samples:
            # controle olhando o true
            x_for_ctrl = getattr(sim, "x_true", sim.x_est)
            v, w, wp_idx = waypoint_controller(x_for_ctrl, waypoints, getattr(sim, "wp_idx", 0), v_max=v_cmd, w_max=w_cmd)
            setattr(sim, "wp_idx", wp_idx)

            # passo do sim
            sim.step(v, w)

            t += float(sim.dt)

            # condição de fim (acabou rota)
            if waypoints is not None and wp_idx >= len(waypoints):
                break

            # amostra
            if t + 1e-9 >= next_sample_t:
                # pega pose true e calcula tag escolhida
                x_true = getattr(sim, "x_true", sim.x_est)
                tag_xy = _tag_xy_from_pose(np.array(x_true, dtype=float), baseline, cfg.tag_mode)

                # mede UWB do pipeline diretamente (bruto)
                ranges_m, sigmas_m = shared_uwb.pipeline.measure_ranges_and_sigmas(
                    x_state=x_true, anchors=anchors, l=l, tag=cfg.tag_mode
                )

                # ranges_m e sigmas_m devem ter shape (N,)
                if len(ranges_m) != nA:
                    # mantém robusto
                    ranges_m = np.resize(np.array(ranges_m, dtype=float), (nA,))
                    sigmas_m = np.resize(np.array(sigmas_m, dtype=float), (nA,))

                if cfg.include_time:
                    row = [f"{t:.6f}"]
                else:
                    row = []

                for i in range(nA):
                    row.append(f"{float(ranges_m[i]):.4f}")
                    row.append(f"{float(sigmas_m[i]):.4f}")

                f.write("\t".join(row) + "\n")

                n_samples += 1
                next_sample_t += cfg.dt_sample

    meta = {
        "anchors_xy": [(float(x), float(y)) for (x, y) in shared_uwb.anchors_xy],
        "tag_mode": cfg.tag_mode,
        "dt_sample": cfg.dt_sample,
        "sim_dt": float(sim.dt),
        "seed": int(getattr(shared_uwb, "seed", 0)),
        "n_anchors": int(nA),
        "n_samples": int(n_samples),
        "waypoints": waypoints.tolist(),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_txt