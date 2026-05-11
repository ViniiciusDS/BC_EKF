from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class BcPrepResult:
    ok: bool
    message: str = ""

    bc_ekf_data: Optional[dict] = None

    batch_dists: Optional[np.ndarray] = None
    batch_devs: Optional[np.ndarray] = None

    dataset_route: Optional[np.ndarray] = None
    real_odom_path: Optional[np.ndarray] = None

    real_range_matrix: Optional[np.ndarray] = None
    real_sigma_matrix: Optional[np.ndarray] = None
    real_timestamps: Optional[list[float]] = None
    real_anchor_ids: Optional[list[int]] = None


def route_xy_to_pose_xytheta(route_xy):
    """
    Converte uma rota XY em poses XYTheta.
    A orientação é calculada pela direção entre pontos consecutivos.
    """
    route_xy = np.asarray(route_xy, dtype=float)

    if route_xy.ndim != 2 or route_xy.shape[1] < 2 or len(route_xy) == 0:
        raise ValueError("Rota inválida para BC-EKF")

    poses = np.zeros((len(route_xy), 3), dtype=float)
    poses[:, :2] = route_xy[:, :2]

    if len(route_xy) == 1:
        return poses

    for i in range(len(route_xy) - 1):
        dx = route_xy[i + 1, 0] - route_xy[i, 0]
        dy = route_xy[i + 1, 1] - route_xy[i, 1]
        poses[i, 2] = np.arctan2(dy, dx)

    poses[-1, 2] = poses[-2, 2]
    return poses


def pose_xytheta_to_vw(poses_xytheta, T):
    """
    Converte poses XYTheta em odometria [v, w] usando diferenças finitas.
    Retorna matriz 2 x M.
    """
    poses = np.asarray(poses_xytheta, dtype=float)

    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("Poses inválidas para BC-EKF")

    M = poses.shape[0]
    odom = np.zeros((2, M), dtype=float)

    if M < 2:
        return odom

    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T inválido para conversão pose -> odometria")

    for k in range(1, M):
        dx = poses[k, 0] - poses[k - 1, 0]
        dy = poses[k, 1] - poses[k - 1, 1]
        ds = float(np.hypot(dx, dy))

        dtheta = poses[k, 2] - poses[k - 1, 2]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        odom[0, k] = ds / T
        odom[1, k] = dtheta / T

    return odom


def guess_sampled_traj_sidecar(dataset_path: str) -> str | None:
    """
    Tenta encontrar o arquivo sidecar de trajetória amostrada associado
    ao dataset, seguindo a convenção:
        dataset_xxx.txt -> dataset_xxx_traj.csv
    """
    if not dataset_path:
        return None

    base, _ = os.path.splitext(dataset_path)
    candidate = base + "_traj.csv"
    return candidate if os.path.exists(candidate) else None


def load_sampled_traj_csv(path: str):
    """
    Carrega um CSV de trajetória amostrada.

    Formatos aceitos:
    1) CSV com cabeçalho: x,y,theta
    2) texto separado por espaço: t x y theta
    """
    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)

        if "," in sample:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                    th = float(row["theta"])
                except Exception:
                    continue
                rows.append([x, y, th])
        else:
            _header = f.readline().strip().split()
            for line in f:
                line = line.strip()
                if not line:
                    continue

                vals = line.split()
                if len(vals) < 4:
                    continue

                try:
                    _, x, y, th = map(float, vals[:4])
                except Exception:
                    continue

                rows.append([x, y, th])

    if not rows:
        raise ValueError(f"Trajetória amostrada vazia: {path}")

    return np.asarray(rows, dtype=float)


def is_bc_uwb_rows(rows) -> bool:
    if not rows:
        return False

    sample = rows[0]
    keys = {str(k).strip().lower() for k in sample.keys()}

    return (
        "range_front" in keys
        and "sigma_front" in keys
        and "range_rear" in keys
        and "sigma_rear" in keys
    )


def expand_bc_uwb_rows_if_needed(rows):
    """
    Se o arquivo UWB estiver em formato BC:
        timestamp, anchor_id, range_front, sigma_front, range_rear, sigma_rear

    expande para duas linhas simples:
        tag=front
        tag=rear

    Se já estiver em formato simples, devolve como está.
    """
    if not rows:
        return rows

    if not is_bc_uwb_rows(rows):
        return rows

    expanded = []

    for row in rows:
        ts = row.get("timestamp", row.get("timestamp_s", ""))
        aid = row.get("anchor_id", row.get("anchor", row.get("id", "")))

        expanded.append({
            "timestamp": ts,
            "anchor_id": aid,
            "range": row["range_front"],
            "sigma": row["sigma_front"],
            "tag": "front",
        })

        expanded.append({
            "timestamp": ts,
            "anchor_id": aid,
            "range": row["range_rear"],
            "sigma": row["sigma_rear"],
            "tag": "rear",
        })

    return expanded


def prepare_simulated_bc_ekf_data(
    *,
    batch_dists,
    batch_devs,
    dataset_anchors,
    dataset_path: str | None,
    dataset_route,
    route_waypoints,
    cfg: Any,
    resample_polyline_fn: Callable[[Any, int], Any],
) -> BcPrepResult:
    """
    Prepara um dataset simulado do tipo BC para o algoritmo bc_ekf.

    Entrada:
        batch_dists: M x 2N, colunas [A0_front, A0_rear, A1_front, A1_rear, ...]
        batch_devs:  M x 2N ou None
        dataset_anchors: N x 3

    Saída:
        bc_ekf_data com z_hist 2N x M;
        batch_dists reduzido para M x N usando somente tag frontal;
        batch_devs reduzido para M x N.
    """
    if batch_dists is None:
        return BcPrepResult(False, "BC-EKF simulado: distâncias não carregadas")

    if dataset_anchors is None:
        return BcPrepResult(False, "BC-EKF simulado requer âncoras carregadas")

    full_dists = np.asarray(batch_dists, dtype=float)
    full_devs = np.asarray(batch_devs, dtype=float) if batch_devs is not None else None

    if full_dists.ndim != 2:
        return BcPrepResult(False, "Dataset BC inválido: matriz de distâncias inválida")

    M, n_cols = full_dists.shape

    if n_cols % 2 != 0:
        return BcPrepResult(False, "Dataset BC inválido: colunas devem ser front/rear")

    anchors = np.asarray(dataset_anchors, dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] < 2:
        return BcPrepResult(False, "Âncoras inválidas para BC-EKF")

    n_anchors = int(anchors.shape[0])

    if n_cols != 2 * n_anchors:
        return BcPrepResult(
            False,
            f"Dataset BC incompatível: {n_cols} colunas para {n_anchors} âncoras",
        )

    poses = None
    traj_sidecar = guess_sampled_traj_sidecar(dataset_path) if dataset_path else None

    if traj_sidecar is not None:
        try:
            poses = load_sampled_traj_csv(traj_sidecar)
        except Exception:
            poses = None

    if poses is None:
        route = None

        if dataset_route is not None:
            route = np.asarray(dataset_route, dtype=float)
        elif route_waypoints is not None:
            route = np.asarray(route_waypoints, dtype=float)

        if route is None or route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
            return BcPrepResult(False, "BC-EKF requer sidecar *_traj.csv ou rota válida")

        route_xy = route[:, :2]

        if len(route_xy) != M:
            route_xy = resample_polyline_fn(route_xy, M)

        if route_xy is None:
            return BcPrepResult(False, "Falha ao reamostrar rota para BC-EKF")

        poses = route_xy_to_pose_xytheta(route_xy)

    poses = np.asarray(poses, dtype=float)

    if poses.ndim != 2 or poses.shape[1] < 3:
        return BcPrepResult(False, "BC-EKF inválido: trajetória precisa ter x, y, theta")

    poses = poses[:, :3]

    if poses.shape[0] != M:
        return BcPrepResult(
            False,
            f"BC-EKF inválido: trajetória tem {poses.shape[0]} amostras, dataset tem {M}",
        )

    T = float(getattr(cfg, "TIME_STEP", 0.05))
    if not np.isfinite(T) or T <= 0:
        T = 0.05

    odometry_noisy = pose_xytheta_to_vw(poses, T)
    z_hist = full_dists.T

    sigma_uwb = None
    if full_devs is not None:
        vals = full_devs[np.isfinite(full_devs)]
        if vals.size > 0:
            sigma_uwb = float(np.nanmedian(vals))

    if sigma_uwb is None or not np.isfinite(sigma_uwb) or sigma_uwb <= 0:
        sigma_uwb = float(getattr(cfg, "UWB_NOISE_STD", 0.05))

    bc_ekf_data = {
        "T": T,
        "odometry_noisy": odometry_noisy,
        "z_hist": z_hist,
        "l": float(getattr(cfg, "TAG_BASELINE", 0.25)) / 2.0,
        "z_c": float(getattr(cfg, "TAG_HEIGHT", 0.20)),
        "sigma_uwb": sigma_uwb,
        "x0": np.asarray(poses[0], dtype=float).reshape(3,),
    }

    reduced_dists = full_dists[:, 0::2]

    if full_devs is not None:
        reduced_devs = full_devs[:, 0::2]
    else:
        reduced_devs = np.full_like(reduced_dists, sigma_uwb, dtype=float)

    return BcPrepResult(
        True,
        bc_ekf_data=bc_ekf_data,
        batch_dists=reduced_dists,
        batch_devs=reduced_devs,
        dataset_route=poses.copy(),
    )


def prepare_real_bc_ekf_data(
    *,
    encoder_samples,
    uwb_rows,
    dataset_anchors,
    cfg: Any,
    build_pose_path_fn: Callable[[Any], Any],
    resample_pose_path_fn: Callable[[Any, int], Any],
    apply_initial_pose_fn: Callable[[Any], Any],
) -> BcPrepResult:
    """
    Prepara o BC-EKF para dataset real.

    Dependências específicas do DatasetMode entram como callbacks:
    - build_pose_path_fn
    - resample_pose_path_fn
    - apply_initial_pose_fn

    Isso evita mover toda a pipeline real de uma vez.
    """
    if dataset_anchors is None:
        return BcPrepResult(False, "BC-EKF real requer âncoras carregadas")

    if not uwb_rows:
        return BcPrepResult(False, "BC-EKF real: UWB vazio")

    anchors = np.asarray(dataset_anchors, dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] < 2:
        return BcPrepResult(False, "Âncoras inválidas para BC-EKF real")

    n_anchors = int(anchors.shape[0])

    ts_sorted_all = sorted({float(r["timestamp"]) for r in uwb_rows})
    M_all = len(ts_sorted_all)

    if M_all == 0:
        return BcPrepResult(False, "BC-EKF real: nenhum timestamp válido")

    z_hist = np.full((2 * n_anchors, M_all), np.nan, dtype=float)
    sigma_vals = []

    ts_to_idx = {t: k for k, t in enumerate(ts_sorted_all)}

    for row in uwb_rows:
        try:
            t = float(row["timestamp"])
            aid = int(row["anchor_id"])
        except Exception:
            continue

        if aid < 0 or aid >= n_anchors:
            continue

        k = ts_to_idx[t]
        j = aid

        try:
            z_hist[2 * j, k] = float(row["range_front"])
            z_hist[2 * j + 1, k] = float(row["range_rear"])

            sf = float(row["sigma_front"])
            sr = float(row["sigma_rear"])

            if np.isfinite(sf):
                sigma_vals.append(sf)
            if np.isfinite(sr):
                sigma_vals.append(sr)

        except Exception:
            continue

    complete_mask = ~np.isnan(z_hist).any(axis=0)

    if not np.any(complete_mask):
        return BcPrepResult(False, "BC real inválido: nenhum timestamp UWB completo")

    z_hist = z_hist[:, complete_mask]
    ts_sorted = np.asarray(ts_sorted_all, dtype=float)[complete_mask]
    M = int(z_hist.shape[1])

    poses = build_pose_path_fn(encoder_samples)

    if poses is None or len(poses) == 0:
        return BcPrepResult(False, "BC-EKF real: odometria vazia")

    poses = resample_pose_path_fn(poses, M)
    poses = apply_initial_pose_fn(poses)
    poses = np.asarray(poses, dtype=float)

    if poses.ndim != 2 or poses.shape[1] < 3:
        return BcPrepResult(False, "BC-EKF real: poses inválidas")

    poses = poses[:, :3]

    if M > 1:
        dt = np.diff(ts_sorted)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        T = float(np.median(dt)) if dt.size > 0 else float(getattr(cfg, "TIME_STEP", 0.05))
    else:
        T = float(getattr(cfg, "TIME_STEP", 0.05))

    if not np.isfinite(T) or T <= 0:
        T = float(getattr(cfg, "TIME_STEP", 0.05))

    odometry_noisy = pose_xytheta_to_vw(poses, T)

    sigma_uwb = (
        float(np.nanmedian(np.asarray(sigma_vals, dtype=float)))
        if sigma_vals
        else float(getattr(cfg, "UWB_NOISE_STD", 0.05))
    )

    if not np.isfinite(sigma_uwb) or sigma_uwb <= 0:
        sigma_uwb = float(getattr(cfg, "UWB_NOISE_STD", 0.05))

    bc_ekf_data = {
        "T": T,
        "odometry_noisy": odometry_noisy,
        "z_hist": z_hist,
        "l": float(getattr(cfg, "TAG_BASELINE", 0.25)) / 2.0,
        "z_c": float(getattr(cfg, "TAG_HEIGHT", 0.20)),
        "sigma_uwb": sigma_uwb,
        "x0": np.asarray(poses[0], dtype=float).reshape(3,),
    }

    batch_dists = z_hist.T[:, 0::2]
    batch_devs = np.full_like(batch_dists, sigma_uwb, dtype=float)

    return BcPrepResult(
        True,
        bc_ekf_data=bc_ekf_data,
        batch_dists=batch_dists,
        batch_devs=batch_devs,
        dataset_route=poses.copy(),
        real_odom_path=poses.copy(),
        real_range_matrix=batch_dists.copy(),
        real_sigma_matrix=batch_devs.copy(),
        real_timestamps=ts_sorted.tolist(),
        real_anchor_ids=list(range(n_anchors)),
    )