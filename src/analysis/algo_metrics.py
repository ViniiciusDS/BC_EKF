from __future__ import annotations

from typing import Mapping, Iterable
import numpy as np


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_error_stats(errs) -> dict | None:
    errs = np.asarray(errs, dtype=float).reshape(-1)
    errs = errs[np.isfinite(errs)]

    if errs.size == 0:
        return None

    return {
        "rmse": float(np.sqrt(np.mean(errs ** 2))),
        "mae": float(np.mean(np.abs(errs))),
        "max_err": float(np.max(errs)),
        "min": float(np.min(errs)),
        "q1": float(np.percentile(errs, 25)),
        "median": float(np.percentile(errs, 50)),
        "q3": float(np.percentile(errs, 75)),
        "max": float(np.max(errs)),
        "mean": float(np.mean(errs)),
        "std": float(np.std(errs)),
        "n": int(errs.size),
    }


def compute_dataset_cluster_stats(
    results: Mapping[str, dict],
    algo_order: Iterable[str] | None = None,
) -> dict[str, dict]:
    """
    Dataset mode:
    mede a dispersão das posições estimadas em torno do centróide 2D.
    """
    stats: dict[str, dict] = {}
    ordered = list(algo_order) if algo_order is not None else list(results.keys())

    for algo in ordered:
        res = results.get(algo)
        if not res:
            continue

        pos = res.get("posicoes", None)
        if pos is None:
            continue

        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or pos.shape[1] < 2:
            continue

        valid = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
        pos = pos[valid]
        if len(pos) == 0:
            continue

        centroid = np.mean(pos[:, :2], axis=0)
        errs = np.linalg.norm(pos[:, :2] - centroid[None, :], axis=1)

        st = build_error_stats(errs)
        if st is not None:
            stats[algo] = st

    return stats


def compute_step_vs_truth_stats(
    trail_true,
    trails: Mapping[str, object],
    algo_order: Iterable[str] | None = None,
) -> dict[str, dict] | None:
    """
    Step mode:
    mede erro 2D entre trajetória estimada e ground truth.
    """
    if trail_true is None or len(trail_true) == 0:
        return None

    gt = np.asarray(trail_true, dtype=float)
    if gt.ndim != 2 or gt.shape[1] < 2:
        return None

    stats: dict[str, dict] = {}
    ordered = list(algo_order) if algo_order is not None else list(trails.keys())

    for algo in ordered:
        if algo not in trails:
            continue

        pos = np.asarray(trails[algo], dtype=float)
        if pos.ndim != 2 or len(pos) == 0 or pos.shape[1] < 2:
            continue

        n = min(len(gt), len(pos))
        gt_n = gt[:n, :2]
        pos_n = pos[:n, :2]

        valid = np.isfinite(pos_n[:, 0]) & np.isfinite(pos_n[:, 1])
        gt_n = gt_n[valid]
        pos_n = pos_n[valid]
        if len(pos_n) == 0:
            continue

        errs = np.linalg.norm(pos_n - gt_n, axis=1)

        st = build_error_stats(errs)
        if st is not None:
            stats[algo] = st

    return stats if stats else None


def rank_algorithms(
    stats: Mapping[str, dict] | None,
    *,
    selected: Mapping[str, bool] | None = None,
    metric: str = "rmse",
) -> list[tuple[str, dict]]:
    """
    Ranking crescente: menor métrica = melhor.
    """
    if not stats:
        return []

    rows: list[tuple[str, dict]] = []
    for algo, st in stats.items():
        if selected is not None and not selected.get(algo, False):
            continue

        score = _safe_float(st.get(metric, np.inf), np.inf)
        if not np.isfinite(score):
            continue

        rows.append((algo, st))

    rows.sort(
        key=lambda item: (
            _safe_float(item[1].get(metric, np.inf), np.inf),
            _safe_float(item[1].get("mae", np.inf), np.inf),
            _safe_float(item[1].get("max_err", np.inf), np.inf),
            item[0],
        )
    )
    return rows


def build_ranking_summary(
    stats: Mapping[str, dict] | None,
    *,
    selected: Mapping[str, bool] | None = None,
    metric: str = "rmse",
    top_k: int = 3,
) -> list[dict]:
    ranked = rank_algorithms(stats, selected=selected, metric=metric)
    out = []

    for i, (algo, st) in enumerate(ranked[:top_k], start=1):
        out.append(
            {
                "rank": i,
                "algo": algo,
                "metric": metric,
                "score": _safe_float(st.get(metric, np.nan), np.nan),
                "mae": _safe_float(st.get("mae", np.nan), np.nan),
                "max_err": _safe_float(st.get("max_err", np.nan), np.nan),
                "n": int(st.get("n", 0) or 0),
            }
        )
    return out