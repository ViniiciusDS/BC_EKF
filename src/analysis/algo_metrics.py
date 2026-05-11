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

def normalize_xy_track(pos):
    pos = np.asarray(pos, dtype=float)

    if pos.ndim == 2 and pos.shape[0] in (2, 3) and pos.shape[1] > pos.shape[0]:
        pos = pos.T

    if pos.ndim != 2 or pos.shape[1] < 2:
        return None

    pos = pos[:, :2]
    valid = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
    pos = pos[valid]

    return pos if len(pos) > 0 else None


def resample_polyline(points, n_samples):
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None

    pts = pts[:, :2]
    n_samples = int(n_samples)

    if n_samples <= 0:
        return None

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    if total <= 1e-12:
        return np.repeat(pts[:1], n_samples, axis=0)

    s_query = np.linspace(0.0, total, n_samples)
    out = np.zeros((len(s_query), 2), dtype=float)

    for i, s in enumerate(s_query):
        j = np.searchsorted(cum, s, side="right") - 1
        j = max(0, min(j, len(seg) - 1))

        s0 = cum[j]
        s1 = cum[j + 1]
        p0 = pts[j]
        p1 = pts[j + 1]

        if abs(s1 - s0) < 1e-12:
            out[i] = p0
        else:
            a = (s - s0) / (s1 - s0)
            out[i] = (1.0 - a) * p0 + a * p1

    return out


def distances_to_polyline(points, polyline):
    pts = normalize_xy_track(points)
    poly = normalize_xy_track(polyline)

    if pts is None or poly is None or len(poly) < 2:
        return np.asarray([], dtype=float)

    all_dists = []

    for i in range(len(poly) - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))

        if denom <= 1e-12:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            t = ((pts - a) @ ab) / denom
            t = np.clip(t, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(pts - proj, axis=1)

        all_dists.append(d)

    return np.min(np.vstack(all_dists), axis=0)


def build_error_stats_with_p95(errs):
    st = build_error_stats(errs)
    if st is None:
        return None

    err = np.asarray(errs, dtype=float).reshape(-1)
    err = err[np.isfinite(err)]

    if err.size == 0:
        return None

    p95 = float(np.percentile(err, 95))
    err95 = err[err <= p95]

    if err95.size == 0:
        err95 = err

    st["p95"] = p95
    st["rmse_p95"] = float(np.sqrt(np.mean(err95 ** 2)))
    st["mae_p95"] = float(np.mean(np.abs(err95)))
    st["max_p95"] = float(np.max(err95))
    st["errors"] = err

    return st


def compute_track_vs_polyline_stats(results, reference_polyline, algo_order=None):
    stats = {}
    ordered = list(algo_order) if algo_order is not None else list(results.keys())

    ref = normalize_xy_track(reference_polyline)
    if ref is None or len(ref) < 2:
        return None

    for algo in ordered:
        res = results.get(algo)
        if not isinstance(res, dict):
            continue

        pos = normalize_xy_track(res.get("posicoes", None))
        if pos is None:
            continue

        err = distances_to_polyline(pos, ref)
        st = build_error_stats_with_p95(err)

        if st is not None:
            st["algo"] = algo
            stats[algo] = st

    return stats if stats else None


def compute_track_vs_synced_reference_stats(results, reference_polyline, algo_order=None):
    stats = {}
    ordered = list(algo_order) if algo_order is not None else list(results.keys())

    ref = normalize_xy_track(reference_polyline)
    if ref is None or len(ref) < 2:
        return None

    for algo in ordered:
        res = results.get(algo)
        if not isinstance(res, dict):
            continue

        pos = normalize_xy_track(res.get("posicoes", None))
        if pos is None or len(pos) < 2:
            continue

        ref_sync = resample_polyline(ref, len(pos))
        if ref_sync is None or len(ref_sync) != len(pos):
            continue

        err = np.linalg.norm(pos - ref_sync, axis=1)
        st = build_error_stats_with_p95(err)

        if st is not None:
            st["algo"] = algo
            stats[algo] = st

    return stats if stats else None


def compute_track_vs_sampled_reference_stats(results, reference_xy, algo_order=None):
    stats = {}
    ordered = list(algo_order) if algo_order is not None else list(results.keys())

    ref = normalize_xy_track(reference_xy)
    if ref is None:
        return None

    for algo in ordered:
        res = results.get(algo)
        if not isinstance(res, dict):
            continue

        pos = normalize_xy_track(res.get("posicoes", None))
        if pos is None:
            continue

        n = min(len(pos), len(ref))
        if n <= 0:
            continue

        err = np.linalg.norm(pos[:n] - ref[:n], axis=1)
        st = build_error_stats_with_p95(err)

        if st is not None:
            st["algo"] = algo
            stats[algo] = st

    return stats if stats else None