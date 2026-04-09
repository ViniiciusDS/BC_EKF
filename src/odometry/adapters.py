from __future__ import annotations

from typing import Sequence


def group_aligned_rows_by_timestamp(aligned_rows: Sequence[dict]) -> list[dict]:
    """
    Agrupa linhas alinhadas por timestamp.

    Saída:
    [
        {
            "timestamp_s": ...,
            "rows": [...],
        },
        ...
    ]
    """
    grouped = {}
    for row in aligned_rows:
        t = float(row["timestamp_s"])
        grouped.setdefault(t, []).append(row)

    out = []
    for t in sorted(grouped.keys()):
        out.append(
            {
                "timestamp_s": t,
                "rows": grouped[t],
            }
        )
    return out


def extract_anchor_ids(aligned_rows: Sequence[dict]) -> list[int]:
    ids = sorted({int(row["anchor_id"]) for row in aligned_rows})
    return ids


def extract_odometry_path(aligned_rows: Sequence[dict]) -> list[tuple[float, float]]:
    """
    Extrai uma trajetória 2D única a partir do dataset alinhado.
    Considera uma pose por timestamp.
    """
    grouped = group_aligned_rows_by_timestamp(aligned_rows)
    path = []

    for block in grouped:
        rows = block["rows"]
        if not rows:
            continue
        row0 = rows[0]
        path.append((float(row0["odom_x"]), float(row0["odom_y"])))

    return path


def build_range_sigma_matrices(aligned_rows: Sequence[dict]) -> dict:
    """
    Monta matrizes por timestamp x anchor_id.

    Retorna:
    {
        "timestamps_s": [...],
        "anchor_ids": [...],
        "ranges": [[...], [...], ...],
        "sigmas": [[...], [...], ...],
        "odom_xy": [[x,y], ...],
        "odom_theta": [...],
    }
    """
    grouped = group_aligned_rows_by_timestamp(aligned_rows)
    anchor_ids = extract_anchor_ids(aligned_rows)
    col_index = {aid: i for i, aid in enumerate(anchor_ids)}

    timestamps_s = []
    ranges = []
    sigmas = []
    odom_xy = []
    odom_theta = []

    for block in grouped:
        t = float(block["timestamp_s"])
        rows = block["rows"]

        rvec = [float("nan")] * len(anchor_ids)
        svec = [float("nan")] * len(anchor_ids)

        ref = rows[0]
        ox = float(ref["odom_x"])
        oy = float(ref["odom_y"])
        oth = float(ref["odom_theta"])

        for row in rows:
            aid = int(row["anchor_id"])
            j = col_index[aid]
            rvec[j] = float(row["range"])
            svec[j] = float(row.get("sigma", float("nan")))

        timestamps_s.append(t)
        ranges.append(rvec)
        sigmas.append(svec)
        odom_xy.append([ox, oy])
        odom_theta.append(oth)

    return {
        "timestamps_s": timestamps_s,
        "anchor_ids": anchor_ids,
        "ranges": ranges,
        "sigmas": sigmas,
        "odom_xy": odom_xy,
        "odom_theta": odom_theta,
    }