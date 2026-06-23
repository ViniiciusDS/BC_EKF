# src/uwb/algoritmos_estaticos.py
"""
Algoritmos de localização UWB — versões estáticas (sem estado).

Tradução fiel dos algoritmos do artigo CBA 2024 (MATLAB → Python):
  - trilaterate3d  : Trilateração clássica com 4 âncoras (sistema linear 3x3)
  - lms            : Mínimos Quadrados Lineares com N âncoras
  - gauss_newton   : Gauss-Newton iterativo (chamado 'errodist' no MATLAB)
  - lmsp           : LMS Ponderado por desvio padrão (W = 1/σ²)

Cada função aceita uma única medição (vetor d) OU pode ser usada via run_batch()
para processar um dataset inteiro.

"""
from __future__ import annotations
import numpy as np
from typing import Optional
import warnings
from itertools import combinations
import time


from src.bc_ekf import run_bc_ekf_from_data


#######################################################
# ÂNCORAS REAIS DO LABORATÓRIO (preset CBA 2024)
#######################################################
ANCHORS_LAB_CBA = np.array([
    [7.500, 5.150, 2.815],   # A1
    [1.725, 0.265, 1.145],   # A2
    [6.530, 8.890, 1.265],   # A3
    [7.200, 0.938, 1.200],   # A4
    [4.550, 0.225, 2.670],   # A5
    [0.560, 8.400, 1.240],   # A6
    [0.052, 4.720, 2.840],   # A8
    [2.922, 9.120, 2.755],   # A9
], dtype=float)               # shape (8, 3)

SALA_LAB = (7.86, 9.52)      # largura × comprimento (m)

#######################################################
# PARÂMETROS DE CONFIGURAÇÃO PARA VERSÕES RÁPIDAS (FAST)
#######################################################
FAST_MAX_ITER_GN = 5
FAST_MAX_ITER_WLS = 8
FAST_MAX_ITER_RKFWLS = 4
FAST_MAX_ITER_RWLS = 4
FAST_MAX_STEP = 0.5
FAST_DAMPING = 1e-2
MAX_TRIPLETS_TO_TEST = 20

##########################################################
# 1. TRILATERAÇÃO CLÁSSICA — exige exatamente 4 âncoras
##########################################################
def trilaterate3d(
    anchors_4x3: np.ndarray,
    distances_4: np.ndarray,
) -> np.ndarray:
    """
    Trilateração 3D clássica com 4 âncoras (sistema linear 3×3).

    Resolve:
        A_mat @ (P - P1) = b
    onde:
        A_mat = 2 * [(P2-P1); (P3-P1); (P4-P1)]
        b[i]  = ||Pi+1 - P1||² - d[i+1]² + d[0]²

    Parâmetros:
        anchors_4x3 : array (4, 3) — posições das âncoras
        distances_4 : array (4,)   — distâncias medidas

    Retorna:
        P : array (3,) — posição estimada
    """
    anchors_4x3 = np.asarray(anchors_4x3, dtype=float)
    distances_4 = np.asarray(distances_4, dtype=float)

    if anchors_4x3.shape[0] < 4:
        raise ValueError(f"trilaterate3d exige ≥ 4 âncoras, recebeu {anchors_4x3.shape[0]}")

    P1, P2, P3, P4 = anchors_4x3[0], anchors_4x3[1], anchors_4x3[2], anchors_4x3[3]
    d1, d2, d3, d4 = distances_4[0], distances_4[1], distances_4[2], distances_4[3]

    A_mat = 2.0 * np.vstack([P2 - P1, P3 - P1, P4 - P1])   # (3, 3)

    b = np.array([
        np.dot(P2 - P1, P2 - P1) - d2**2 + d1**2,
        np.dot(P3 - P1, P3 - P1) - d3**2 + d1**2,
        np.dot(P4 - P1, P4 - P1) - d4**2 + d1**2,
    ])

    try:
        P_rel = np.linalg.solve(A_mat, b)
    except np.linalg.LinAlgError:
        P_rel = np.linalg.lstsq(A_mat, b, rcond=None)[0]

    return P_rel + P1

def _is_plausible_position_fast(p, anchors_3d, distances_N, slack=3.0):
    p = np.asarray(p, dtype=float).reshape(-1)

    if p.size < 2 or not np.all(np.isfinite(p[:2])):
        return False

    d = np.asarray(distances_N, dtype=float)
    finite_d = d[np.isfinite(d)]

    if finite_d.size == 0:
        return True

    max_range = float(np.nanmax(finite_d)) + float(slack)

    center = np.nanmean(anchors_3d[:, :2], axis=0)
    dist_center = float(np.linalg.norm(p[:2] - center))

    if dist_center > 3.0 * max_range:
        return False

    return True

_TRIPLET_CACHE = {}


def _triangle_area_xy(p1, p2, p3):
    p1 = np.asarray(p1[:2], dtype=float)
    p2 = np.asarray(p2[:2], dtype=float)
    p3 = np.asarray(p3[:2], dtype=float)

    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def precompute_best_triplets(anchors_Nx3, max_triplets=10, min_area=1e-6):
    from itertools import combinations

    anchors = _as_anchors_3d(anchors_Nx3)

    scored = []

    for tri in combinations(range(len(anchors)), 3):
        p1, p2, p3 = anchors[list(tri)]

        area = _triangle_area_xy(p1, p2, p3)

        if area >= min_area:
            scored.append((area, tri))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [tri for _, tri in scored[:max_triplets]]


def get_cached_best_triplets(anchors_Nx3, max_triplets=10):
    anchors = _as_anchors_3d(anchors_Nx3)
    key = tuple(np.round(anchors.reshape(-1), 6))

    cache_key = (key, max_triplets)

    if cache_key not in _TRIPLET_CACHE:
        _TRIPLET_CACHE[cache_key] = precompute_best_triplets(
            anchors,
            max_triplets=max_triplets,
        )

    return _TRIPLET_CACHE[cache_key]
##########################################################
# 1B. TRILATERAÇÃO GEOMÉTRICA — Sang et al. (2019)
##########################################################

def _range_residual_score(p, anchors, distances):
    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)
    p = np.asarray(p, dtype=float)

    valid = (
        np.isfinite(distances)
        & np.all(np.isfinite(anchors), axis=1)
    )

    if valid.sum() == 0:
        return np.inf

    pred = np.linalg.norm(anchors[valid] - p, axis=1)
    err = pred - distances[valid]

    return float(np.sqrt(np.mean(err**2)))


def _solve_trilat_geo_triplet_sang2019(
    anchors_3x3,
    distances_3,
    *,
    score_anchors=None,
    score_distances=None,
    eps=1e-9,
):
    """
    Resolve a trilateração geométrica de Sang et al. (2019)
    para uma trinca de âncoras.

    A1 é deslocada para a origem, A2 define o eixo local X,
    e A3 define o plano local XY.

    Retorna uma posição 3D.
    """
    A = np.asarray(anchors_3x3, dtype=float)
    d = np.asarray(distances_3, dtype=float)

    if A.shape[0] < 3 or A.shape[1] < 3:
        raise ValueError("Trilateração geométrica requer 3 âncoras 3D")

    if len(d) < 3:
        raise ValueError("Trilateração geométrica requer 3 distâncias")

    A1, A2, A3 = A[0], A[1], A[2]
    d1, d2, d3 = float(d[0]), float(d[1]), float(d[2])

    ex_vec = A2 - A1
    U = float(np.linalg.norm(ex_vec))

    if U < eps:
        raise ValueError("A1 e A2 coincidentes na trilateração geométrica")

    ex = ex_vec / U

    A3_rel = A3 - A1
    Vx = float(np.dot(A3_rel, ex))

    ey_vec = A3_rel - Vx * ex
    Vy = float(np.linalg.norm(ey_vec))

    if Vy < eps:
        raise ValueError("Âncoras colineares na trilateração geométrica")

    ey = ey_vec / Vy

    ez = np.cross(ex, ey)
    ez_norm = float(np.linalg.norm(ez))

    if ez_norm < eps:
        raise ValueError("Base local degenerada na trilateração geométrica")

    ez = ez / ez_norm

    # Equações apresentadas por Sang et al. (2019)
    xt = (d1**2 - d2**2 + U**2) / (2.0 * U)
    yt = (d1**2 - d3**2 + Vx**2 + Vy**2 - 2.0 * xt * Vx) / (2.0 * Vy)

    z2 = d1**2 - xt**2 - yt**2

    # Em dados reais ruidosos, z2 pode ficar levemente negativo.
    if z2 < -1.0:
        raise ValueError("Trilateração geométrica sem solução real consistente")

    z_abs = float(np.sqrt(max(0.0, z2)))

    p_base = A1 + xt * ex + yt * ey

    candidates = [
        p_base + z_abs * ez,
        p_base - z_abs * ez,
    ]

    # Se houver mais âncoras, usa-as para resolver a ambiguidade de sinal.
    if score_anchors is not None and score_distances is not None:
        scores = [
            _range_residual_score(c, score_anchors, score_distances)
            for c in candidates
        ]
        return candidates[int(np.argmin(scores))]

    # Sem quarta âncora, retorna a solução positiva.
    return candidates[0]


def trilat_geo_sang2019(
    anchors_Nx3: np.ndarray,
    distances_N: np.ndarray,
) -> np.ndarray:
    """
    Trilateração geométrica baseada em Sang et al. (2019).

    Usa as três primeiras medições válidas. Quando há mais de três âncoras,
    usa as demais apenas para escolher o sinal de z que minimiza o resíduo.
    """
    anchors = np.asarray(anchors_Nx3, dtype=float)
    distances = np.asarray(distances_N, dtype=float)

    if anchors.ndim != 2 or anchors.shape[0] < 3:
        raise ValueError("trilat_geo_sang2019 exige ≥ 3 âncoras")

    if anchors.shape[1] == 2:
        anchors = np.column_stack([anchors, np.zeros(len(anchors))])

    valid = (
        np.isfinite(distances)
        & np.all(np.isfinite(anchors), axis=1)
    )

    idx = np.where(valid)[0]

    if len(idx) < 3:
        raise ValueError("trilat_geo_sang2019 exige ≥ 3 ranges válidos")

    triplet = idx[:3]

    return _solve_trilat_geo_triplet_sang2019(
        anchors[triplet],
        distances[triplet],
        score_anchors=anchors[idx],
        score_distances=distances[idx],
    )


def trilat_geo_triplet_sang2019_fast(
    anchors_Nx3,
    distances_N,
    triplets=None,
):
    anchors = _as_anchors_3d(anchors_Nx3)
    d = np.asarray(distances_N, dtype=float)

    if triplets is None:
        triplets = get_cached_best_triplets(anchors, max_triplets=10)

    best_p = None
    best_score = np.inf

    valid_all = np.isfinite(d)

    for tri in triplets:
        tri = np.asarray(tri, dtype=int)

        if not np.all(valid_all[tri]):
            continue

        try:
            p = _solve_trilat_geo_triplet_sang2019(
                anchors[tri],
                d[tri],
                score_anchors=anchors,
                score_distances=d,
            )

            score = _range_residual_score(p, anchors, d)

            if score < best_score:
                best_score = score
                best_p = p

        except Exception:
            continue

    if best_p is None:
        raise ValueError("Nenhuma trinca válida")

    return best_p

def trilat_geo_sang2019_batch(anchors_Nx3, distances_MxN):
    """
    Versão vetorizada da trilateração geométrica Sang2019
    usando as três primeiras âncoras.

    Retorna:
        posicoes Mx3
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    D = np.asarray(distances_MxN, dtype=float)

    if D.ndim != 2:
        raise ValueError("distances_MxN deve ser matriz MxN")

    if anchors.shape[0] < 3 or D.shape[1] < 3:
        raise ValueError("trilat_geo_sang2019_batch exige pelo menos 3 âncoras")

    A1 = anchors[0]
    A2 = anchors[1]
    A3 = anchors[2]

    d1 = D[:, 0]
    d2 = D[:, 1]
    d3 = D[:, 2]

    ex_vec = A2 - A1
    U = float(np.linalg.norm(ex_vec))

    if U < 1e-9:
        raise ValueError("A1 e A2 coincidentes")

    ex = ex_vec / U

    A3_rel = A3 - A1
    Vx = float(np.dot(A3_rel, ex))

    ey_vec = A3_rel - Vx * ex
    Vy = float(np.linalg.norm(ey_vec))

    if Vy < 1e-9:
        raise ValueError("Âncoras colineares")

    ey = ey_vec / Vy

    ez = np.cross(ex, ey)
    ez_norm = float(np.linalg.norm(ez))

    if ez_norm < 1e-9:
        raise ValueError("Base degenerada")

    ez = ez / ez_norm

    xt = (d1**2 - d2**2 + U**2) / (2.0 * U)
    yt = (d1**2 - d3**2 + Vx**2 + Vy**2 - 2.0 * xt * Vx) / (2.0 * Vy)

    z2 = d1**2 - xt**2 - yt**2
    z_abs = np.sqrt(np.maximum(0.0, z2))

    # Para seu simulador 2D, usar z positivo ou negativo não muda x,y.
    X = (
        A1[None, :]
        + xt[:, None] * ex[None, :]
        + yt[:, None] * ey[None, :]
        + z_abs[:, None] * ez[None, :]
    )

    bad = ~np.all(np.isfinite(D[:, :3]), axis=1)
    X[bad] = np.nan

    return X

###########################################################
# 2. LMS — Mínimos Quadrados Lineares com N âncoras
###########################################################
def _as_anchors_3d(anchors_Nx3):
    anchors = np.asarray(anchors_Nx3, dtype=float)

    if anchors.ndim != 2 or anchors.shape[0] < 3 or anchors.shape[1] < 2:
        raise ValueError("São necessárias pelo menos 3 âncoras com coordenadas x,y")

    if anchors.shape[1] == 2:
        anchors = np.column_stack([anchors, np.zeros(len(anchors))])

    return anchors[:, :3]


def _valid_range_rows(anchors, distances):
    d = np.asarray(distances, dtype=float)
    A = np.asarray(anchors, dtype=float)

    valid = np.isfinite(d) & np.all(np.isfinite(A), axis=1)

    return A[valid], d[valid], np.where(valid)[0]


def _safe_lstsq(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x

def lms(
    anchors_Nx3: np.ndarray,
    distances_N: np.ndarray,
) -> np.ndarray:
    """
    Localização por Mínimos Quadrados Lineares com N âncoras.

    Lineariza o sistema de equações de esfera em torno da âncora A1:
        dAnchors = A[1:] - A[0]
        b = 0.5 * ((d[0]² - d[j]²) + ||Aj - A0||²)
        P = (dAnchors' @ dAnchors)⁻¹ @ dAnchors' @ b + A[0]

    Parâmetros:
        anchors_Nx3 : array (N, 3) — posições das âncoras (N ≥ 2)
        distances_N : array (N,)   — distâncias medidas

    Retorna:
        P : array (3,) — posição estimada
    """
    anchors_Nx3 = np.asarray(anchors_Nx3, dtype=float)
    distances_N = np.asarray(distances_N, dtype=float)
    N = anchors_Nx3.shape[0]

    if N < 2:
        raise ValueError(f"lms exige ≥ 2 âncoras, recebeu {N}")

    A0 = anchors_Nx3[0]
    d0 = distances_N[0]

    # Diferença de posições em relação à âncora 0
    dA = anchors_Nx3[1:] - A0              # (N-1, 3)
    # Quadrado das distâncias entre âncoras
    dist1j = np.sum(dA**2, axis=1)         # (N-1,)
    # Lado direito da equação linearizada
    b = 0.5 * ((d0**2 - distances_N[1:]**2) + dist1j)   # (N-1,)

    # Solução por pseudoinversa (equivalente a inv(dA'dA) @ dA' @ b)
    P_rel, _, _, _ = np.linalg.lstsq(dA, b, rcond=None)

    return P_rel + A0

def ls_sang2019(anchors_Nx3, distances_N):
    """
    Multilateração por mínimos quadrados em forma fechada,
    baseada em Sang et al. (2019).

    Retorna posição 3D [x, y, z].
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    anchors, d, _ = _valid_range_rows(anchors, distances_N)

    if len(d) < 3:
        raise ValueError("ls_sang2019 exige pelo menos 3 ranges válidos")

    p1 = anchors[0]
    d1 = d[0]

    A = anchors[1:] - p1

    b = 0.5 * (
        d1**2
        - d[1:]**2
        + np.sum(anchors[1:] ** 2, axis=1)
        - np.sum(p1 ** 2)
    )

    return _safe_lstsq(A, b)

def ls_li2023_2d(anchors_Nx3, distances_N):
    anchors = _as_anchors_3d(anchors_Nx3)
    anchors, d, _ = _valid_range_rows(anchors, distances_N)

    if len(d) < 3:
        raise ValueError("ls_li2023_2d exige pelo menos 3 ranges válidos")

    pref = anchors[-1, :2]
    dref = d[-1]

    ai = anchors[:-1, :2]
    di = d[:-1]

    A = ai - pref
    C = (
        dref**2
        - di**2
        + np.sum(ai**2, axis=1)
        - np.sum(pref**2)
    )

    x2 = 0.5 * _safe_lstsq(A, C)
    return np.array([x2[0], x2[1], 0.0], dtype=float)

def ls_li2023(anchors_Nx3, distances_N):
    """
    Least Squares baseado em Li et al. (2023).

    Usa a última âncora válida como referência.
    Retorna posição 3D [x, y, z].
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    anchors, d, _ = _valid_range_rows(anchors, distances_N)

    if len(d) < 3:
        raise ValueError("ls_li2023 exige pelo menos 3 ranges válidos")

    pref = anchors[-1]
    dref = d[-1]

    ai = anchors[:-1]
    di = d[:-1]

    A = ai - pref

    C = (
        dref**2
        - di**2
        + np.sum(ai**2, axis=1)
        - np.sum(pref**2)
    )

    x = 0.5 * _safe_lstsq(A, C)
    return x

def ls_gn_li2023(
    anchors_Nx3,
    distances_N,
    *,
    x0=None,
    max_iter=5,
    tol=1e-5,
):
    anchors = _as_anchors_3d(anchors_Nx3)
    anchors, d, _ = _valid_range_rows(anchors, distances_N)

    if len(d) < 3:
        raise ValueError("ls_gn_li2023 exige pelo menos 3 ranges válidos")

    if x0 is None:
        x = np.asarray(ls_li2023(anchors, d), dtype=float).reshape(-1)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1)

    if x.size < 3:
        x = np.pad(x, (0, 3 - x.size), constant_values=0.0)

    x = x[:3].astype(float)

    for _ in range(max_iter):
        diff = anchors - x[None, :]
        pred = np.linalg.norm(diff, axis=1)

        valid = np.isfinite(pred) & (pred > 1e-9)

        if valid.sum() < 3:
            break

        B = diff[valid] / pred[valid, None]
        L = pred[valid] - d[valid]

        delta = _safe_lstsq(B, L)

        if np.linalg.norm(delta) > 0.5:
            delta = delta * (0.5 / np.linalg.norm(delta))

        x_new = x + delta

        if not np.all(np.isfinite(x_new)):
            break

        if np.linalg.norm(delta) < tol:
            x = x_new
            break

        x = x_new

    return x

def ls_sang2019_batch(anchors_Nx3, distances_MxN):
    """
    Versão vetorizada do LS Sang2019.

    Calcula todas as amostras de uma vez:
        X = B @ pinv(A).T

    Retorna:
        posicoes Mx3
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    D = np.asarray(distances_MxN, dtype=float)

    if D.ndim != 2:
        raise ValueError("distances_MxN deve ser matriz MxN")

    M, N = D.shape

    if anchors.shape[0] != N:
        raise ValueError("Número de âncoras incompatível com matriz de ranges")

    p1 = anchors[0]
    d1 = D[:, 0]

    A = anchors[1:] - p1  # (N-1) x 3

    B = 0.5 * (
        d1[:, None] ** 2
        - D[:, 1:] ** 2
        + np.sum(anchors[1:] ** 2, axis=1)[None, :]
        - np.sum(p1 ** 2)
    )  # M x (N-1)

    pinv_A = np.linalg.pinv(A)  # 3 x (N-1)

    X = B @ pinv_A.T  # M x 3

    X[~np.all(np.isfinite(D), axis=1)] = np.nan

    return X


def ls_li2023_batch(anchors_Nx3, distances_MxN):
    """
    Versão vetorizada do LS Li2023.

    Usa a última âncora como referência.
    Retorna:
        posicoes Mx3
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    D = np.asarray(distances_MxN, dtype=float)

    if D.ndim != 2:
        raise ValueError("distances_MxN deve ser matriz MxN")

    M, N = D.shape

    if anchors.shape[0] != N:
        raise ValueError("Número de âncoras incompatível com matriz de ranges")

    pref = anchors[-1]
    dref = D[:, -1]

    ai = anchors[:-1]
    di = D[:, :-1]

    A = ai - pref  # (N-1) x 3

    C = (
        dref[:, None] ** 2
        - di ** 2
        + np.sum(ai ** 2, axis=1)[None, :]
        - np.sum(pref ** 2)
    )  # M x (N-1)

    pinv_A = np.linalg.pinv(A)

    X = 0.5 * (C @ pinv_A.T)

    X[~np.all(np.isfinite(D), axis=1)] = np.nan

    return X

##################################################################
# 3. GAUSS-NEWTON — Mínimos Quadrados Não-Lineares iterativos
##################################################################
def gauss_newton(
    anchors_Nx3: np.ndarray,
    distances_N: np.ndarray,
    p_init:      Optional[np.ndarray] = None,
    n_iter:      int = 6,
) -> np.ndarray:
    """
    Refinamento iterativo por Gauss-Newton (Mínimos Quadrados Não-Lineares).

    A cada iteração:
        D[i]   = ||p - anchor_i||
        fe[i]  = D[i] - d[i]             (erro de distância)
        J[i,:] = (p - anchor_i) / D[i]   (Jacobiana)
        p      = p - (J'J)⁻¹ J' fe       (passo de Newton)

    Parâmetros:
        anchors_Nx3 : array (N, 3) — posições das âncoras
        distances_N : array (N,)   — distâncias medidas
        p_init      : array (3,)   — estimativa inicial (default: LMS)
        n_iter      : int           — número de iterações (default: 6)

    Retorna:
        P : array (3,) — posição refinada
    """
    anchors_Nx3 = np.asarray(anchors_Nx3, dtype=float)
    distances_N = np.asarray(distances_N, dtype=float)

    # Estimativa inicial: usa LMS se não fornecida
    if p_init is None:
        try:
            p = lms(anchors_Nx3, distances_N)
        except Exception:
            p = anchors_Nx3.mean(axis=0)
    else:
        p = np.asarray(p_init, dtype=float).copy()

    N = anchors_Nx3.shape[0]

    for _ in range(n_iter):
        # Vetor de erro
        diff = p - anchors_Nx3                 # (N, 3)
        D    = np.linalg.norm(diff, axis=1)    # (N,)

        # Guarda singular (evita divisão por zero)
        singular = D < 1e-10
        if singular.any():
            D[singular] = 1e-10

        fe = D - distances_N                   # (N,)
        J  = diff / D[:, np.newaxis]           # (N, 3) — Jacobiana

        # Passo de Gauss-Newton: p = p - (J'J)^-1 J' fe
        try:
            delta = np.linalg.solve(J.T @ J, J.T @ fe)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(J.T @ J, J.T @ fe, rcond=None)

        p = p - delta

    return p

def _initial_guess_for_gn(anchors_Nx3, distances_N, fallback_center=True):
    """
    Estimativa inicial para métodos GN.

    Prioridade:
    1. LS Sang2019;
    2. LMS atual;
    3. centro das âncoras.
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    d = np.asarray(distances_N, dtype=float)

    try:
        x0 = np.asarray(ls_sang2019(anchors, d), dtype=float).reshape(-1)
    except Exception:
        try:
            x0 = np.asarray(lms(anchors, d), dtype=float).reshape(-1)
        except Exception:
            if not fallback_center:
                raise
            x0 = np.nanmean(anchors, axis=0)

    if x0.size < 3:
        x0 = np.pad(x0, (0, 3 - x0.size), constant_values=0.0)

    return x0[:3].astype(float)


def _gn_covariance_and_dop(J, sigma2=1.0, damping=0.0):
    """
    Calcula Qx, PDOP, HDOP e VDOP a partir da matriz Jacobiana.
    """
    J = np.asarray(J, dtype=float)

    H = J.T @ J

    if damping and damping > 0:
        H = H + float(damping) * np.eye(H.shape[0])

    try:
        Q = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Q = np.linalg.pinv(H)

    Qs = float(sigma2) * Q

    q11 = float(Qs[0, 0]) if Qs.shape[0] > 0 else np.nan
    q22 = float(Qs[1, 1]) if Qs.shape[0] > 1 else np.nan
    q33 = float(Qs[2, 2]) if Qs.shape[0] > 2 else np.nan

    pdop = float(np.sqrt(max(q11 + q22 + q33, 0.0)))
    hdop = float(np.sqrt(max(q11 + q22, 0.0)))
    vdop = float(np.sqrt(max(q33, 0.0)))

    return Qs, pdop, hdop, vdop


def gauss_newton_wang2020(
    anchors_Nx3,
    distances_N,
    p_init=None,
    *,
    n_iter=FAST_MAX_ITER_GN,
    tol=1e-5,
    damping=0.0,
    max_step=None,
    return_info=False,
):
    """
    Gauss-Newton para posicionamento UWB baseado na formulação de
    Wang et al. (2020), Appl. Sci. 10, 273.

    Modelo:
        L_i = d_i(X) + eps_i

    Residual:
        v_i = d_i(X) - L_i

    Jacobiana:
        J_i = [(x-x_i), (y-y_i), (z-z_i)] / d_i(X)

    Atualização usada:
        X_{k+1} = X_k - delta
        delta = (J^T J)^-1 J^T v

    O sinal segue a convenção residual = predito - medido.
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    d = np.asarray(distances_N, dtype=float)

    valid = np.isfinite(d) & np.all(np.isfinite(anchors), axis=1)
    anchors = anchors[valid]
    d = d[valid]

    if len(d) < 3:
        raise ValueError("gauss_newton_wang2020 exige pelo menos 3 ranges válidos")

    if p_init is None:
        x = _initial_guess_for_gn(anchors, d)
    else:
        x = np.asarray(p_init, dtype=float).reshape(-1)
        if x.size < 3:
            x = np.pad(x, (0, 3 - x.size), constant_values=0.0)
        x = x[:3].astype(float)

    info = {
        "iterations": 0,
        "converged": False,
        "last_step_norm": np.nan,
        "sigma2": np.nan,
        "pdop": np.nan,
        "hdop": np.nan,
        "vdop": np.nan,
        "lambda_mahal": np.nan,
        "accepted": True,
    }

    last_J = None
    last_residual = None
    last_delta = None

    for it in range(int(n_iter)):
        diff = x[None, :] - anchors
        pred = np.linalg.norm(diff, axis=1)

        safe = np.isfinite(pred) & (pred > 1e-9)
        if safe.sum() < 3:
            break

        J = diff[safe] / pred[safe, None]
        residual = pred[safe] - d[safe]

        H = J.T @ J
        g = J.T @ residual

        if damping and damping > 0:
            H = H + float(damping) * np.eye(H.shape[0])

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(H, g, rcond=None)

        step_norm = float(np.linalg.norm(delta))

        if max_step is not None and step_norm > float(max_step):
            delta = delta * (float(max_step) / step_norm)
            step_norm = float(np.linalg.norm(delta))

        x_new = x - delta

        if not np.all(np.isfinite(x_new)):
            break

        x = x_new

        last_J = J
        last_residual = residual
        last_delta = delta

        info["iterations"] = it + 1
        info["last_step_norm"] = step_norm

        if step_norm < tol:
            info["converged"] = True
            break

    if last_J is not None and last_residual is not None:
        dof = max(len(last_residual) - 3, 1)
        sigma2 = float(np.sum(last_residual**2) / dof)

        Qx, pdop, hdop, vdop = _gn_covariance_and_dop(
            last_J,
            sigma2=max(sigma2, 1e-12),
            damping=damping,
        )

        info["sigma2"] = sigma2
        info["pdop"] = pdop
        info["hdop"] = hdop
        info["vdop"] = vdop

        if last_delta is not None:
            try:
                Qx_inv = np.linalg.pinv(Qx)
                lam = float(last_delta.T @ Qx_inv @ last_delta)
            except Exception:
                lam = np.nan

            info["lambda_mahal"] = lam

    if return_info:
        return x, info

    return x


def gauss_newton_wang2020_damped(
    anchors_Nx3,
    distances_N,
    p_init=None,
    *,
    n_iter=FAST_MAX_ITER_GN,
    tol=1e-5,
):
    """
    Variante prática estabilizada do GN Wang2020.

    Mantém a formulação GN, mas usa:
    - damping tipo Levenberg;
    - limite de passo.

    Útil para evitar saltos em geometrias ruins ou ranges ruidosos.
    """
    return gauss_newton_wang2020(
        anchors_Nx3,
        distances_N,
        p_init=p_init,
        n_iter=n_iter,
        tol=tol,
        damping=FAST_DAMPING,
        max_step=FAST_MAX_STEP,
        return_info=False,
    )


def gauss_newton_wang2020_mahalanobis(
    anchors_Nx3,
    distances_N,
    p_init=None,
    *,
    n_iter=FAST_MAX_ITER_GN,
    tol=1e-5,
    gamma=3.84,
    return_info=False,
):
    """
    GN Wang2020 com teste de Mahalanobis.

    Observação:
    O artigo usa a distância de Mahalanobis para avaliar se a
    linearização é suficientemente aproximada e se o viés do estimador
    pode ser relevante. Aqui o teste é usado como diagnóstico numérico
    e como proteção prática contra passos ruins.

    gamma=3.84 corresponde ao nível de significância alpha=0.05
    citado no artigo.
    """
    x, info = gauss_newton_wang2020(
        anchors_Nx3,
        distances_N,
        p_init=p_init,
        n_iter=n_iter,
        tol=tol,
        damping=FAST_DAMPING,
        max_step=FAST_MAX_STEP,
        return_info=True,
    )

    lam = info.get("lambda_mahal", np.nan)

    if np.isfinite(lam) and lam > float(gamma):
        info["accepted"] = False
    else:
        info["accepted"] = True

    if return_info:
        return x, info

    return x

def gauss_newton_wang2020_2d(
    anchors_Nx3,
    distances_N,
    p_init=None,
    *,
    fixed_z=0.0,
    n_iter=FAST_MAX_ITER_GN,
    tol=1e-5,
    damping=0.0,
    max_step=None,
    return_info=False,
):
    """
    Gauss-Newton UWB baseado em Wang et al. (2020), mas restrito ao plano XY.

    Estima apenas x,y e mantém z fixo. Isso é mais adequado para o simulador 2D,
    evitando instabilidade no eixo z quando as âncoras são coplanares ou quase coplanares.
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    d = np.asarray(distances_N, dtype=float)

    valid = np.isfinite(d) & np.all(np.isfinite(anchors), axis=1)
    anchors = anchors[valid]
    d = d[valid]

    if len(d) < 3:
        raise ValueError("gauss_newton_wang2020_2d exige pelo menos 3 ranges válidos")

    if p_init is None:
        try:
            x0 = np.asarray(ls_li2023_2d(anchors, d), dtype=float).reshape(-1)
        except Exception:
            try:
                x0 = np.asarray(lms(anchors, d), dtype=float).reshape(-1)
            except Exception:
                x0 = np.nanmean(anchors, axis=0)

        xy = x0[:2].astype(float)
    else:
        x0 = np.asarray(p_init, dtype=float).reshape(-1)
        xy = x0[:2].astype(float)

    info = {
        "iterations": 0,
        "converged": False,
        "last_step_norm": np.nan,
        "sigma2": np.nan,
        "hdop": np.nan,
        "accepted": True,
    }

    last_J = None
    last_residual = None
    last_delta = None

    for it in range(int(n_iter)):
        dx = xy[0] - anchors[:, 0]
        dy = xy[1] - anchors[:, 1]
        dz = float(fixed_z) - anchors[:, 2]

        pred = np.sqrt(dx * dx + dy * dy + dz * dz)

        safe = np.isfinite(pred) & (pred > 1e-9)
        if safe.sum() < 3:
            break

        residual = pred[safe] - d[safe]

        J = np.column_stack([
            dx[safe] / pred[safe],
            dy[safe] / pred[safe],
        ])

        H = J.T @ J
        g = J.T @ residual

        if damping and damping > 0:
            H = H + float(damping) * np.eye(2)

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(H, g, rcond=None)

        step_norm = float(np.linalg.norm(delta))

        if max_step is not None and step_norm > float(max_step):
            delta = delta * (float(max_step) / step_norm)
            step_norm = float(np.linalg.norm(delta))

        xy_new = xy - delta

        if not np.all(np.isfinite(xy_new)):
            break

        xy = xy_new

        last_J = J
        last_residual = residual
        last_delta = delta

        info["iterations"] = it + 1
        info["last_step_norm"] = step_norm

        if step_norm < tol:
            info["converged"] = True
            break

    if last_J is not None and last_residual is not None:
        dof = max(len(last_residual) - 2, 1)
        sigma2 = float(np.sum(last_residual ** 2) / dof)

        H = last_J.T @ last_J
        if damping and damping > 0:
            H = H + float(damping) * np.eye(2)

        try:
            Qxy = sigma2 * np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Qxy = sigma2 * np.linalg.pinv(H)

        info["sigma2"] = sigma2
        info["hdop"] = float(np.sqrt(max(Qxy[0, 0] + Qxy[1, 1], 0.0)))

        if last_delta is not None:
            try:
                lam = float(last_delta.T @ np.linalg.pinv(Qxy) @ last_delta)
            except Exception:
                lam = np.nan
            info["lambda_mahal"] = lam

    out = np.array([xy[0], xy[1], float(fixed_z)], dtype=float)

    if return_info:
        return out, info

    return out
######################################################
# 4. LMSP — LMS Ponderado (Weighted Least Squares)
######################################################
def lmsp(
    anchors_Nx3:  np.ndarray,
    distances_N:  np.ndarray,
    deviations_N: np.ndarray,
) -> np.ndarray:
    """
    Localização por Mínimos Quadrados Ponderados (WLS / LMSP).

    Igual ao LMS, mas usa matriz de pesos W = diag(1/σ²):
        xvar[j] = σ[j]² + σ[0]²     (variância combinada)
        W       = diag(1/xvar)
        P       = (dA' W dA)⁻¹ dA' W b + A[0]

    O peso maior é atribuído às medições com menor desvio padrão.

    Parâmetros:
        anchors_Nx3  : array (N, 3) — posições das âncoras
        distances_N  : array (N,)   — distâncias medidas
        deviations_N : array (N,)   — desvios padrão das medições

    Retorna:
        P : array (3,) — posição estimada
    """
    anchors_Nx3  = np.asarray(anchors_Nx3,  dtype=float)
    distances_N  = np.asarray(distances_N,  dtype=float)
    deviations_N = np.asarray(deviations_N, dtype=float)
    N = anchors_Nx3.shape[0]

    if N < 2:
        raise ValueError(f"lmsp exige ≥ 2 âncoras, recebeu {N}")

    A0   = anchors_Nx3[0]
    d0   = distances_N[0]
    sig0 = deviations_N[0]

    dA     = anchors_Nx3[1:] - A0
    dist1j = np.sum(dA**2, axis=1)
    b      = 0.5 * ((d0**2 - distances_N[1:]**2) + dist1j)

    # Variância combinada: σ_j² + σ_0²
    xvar = deviations_N[1:]**2 + sig0**2
    
    # Clamp para evitar peso infinito
    xvar = np.maximum(xvar, 1e-12)
    w    = 1.0 / xvar                           # (N-1,)
    W    = np.diag(w)

    A_w = dA.T @ W @ dA      # (3, 3)
    b_w = dA.T @ W @ b       # (3,)

    try:
        P_rel = np.linalg.solve(A_w, b_w)
    except np.linalg.LinAlgError:
        P_rel, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)

    return P_rel + A0

def wls_sigma_gn(
    anchors_Nx3,
    distances_N,
    deviations_N=None,
    x0=None,
    max_iter=10,
    tol=1e-6,
    damping=1e-3,
    max_step=0.75,
):
    """
    WLS não linear por Gauss-Newton/Levenberg usando pesos 1/sigma².

    Versão estabilizada:
    - limita sigma mínimo;
    - adiciona damping na matriz normal;
    - limita o tamanho do passo iterativo.
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    anchors, d, valid_idx = _valid_range_rows(anchors, distances_N)

    if len(d) < 3:
        raise ValueError("wls_sigma_gn exige pelo menos 3 ranges válidos")

    if deviations_N is not None:
        sigma = np.asarray(deviations_N, dtype=float)[valid_idx]
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-3), sigma, 0.05)
    else:
        sigma = np.full(len(d), 0.05, dtype=float)

    # Evita pesos absurdamente grandes.
    sigma = np.clip(sigma, 0.01, 2.0)
    w = 1.0 / (sigma ** 2)

    if x0 is None:
        try:
            x = np.asarray(ls_sang2019(anchors, d), dtype=float).reshape(-1)
        except Exception:
            x = np.mean(anchors, axis=0)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1)

    if x.size < 3:
        x = np.pad(x, (0, 3 - x.size), constant_values=0.0)

    x = x[:3].astype(float)

    for _ in range(max_iter):
        diff = x[None, :] - anchors
        pred = np.linalg.norm(diff, axis=1)

        valid = np.isfinite(pred) & (pred > 1e-9)

        if valid.sum() < 3:
            break

        J = diff[valid] / pred[valid, None]
        r = d[valid] - pred[valid]
        wv = w[valid]

        JT_W = J.T * wv
        Hn = JT_W @ J
        g = JT_W @ r

        Hn = Hn + damping * np.eye(Hn.shape[0])

        try:
            delta = np.linalg.solve(Hn, g)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(Hn, g, rcond=None)

        step_norm = float(np.linalg.norm(delta))

        if step_norm > max_step:
            delta = delta * (max_step / step_norm)

        x_new = x + delta

        if not np.all(np.isfinite(x_new)):
            break

        if np.linalg.norm(delta) < tol:
            x = x_new
            break

        x = x_new

    return x

def wls_rkf_fan2022_batch(
    anchors_Nx3,
    distances_MxN,
    *,
    dt=0.05,
    sigma_m=0.02,
    sigma_u=0.5,
    chi2_threshold=6.2,
):
    """
    Implementação prática do WLS-RKF de Fan e Du.

    Ideia:
    - Um KF 1D por âncora estima range e velocidade do range.
    - Mahalanobis detecta provável NLOS.
    - Medição NLOS recebe peso menor.
    - Posição é resolvida por WLS-GN usando ranges filtrados/preditos.

    Retorna:
        posicoes Mx3
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    D = np.asarray(distances_MxN, dtype=float)

    if D.ndim != 2:
        raise ValueError("distances_MxN deve ser matriz MxN")

    M, N = D.shape

    if len(anchors) != N:
        raise ValueError("Número de âncoras incompatível com matriz de ranges")

    # Estado por âncora: [range, range_rate]
    X = np.zeros((N, 2), dtype=float)
    P = np.zeros((N, 2, 2), dtype=float)

    # Inicialização
    first = D[0]
    for i in range(N):
        ri = first[i]
        if not np.isfinite(ri):
            ri = np.nanmedian(D[:, i])
        if not np.isfinite(ri):
            ri = 1.0

        X[i] = [ri, 0.0]
        P[i] = np.diag([sigma_m**2, 1.0])

    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)

    # Modelo simples de aceleração/variação de range
    G = np.array([[0.5 * dt * dt], [dt]], dtype=float)
    Q = (sigma_u ** 2) * (G @ G.T)
    R = sigma_m ** 2

    posicoes = np.full((M, 3), np.nan, dtype=float)

    last_pos = None

    for k in range(M):
        ranges_used = np.full(N, np.nan, dtype=float)
        weights = np.ones(N, dtype=float)

        # 1) Predição e identificação NLOS por âncora
        nlos_flags = np.zeros(N, dtype=bool)

        for i in range(N):
            # Predict
            X_pred = A @ X[i]
            P_pred = A @ P[i] @ A.T + Q

            r_meas = D[k, i]
            d_pred = float(X_pred[0])

            S = float(P_pred[0, 0] + R)

            if not np.isfinite(r_meas):
                X[i] = X_pred
                P[i] = P_pred
                ranges_used[i] = d_pred
                weights[i] = 0.1
                continue

            gamma = ((r_meas - d_pred) ** 2) / max(S, 1e-12)

            # Critério do artigo: gamma > limiar e range medido maior que previsto
            is_nlos = (gamma > chi2_threshold) and (r_meas > d_pred)
            nlos_flags[i] = is_nlos

            if is_nlos:
                # NLOS: usa range previsto e peso reduzido
                ranges_used[i] = d_pred
                weights[i] = np.sqrt(chi2_threshold / max(gamma, 1e-12))

                # Não atualiza agora; será atualizado após a posição WLS
                X[i] = X_pred
                P[i] = P_pred

            else:
                # LOS: atualiza KF normalmente com r_meas
                K = P_pred[:, 0] / max(S, 1e-12)
                innovation = r_meas - d_pred

                X_upd = X_pred + K * innovation

                KH = np.zeros((2, 2), dtype=float)
                KH[:, 0] = K
                P_upd = (np.eye(2) - KH) @ P_pred

                X[i] = X_upd
                P[i] = P_upd

                ranges_used[i] = float(X_upd[0])
                weights[i] = 1.0

        # 2) Resolve posição por WLS-GN com pesos do WLS-RKF
        try:
            sigma_eff = 1.0 / np.maximum(weights, 1e-3)
            p = wls_sigma_gn(
                anchors,
                ranges_used,
                deviations_N=sigma_eff,
                x0=last_pos,
                max_iter=4,
                tol=1e-5,
                damping=1e-2,
                max_step=0.5,
            )
            p = np.asarray(p, dtype=float).reshape(-1)
            posicoes[k] = p[:3]
            last_pos = posicoes[k].copy()

        except Exception:
            if last_pos is not None:
                posicoes[k] = last_pos

        # 3) Para âncoras NLOS, atualiza KF com distância baseada na posição estimada
        if np.all(np.isfinite(posicoes[k])):
            for i in range(N):
                if not nlos_flags[i]:
                    continue

                y = float(np.linalg.norm(posicoes[k] - anchors[i]))
                d_pred = float(X[i, 0])
                S = float(P[i, 0, 0] + R)

                K = P[i, :, 0] / max(S, 1e-12)
                innovation = y - d_pred

                X[i] = X[i] + K * innovation

                KH = np.zeros((2, 2), dtype=float)
                KH[:, 0] = K
                P[i] = (np.eye(2) - KH) @ P[i]

    return posicoes

def rwls_wang2017_approx(
    anchors_Nx3,
    distances_N,
    deviations_N=None,
    x0=None,
    rho=1.5,
    max_iter=6,
):
    """
    Aproximação prática inspirada em Wang et al. (2017).

    Não é a implementação SOCP fiel do artigo.
    Esta versão apenas reduz pesos de ranges com resíduo positivo grande,
    evitando divergência numérica.
    """
    anchors = _as_anchors_3d(anchors_Nx3)
    anchors, d, valid_idx = _valid_range_rows(anchors, distances_N)

    if len(d) < 3:
        raise ValueError("rwls_wang2017_approx exige pelo menos 3 ranges válidos")

    if deviations_N is not None:
        sigma = np.asarray(deviations_N, dtype=float)[valid_idx]
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-3), sigma, 0.05)
    else:
        sigma = np.full(len(d), 0.05, dtype=float)

    sigma = np.clip(sigma, 0.02, 2.0)

    if x0 is None:
        try:
            x = np.asarray(ls_sang2019(anchors, d), dtype=float).reshape(-1)
        except Exception:
            x = np.mean(anchors, axis=0)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1)

    if x.size < 3:
        x = np.pad(x, (0, 3 - x.size), constant_values=0.0)

    x = x[:3].astype(float)

    for _ in range(max_iter):
        pred = np.linalg.norm(x[None, :] - anchors, axis=1)
        residual = d - pred

        if not np.all(np.isfinite(residual)):
            break

        # NLOS em TOA/UWB tende a range positivo: d medido > d previsto.
        positive_residual = np.maximum(residual, 0.0)

        # Peso robusto conservador.
        robust_factor = 1.0 / (1.0 + (positive_residual / max(rho, 1e-6)) ** 2)
        robust_factor = np.clip(robust_factor, 0.05, 1.0)

        sigma_eff = sigma / np.sqrt(robust_factor)

        x_new = wls_sigma_gn(
            anchors,
            d,
            deviations_N=sigma_eff,
            x0=x,
            max_iter=4,
            max_step=0.5,
            damping=1e-2,
        )

        x_new = np.asarray(x_new, dtype=float).reshape(-1)[:3]

        if not np.all(np.isfinite(x_new)):
            break

        if np.linalg.norm(x_new - x) > 2.0:
            # Evita salto absurdo.
            break

        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break

        x = x_new

    return x


#######################################################
# BATCH — processa um dataset inteiro de uma vez
#######################################################
def run_batch(
    anchors_Nx3:  np.ndarray,
    distances:    np.ndarray,
    deviations:   Optional[np.ndarray] = None,
    algoritmos:   Optional[list[str]] = None,
    p_true:       Optional[np.ndarray] = None,
    bc_ekf_data:  Optional[dict]   = None,
) -> dict:
    """
    Processa um dataset completo com múltiplos algoritmos.

    Versão otimizada:
    - algoritmos vetorizados rodam uma vez;
    - algoritmos batch rodam uma vez;
    - algoritmos amostra-a-amostra têm apenas um loop M;
    - evita loops M x M acidentais.
    """
    anchors_Nx3 = np.asarray(anchors_Nx3, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if distances.ndim != 2:
        raise ValueError(f"distances deve ser matriz MxN, shape={distances.shape}")

    M, N = distances.shape

    if deviations is not None:
        deviations = np.asarray(deviations, dtype=float)

    if algoritmos is None:
        algoritmos = ["trilaterate3d", "lms", "gauss_newton", "lmsp", "bc_ekf"]

    # Remove métodos que dependem de deviations quando não há deviations.
    needs_dev = {"lmsp", "wls_sigma", "rwls_wang2017"}

    if deviations is None:
        filtered = []
        for a in algoritmos:
            if a in needs_dev:
                warnings.warn(f"{a} removido do batch: deviations não fornecido")
            else:
                filtered.append(a)
        algoritmos = filtered

    resultados = {}
    anchors_3d_cached = _as_anchors_3d(anchors_Nx3)

    def _new_posicoes():
        return np.full((M, 3), np.nan, dtype=float)

    def _save_position(posicoes, idx, p):
        p = np.asarray(p, dtype=float).reshape(-1)

        if p.size < 2:
            raise ValueError("posição inválida")

        if p.size < 3:
            p = np.pad(p, (0, 3 - p.size), constant_values=0.0)

        posicoes[idx] = p[:3]

    def _save_trajectory(posicoes, traj):
        traj = np.asarray(traj, dtype=float)

        if traj.ndim != 2 or traj.shape[0] != M or traj.shape[1] < 2:
            raise ValueError(f"trajetória inválida: shape={traj.shape}")

        if traj.shape[1] >= 3:
            posicoes[:, :] = traj[:, :3]
        else:
            posicoes[:, :2] = traj[:, :2]
            posicoes[:, 2] = 0.0

    for nome in algoritmos:
        t0_algo = time.perf_counter()
        posicoes = _new_posicoes()

        try:
            # =====================================================
            # BC-EKF — batch próprio
            # =====================================================
            if nome == "bc_ekf":
                if bc_ekf_data is None:
                    raise ValueError("bc_ekf requer bc_ekf_data")

                anchors_bc = np.asarray(anchors_Nx3, dtype=float).T  # 3xN

                x_hist_est = run_bc_ekf_from_data(
                    T=bc_ekf_data["T"],
                    anchors=anchors_bc,
                    odometry_noisy=bc_ekf_data["odometry_noisy"],
                    z_hist=bc_ekf_data["z_hist"],
                    l=bc_ekf_data["l"],
                    z_c=bc_ekf_data["z_c"],
                    sigma_uwb=bc_ekf_data["sigma_uwb"],
                    x0=bc_ekf_data.get("x0", None),
                )

                _save_trajectory(posicoes, x_hist_est.T)

            # =====================================================
            # Métodos vetorizados — rodam uma única vez
            # =====================================================
            elif nome == "trilat_geo_sang2019":
                traj = trilat_geo_sang2019_batch(anchors_Nx3, distances)
                _save_trajectory(posicoes, traj)

            elif nome == "ls_sang2019":
                traj = ls_sang2019_batch(anchors_Nx3, distances)
                _save_trajectory(posicoes, traj)

            elif nome == "ls_li2023":
                traj = ls_li2023_batch(anchors_Nx3, distances)
                _save_trajectory(posicoes, traj)

            # =====================================================
            # WLS-RKF — batch próprio, roda uma única vez
            # =====================================================
            elif nome == "wls_rkf_fan2022":
                traj = wls_rkf_fan2022_batch(
                    anchors_Nx3,
                    distances,
                    dt=0.05,
                )
                _save_trajectory(posicoes, traj)

            # =====================================================
            # Métodos amostra-a-amostra simples
            # =====================================================
            elif nome == "trilaterate3d":
                for i in range(M):
                    try:
                        p = trilaterate3d(anchors_Nx3[:4], distances[i, :4])
                        _save_position(posicoes, i, p)
                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            elif nome == "lms":
                for i in range(M):
                    try:
                        p = lms(anchors_Nx3, distances[i])
                        _save_position(posicoes, i, p)
                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            elif nome == "gauss_newton":
                p_init_gn = None

                for i in range(M):
                    try:
                        if p_init_gn is None:
                            p_init_gn = lms(anchors_Nx3, distances[i])

                        p = gauss_newton(
                            anchors_Nx3,
                            distances[i],
                            p_init=p_init_gn,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)
                        p_init_gn = p.copy()

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            elif nome == "lmsp":
                for i in range(M):
                    try:
                        dev_i = deviations[i] if deviations is not None else None

                        if dev_i is None:
                            raise ValueError("lmsp requer deviations")

                        p = lmsp(anchors_Nx3, distances[i], dev_i)
                        _save_position(posicoes, i, p)

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            # =====================================================
            # Trilateração geométrica com seleção de trinca
            # Agora tem apenas UM loop M, não M x M.
            # =====================================================
            elif nome == "trilat_geo_triplet_sang2019":
                triplets = get_cached_best_triplets(
                    anchors_Nx3,
                    max_triplets=MAX_TRIPLETS_TO_TEST,
                )

                for i in range(M):
                    try:
                        p = trilat_geo_triplet_sang2019_fast(
                            anchors_Nx3,
                            distances[i],
                            triplets=triplets,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            # =====================================================
            # LS + GN Li2023
            # Agora tem apenas UM loop M.
            # =====================================================
            elif nome == "ls_gn_li2023":
                last_p = None

                for i in range(M):
                    try:
                        p = ls_gn_li2023(
                            anchors_Nx3,
                            distances[i],
                            x0=last_p,
                            max_iter=FAST_MAX_ITER_GN,
                            tol=1e-5,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)
                        last_p = p.copy()

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            # =====================================================
            # WLS sigma
            # Agora tem apenas UM loop M.
            # =====================================================
            elif nome == "wls_sigma":
                last_p = None

                for i in range(M):
                    try:
                        dev_i = deviations[i] if deviations is not None else None

                        p = wls_sigma_gn(
                            anchors_Nx3,
                            distances[i],
                            deviations_N=dev_i,
                            x0=last_p,
                            max_iter=FAST_MAX_ITER_WLS,
                            tol=1e-5,
                            damping=FAST_DAMPING,
                            max_step=FAST_MAX_STEP,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)
                        last_p = p.copy()

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            # =====================================================
            # RWLS Wang2017 aproximado
            # Agora tem apenas UM loop M.
            # =====================================================
            elif nome == "rwls_wang2017":
                last_p = None

                for i in range(M):
                    try:
                        dev_i = deviations[i] if deviations is not None else None

                        p = rwls_wang2017_approx(
                            anchors_Nx3,
                            distances[i],
                            deviations_N=dev_i,
                            x0=last_p,
                            max_iter=FAST_MAX_ITER_RWLS,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)
                        last_p = p.copy()

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            # =====================================================
            # GN Wang2020 com damping
            # Agora tem apenas UM loop M.
            # =====================================================
            elif nome == "gn_wang2020":
                last_p = None

                for i in range(M):
                    try:
                        if last_p is None:
                            last_p = _initial_guess_for_gn(anchors_Nx3, distances[i])

                        p = gauss_newton_wang2020_2d(
                            anchors_Nx3,
                            distances[i],
                            p_init=last_p,
                            fixed_z=0.0,
                            n_iter=FAST_MAX_ITER_GN,
                            tol=1e-5,
                            damping=0.20,
                            max_step=FAST_MAX_STEP,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)
                        last_p = p.copy()

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            # =====================================================
            # GN Wang2020 com damping
            # Agora tem apenas UM loop M.
            # =====================================================
            elif nome == "gn_wang2020_damped":
                last_p = None

                for i in range(M):
                    try:
                        if last_p is None:
                            last_p = _initial_guess_for_gn(anchors_Nx3, distances[i])

                        p = gauss_newton_wang2020_damped(
                            anchors_Nx3,
                            distances[i],
                            p_init=last_p,
                            n_iter=FAST_MAX_ITER_GN,
                            tol=1e-5,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)
                        last_p = p.copy()

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            # =====================================================
            # GN Wang2020 com teste de Mahalanobis
            # Agora tem apenas UM loop M.
            # =====================================================
            elif nome == "gn_wang2020_mahal":
                last_p = None

                for i in range(M):
                    try:
                        if last_p is None:
                            last_p = _initial_guess_for_gn(anchors_Nx3, distances[i])

                        p, info = gauss_newton_wang2020_mahalanobis(
                            anchors_Nx3,
                            distances[i],
                            p_init=last_p,
                            n_iter=FAST_MAX_ITER_GN,
                            tol=1e-5,
                            gamma=3.84,
                            return_info=True,
                        )

                        p = np.asarray(p, dtype=float).reshape(-1)

                        # Se o teste indicar linearização ruim, mantém a última posição.
                        # Isso é uma adaptação prática para o simulador.
                        if not info.get("accepted", True) and last_p is not None:
                            p = last_p.copy()

                        if not _is_plausible_position_fast(
                            p,
                            anchors_3d_cached,
                            distances[i],
                        ):
                            raise ValueError("posição implausível")

                        _save_position(posicoes, i, p)
                        last_p = p.copy()

                    except Exception:
                        if i > 0:
                            posicoes[i] = posicoes[i - 1]

            else:
                warnings.warn(f"Algoritmo desconhecido no batch: {nome}")
                posicoes[:] = np.nan

        except Exception as e:
            warnings.warn(f"[{nome}] batch: {e}")
            posicoes[:] = np.nan

        finally:
            elapsed = time.perf_counter() - t0_algo
            print(f"[RUN_BATCH_TIME] {nome}: {elapsed:.3f}s")

        # =====================================================
        # Métricas opcionais com p_true
        # =====================================================
        rmse_xy = rmse_xyz = None

        if p_true is not None:
            p_true_arr = np.asarray(p_true, dtype=float)

            n = min(len(posicoes), len(p_true_arr))
            pos_xy = np.asarray(posicoes[:n, :2], dtype=float)

            if p_true_arr.ndim == 2 and p_true_arr.shape[1] >= 2:
                true_xy = np.asarray(p_true_arr[:n, :2], dtype=float)
            else:
                raise ValueError(
                    f"p_true inválido para cálculo de erro: shape={p_true_arr.shape}"
                )

            valid = (
                np.all(np.isfinite(pos_xy), axis=1)
                & np.all(np.isfinite(true_xy), axis=1)
            )

            if np.any(valid):
                err_xy = pos_xy[valid] - true_xy[valid]
                err_pos = np.linalg.norm(err_xy, axis=1)
                rmse_xy = float(np.sqrt(np.mean(err_pos**2)))

            if (
                p_true_arr.ndim == 2
                and p_true_arr.shape[1] >= 3
                and posicoes.shape[1] >= 3
            ):
                pos_xyz = np.asarray(posicoes[:n, :3], dtype=float)
                true_xyz = np.asarray(p_true_arr[:n, :3], dtype=float)

                valid3 = (
                    np.all(np.isfinite(pos_xyz), axis=1)
                    & np.all(np.isfinite(true_xyz), axis=1)
                )

                if np.any(valid3):
                    err_xyz = pos_xyz[valid3] - true_xyz[valid3]
                    rmse_xyz = float(np.sqrt(np.mean(np.sum(err_xyz**2, axis=1))))
        
        valid_count = int(np.sum(np.all(np.isfinite(posicoes[:, :2]), axis=1)))
        print(f"[RUN_BATCH_VALID] {nome}: {valid_count}/{M}")

        resultados[nome] = {
            "posicoes": posicoes,
            "rmse_xy": rmse_xy,
            "rmse_xyz": rmse_xyz,
        }

    return resultados


##############################################################
# LEITOR DE DADOS REAIS — formato do ensaio do laboratório
##############################################################
def carregar_ensaio_lab(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega o arquivo Dados_mesclados.txt do ensaio de laboratório.

    Formato: linhas com 16 colunas — pares (distância, desvio) por âncora:
        col 0 = d_A1, col 1 = σ_A1, col 2 = d_A2, col 3 = σ_A2, ...

    Retorna:
        distances  : array (M, 8) — distâncias medidas (m)
        deviations : array (M, 8) — desvios padrão (m)
    """
    data = []
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            try:
                vals = [float(v) for v in line.split("\t")]
                if len(vals) == 16:
                    data.append(vals)
            except ValueError:
                pass

    if not data:
        raise ValueError(f"Nenhuma linha válida encontrada em {caminho}")

    arr = np.array(data, dtype=float)
    distances  = arr[:, 0::2]   # colunas pares  (0,2,4,...,14)
    deviations = arr[:, 1::2]   # colunas ímpares (1,3,5,...,15)
    return distances, deviations