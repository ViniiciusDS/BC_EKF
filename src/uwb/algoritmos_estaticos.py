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


def trilat_geo_triplet_sang2019(
    anchors_Nx3: np.ndarray,
    distances_N: np.ndarray,
) -> np.ndarray:
    """
    Extensão prática da trilateração geométrica de Sang et al. (2019).

    Testa todas as trincas válidas de âncoras e escolhe a solução
    com menor resíduo em relação a todas as âncoras disponíveis.

    Essa versão é mais robusta em cenários com mais de três âncoras.
    """
    anchors = np.asarray(anchors_Nx3, dtype=float)
    distances = np.asarray(distances_N, dtype=float)

    if anchors.ndim != 2 or anchors.shape[0] < 3:
        raise ValueError("trilat_geo_triplet_sang2019 exige ≥ 3 âncoras")

    if anchors.shape[1] == 2:
        anchors = np.column_stack([anchors, np.zeros(len(anchors))])

    valid = (
        np.isfinite(distances)
        & np.all(np.isfinite(anchors), axis=1)
    )

    idx = np.where(valid)[0]

    if len(idx) < 3:
        raise ValueError("trilat_geo_triplet_sang2019 exige ≥ 3 ranges válidos")

    best_p = None
    best_score = np.inf

    for triplet in combinations(idx, 3):
        triplet = np.asarray(triplet, dtype=int)

        try:
            p = _solve_trilat_geo_triplet_sang2019(
                anchors[triplet],
                distances[triplet],
                score_anchors=anchors[idx],
                score_distances=distances[idx],
            )

            score = _range_residual_score(p, anchors[idx], distances[idx])

            if score < best_score:
                best_score = score
                best_p = p

        except Exception:
            continue

    if best_p is None:
        raise ValueError("Nenhuma trinca válida para trilateração geométrica")

    return best_p

###########################################################
# 2. LMS — Mínimos Quadrados Lineares com N âncoras
###########################################################
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

    Parâmetros:
        anchors_Nx3  : array (N, 3)          — posições das âncoras
        distances    : array (M, N)           — M amostras × N distâncias
        deviations   : array (M, N) ou None   — desvios padrão (para LMSP)
        algoritmos   : lista de nomes a rodar, ex: ["lms", "gauss_newton"]
                       None → roda todos
        p_true       : array (M, 3) ou None   — posição real (para calcular RMSE)

    Retorna:
        dict com chaves = nome do algoritmo, valores = dict com:
            "posicoes" : array (M, 3)
            "rmse_xy"  : float ou None
            "rmse_xyz" : float ou None
    """
    distances  = np.asarray(distances,  dtype=float)
    M = distances.shape[0]
    N = anchors_Nx3.shape[0]

    if algoritmos is None:
        algoritmos = ["trilaterate3d", "lms", "gauss_newton", "lmsp", "bc_ekf"]

    # Remove lmsp se não temos desvios
    if deviations is None and "lmsp" in algoritmos:
        algoritmos = [a for a in algoritmos if a != "lmsp"]
        warnings.warn("lmsp removido do batch: deviations não fornecido")

    resultados = {}

    for nome in algoritmos:
        posicoes = np.zeros((M, 3))
        p_init_gn = None   # warm start do Gauss-Newton

        if nome == "bc_ekf":
            try:
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
                posicoes = x_hist_est.T  # (M, 3)
                
            except Exception as e:
                print(f"[bc_ekf] batch error: {e}")
                warnings.warn(f"[bc_ekf] batch: {e}")
                posicoes[:] = np.nan

        else:
            for i in range(M):
                d = distances[i]
                dev = deviations[i] if deviations is not None else None

                try:
                    if nome == "trilaterate3d":
                        posicoes[i] = trilaterate3d(anchors_Nx3[:4], d[:4])

                    elif nome == "lms":
                        posicoes[i] = lms(anchors_Nx3, d)
                        p_init_gn = posicoes[i]   # salva para usar como init do GN

                    elif nome == "gauss_newton":
                        # warm start: usa LMS se disponível, senão calcula aqui
                        if p_init_gn is None:
                            p_init_gn = lms(anchors_Nx3, d)
                        posicoes[i] = gauss_newton(anchors_Nx3, d, p_init=p_init_gn)
                        p_init_gn = posicoes[i]   # atualiza warm start com resultado anterior

                    elif nome == "lmsp":
                        posicoes[i] = lmsp(anchors_Nx3, d, dev)

                    elif nome == "trilat_geo_sang2019":
                        for k in range(M):
                            try:
                                posicoes[k] = trilat_geo_sang2019(
                                    anchors_Nx3,
                                    distances[k],
                                )
                            except Exception:
                                posicoes[k] = np.nan


                    elif nome == "trilat_geo_triplet_sang2019":
                        for k in range(M):
                            try:
                                posicoes[k] = trilat_geo_triplet_sang2019(
                                    anchors_Nx3,
                                    distances[k],
                                )
                            except Exception:
                                posicoes[k] = np.nan

                except Exception as e:
                    # Em caso de falha numérica, mantém última posição conhecida
                    if i > 0:
                        posicoes[i] = posicoes[i - 1]
                    warnings.warn(f"[{nome}] amostra {i}: {e}")

        # Calcula RMSE se temos ground truth
        rmse_xy = rmse_xyz = None
        if p_true is not None:
            p_true_arr = np.asarray(p_true, dtype=float)

            n = min(len(posicoes), len(p_true_arr))
            pos_xy = np.asarray(posicoes[:n, :2], dtype=float)

            if p_true_arr.ndim == 2 and p_true_arr.shape[1] >= 2:
                true_xy = np.asarray(p_true_arr[:n, :2], dtype=float)
            else:
                raise ValueError(f"p_true inválido para cálculo de erro: shape={p_true_arr.shape}")

            err_xy = pos_xy - true_xy
            err_pos = np.linalg.norm(err_xy, axis=1)

            # RMSE posicional 2D
            rmse_xy = float(np.sqrt(np.mean(err_pos**2)))

            # rmse_xyz só faz sentido se ambos tiverem 3 colunas
            if p_true_arr.ndim == 2 and p_true_arr.shape[1] >= 3 and posicoes.shape[1] >= 3:
                err_xyz = np.asarray(posicoes[:n, :3], dtype=float) - np.asarray(p_true_arr[:n, :3], dtype=float)
                rmse_xyz = float(np.sqrt(np.mean(np.sum(err_xyz**2, axis=1))))
            else:
                rmse_xyz = None

        resultados[nome] = {
            "posicoes":  posicoes,
            "rmse_xy":   rmse_xy,
            "rmse_xyz":  rmse_xyz,
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