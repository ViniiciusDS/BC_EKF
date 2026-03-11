# src/uwb/algoritmo_step.py
"""
Algoritmos de localização UWB — versões por step (com estado).

Cada classe processa UMA medição por vez e mantém histórico interno.
São os mesmos algoritmos de algoritmos_estaticos.py, encapsulados para
integração com o loop de simulação (Simulator.step) e testes Monte Carlo.

Uso típico:
    loc = LocalizadorLMS(anchors)
    for dist, dev in stream:
        pos = loc.step(dist, dev)   # → np.ndarray[3] ou None

    # Acessar histórico
    hist = np.array(loc.historico)   # (N, 3)
    rmse = loc.rmse_xy(p_true)

Registry de algoritmos (para UI):
    from src.uwb.algoritmo_step import ALGORITMOS
    loc = ALGORITMOS["lms"](anchors)
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Type, List
from abc import ABC, abstractmethod

from src.uwb.algoritmos_estaticos import (
    trilaterate3d, lms, gauss_newton, lmsp
)



#####################
# BASE
#####################
class LocalizadorBase(ABC):
    """
    Interface comum para todos os localizadores.

    Subclasses implementam _compute(distances, deviations) → np.ndarray[3].
    Esta base cuida do histórico, contadores e RMSE.
    """

    #: Nome legível para UI / logs
    NOME: str = "base"
    #: Cor padrão para plots (RGB 0-255)
    COR: tuple = (100, 100, 100)

    def __init__(
        self,
        anchors: np.ndarray,
        nome_override: Optional[str] = None,
    ) -> None:
        self.anchors = np.asarray(anchors, dtype=float)   # (N, 3)
        if nome_override:
            self.NOME = nome_override
        self.reset()

    def reset(self) -> None:
        """Limpa histórico e reinicia o estado interno."""
        self.historico: list[np.ndarray] = []
        self.n_total:   int  = 0
        self.n_falhas:  int  = 0
        self._ultimo:   Optional[np.ndarray] = None
        self._reset_interno()

    def _reset_interno(self) -> None:
        """Hook para subclasses com estado (ex: warm start, EKF)."""
        pass


    def step(
        self,
        distances:  np.ndarray,
        deviations: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Processa uma única medição.

        Parâmetros:
            distances  : array (N,) — distâncias por âncora (m)
            deviations : array (N,) — desvios padrão (m); None se não disponível

        Retorna:
            pos : array (3,) — posição estimada, ou None se falhou
        """
        distances = np.asarray(distances, dtype=float)
        if deviations is not None:
            deviations = np.asarray(deviations, dtype=float)

        self.n_total += 1

        try:
            pos = self._compute(distances, deviations)
            pos = np.asarray(pos, dtype=float)

            if not np.all(np.isfinite(pos)):
                raise ValueError(f"Posição não-finita: {pos}")

            self._ultimo = pos
            self.historico.append(pos.copy())
            return pos

        except Exception as e:
            self.n_falhas += 1
            # Fallback: repete última posição conhecida
            if self._ultimo is not None:
                self.historico.append(self._ultimo.copy())
                return self._ultimo.copy()
            return None

    # Métricas
    def rmse_xy(self, p_true: Optional[np.ndarray]) -> Optional[float]:
        """RMSE 2D (X,Y) em metros."""
        if not self.historico or p_true is None:
            return None
        p_true = np.asarray(p_true, dtype=float)
        if p_true.ndim == 0 or p_true.size == 0:
            return None
        hist = np.array(self.historico)
        n = min(len(hist), len(p_true))
        err = hist[:n, :2] - p_true[:n, :2]
        return float(np.sqrt(np.mean(err**2)))

    def rmse_xyz(self, p_true: Optional[np.ndarray]) -> Optional[float]:
        """RMSE 3D (X,Y,Z) em metros."""
        if not self.historico or p_true is None:
            return None
        p_true = np.asarray(p_true, dtype=float)
        if p_true.ndim == 0 or p_true.size == 0:
            return None
        hist = np.array(self.historico)
        n = min(len(hist), len(p_true))
        err = hist[:n, :3] - p_true[:n, :3]
        return float(np.sqrt(np.mean(err**2)))

    def taxa_falha(self) -> float:
        """Fração de medições que falharam (0.0 – 1.0)."""
        if self.n_total == 0:
            return 0.0
        return self.n_falhas / self.n_total

    def resumo(self) -> dict:
        """Dict com métricas resumidas para logs e UI."""
        return {
            "algoritmo":  self.NOME,
            "n_total":    self.n_total,
            "n_falhas":   self.n_falhas,
            "taxa_falha": f"{self.taxa_falha()*100:.1f}%",
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n_total}, falhas={self.n_falhas})"


#############################
# 1. TRILATERAÇÃO CLÁSSICA
##############################
class LocalizadorTrilateracao(LocalizadorBase):
    """
    Trilateração 3D clássica — usa apenas as 4 primeiras âncoras.

    Vantagem: simples, determinístico.
    Desvantagem: não aproveita âncoras extras; sensível à geometria das 4 usadas.
    """
    NOME = "trilaterate3d"
    COR  = (220, 50, 50)   # vermelho

    def _compute(self, distances, deviations):
        return trilaterate3d(self.anchors[:4], distances[:4])


#########################
# 2. LMS LINEAR
#########################
class LocalizadorLMS(LocalizadorBase):
    """
    Mínimos Quadrados Lineares com N âncoras.

    Vantagem: usa todas as âncoras, rápido, closed-form.
    Desvantagem: linearização introduz erro de aproximação.
    """
    NOME = "lms"
    COR  = (160, 30, 180)   # magenta/roxo

    def _compute(self, distances, deviations):
        return lms(self.anchors, distances)


######################################
# 3. GAUSS-NEWTON (NNLS iterativo)
######################################
class LocalizadorGaussNewton(LocalizadorBase):
    """
    Refinamento iterativo por Gauss-Newton.

    Usa LMS como ponto inicial (warm start). A convergência é rápida (~6 iter).
    Mantém a última posição estimada como warm start da próxima iteração,
    o que melhora a estabilidade para movimentos suaves.

    Parâmetros:
        n_iter : número de iterações Gauss-Newton (default: 6)
    """
    NOME = "gauss_newton"
    COR  = (30, 30, 30)   # preto

    def __init__(self, anchors, n_iter: int = 6, **kw):
        self.n_iter = n_iter
        super().__init__(anchors, **kw)

    def _reset_interno(self):
        self._p_warm: Optional[np.ndarray] = None

    def _compute(self, distances, deviations):
        # LMS como chute inicial se não temos warm start
        p_init = self._p_warm if self._p_warm is not None else lms(self.anchors, distances)
        pos = gauss_newton(self.anchors, distances, p_init=p_init, n_iter=self.n_iter)
        self._p_warm = pos   # warm start para próximo step
        return pos


###############################
# 4. LMSP — LMS PONDERADO
###############################
class LocalizadorLMSP(LocalizadorBase):
    """
    LMS Ponderado por desvio padrão (W = 1/σ²).

    Desvio padrão fornecido pelo hardware UWB (chips DW1000 exportam esta info).
    Âncoras com maior incerteza recebem peso menor na estimativa.

    Fallback: se deviations não for fornecido, usa LMS simples.
    """
    NOME = "lmsp"
    COR  = (30, 80, 200)   # azul

    def _compute(self, distances, deviations):
        if deviations is None:
            # Fallback silencioso para LMS sem peso
            return lms(self.anchors, distances)
        return lmsp(self.anchors, distances, deviations)


#####################################################################
# 5. BC-EKF WRAPPER — integra o filtro existente na interface step
######################################################################
class LocalizadorBCEKF(LocalizadorBase):
    """
    Wrapper do BC-EKF existente na interface LocalizadorBase.

    Diferente dos algoritmos estáticos, o EKF é dinâmico — precisa de:
        odometria (v_cmd, w_cmd)  → fornecida via step()
        posição anterior           → mantida internamente

    Use step_ekf() para fornecer velocidades, ou step() para só UWB
    (nesse caso usa v=0, w=0 — adequado para robô parado ou dataset estático).

    Parâmetros adicionais:
        baseline : distância entre as tags (m)
        z_c      : altura das tags (m)
        dt       : passo de tempo (s)
        Q, R     : covariâncias do EKF
    """
    NOME = "bc_ekf"
    COR  = (255, 140, 0)   # laranja

    def __init__(
        self,
        anchors:  np.ndarray,
        baseline: float = 0.65,
        z_c:      float = 0.50,
        dt:       float = 0.05,
        Q:        Optional[np.ndarray] = None,
        R:        Optional[np.ndarray] = None,
        x0:       Optional[np.ndarray] = None,
        **kw,
    ) -> None:
        self.baseline = baseline
        self.z_c      = z_c
        self.dt       = dt
        N = np.asarray(anchors).shape[1] if np.asarray(anchors).ndim == 2 and np.asarray(anchors).shape[0] == 3 else np.asarray(anchors).shape[0]
        # Suporte a anchors (3, N) e (N, 3)
        anc = np.asarray(anchors, dtype=float)
        if anc.ndim == 2 and anc.shape[0] == 3:
            anc = anc.T   # normaliza para (N, 3)
        N = anc.shape[0]
        self._Q = Q if Q is not None else np.diag([1e-4, 1e-4, 1e-4])
        self._R = R if R is not None else np.eye(2 * N) * 0.05**2
        self._x0 = x0 if x0 is not None else np.array([0.0, 0.0, 0.0])
        self._v_cmd = 0.0
        self._w_cmd = 0.0
        super().__init__(anc, **kw)

    def _reset_interno(self):
        self._x_est = self._x0.copy() if hasattr(self, '_x0') else np.zeros(3)
        self._P     = np.diag([0.1, 0.1, 0.1])
        self._v_cmd = 0.0
        self._w_cmd = 0.0

    def set_odometry(self, v: float, w: float) -> None:
        """Fornece odometria para o próximo step."""
        self._v_cmd = float(v)
        self._w_cmd = float(w)

    def step_ekf(
        self,
        distances:  np.ndarray,
        v_cmd:      float,
        w_cmd:      float,
        deviations: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Step com odometria explícita."""
        self._v_cmd = float(v_cmd)
        self._w_cmd = float(w_cmd)
        return self.step(distances, deviations)

    def _compute(self, distances, deviations):
        from src.bc_ekf import run_bc_ekf_step
        
        # anchors em formato (3, N) para o EKF
        anc_3xN = self.anchors.T
        l = self.baseline / 2.0
        
        try:
            x_next, P_next = run_bc_ekf_step(  # ← SEM o "_"
                self._x_est, self._P,
                np.array([self._v_cmd, self._w_cmd]),
                distances,
                anc_3xN, l, self.z_c,
                self._Q, self._R,
                dt=self.dt,
            )
            
            self._x_est = x_next
            self._P     = P_next
            
            # Retorna posição (x, y) + z fixo (EKF é 2D)
            result = np.array([x_next[0], x_next[1], self.z_c])
            return result
        
        except Exception as e:
            print(f"[BC-EKF] ERRO: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise


######################################################################
# REGISTRY — mapeamento nome → classe para uso na UI e Monte Carlo
######################################################################
ALGORITMOS: Dict[str, Type[LocalizadorBase]] = {
    "trilaterate3d": LocalizadorTrilateracao,
    "lms":           LocalizadorLMS,
    "gauss_newton":  LocalizadorGaussNewton,
    "lmsp":          LocalizadorLMSP,
    "bc_ekf":        LocalizadorBCEKF,
}

#: Nomes legíveis para UI (labels dos botões/legendas)
NOMES_UI: Dict[str, str] = {
    "trilaterate3d": "(a) Trilateração",
    "lms":           "(b) LMS",
    "gauss_newton":  "(c) Gauss-Newton",
    "lmsp":          "(d) LMS Ponderado",
    "bc_ekf":        "(e) BC-EKF",
}


def criar_localizadores(
    anchors:    np.ndarray,
    algoritmos: Optional[list[str]] = None,
    **kwargs,
) -> Dict[str, LocalizadorBase]:
    """
    Factory: cria um dict de localizadores prontos para uso.

    Parâmetros:
        anchors    : posições das âncoras
        algoritmos : lista de nomes; None → todos exceto bc_ekf
        **kwargs   : repassados para os construtores

    Exemplo:
        locs = criar_localizadores(anchors, ["lms", "gauss_newton"])
        for dist, dev in stream:
            for nome, loc in locs.items():
                pos = loc.step(dist, dev)
    """
    if algoritmos is None:
        algoritmos = ["trilaterate3d", "lms", "gauss_newton", "lmsp"]

    return {
        nome: ALGORITMOS[nome](anchors, **kwargs)
        for nome in algoritmos
        if nome in ALGORITMOS
    }


########################################################
# RUNNER MONTE CARLO (helper para testes massivos)
########################################################

def monte_carlo(
    algoritmo_nome: str,
    anchors_Nx3: np.ndarray,
    distances: np.ndarray,
    deviations: Optional[np.ndarray] = None,
    p_true: Optional[np.ndarray] = None,
    odometry: Optional[List[tuple]] = None
) -> tuple:
    """Executa algoritmo em batch de medições."""
    
    # Cria localizador
    if algoritmo_nome == "bc_ekf":
        
        loc = ALGORITMOS[algoritmo_nome](
            anchors_Nx3,
            baseline=0.65,
            z_c=0.5,
            dt=0.05,
            Q=np.diag([1e-4, 1e-4, 1e-4]),
            R=np.eye(2 * len(anchors_Nx3)) * (0.05**2)
        )

    else:
        loc = ALGORITMOS[algoritmo_nome](anchors_Nx3)
    
    posicoes = []
    
    for i, d in enumerate(distances):
        if algoritmo_nome == "bc_ekf" and i < 5:  # Só primeiros 5 steps          
            if odometry:
                v, w = odometry[i]
                loc.set_odometry(v, w)
        
        elif algoritmo_nome == "bc_ekf" and odometry:
            v, w = odometry[i]
            loc.set_odometry(v, w)
        
        try:
            pos = loc.step(d, deviations[i] if deviations is not None else None)
            
            if pos is not None:
                posicoes.append(pos[:2])
            else:
                if algoritmo_nome == "bc_ekf" and i < 5:
                    print(f"[monte_carlo]   WARNING: pos is None!")
                posicoes.append([np.nan, np.nan])
        
        except Exception as e:
            if algoritmo_nome == "bc_ekf" and i < 5:
                print(f"[monte_carlo]   ERRO no step: {e}")
                import traceback
                traceback.print_exc()
            posicoes.append([np.nan, np.nan])
    
    posicoes_arr = np.array(posicoes)
    
    # Calcula erros
    if p_true is not None:
        errors = np.linalg.norm(posicoes_arr - p_true, axis=1)
    else:
        errors = None
    
    return posicoes_arr, errors