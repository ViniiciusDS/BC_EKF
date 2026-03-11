# src/experiments/monte_carlo_runner.py
"""
Runner de experimentos Monte Carlo.

Executa múltiplas simulações com diferentes seeds,
coleta estatísticas e gera relatórios.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import json
import time
import math
import os


from src.simulator import Simulator
from src.uwb.algoritmos_step import criar_localizadores, monte_carlo
from src.uwb.node_params_serialization import dict_to_node_params
from src.environment import Environment


@dataclass
class ResultadoMC:
    """Resultado de uma rodada de Monte Carlo."""
    algoritmo:  str
    rmse_xy:    float
    rmse_xyz:   float
    taxa_falha: float
    posicoes:   np.ndarray = field(repr=False)

@dataclass
class MonteCarloConfig:
    """Configuração de um experimento Monte Carlo."""
    # Arquivos
    route_file: str           # "ensaio_1.json"
    anchors_file: str         # "cba2024_sala1.json"
    
    # Algoritmos
    algoritmos: List[str]     # ["lms", "gauss_newton", "lmsp"]
    
    # Seeds
    seeds: List[int]          # [42, 123, 456, ...]

    # Mapa
    map_file: str
    
    # Flags perfect
    perfect_motion: bool = True
    perfect_odometry: bool = False
    perfect_uwb: bool = False
    perfect_filter_model: bool = False
    
    # Output
    output_dir: str = "resultados/monte_carlo"
    experiment_name: str = "exp_001"


@dataclass
class MonteCarloProgress:
    """Estado do progresso do Monte Carlo."""
    total_runs: int
    completed_runs: int
    current_seed: int
    elapsed_time: float
    eta_seconds: float
    intermediate_data: Optional[dict] = None  # Para dados parciais durante a execução
    
    @property
    def progress(self) -> float:
        """0.0 - 1.0"""
        if self.total_runs == 0:
            return 0.0
        return self.completed_runs / self.total_runs


@dataclass
class MonteCarloResults:
    """Resultados completos de um Monte Carlo."""
    config: MonteCarloConfig
    resultados_por_algo: Dict[str, List[ResultadoMC]]  # nome → [resultado seed1, seed2, ...]
    execution_time_s: float
    
    def estatisticas(self) -> Dict[str, Dict]:
        """Calcula estatísticas agregadas."""
        stats = {}
        
        for algo, resultados in self.resultados_por_algo.items():
            rmse_xy_array = np.array([r.rmse_xy for r in resultados])
            rmse_xyz_array = np.array([r.rmse_xyz for r in resultados])
            taxa_falha_array = np.array([r.taxa_falha for r in resultados])
            
            stats[algo] = {
                "rmse_xy_mean": float(np.mean(rmse_xy_array)),
                "rmse_xy_std": float(np.std(rmse_xy_array)),
                "rmse_xy_min": float(np.min(rmse_xy_array)),
                "rmse_xy_max": float(np.max(rmse_xy_array)),
                "rmse_xy_median": float(np.median(rmse_xy_array)),
                
                "rmse_xyz_mean": float(np.mean(rmse_xyz_array)),
                "rmse_xyz_std": float(np.std(rmse_xyz_array)),
                
                "taxa_falha_mean": float(np.mean(taxa_falha_array)),
                "taxa_falha_max": float(np.max(taxa_falha_array)),
                
                "n_runs": len(resultados),
            }
        
        return stats
    
    def export_csv(self, path: str):
        """Exporta resultados para CSV."""
        import pandas as pd
        
        rows = []
        for algo, resultados in self.resultados_por_algo.items():
            for i, (seed, r) in enumerate(zip(self.config.seeds, resultados)):
                rows.append({
                    "algoritmo": algo,
                    "seed": seed,
                    "run": i,
                    "rmse_xy": r.rmse_xy,
                    "rmse_xyz": r.rmse_xyz,
                    "taxa_falha": r.taxa_falha,
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"Resultados exportados: {path}")


class MonteCarloRunner:
    """Executa experimentos Monte Carlo com callback de progresso."""
    
    def __init__(
        self,
        config: MonteCarloConfig,
        shared_uwb,
        progress_callback=None,
    ):
        self.config = config
        self.shared_uwb = shared_uwb
        self.progress_callback = progress_callback
        
        # Carrega configuração
        self._load_config()
    
    def _load_config(self):
        """Carrega rota e âncoras dos arquivos."""
        # Carrega rota
        route_path = Path("routes") / self.config.route_file
        with open(route_path) as f:
            route_data = json.load(f)
        self.waypoints = np.array(route_data["waypoints"])
        self.robot_cfg = route_data.get("robot_config", {})
        
        # Carrega âncoras (com parâmetros)
        anchors_path = Path("anchor_sets") / self.config.anchors_file
        with open(anchors_path) as f:
            anchor_data = json.load(f)
        
        # Atualiza shared_uwb
        self.shared_uwb.anchors_xy = anchor_data["anchors_xy"]
        if "anchor_params" in anchor_data:
            self.shared_uwb.anchor_params = {
                int(k): dict_to_node_params(v)
                for k, v in anchor_data["anchor_params"].items()
            }
        self.shared_uwb.reindex_anchor_params()
        self.shared_uwb.sync_pipeline_from_state()
    
    def run(self) -> MonteCarloResults:
        """Executa o Monte Carlo completo."""
        start_time = time.time()
        total_runs = len(self.config.seeds)
        
        # Dicionário para acumular resultados
        resultados_acumulados = {algo: [] for algo in self.config.algoritmos}
        
        for i, seed in enumerate(self.config.seeds):
            # Atualiza seed do pipeline
            self.shared_uwb.pipeline.seed = seed
            self.shared_uwb.sync_pipeline_from_state()

            env = None
            if getattr(self.config, "map_file", ""):
                map_path = os.path.join("maps", self.config.map_file)
                if os.path.exists(map_path):
                    try:
                        env = Environment.load_json(map_path)
                        print(f"[MC Runner] Mapa carregado: {map_path}")
                    except Exception as e:
                        print(f"[MC Runner] Erro ao carregar mapa '{map_path}': {e}")
                else:
                    print(f"[MC Runner] Mapa não encontrado: {map_path}")
            
            # Cria simulador
            sim = Simulator(
                anchors=self.shared_uwb.anchors_np3(),
                env=env,
                baseline=0.65,
                z_c=0.5,
                Q=np.diag([1e-4, 1e-4, 1e-4]),
                R=np.eye(self.shared_uwb.anchors_np3().shape[1] * 2) * 0.05**2,
                dt=0.05,
                uwb_pipeline=self.shared_uwb.pipeline,
            )

            # Inicializa robô no primeiro waypoint (ao invés de 0,0)
            if len(self.waypoints) > 0:
                sim.robot.x = float(self.waypoints[0][0])
                sim.robot.y = float(self.waypoints[0][1])
                
                # Orientação inicial (direção para o próximo waypoint)
                if len(self.waypoints) > 1:
                    dx = self.waypoints[1][0] - self.waypoints[0][0]
                    dy = self.waypoints[1][1] - self.waypoints[0][1]
                    sim.robot.theta = float(np.arctan2(dy, dx))
                else:
                    sim.robot.theta = 0.0
            
            # Roda simulação
            sim_results = self._run_single_simulation(sim)
            
            # Processa com algoritmos
            algo_results = self._run_algorithms(sim_results)
            
            # Acumula
            for algo, resultado in algo_results.items():
                resultados_acumulados[algo].append(resultado)
            
            # Callback de progresso
            if self.progress_callback:
                elapsed = time.time() - start_time
                runs_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_runs - (i + 1)) / runs_per_sec if runs_per_sec > 0 else 0
                
                # ========== EXTRAI POSIÇÕES PARA VISUALIZAÇÃO ==========
                visualization_data = {}
                for algo, resultado in algo_results.items():
                    if hasattr(resultado, 'posicoes') and resultado.posicoes is not None:
                        positions = resultado.posicoes
                        
                        if len(positions) > 0:
                            # Amostra para não sobrecarregar (máximo 100 pontos por run)
                            step = max(1, len(positions) // 100)
                            sampled = positions[::step]
                            
                            # Converte para lista
                            visualization_data[algo] = sampled.tolist() if hasattr(sampled, 'tolist') else list(sampled)
                        else:
                            print(f"[MC Runner]   -> ATENÇÃO: array vazio!")
                    else:
                        print(f"[MC Runner] Run {i+1}: {algo} NÃO tem posicoes!")
                
                progress = MonteCarloProgress(
                    total_runs=total_runs,
                    completed_runs=i + 1,
                    current_seed=seed,
                    elapsed_time=elapsed,
                    eta_seconds=eta,
                    intermediate_data={'positions': visualization_data} if visualization_data else None,
                )
                
                self.progress_callback(progress)
        
        execution_time = time.time() - start_time
        
        return MonteCarloResults(
            config=self.config,
            resultados_por_algo=resultados_acumulados,
            execution_time_s=execution_time,
        )
    
    def _run_single_simulation(self, sim: Simulator) -> dict:
        """Roda uma simulação completa seguindo os waypoints."""
        
        results = {
            "true_trajectory": [],  # Lista de (x, y, theta)
            "distances": [],        # Lista de medições UWB
            "deviations": [],       # Lista de desvios (se disponível)
            "odometry": [],         # Lista de (v, w) comandados
        }
        
        # Configuração
        max_steps = 1000
        wp_idx = 0
        waypoint_reached_dist = 0.3  # metros
        
        # Loop de simulação
        for step in range(max_steps):
            # ========================================
            # 1. CALCULA VELOCIDADES DE CONTROLE
            # ========================================
            if len(self.waypoints) > 0:
                # Waypoint atual
                wp = self.waypoints[wp_idx % len(self.waypoints)]
                target_x, target_y = wp[0], wp[1]
                
                # Posição e orientação atual do robô
                current_x = sim.robot.x
                current_y = sim.robot.y
                current_theta = sim.robot.theta
                
                # Calcula distância e ângulo até o waypoint
                dx = target_x - current_x
                dy = target_y - current_y
                dist_to_waypoint = math.hypot(dx, dy)
                angle_to_waypoint = math.atan2(dy, dx)
                
                # Diferença de ângulo (normalizada entre -pi e pi)
                angle_diff = math.atan2(
                    math.sin(angle_to_waypoint - current_theta),
                    math.cos(angle_to_waypoint - current_theta)
                )
                
                # Comandos de velocidade (controlador proporcional simples)
                v_cmd = min(dist_to_waypoint, 0.30)  # Limita velocidade linear
                w_cmd = float(np.clip(angle_diff * 2.0, -1.2, 1.2))  # Limita velocidade angular
                
                # Avança para próximo waypoint se chegou perto
                if dist_to_waypoint < waypoint_reached_dist:
                    wp_idx += 1
                    
                    # Para ao completar a rota
                    if wp_idx >= len(self.waypoints):
                        break
            else:
                # Sem waypoints: parado
                v_cmd, w_cmd = 0.0, 0.0
            
            # ========================================
            # 2. EXECUTA PASSO DE SIMULAÇÃO
            # ========================================
            result = sim.step(v_cmd, w_cmd)
            
            # Posição verdadeira
            x_true, y_true, theta_true = result["true"]
            results["true_trajectory"].append((x_true, y_true, theta_true))
            
            # ========================================
            # 3. EXTRAI MEDIÇÕES UWB
            # ========================================
            # sim.step() gera medições internamente via pipeline
            
            # Re-gera medições (mais confiável)
            if hasattr(sim, 'anchors') and sim.anchors is not None:
                x_state = np.array([x_true, y_true, theta_true], dtype=float)

                # Mede front e rear separadamente (sua função suporta "front" | "rear" | "mid")
                z_f, s_f = sim.uwb_pipeline.measure_ranges_and_sigmas(
                    x_state=x_state,
                    anchors=sim.anchors,
                    l=sim.l,
                    tag="front",
                    return_meta=False,
                )
                z_r, s_r = sim.uwb_pipeline.measure_ranges_and_sigmas(
                    x_state=x_state,
                    anchors=sim.anchors,
                    l=sim.l,
                    tag="rear",
                    return_meta=False,
                )

                # Intercala no formato que você já usa: [front0, rear0, front1, rear1, ...]
                N = z_f.shape[0]
                z_k = np.empty((2 * N,), dtype=float)
                s_k = np.empty((2 * N,), dtype=float)

                z_k[0::2] = z_f
                z_k[1::2] = z_r
                s_k[0::2] = s_f
                s_k[1::2] = s_r

                results["distances"].append(z_k)
                results["deviations"].append(s_k)
            
            # ========================================
            # 4. SALVA ODOMETRIA
            # ========================================
            results["odometry"].append((v_cmd, w_cmd))
            
            # Progresso (opcional: callback intermediário a cada 50 steps)
            if step % 50 == 0 and step > 0:
                print(f"  Simulação: {step}/{max_steps} steps")
        
        # Converte listas para arrays
        results["true_trajectory"] = np.array(results["true_trajectory"])
        
        # Distances: lista de arrays → array 2D
        if results["distances"]:
            results["distances"] = np.array(results["distances"])
        else:
            results["distances"] = np.array([])
        
        # Deviations: pode ser None
        if results["deviations"] and results["deviations"][0] is not None:
            results["deviations"] = np.array(results["deviations"])
        else:
            results["deviations"] = None
        
        # Odometry
        results["odometry"] = results["odometry"]  # Lista de tuplas
        
        print(f"  Simulação completa: {len(results['distances'])} amostras")
        
        return results
    
    def _run_algorithms(self, sim_results: dict) -> Dict[str, ResultadoMC]:
        """Processa dados com todos os algoritmos."""
        from src.uwb.algoritmos_step import monte_carlo
        
        # Tag frontal (índices pares)
        distances_full = sim_results["distances"]  # (M, 2N)
        distances_front = distances_full[:, 0::2]   # (M, N)

        # Ground truth (x, y)
        p_true = sim_results["true_trajectory"][:, :2]  # (M, 2)
        
        # Desvios
        deviations = sim_results["deviations"]

        deviations_full = deviations  # (M, 2N) ou None

        deviations_front = None
        if deviations_full is not None:
            deviations_front = deviations_full[:, 0::2]   # (M, N)
        
        # Odometria
        odometry = sim_results["odometry"]
        
        # Âncoras
        anchors_Nx3 = self.shared_uwb.anchors_np3().T  # (N, 3)
        
        # ========================================
        # LOOP: Roda cada algoritmo separadamente
        # ========================================
        resultados = {}
        
        for algo_nome in self.config.algoritmos:
            try:
                if algo_nome == "bc_ekf":
                    distances_to_use = distances_full  # (M, 2N) - ambas as tags
                    deviations_to_use = deviations_full  # (M, 2N) - ambas as tags

                else:
                    distances_to_use = distances_front  # (M, N) - só frontal
                    deviations_to_use = deviations_front  # (M, N) - só frontal

                # Chama monte_carlo para UM algoritmo
                posicoes_arr, errors = monte_carlo(
                    algo_nome,        # ← Nome do algoritmo (string)
                    anchors_Nx3,      # ← Âncoras
                    distances_to_use,  # ← Distâncias
                    deviations_to_use,       # ← Desvios
                    p_true,           # ← Ground truth
                    odometry=odometry if algo_nome == "bc_ekf" else None
                )
                

                # Calcula métricas
                if errors is not None:
                    # Remove NaNs antes de calcular RMSE
                    valid_errors = errors[~np.isnan(errors)]
                    rmse_xy = float(np.sqrt(np.mean(valid_errors**2))) if len(valid_errors) > 0 else np.nan
                else:
                    rmse_xy = np.nan
                
                # Cria ResultadoMC
                
                resultados[algo_nome] = ResultadoMC(
                    algoritmo=algo_nome,
                    rmse_xy=rmse_xy,
                    rmse_xyz=rmse_xy,  # Simplificado (2D)
                    taxa_falha=0.0,    # TODO: calcular da função
                    posicoes=posicoes_arr
                )
                
                print(f"[MC]   {algo_nome}: RMSE = {rmse_xy:.4f}m")
            
            except Exception as e:
                print(f"[MC] ERRO em {algo_nome}: {e}")
                import traceback
                traceback.print_exc()
                
                # Resultado vazio
                resultados[algo_nome] = ResultadoMC(
                    algoritmo=algo_nome,
                    rmse_xy=np.nan,
                    rmse_xyz=np.nan,
                    taxa_falha=1.0,
                    posicoes=np.array([])
                )
        
        return resultados

