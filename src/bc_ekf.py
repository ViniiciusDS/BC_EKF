# bc_ekf.py
# EKF (passo unico + wrapper) para robô diferencial com UWB
import numpy as np
from typing import Optional

from src.fusion.uwb_ekf_update import (
    correct_uwb_ranges, compact_uwb_diagnostic,
)

def run_bc_ekf(
    T,
    t_final,
    anchors,
    v_true=0.3,
    w_true=np.deg2rad(7.5),
    l=0.65/2,
    z_c=0.5,
    sigma_v=0.02,
    sigma_w=0.05,
    sigma_uwb=np.sqrt(0.0025),
    x0=None,
    tag_offsets_xyz=None,
):
    """
    Executa o EKF em uma simulação com velocidades constantes.
    """
    t = np.arange(0, t_final + T, T)
    num_anchors = anchors.shape[1]
    update_ratio = max(1, int((1/T) / 5))  # Evita divisor zero se T>0.2s

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    if x0 is None:
        x_hist_true[:, 0] = [2.5, 0.0, 0.0]
    else:
        x_hist_true[:, 0] = np.asarray(x0, dtype=float).reshape(3,)

    for k in range(1, len(t)):
        theta = x_hist_true[2,k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_true*T*np.cos(theta + w_true*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_true*T*np.sin(theta + w_true*T/2)
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k-1] + w_true*T), np.cos(x_hist_true[2,k-1] + w_true*T))

    # Odometria ruidosa
    input_noisy = np.vstack([
        v_true + sigma_v * np.random.randn(len(t)),
        w_true + sigma_w * np.random.randn(len(t))
    ])

    # Medidas UWB
    z_hist = _generate_uwb_measurements(x_hist_true, anchors, l, z_c, sigma_uwb, tag_offsets_xyz=tag_offsets_xyz)

    # Estimativa inicial e covariância
    x_est = x_hist_true[:, 0].copy()
    P = np.diag([0.1,0.1,0.1])
    Q = np.diag([1e-4]*3)
    R = np.diag([sigma_uwb**2]*(2*num_anchors))
    x_hist_est = np.zeros((3, len(t)))
    x_hist_est[:,0] = x_est

    # Loop EKF
    for k in range(1, len(t)):
        v_k = input_noisy[0,k]
        w_k = input_noisy[1,k]
        # Simulador sintético: passo constante T. O código anterior
        # referenciava 'time_axis' sem defini-lo nesse wrapper.
        x_pred, A_k = _predict_state(x_est, v_k, w_k, T)
        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio == 0:
            h_pred, H_k = _measurement_model(
                x_pred, anchors, l, z_c, tag_offsets_xyz=tag_offsets_xyz
            )
            x_est, P, _ = correct_uwb_ranges(
                x_pred, P_pred, z_hist[:,k], h_pred, H_k, R
            )
        else:
            x_est = x_pred
            P = P_pred

        x_hist_est[:,k] = x_est

    rmse_pos, rmse_heading = _compute_rmse(x_hist_true, x_hist_est)

    return rmse_pos, rmse_heading, t, x_hist_true, x_hist_est


def run_bc_ekf_custom_commands(
    T,
    t_final,
    anchors,
    v_commands,
    w_commands,
    l=0.65/2,
    z_c=0.5,
    sigma_v=0.02,
    sigma_w=0.05,
    sigma_uwb=np.sqrt(0.0025),
    x0=None,
    tag_offsets_xyz=None,
):
    """
    Executa EKF com comandos de velocidade variáveis ao longo do tempo.
    """
    t = np.arange(0, t_final, T)
    num_anchors = anchors.shape[1]
    update_ratio = max(1, int((1/T)/5))

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    if x0 is None:
        x_hist_true[:, 0] = [2.5, 0.0, 0.0]
    else:
        x_hist_true[:, 0] = np.asarray(x0, dtype=float).reshape(3,)

    for k in range(1, len(t)):
        v_k = v_commands[k]
        w_k = w_commands[k]
        theta = x_hist_true[2,k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_k*T*np.cos(theta + w_k*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_k*T*np.sin(theta + w_k*T/2)
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k-1]+w_k*T), np.cos(x_hist_true[2,k-1]+w_k*T))

    # Odometria ruidosa
    input_noisy = np.vstack([
        v_commands + sigma_v*np.random.randn(len(t)),
        w_commands + sigma_w*np.random.randn(len(t))
    ])

    # Medidas UWB
    z_hist = _generate_uwb_measurements(x_hist_true, anchors, l, z_c, sigma_uwb, tag_offsets_xyz=tag_offsets_xyz)

    # Inicialização
    x_est = x_hist_true[:, 0].copy()
    P = np.diag([0.1,0.1,0.1])
    Q = np.diag([1e-4]*3)
    R = np.diag([sigma_uwb**2]*(2*num_anchors))
    x_hist_est = np.zeros((3, len(t)))
    x_hist_est[:,0] = x_est

    for k in range(1, len(t)):
        v_k = input_noisy[0,k]
        w_k = input_noisy[1,k]
        x_pred, A_k = _predict_state(x_est, v_k, w_k, T)
        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio == 0:
            h_pred,H_k=_measurement_model(
                x_pred,anchors,l,z_c,tag_offsets_xyz=tag_offsets_xyz
            )
            x_est,P,_=correct_uwb_ranges(
                x_pred,P_pred,z_hist[:,k],h_pred,H_k,R
            )
        else:
            x_est=x_pred
            P=P_pred

        x_hist_est[:,k] = x_est

    rmse_pos, rmse_heading = _compute_rmse(x_hist_true, x_hist_est)

    return rmse_pos, rmse_heading, t, x_hist_true, x_hist_est

def run_bc_ekf_from_data(
    T,
    anchors,
    odometry_noisy,
    z_hist,
    l,
    z_c,
    sigma_uwb,
    x0=None,
    update_hz=5.0,
    tag_offsets_xyz=None,
    timestamps_s=None,
    *,
    min_valid_ranges=1,
    nis_gate_per_dof=None,
    range_gate_sigma=None,
    return_diagnostics=False,
):
    """BC-EKF de duas tags: odometria contínua e correção UWB parcial.

    Compatibilidade:
    - Sem timestamps_s: usa T constante e agenda legacy por
      update_hz. A saída padrão continua sendo matriz (3,N).
    - Com timestamps_s: usa dt real em cada transição, escala
      o ruído de processo por dt/T e processa cada linha que
      contiver pelo menos uma medição UWB nova e válida.
      Ausências (NaN/Inf/range<=0) não invalidam os demais ranges.

    Gating por NIS/dof e resíduos individuais é opcional e DESLIGADO
    por padrão. NIS descreve consistência estatística do EKF, não GT.
    return_diagnostics=True retorna (poses, {summary, steps}).
    """
    T=float(T)
    if not np.isfinite(T) or T<=0:
        raise ValueError(f"T inválido no BC-EKF: {T}")
    update_hz=float(update_hz)
    if not np.isfinite(update_hz) or update_hz<=0:
        raise ValueError("update_hz deve ser positivo e finito")
    sigma_uwb=float(sigma_uwb)
    if not np.isfinite(sigma_uwb) or sigma_uwb<=0:
        raise ValueError("sigma_uwb precisa ser positivo e finito")
    if type(min_valid_ranges) is not int or min_valid_ranges<1:
        raise ValueError("min_valid_ranges deve ser inteiro positivo")
    for name,value in (("nis_gate_per_dof",nis_gate_per_dof),
                       ("range_gate_sigma",range_gate_sigma)):
        if value is not None and (not np.isscalar(value)
                or not np.isfinite(value) or float(value)<=0):
            raise ValueError(f"{name} deve ser positivo ou None")

    anchors=np.asarray(anchors,dtype=float)
    odo=np.asarray(odometry_noisy,dtype=float)
    z=np.asarray(z_hist,dtype=float)
    if anchors.ndim!=2 or anchors.shape[0]!=3 or not np.isfinite(anchors).all():
        raise ValueError("anchors deve ser matriz 3xA finita")
    if odo.ndim!=2 or odo.shape[0]!=2 or not np.isfinite(odo).all():
        raise ValueError("odometry_noisy deve ser matriz 2xN finita")
    num_steps=odo.shape[1]
    if num_steps<1:
        raise ValueError("odometry_noisy precisa de ao menos uma amostra")
    m=2*anchors.shape[1]
    if z.shape!=(m,num_steps):
        raise ValueError(
            f"z_hist possui shape {z.shape}, esperado ({m},{num_steps})"
        )

    time_axis=None
    if timestamps_s is not None:
        time_axis=np.asarray(timestamps_s,dtype=float)
        if (time_axis.shape!=(num_steps,) or not np.isfinite(time_axis).all()
                or np.any(np.diff(time_axis)<=0)):
            raise ValueError(
                "timestamps_s precisa conter N instantes crescentes e finitos"
            )
    initial=np.array([2.5,0.,0.],dtype=float) if x0 is None else (
        np.asarray(x0,dtype=float).reshape(3).copy()
    )
    if not np.isfinite(initial).all():
        raise ValueError("x0 precisa ser finito")

    x_est=initial
    P=np.diag([.1,.1,.1])
    Q=np.diag([1e-4]*3)
    R=np.eye(m)*(sigma_uwb*sigma_uwb)
    states=np.zeros((3,num_steps),dtype=float)
    states[:,0]=x_est
    update_ratio=max(1,int(round((1./update_hz)/T)))
    summary={
        "pcnt_samples":int(num_steps),
        "anchor_count":int(anchors.shape[1]),
        "sampling_mode":"real_dt" if time_axis is not None else "fixed_dt_legacy",
        "nominal_T_s":T,
        "scheduled_update_ratio":None if time_axis is not None else update_ratio,
        "updates_accepted":0,
        "partial_updates_accepted":0,
        "nis_gate_rejected":0,
        "range_gate_rejected":0,
        "insufficient_ranges":0,
        "prediction_only_steps":0,
        "max_nis_per_dof":None,
        "max_innovation_rms_m":None,
        "min_dt_s":None,
        "max_dt_s":None,
    }
    reports=[]
    for k in range(1,num_steps):
        dt=float(time_axis[k]-time_axis[k-1]) if time_axis is not None else T
        x_pred,A=_predict_state(x_est,odo[0,k],odo[1,k],dt)
        # Q é definido por passo nominal T, por isso escala com dt.
        q_effective=Q*(dt/T) if time_axis is not None else Q
        P_pred=A@P@A.T+q_effective
        if summary["min_dt_s"] is None or dt<summary["min_dt_s"]:
            summary["min_dt_s"]=dt
        if summary["max_dt_s"] is None or dt>summary["max_dt_s"]:
            summary["max_dt_s"]=dt

        should_update=(time_axis is not None or k%update_ratio==0)
        values=z[:,k]
        if should_update and np.any(np.isfinite(values)&(values>0)):
            h,H=_measurement_model(
                x_pred,anchors,l,z_c,tag_offsets_xyz=tag_offsets_xyz
            )
            x_est,P,detail=correct_uwb_ranges(
                x_pred,P_pred,values,h,H,R,
                min_valid_ranges=min_valid_ranges,
                nis_gate_per_dof=nis_gate_per_dof,
                range_gate_sigma=range_gate_sigma,
            )
        else:
            x_est,P=x_pred,P_pred
            detail={
                "status":("not_scheduled" if not should_update
                          else "no_valid_ranges"),
                "total_ranges":m,
                "valid_ranges":int(np.count_nonzero(
                    np.isfinite(values)&(values>0)
                )),
                "used_ranges":0,"dropped_ranges":m,
                "used_indices":np.empty(0,dtype=int),
                "nis":None,"nis_per_dof":None,
                "innovation_rms_m":None,
                "max_abs_innovation_m":None,
                "max_standardized_innovation":None,
            }
        status=detail["status"]
        if status=="accepted":
            summary["updates_accepted"]+=1
            if detail["used_ranges"]<m:
                summary["partial_updates_accepted"]+=1
        else:
            summary["prediction_only_steps"]+=1
            if status=="nis_gate_rejected":
                summary["nis_gate_rejected"]+=1
            elif status=="range_gate_rejected":
                summary["range_gate_rejected"]+=1
            elif status=="insufficient_ranges":
                summary["insufficient_ranges"]+=1
        for key,name in (("max_nis_per_dof","nis_per_dof"),
                         ("max_innovation_rms_m","innovation_rms_m")):
            v=detail[name]
            if v is not None:
                summary[key]=v if summary[key] is None else max(summary[key],v)
        states[:,k]=x_est
        if return_diagnostics:
            reports.append(compact_uwb_diagnostic(
                detail,index=k,
                timestamp_s=float(time_axis[k]) if time_axis is not None else k*T,
                dt_s=dt,
            ))
    if not return_diagnostics:
        return states
    return states,{
        "schema":"bc_ekf.range_innovations.v1",
        "summary":summary,
        "steps":reports,
        "config":{
            "min_valid_ranges":min_valid_ranges,
            "nis_gate_per_dof":nis_gate_per_dof,
            "range_gate_sigma":range_gate_sigma,
            "sigma_uwb_m":sigma_uwb,
        },
        "caveats":[
            "NIS é estatística de consistência, não RMSE nem ground truth.",
            "Limiares de gating são opcionais; sem ajuste de Q/R "
            "e sem sincronização física UWB não garantem rejeição ótima.",
        ],
    }


def run_bc_ekf_step(
        x_est: np.ndarray,
        P: np.ndarray, 
        u_k: np.ndarray, 
        z_k: np.ndarray, 
        anchors: np.ndarray, 
        l: float, 
        z_c: float, 
        Q: np.ndarray, 
        R: np.ndarray, 
        dt: Optional[float] = None, 
        debug: bool = False,
        tag_offsets_xyz=None,
        min_valid_ranges=1,
        nis_gate_per_dof=None,
        range_gate_sigma=None,
        ):
    """
    Executa UM passo do BC-EKF (predição +, opcionalmente, correção).
    - Aceita z_k vazio e ignora a correção se não houver medições.
    - dt explicito para coerencia com o modelo discreto.
    - Quando debug=True, retorna também um dicionário de diagnósticos com:
      {'innov': y, 'S': S, 'K': K, 'h': h, 'H': H}

    Args:
    x_est: estado estimado corrente (shape (3,))
    P: matriz de covariância atual (3x3)
    u_k: comando de controle [v, w]
    z_k: vetor de medições UWB (2*N anchors) ou vazio
    anchors: matriz 3xN com posições das âncoras
    l: metade do baseline entre as tags
    z_c: altura das tags
    Q: covariância do processo (3x3)
    R: covariância das medições (2N x 2N ou vazia)
    dt: passo de tempo
    debug: se True, retorna também dicionário de debug

    Retornos:
      - Se debug=False (padrão): (x_next, P_next)
      - Se debug=True: (x_next, P_next, diag_dict)
    """
    if dt is None:
        raise ValueError("run_bc_ekf_step: dt é None; passe dt=... (time step)")
    v, w = u_k

    # --- Predição discreta com dt ---
    theta = x_est[2]
    dx   = v * dt * np.cos(theta + w*dt/2.0)
    dy   = v * dt * np.sin(theta + w*dt/2.0)
    dth  = w * dt

    x_pred = x_est + np.array([dx, dy, dth])
    x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

    # Jacobiano do processo (discreto) w.r.t. estado
    A = np.array([
        [1, 0, -v * dt * np.sin(theta + w*dt/2.0)],
        [0, 1,  v * dt * np.cos(theta + w*dt/2.0)],
        [0, 0, 1]
    ])
    P_pred = A @ P @ A.T + Q

    # --- Correção UWB ---
    # Em step, R None / z None preservam o comportamento legado:
    # somente a predição. O debug agora informa status e dt.
    def prediction_only(status, warning=None):
        if not debug:
            return x_pred,P_pred
        result={
            "innov":None,"S":None,"K":None,"h":None,"H":None,
            "x_pred":x_pred.copy(),"P_pred":P_pred.copy(),
            "dt":float(dt),"status":status,
            "nis":None,"nis_per_dof":None,
            "used_indices":np.empty(0,dtype=int),"valid_ranges":0,
        }
        if warning is not None:
            result["warn"]=warning
        return x_pred,P_pred,result

    if z_k is None or R is None or np.size(R)==0 or np.size(z_k)==0:
        return prediction_only("no_valid_ranges")
    z=np.asarray(z_k,dtype=float)
    if z.shape!=(2*anchors.shape[1],):
        raise ValueError("z_k deve ter 2 ranges por âncora")
    matrix=np.asarray(R,dtype=float)
    if matrix.shape!=(len(z),len(z)):
        raise ValueError("R deve ser matriz 2A x 2A")
    if not np.any(np.isfinite(z)&(z>0)):
        return prediction_only("no_valid_ranges")

    try:
        h,H=_measurement_model(
            x_pred,anchors,l,z_c,tag_offsets_xyz=tag_offsets_xyz
        )
    except ValueError as exc:
        if "Distância tag-âncora" not in str(exc):
            raise
        return prediction_only("geometry_unavailable",str(exc))

    x_upd,P_upd,detail=correct_uwb_ranges(
        x_pred,P_pred,z,h,H,matrix,
        min_valid_ranges=min_valid_ranges,
        nis_gate_per_dof=nis_gate_per_dof,
        range_gate_sigma=range_gate_sigma,
    )
    if not debug:
        return x_upd,P_upd
    return x_upd,P_upd,{
        "innov":detail["innovation"],
        "S":detail["S"],"K":detail["K"],
        "h":detail["h"],"H":detail["H"],
        "x_pred":x_pred.copy(),"P_pred":P_pred.copy(),
        "dt":float(dt),
        "status":detail["status"],
        "used_indices":detail["used_indices"].copy(),
        "valid_ranges":detail["valid_ranges"],
        "used_ranges":detail["used_ranges"],
        "nis":detail["nis"],
        "nis_per_dof":detail["nis_per_dof"],
        "innovation_rms_m":detail["innovation_rms_m"],
        "max_abs_innovation_m":detail["max_abs_innovation_m"],
    }

# ======================
# Funções auxiliares
# ======================
def _predict_state(x_est, v, w, T):
    """
    Prediz o próximo estado do robô com base nas entradas de velocidade e no modelo de movimento diferencial.
    
    Args:
        x_est (ndarray): Estado estimado atual [x, y, theta].
        v (float): Velocidade linear.
        w (float): Velocidade angular.
        T (float): Intervalo de tempo (s).
        
    Returns:
        x_pred (ndarray): Estado previsto [x, y, theta].
        A_k (ndarray): Jacobiano da função de transição de estado.
    """
    theta = x_est[2]
    dx = v * T * np.cos(theta + w * T / 2)
    dy = v * T * np.sin(theta + w * T / 2)
    dtheta = w * T

    x_pred = x_est + np.array([dx, dy, dtheta])
    x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))  # normaliza ângulo

    A_k = np.array([
        [1, 0, -v * T * np.sin(theta + w * T / 2)],
        [0, 1,  v * T * np.cos(theta + w * T / 2)],
        [0, 0, 1]
    ])
    return x_pred, A_k


def _resolved_tag_offsets(l, z_c, tag_offsets_xyz=None):
    """Matriz 2x3 em referencial robô (x frente, y esquerda, z cima).

    Ausência da opção reproduz o modelo histórico: frente/trás
    com distância 2*l e altura z_c. Não reordena os canais do log.
    """
    if tag_offsets_xyz is None:
        offsets = np.array([[l, 0., z_c], [-l, 0., z_c]], dtype=float)
    else:
        offsets = np.asarray(tag_offsets_xyz, dtype=float)
    if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
        raise ValueError("tag_offsets_xyz exige duas coordenadas XYZ finitas")
    return offsets


def _measurement_model(x_pred, anchors, l, z_c, tag_offsets_xyz=None):
    """Ranges e Jacobiano para tags XYZ arbitrárias, com canais fixos.

    Para cada tag j, offset local (a,b,c) vira no mundo:
    xt=x+cos(theta)*a-sin(theta)*b
    yt=y+sin(theta)*a+cos(theta)*b
    zt=c
    h_ij=norm([xt,yt,zt] - anchor_i)
    d(xt)/dtheta=-sin(theta)*a-cos(theta)*b
    d(yt)/dtheta= cos(theta)*a-sin(theta)*b

    Ordem h/H: [âncora0_canal0, âncora0_canal1, ...].
    """
    anchors=np.asarray(anchors,dtype=float)
    if anchors.ndim!=2 or anchors.shape[0]!=3 or not np.isfinite(anchors).all():
        raise ValueError("anchors deve ser matriz 3xN finita")
    p=np.asarray(x_pred,dtype=float).reshape(3)
    offsets=_resolved_tag_offsets(l,z_c,tag_offsets_xyz)
    ct,st=np.cos(p[2]),np.sin(p[2])
    h=np.empty(2*anchors.shape[1],dtype=float)
    H=np.empty((2*anchors.shape[1],3),dtype=float)
    for i in range(anchors.shape[1]):
        for j,(ox,oy,oz) in enumerate(offsets):
            tx=p[0]+ct*ox-st*oy
            ty=p[1]+st*ox+ct*oy
            dtx=-st*ox-ct*oy
            dty= ct*ox-st*oy
            delta=np.array([tx-anchors[0,i],ty-anchors[1,i],oz-anchors[2,i]])
            dist=float(np.linalg.norm(delta))
            if dist<1e-9:
                raise ValueError("Distância tag-âncora ~ 0")
            k=2*i+j
            h[k]=dist
            H[k,0]=delta[0]/dist
            H[k,1]=delta[1]/dist
            H[k,2]=(delta[0]*dtx+delta[1]*dty)/dist
    return h,H


def _generate_uwb_measurements(
    x_hist, anchors, l, z_c, sigma_uwb, tag_offsets_xyz=None
):
    """Gera 2N ranges por passo usando a MESMA geometria do filtro.

    Importa o ruído sintético do projeto somente quando usado, para
    permitir testes NumPy-only do BC-EKF com dados reais no CI.
    """
    import src.utils as utils
    num_anchors=anchors.shape[1]
    z_hist=np.empty((2*num_anchors,x_hist.shape[1]),dtype=float)
    for k in range(x_hist.shape[1]):
        expected,_=_measurement_model(
            x_hist[:,k],anchors,l,z_c,tag_offsets_xyz=tag_offsets_xyz
        )
        for j,dist in enumerate(expected):
            z_hist[j,k]=utils.apply_uwb_errors(dist,sigma_uwb)
    return z_hist


def _compute_rmse(x_true, x_est):
    """
    Calcula RMSE de posição e orientação ao longo da trajetória.
    
    Args:
        x_true (ndarray): Trajetória real.
        x_est (ndarray): Trajetória estimada.
        
    Returns:
        rmse_pos (float): RMSE da posição Euclidiana.
        rmse_heading (float): RMSE do heading (graus).
    """
    error = x_true - x_est
    error[2, :] = np.arctan2(np.sin(error[2, :]), np.cos(error[2, :]))
    pos_error = np.linalg.norm(error[0:2, :], axis=0)
    heading_error_deg = np.abs(error[2, :]) * (180 / np.pi)
    rmse_pos = np.sqrt(np.mean(pos_error ** 2))
    rmse_heading = np.sqrt(np.mean(heading_error_deg ** 2))
    return rmse_pos, rmse_heading