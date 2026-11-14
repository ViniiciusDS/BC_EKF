"""Fusão EKF de odometria diferencial com ranges de UMA tag UWB.

Implementação NumPy independente do Pygame, do PCNT e do LAB7A.
Entradas em SI:
- odometria: timestamps N, v e omega N (velocidades do intervalo k-1→k);
- UWB: ranges MxA, índices de observação crescentes no eixo de N passos,
  sigmas MxA opcionais, âncoras Ax3;
- pose inicial (x,y,theta) e deslocamento rígido da tag no robô.

Não localiza por LS antes da correção: atualiza diretamente cada conjunto
de ranges contemporâneos. Não há interpolação de leituras nem outlier
rejection por inovação (assunto do PR robustez). O filtro não fornece GT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

import numpy as np


def _nonnegative_number(name, value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} deve ser número finito {'positivo' if positive else 'não negativo'}")
    return float(value)


def _wrap(theta):
    """Ângulo em [-pi, pi] sem depender do histórico de giro acumulado."""
    return math.atan2(math.sin(theta), math.cos(theta))


@dataclass(frozen=True)
class SingleTagEkfConfig:
    """Parâmetros independentes de qualquer hardware ou cenário de ensaio.

    Os coeficientes de ruído de processo seguem distância percorrida,
    rotação acumulada e tempo (variâncias aditivas). Sigmas UWB do
    firmware NÃO são automaticamente considerados bem calibrados.
    """
    tag_height_m: float = 0.5
    tag_offset_x_m: float = 0.0
    tag_offset_y_m: float = 0.0
    initial_position_sigma_m: float = 0.6
    initial_heading_sigma_deg: float = 35.0
    range_sigma_floor_m: float = 0.15
    use_reported_sigma: bool = True
    motion_sigma_m_sqrt_m: float = 0.08
    turn_sigma_rad_sqrt_rad: float = 0.12
    stationary_sigma_m_sqrt_s: float = 0.005
    heading_sigma_rad_sqrt_s: float = 0.02
    min_ranges_for_update: int = 3

    def __post_init__(self):
        for name in (
            "tag_height_m", "initial_position_sigma_m",
            "initial_heading_sigma_deg", "range_sigma_floor_m",
            "motion_sigma_m_sqrt_m", "turn_sigma_rad_sqrt_rad",
            "stationary_sigma_m_sqrt_s", "heading_sigma_rad_sqrt_s",
        ):
            _nonnegative_number(
                name, getattr(self, name),
                positive=name in ("initial_position_sigma_m",
                                  "initial_heading_sigma_deg", "range_sigma_floor_m")
            )
        for name in ("tag_offset_x_m", "tag_offset_y_m"):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError(f"{name} deve ser número finito")
        if type(self.use_reported_sigma) is not bool:
            raise ValueError("use_reported_sigma deve ser booleano")
        if type(self.min_ranges_for_update) is not int or self.min_ranges_for_update < 1:
            raise ValueError("min_ranges_for_update deve ser inteiro >=1")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("single_tag_ekf deve ser objeto JSON")
        names={f.name for f in fields(cls)}
        unknown=set(data)-names
        if unknown:
            raise ValueError("Campos desconhecidos em single_tag_ekf: "
                             + ", ".join(sorted(unknown)))
        return cls(**data)


def load_single_tag_ekf_config(anchor_json: str | Path):
    """Configuração opcional para datasets que possuem odometria + UMA tag.

    Sem campo, usa defaults conservadores. tag_height_m é configuração
    da posição física da tag; deve ser conferida no referencial XYZ.
    """
    with Path(anchor_json).open("r", encoding="utf-8") as src:
        root=json.load(src)
    if not isinstance(root, dict):
        raise ValueError("JSON de âncoras precisa ser objeto")
    explicit=root.get("single_tag_ekf", {})
    if not isinstance(explicit,dict):
        raise ValueError("single_tag_ekf deve ser objeto JSON")
    params=dict(explicit)
    if "tag_height_m" not in params:
        # A altura usada para validar os ranges também descreve a
        # posição física da tag; reutilizar SOMENTE se explícita
        # no bloco de pré-processamento. Nunca assumir altura LAB7A.
        pre=root.get("uwb_preprocessing", {})
        height=pre.get("tag_heights_m") if isinstance(pre,dict) else None
        if type(height) in (int,float):
            params["tag_height_m"]=height
        else:
            raise ValueError(
                "EKF 1 tag exige tag_height_m no bloco single_tag_ekf "
                "ou tag_heights_m escalar em uwb_preprocessing"
            )
    return SingleTagEkfConfig.from_dict(params)


@dataclass(frozen=True)
class SingleTagEkfResult:
    """Histórico N×3, posições alinhadas aos M snapshots e diagnósticos."""
    poses_xytheta: np.ndarray
    sampled_xytheta: np.ndarray
    covariances: np.ndarray
    sample_indices: np.ndarray
    updates_applied: np.ndarray
    used_ranges_per_epoch: np.ndarray
    innovation_rms_per_epoch_m: np.ndarray
    summary: dict


def _validate_inputs(
    pcnt_timestamps_s, v_m_s, omega_rad_s, anchors_xyz, ranges_m,
    sample_indices, sigma_m, sample_timestamps_s, initial_pose,
):
    t=np.asarray(pcnt_timestamps_s,dtype=float)
    v=np.asarray(v_m_s,dtype=float)
    w=np.asarray(omega_rad_s,dtype=float)
    a=np.asarray(anchors_xyz,dtype=float)
    d=np.asarray(ranges_m,dtype=float)
    idx=np.asarray(sample_indices)
    pose=np.asarray(initial_pose,dtype=float)

    if (t.ndim!=1 or len(t)<2 or not np.isfinite(t).all()
            or np.any(np.diff(t)<=0)):
        raise ValueError("PCNT timestamps precisam ser crescentes e finitos (>=2)")
    if (v.shape!=t.shape or w.shape!=t.shape or
            not np.isfinite(v).all() or not np.isfinite(w).all()):
        raise ValueError("v/omega precisam ter N amostras finitas alinhadas ao PCNT")
    if (a.ndim!=2 or a.shape[1]!=3 or len(a)<1 or not np.isfinite(a).all()):
        raise ValueError("âncoras precisam ser Ax3, finitas")
    if d.ndim!=2 or d.shape[0]<1 or d.shape[1]!=len(a):
        raise ValueError("ranges precisam ser MxA (uma tag e A âncoras)")
    if (idx.ndim!=1 or idx.shape!=(len(d),)
            or idx.dtype.kind not in ("i","u") or idx[0]<0
            or idx[-1]>=len(t) or np.any(np.diff(idx)<=0)):
        raise ValueError("índices UWB devem ser inteiros, únicos e crescentes dentro do PCNT")
    if pose.shape!=(3,) or not np.isfinite(pose).all():
        raise ValueError("pose inicial precisa ser (x,y,theta) finita")
    sig=None if sigma_m is None else np.asarray(sigma_m,dtype=float)
    if sig is not None and sig.shape!=d.shape:
        raise ValueError("sigmas UWB devem ter a mesma forma MxA dos ranges")
    if sample_timestamps_s is not None:
        observed=np.asarray(sample_timestamps_s,dtype=float)
        if (observed.shape!=(len(idx),) or not np.isfinite(observed).all()
                or not np.allclose(observed,t[idx],rtol=0,atol=1e-6)):
            raise ValueError("timestamps UWB não correspondem aos índices PCNT")
    return t,v,w,a,d,idx.astype(np.int64,copy=True),sig,pose


def _predict(x, P, ds, dtheta, dt, cfg):
    """Modelo cinemático do adaptador PCNT e Jacobianas de propagação."""
    midpoint=x[2]+0.5*dtheta
    cos_t,sin_t=math.cos(midpoint),math.sin(midpoint)
    propagated=x.copy()
    propagated[0]+=ds*cos_t
    propagated[1]+=ds*sin_t
    propagated[2]=_wrap(x[2]+dtheta)
    F=np.eye(3)
    F[0,2]=-ds*sin_t
    F[1,2]= ds*cos_t

    # Variância proporcional à distância/rotação e ao tempo, sem
    # atribuir ruído constante imenso a cada linha PCNT parada.
    var_d=(cfg.motion_sigma_m_sqrt_m**2*abs(ds)
           + cfg.stationary_sigma_m_sqrt_s**2*dt)
    var_th=(cfg.turn_sigma_rad_sqrt_rad**2*abs(dtheta)
            +cfg.heading_sigma_rad_sqrt_s**2*dt)
    G=np.array([
        [cos_t,-.5*ds*sin_t],
        [sin_t, .5*ds*cos_t],
        [0.,1.],
    ],dtype=float)
    Q=(G * np.array([var_d,var_th])[None,:])@G.T
    covariance=F@P@F.T+Q
    return propagated,.5*(covariance+covariance.T)


def _ranges_and_jacobian(x, anchors, anchor_idx, cfg):
    """Distância 3D às âncoras, posição rígida da tag, H de x,y,theta."""
    angle=x[2]
    ct,st=math.cos(angle),math.sin(angle)
    ox,oy=cfg.tag_offset_x_m,cfg.tag_offset_y_m
    xt=x[0]+ct*ox-st*oy
    yt=x[1]+st*ox+ct*oy
    dxt=-st*ox-ct*oy
    dyt=ct*ox-st*oy
    da=xt-anchors[anchor_idx,0]
    db=yt-anchors[anchor_idx,1]
    dc=cfg.tag_height_m-anchors[anchor_idx,2]
    expected=np.sqrt(da*da+db*db+dc*dc)
    good=expected>1e-10
    if not np.all(good):
        return expected,None
    H=np.column_stack((
        da/expected,
        db/expected,
        (da*dxt+db*dyt)/expected,
    ))
    return expected,H


def run_single_tag_range_ekf(
    *, timestamps_s, v_m_s, omega_rad_s,
    anchors_xyz, ranges_m, sample_indices, initial_pose,
    deviations_m=None, sample_timestamps_s=None,
    config: SingleTagEkfConfig | dict | None=None,
) -> SingleTagEkfResult:
    """EKF causal: predição em cada PCNT, correção SOMENTE nos índices UWB.

    Correção em lote (não ordena âncoras). Joseph form garante
    covariância simétrica/semidefinida positiva numericamente.
    Quando restam menos ranges que min_ranges_for_update, não inventa
    dados: apenas prediz e registra a ausência da correção.
    """
    cfg=(SingleTagEkfConfig() if config is None else
         SingleTagEkfConfig.from_dict(config) if isinstance(config,dict)
         else config)
    if not isinstance(cfg,SingleTagEkfConfig):
        raise ValueError("config do EKF inválida")
    t,v,w,anchors,d,idx,sig,x=_validate_inputs(
        timestamps_s,v_m_s,omega_rad_s,anchors_xyz,ranges_m,
        sample_indices,deviations_m,sample_timestamps_s,initial_pose
    )
    if cfg.min_ranges_for_update > len(anchors):
        raise ValueError("min_ranges_for_update excede número de âncoras")
    x=x.copy()
    x[2]=_wrap(x[2])
    P=np.diag([
        cfg.initial_position_sigma_m**2,
        cfg.initial_position_sigma_m**2,
        math.radians(cfg.initial_heading_sigma_deg)**2,
    ])
    n=len(t)
    m=len(idx)
    trajectory=np.empty((n,3),dtype=float)
    covariance=np.empty((n,3,3),dtype=float)
    updates=np.zeros(m,dtype=bool)
    used=np.zeros(m,dtype=np.int64)
    innovation_rms=np.full(m,np.nan)
    missing_epochs=0
    sigma_fallback_count=0
    pointer=0

    for k in range(n):
        if k:
            dt=t[k]-t[k-1]
            x,P=_predict(x,P,float(v[k]*dt),float(w[k]*dt),float(dt),cfg)
        if pointer<m and int(idx[pointer])==k:
            row=d[pointer]
            valid=np.isfinite(row)&(row>0)
            # Um range sem sigma não é descartado: usa o floor
            # explícito. O valor original não é corrigido/inventado.
            valid_indices=np.flatnonzero(valid)
            if len(valid_indices)<cfg.min_ranges_for_update:
                missing_epochs+=1
            else:
                predicted,H=_ranges_and_jacobian(x,anchors,valid_indices,cfg)
                if H is None:
                    missing_epochs+=1
                else:
                    z=row[valid_indices]
                    residual=z-predicted
                    used_sigma=np.full(len(z),cfg.range_sigma_floor_m)
                    if cfg.use_reported_sigma and sig is not None:
                        reported=sig[pointer,valid_indices]
                        okay=np.isfinite(reported)&(reported>0)
                        sigma_fallback_count+=int(np.count_nonzero(~okay))
                        used_sigma[okay]=np.maximum(
                            cfg.range_sigma_floor_m,reported[okay]
                        )
                    R=np.diag(used_sigma*used_sigma)
                    innovation_cov=H@P@H.T+R
                    try:
                        K=np.linalg.solve(innovation_cov,H@P).T
                    except np.linalg.LinAlgError:
                        missing_epochs+=1
                    else:
                        x=x+K@residual
                        x[2]=_wrap(x[2])
                        J=np.eye(3)-K@H
                        P=J@P@J.T+K@R@K.T
                        P=.5*(P+P.T)
                        updates[pointer]=True
                        used[pointer]=len(z)
                        innovation_rms[pointer]=float(
                            np.sqrt(np.mean(residual*residual))
                        )
            pointer+=1
        trajectory[k]=x
        covariance[k]=P

    if pointer!=m:
        raise AssertionError("Observações UWB não foram totalmente processadas")
    return SingleTagEkfResult(
        poses_xytheta=trajectory,
        sampled_xytheta=trajectory[idx].copy(),
        covariances=covariance,
        sample_indices=idx.copy(),
        updates_applied=updates,
        used_ranges_per_epoch=used,
        innovation_rms_per_epoch_m=innovation_rms,
        summary={
            "schema":"bc_ekf.single_tag_range_ekf.v1",
            "method":"ekf_raw_ranges_single_tag",
            "config":asdict(cfg),
            "pcnt_samples":int(n),
            "uwb_epochs":int(m),
            "corrections_applied":int(np.count_nonzero(updates)),
            "predict_only_uwb_epochs":int(m-np.count_nonzero(updates)),
            "insufficient_or_unsolved_epochs":int(missing_epochs),
            "total_ranges_used":int(np.sum(used)),
            "sigma_fallback_ranges":int(sigma_fallback_count),
            "innovation_rms_mean_m":(
                float(np.nanmean(innovation_rms))
                if np.any(np.isfinite(innovation_rms)) else None
            ),
            "note":[
                "Ranges são do timestamp PCNT de registro, não necessariamente timestamp físico de ranging",
                "Filtro causal: sem interpolações ou ground truth",
                "Heading de uma tag central não é diretamente observado em uma única correção",
                "Sem inovação gating/Huber: métricas não garantem robustez a outliers",
                "Sigmas de firmware têm piso configurável e podem não representar covariância calibrada",
            ],
        },
    )
