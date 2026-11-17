"""Auditoria NÃO destrutiva da orientação PCNT versus duas tags UWB.

Sem Pygame, SciPy, BC-EKF, mapa nominal ou ground truth.
1. Reconstrói XY de cada tag diretamente por WLS/Huber em ranges 3D.
2. Obtém heading observável da linha entre as antenas com offsets XYZ
   arbitrários, sem incorporar a odometria na estimação UWB.
3. Compara heading UWB (circular) ao heading PCNT (contínuo).
4. Mantém todos os índices PCNT e retorna quality flags por snapshot.

Avisos metrológicos:
- duas tags distantes só 36 cm amplificam erros angulares;
- timestamp UWB corresponde à linha PCNT de registro, não há
  garantia de simultaneidade física do ranging entre duas tags;
- covariância calculada por WLS é nominal, não incerteza calibrada;
- diferença UWB−odometria é DISCORDÂNCIA, nunca erro versus GT;
- o ângulo "desenrolado" só é inequívoco quando variação entre
  snapshots aceitos é inferior a 180° e o sincronismo é adequado.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _number_or_null(value: Any):
    return float(value) if np.isfinite(value) else None


@dataclass(frozen=True)
class HeadingAuditConfig:
    """Parâmetros de qualidade explícitos; filtros NÃO alteram o CSV."""
    sigma_floor_m: float = 0.10
    huber_delta_m: float = 0.20
    min_anchors_per_tag: int = 4
    max_fit_rms_m: float = 0.20
    max_baseline_error_m: float = 0.12
    max_heading_sigma_deg: float = 40.0
    window_s: float = 10.0
    focus_start_s: float = 80.0
    focus_end_s: float = 120.0
    min_good_per_window: int = 2

    def __post_init__(self):
        for name in (
            "sigma_floor_m","huber_delta_m","max_fit_rms_m",
            "max_baseline_error_m","max_heading_sigma_deg",
            "window_s","focus_end_s",
        ):
            x=getattr(self,name)
            if type(x) not in (int,float) or not math.isfinite(x) or x<=0:
                raise ValueError(f"{name} precisa ser número positivo e finito")
        start=self.focus_start_s
        if type(start) not in (int,float) or not math.isfinite(start) or start<0:
            raise ValueError("focus_start_s deve ser não negativo")
        if self.focus_end_s<=start:
            raise ValueError("focus_end_s precisa exceder focus_start_s")
        for name in ("min_anchors_per_tag","min_good_per_window"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f"{name} precisa ser inteiro positivo")


def _fit_xy(anchors, observed, deviations, z_tag, cfg, previous_xy=None):
    """Posição XY independente dos encoders por WLS/Huber multistart.

    Um ponto anterior aceito de UWB pode iniciar a otimização, mas
    NUNCA utiliza a odometria ou a pose BC-EKF como prior.
    """
    ok=np.isfinite(observed)&(observed>0)
    if int(np.count_nonzero(ok))<cfg.min_anchors_per_tag:
        return None
    a=anchors[ok]
    z=observed[ok]
    s=deviations[ok]
    sigma=np.where(np.isfinite(s)&(s>0),s,cfg.sigma_floor_m)
    sigma=np.maximum(sigma,cfg.sigma_floor_m)
    base_w=1/(sigma*sigma)
    h_z=z_tag-a[:,2]

    def residual_jac(xy):
        delta=xy[None,:]-a[:,:2]
        pred=np.sqrt(np.sum(delta*delta,axis=1)+h_z*h_z)
        jac=delta/np.maximum(pred[:,None],1e-9)
        res=z-pred
        return res,jac

    # Várias sementes fixas para evitar mínimos locais, mantendo
    # independência total da odometria e do traçado nominal.
    mid=np.mean(a[:,:2],axis=0)
    seeds=[mid,*[x[:2].copy() for x in a]]
    if previous_xy is not None and np.isfinite(previous_xy).all():
        seeds.insert(0,np.asarray(previous_xy,dtype=float))
    best=None
    for start in seeds:
        xy=np.asarray(start,dtype=float).copy()
        for _ in range(35):
            res,J=residual_jac(xy)
            robust=np.minimum(1.,cfg.huber_delta_m/np.maximum(np.abs(res),1e-12))
            w=base_w*robust
            A=J.T@(w[:,None]*J)
            rhs=J.T@(w*res)
            try:
                step=np.linalg.solve(A+1e-7*np.eye(2),rhs)
            except np.linalg.LinAlgError:
                break
            norm=float(np.linalg.norm(step))
            if norm>.7:
                step*=.7/norm
            xy+=step
            if norm<1e-8:
                break
        res,J=residual_jac(xy)
        # Huber loss mede qualidade mesmo quando há uma âncora ruim.
        ab=np.abs(res)
        loss=np.where(ab<=cfg.huber_delta_m,
                      .5*ab*ab,cfg.huber_delta_m*(ab-.5*cfg.huber_delta_m))
        score=float(np.dot(base_w,loss))
        if not math.isfinite(score):
            continue
        if best is None or score<best[0]:
            best=(score,xy.copy(),res.copy(),J.copy())
    if best is None:
        return None
    _,xy,res,J=best
    robust=np.minimum(1.,cfg.huber_delta_m/np.maximum(np.abs(res),1e-12))
    W=base_w*robust
    A=J.T@(W[:,None]*J)
    if np.linalg.cond(A)>1e12:
        return None
    cov=np.linalg.inv(A)
    if not np.isfinite(cov).all():
        return None
    return {
        "xy":xy,"cov":cov,"rms":float(np.sqrt(np.mean(res**2))),
        "n":int(len(z)),
    }


def _validate_inputs(t,odo,dl,dr,idx,dist,sigma,anchors,offsets):
    t=np.asarray(t,dtype=float)
    odo=np.asarray(odo,dtype=float)
    dl=np.asarray(dl,dtype=float)
    dr=np.asarray(dr,dtype=float)
    idx=np.asarray(idx)
    dist=np.asarray(dist,dtype=float)
    sigma=np.asarray(sigma,dtype=float)
    anchors=np.asarray(anchors,dtype=float)
    offsets=np.asarray(offsets,dtype=float)
    n=len(t) if t.ndim==1 else 0
    if n<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError("timestamps PCNT devem ser crescentes e finitos")
    if (odo.shape!=(n,3) or dl.shape!=(n,) or dr.shape!=(n,)
            or not np.isfinite(odo).all()
            or not np.isfinite(dl).all() or not np.isfinite(dr).all()):
        raise ValueError("odometria/contadores PCNT precisam estar alinhados a N instantes")
    if (idx.ndim!=1 or not len(idx) or idx.dtype.kind not in "iu"
            or idx[0]<0 or idx[-1]>=n or np.any(np.diff(idx)<=0)):
        raise ValueError("índices UWB devem ser inteiros, crescentes e existentes no PCNT")
    if (anchors.ndim!=2 or anchors.shape[1]!=3
            or not np.isfinite(anchors).all() or len(anchors)<3):
        raise ValueError("âncoras exigem Ax3 válidas")
    if (dist.shape!=(len(idx),2,len(anchors))
            or sigma.shape!=dist.shape):
        raise ValueError("ranges e desvios UWB exigem Mx2xA")
    if (offsets.shape!=(2,3) or not np.isfinite(offsets).all()):
        raise ValueError("duas tags precisam de offsets XYZ 2x3 finitos")
    if np.linalg.norm(offsets[1,:2]-offsets[0,:2])<1e-5:
        raise ValueError("tags com mesmo XY não permitem estimar heading planar")
    return t,odo,dl,dr,idx,dist,sigma,anchors,offsets


def _window_summary(t,dl,dr,dtheta,accepted,err,uwb_sigma,first,last,cfg):
    """Resumo por faixa PCNT; sem chamar um intervalo de volta física."""
    mask=(t>=first)&(t<last)
    e=err[mask&accepted]
    s=uwb_sigma[mask&accepted]
    samples=int(np.count_nonzero(mask))
    count=int(len(e))
    center=_number_or_null(np.median(e)) if count else None
    change=_number_or_null(e[-1]-e[0]) if count>=2 else None
    span=0.
    if count>=2:
        stamps=t[mask&accepted]
        span=float(stamps[-1]-stamps[0])
    return {
        "start_s":round(float(first),6),"end_s":round(float(last),6),
        "pcnt_count":samples,"uwb_good_count":count,
        "quality":"ok" if count>=cfg.min_good_per_window else "insufficient_uwb",
        "angular_difference_median_deg":center,
        "angular_difference_change_deg":change,
        "first_to_last_good_s":span if count>=2 else None,
        "uwb_heading_sigma_median_deg":_number_or_null(np.median(s)) if count else None,
        "pcnt_delta_left_ticks":float(np.sum(dl[mask])),
        "pcnt_delta_right_ticks":float(np.sum(dr[mask])),
        "pcnt_turn_deg":float(np.rad2deg(np.sum(dtheta[mask]))),
        "pcnt_abs_turn_deg":float(np.rad2deg(np.sum(np.abs(dtheta[mask])))),
        "pcnt_max_step_deg":float(np.rad2deg(np.max(np.abs(dtheta[mask])))) if samples else None,
    }


def audit_dual_tag_heading(
    *, timestamps_s, odometry_xytheta,
    pcnt_delta_left_ticks, pcnt_delta_right_ticks,
    sample_indices, distances_by_tag, deviations_by_tag,
    anchors_xyz, tag_offsets_xyz,
    config: HeadingAuditConfig | None=None,
) -> dict:
    """Retorna séries e janelas JSON-serializáveis sem modificar entradas.

    heading UWB é obtido da reta entre as duas tags: atan2(tag1-tag0)
    menos a orientação LOCAL desse segmento. A relação independe de
    qual tag é esquerda: é o deslocamento configurado que determina
    o sinal e o offset angular.
    """
    cfg=config if config is not None else HeadingAuditConfig()
    if not isinstance(cfg,HeadingAuditConfig):
        raise ValueError("config precisa ser HeadingAuditConfig")
    t,odo,dl,dr,idx,d,s,anchors,offsets=_validate_inputs(
        timestamps_s,odometry_xytheta,pcnt_delta_left_ticks,
        pcnt_delta_right_ticks,sample_indices,distances_by_tag,
        deviations_by_tag,anchors_xyz,tag_offsets_xyz
    )
    m=len(idx)
    heading_uwb=np.full(m,np.nan)
    heading_odom=odo[idx,2]
    separation=np.full(m,np.nan)
    fit_rms=np.full((m,2),np.nan)
    fit_n=np.zeros((m,2),dtype=int)
    heading_sigma=np.full(m,np.nan)
    accepted=np.zeros(m,dtype=bool)
    reasons=[]
    centers=np.full((m,2),np.nan)
    baseline=float(np.linalg.norm(offsets[1]-offsets[0]))
    planar_baseline=float(np.linalg.norm(offsets[1,:2]-offsets[0,:2]))
    angle_local=math.atan2(
        offsets[1,1]-offsets[0,1],
        offsets[1,0]-offsets[0,0],
    )
    previous=[None,None]
    for i in range(m):
        estimates=[]
        for tag in (0,1):
            result=_fit_xy(
                anchors,d[i,tag],s[i,tag],offsets[tag,2],
                cfg,previous_xy=previous[tag],
            )
            if result is not None:
                previous[tag]=result["xy"]
                fit_rms[i,tag]=result["rms"]
                fit_n[i,tag]=result["n"]
            estimates.append(result)
        if any(p is None for p in estimates):
            reasons.append("insufficient_or_singular_ranges")
            continue
        p0,p1=estimates
        vec=p1["xy"]-p0["xy"]
        planar=float(np.linalg.norm(vec))
        sep=math.hypot(planar,offsets[1,2]-offsets[0,2])
        separation[i]=sep
        centers[i]=(p0["xy"]+p1["xy"])*.5
        if (p0["rms"]>cfg.max_fit_rms_m or
                p1["rms"]>cfg.max_fit_rms_m):
            reasons.append("range_fit_residual")
            continue
        if abs(sep-baseline)>cfg.max_baseline_error_m:
            reasons.append("tag_baseline_inconsistent")
            continue
        if planar<1e-6:
            reasons.append("unobservable_baseline")
            continue
        # Jacobiano de atan2 da posição relativa: [-dy,dx]/r².
        u=np.array([-vec[1],vec[0]])/(planar*planar)
        var=float(u@(p0["cov"]+p1["cov"])@u)
        sigma_deg=math.degrees(math.sqrt(max(0.,var)))
        heading_sigma[i]=sigma_deg
        if sigma_deg>cfg.max_heading_sigma_deg:
            reasons.append("uncertain_tag_heading")
            continue
        heading_uwb[i]=_wrap(math.atan2(vec[1],vec[0])-angle_local)
        accepted[i]=True
        reasons.append("ok")

    # Error circular evita saltos artificiais em theta=±180°.
    wrapped=np.full(m,np.nan)
    wrapped[accepted]=np.arctan2(
        np.sin(heading_uwb[accepted]-heading_odom[accepted]),
        np.cos(heading_uwb[accepted]-heading_odom[accepted]),
    )
    continuous=np.full(m,np.nan)
    if np.any(accepted):
        continuous[accepted]=np.unwrap(wrapped[accepted])
    # Relevante: diferença relativa à primeira época válida; o
    # desalinhamento absoluto inicial NÃO é calibrado por este método.
    drift=np.full(m,np.nan)
    if np.any(accepted):
        drift[accepted]=continuous[accepted]-continuous[accepted][0]
    diff_uwb=np.full(m,np.nan)
    good=np.flatnonzero(accepted)
    if len(good)>1:
        diff_uwb[good[1:]]=np.rad2deg(np.diff(np.unwrap(heading_uwb[good])))
    dtheta_pcnt=np.diff(odo[:,2],prepend=odo[0,2])
    # Para qualquer ângulo contínuo de odometria, diferença de
    # snapshots é somada em PCNT sem retirar múltiplas voltas.
    intervals=[]
    stop=float(t[-1])
    a=0.
    while a<=stop:
        b=a+cfg.window_s
        seg=_window_summary(
            t,dl,dr,dtheta_pcnt,idx_accepted_mask(accepted,idx,len(t)),
            pcnt_aligned(continuous,idx,len(t),degrees=True),
            pcnt_aligned(heading_sigma,idx,len(t)),
            a,b,cfg
        )
        intervals.append(seg)
        a=b
        if len(intervals)>10000:
            raise ValueError("Excesso de janelas: configure window_s maior")
    candidates=[
        x for x in intervals
        if x["quality"]=="ok"
        and x["angular_difference_change_deg"] is not None
    ]
    candidates.sort(
        key=lambda x:abs(x["angular_difference_change_deg"]),reverse=True
    )
    records=[]
    for j,k in enumerate(idx):
        records.append({
            "uwb_epoch":int(j+1),"pcnt_index":int(k),
            "timestamp_s":float(t[k]),
            "valid":bool(accepted[j]),
            "quality_reason":reasons[j],
            "range_fit_rms_t1_m":_number_or_null(fit_rms[j,0]),
            "range_fit_rms_t2_m":_number_or_null(fit_rms[j,1]),
            "valid_anchors_t1":int(fit_n[j,0]),
            "valid_anchors_t2":int(fit_n[j,1]),
            "measured_baseline_m":_number_or_null(separation[j]),
            "heading_sigma_nominal_deg":_number_or_null(heading_sigma[j]),
            "theta_odom_deg":float(np.rad2deg(heading_odom[j])),
            "theta_uwb_wrapped_deg":_number_or_null(np.rad2deg(heading_uwb[j])),
            "difference_wrapped_deg":_number_or_null(np.rad2deg(wrapped[j])),
            "difference_unwrapped_deg":_number_or_null(np.rad2deg(continuous[j])),
            "relative_drift_deg":_number_or_null(np.rad2deg(drift[j])),
            "uwb_delta_since_prev_good_deg":_number_or_null(diff_uwb[j]),
            "uwb_center_x_m":_number_or_null(centers[j,0]),
            "uwb_center_y_m":_number_or_null(centers[j,1]),
        })
    pcnt_rows=[]
    for k in range(len(t)):
        dt=float(t[k]-t[k-1]) if k else None
        pcnt_rows.append({
            "pcnt_index":k,"timestamp_s":float(t[k]),
            "delta_left_ticks":float(dl[k]),
            "delta_right_ticks":float(dr[k]),
            "delta_theta_deg":float(np.rad2deg(dtheta_pcnt[k])),
            "theta_odom_deg":float(np.rad2deg(odo[k,2])),
            "omega_odom_deg_s":float(np.rad2deg(dtheta_pcnt[k])/dt) if dt else None,
            "v_m_s":None, # CLI pode associar v a partir do adaptador
        })
    gaps=np.diff(t[idx[good]]) if len(good)>1 else np.array([])
    warnings=[
        "UWB heading é estimado apenas por ranges; não é GT nem observação perfeitamente sincronizada.",
        "A distância pequena entre tags aumenta o erro de orientação; sigma WLS é apenas nominal.",
        "Os ângulos não foram ajustados para caber na pista ou na odometria.",
        "Diferença angular grande em uma janela não prova salto instantâneo no encoder.",
        "Velocidade do PCNT usa incrementos por amostra; a validação de sinais e parâmetros é separada.",
    ]
    if np.any(gaps>15):
        warnings.append(
            "Uma ou mais lacunas entre headings UWB aceitos excedem 15 s; "
            "desenrolamento angular pode ser ambíguo nelas."
        )
    return {
        "schema":"bc_ekf.dual_tag_heading_audit.v1",
        "config":asdict(cfg),
        "geometry":{
            "offsets_xyz_m":offsets.tolist(),
            "baseline_m":baseline,
            "planar_baseline_m":planar_baseline,
            "tag_order":"UWB t1 -> canal0, t2 -> canal1",
        },
        "summary":{
            "pcnt_samples":int(len(t)),"uwb_epochs":int(m),
            "accepted_heading_epochs":int(np.count_nonzero(accepted)),
            "rejected_heading_epochs":int(m-np.count_nonzero(accepted)),
            "first_accepted_time_s":float(t[idx[good[0]]]) if len(good) else None,
            "last_accepted_time_s":float(t[idx[good[-1]]]) if len(good) else None,
            "max_abs_pcnt_step_deg":float(np.rad2deg(np.max(np.abs(dtheta_pcnt)))),
            "max_gap_between_accepted_uwb_s":float(np.max(gaps)) if len(gaps) else None,
            "quality_reasons":{name:reasons.count(name) for name in sorted(set(reasons))},
            "top_change_windows":candidates[:8],
            "focus_window":[x for x in intervals if
                            x["start_s"]<cfg.focus_end_s
                            and x["end_s"]>cfg.focus_start_s],
            "interpretation":"Desvio angular UWB−odometria, não erro angular verdadeiro.",
        },
        "windows":intervals,
        "uwb_snapshots":records,
        "pcnt_steps":pcnt_rows,
        "warnings":warnings,
    }


def idx_accepted_mask(valid, indices, n):
    arr=np.zeros(n,dtype=bool)
    arr[np.asarray(indices)[valid]]=True
    return arr


def pcnt_aligned(values, indices, n, *, degrees=False):
    arr=np.full(n,np.nan)
    good=np.isfinite(values)
    arr[np.asarray(indices)[good]]=(
        np.rad2deg(values[good]) if degrees else values[good]
    )
    return arr
