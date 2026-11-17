"""Atualização EKF por ranges UWB com observações parciais e diagnósticos.

A geometria rígida das tags é calculada pelo chamador. Este módulo
não conhece laboratório, IDs SPI, número de âncoras ou firmware.
Recebe ranges na ordem do modelo H e uma matriz R configurável.

Não realiza pré-processamento nem altera medições: NaN, Inf e ranges
não positivos são simplesmente removidos da atualização, preservando
os índices originais nos diagnósticos. NIS é consistência estatística
do EKF, NÃO erro versus ground truth.
"""
from __future__ import annotations

import numpy as np


def correct_uwb_ranges(
    x_pred, P_pred, measured_ranges, predicted_ranges, jacobian, R, *,
    min_valid_ranges=1, nis_gate_per_dof=None, range_gate_sigma=None,
):
    """Atualiza [x,y,theta] com qualquer subconjunto de ranges válidos.

    Args:
        x_pred: vetor de estado predito (3,).
        P_pred: covariância do estado predito (3,3).
        measured_ranges: vetor de ranges reais, com ausências via NaN.
        predicted_ranges: saída do modelo rígido h(x_pred).
        jacobian: H(x_pred) (m,3), na MESMA ordem dos ranges.
        R: covariância nominal (m,m), positiva definida.
        min_valid_ranges: número mínimo requerido para atualizar
            (configurável, padrão 1). O prior EKF resolve observação
            parcial, mas atualizações com poucas âncoras têm limites.
        nis_gate_per_dof: limiar opcional sobre NIS/graus de liberdade;
            None DESATIVA descarte global para manter legado.
        range_gate_sigma: limiar opcional para filtrar resíduos
            padronizados individuais pelo S; None não filtra.

    Returns:
        (x_post, P_post, report). O report contém status, índices das
        observações usadas, inovação, S, K, H, NIS, NIS/dof e contagens.
        Não retorna nem inventa pose ground truth.
    """
    x=np.asarray(x_pred,dtype=float)
    P=np.asarray(P_pred,dtype=float)
    z=np.asarray(measured_ranges,dtype=float)
    h=np.asarray(predicted_ranges,dtype=float)
    H=np.asarray(jacobian,dtype=float)
    covariance=np.asarray(R,dtype=float)
    m=z.size
    if (x.shape!=(3,) or P.shape!=(3,3) or z.ndim!=1 or h.shape!=(m,)
            or H.shape!=(m,3) or covariance.shape!=(m,m)):
        raise ValueError("BC-EKF: dimensões incompatíveis de estado, ranges, H ou R")
    if (not np.isfinite(x).all() or not np.isfinite(P).all()
            or not np.isfinite(h).all() or not np.isfinite(H).all()
            or not np.isfinite(covariance).all()):
        raise ValueError("BC-EKF: estado, modelo e R exigem números finitos")
    if (not np.allclose(covariance,covariance.T,rtol=1e-8,atol=1e-10)
            or np.any(np.diag(covariance)<=0)):
        raise ValueError("BC-EKF: R deve ser simétrica, com variâncias positivas")
    if type(min_valid_ranges) is not int or min_valid_ranges<1:
        raise ValueError("min_valid_ranges deve ser inteiro positivo")
    for name,value in (("nis_gate_per_dof",nis_gate_per_dof),
                       ("range_gate_sigma",range_gate_sigma)):
        if value is not None and (not np.isscalar(value)
                or not np.isfinite(value) or float(value)<=0):
            raise ValueError(f"{name} precisa ser número positivo ou None")

    idx=np.flatnonzero(np.isfinite(z)&(z>0))
    valid_before=int(len(idx))
    empty=np.empty(0,dtype=float)
    base={
        "status":"no_valid_ranges",
        "total_ranges":m,
        "valid_ranges":valid_before,
        "used_ranges":0,
        "dropped_ranges":m,
        "used_indices":np.array([],dtype=int),
        "innovation":empty,
        "S":np.empty((0,0)),
        "K":np.empty((3,0)),
        "h":empty,
        "H":np.empty((0,3)),
        "nis":None,
        "nis_per_dof":None,
        "innovation_rms_m":None,
        "max_abs_innovation_m":None,
        "max_standardized_innovation":None,
    }
    if valid_before<min_valid_ranges:
        base["status"]=(
            "no_valid_ranges" if valid_before==0 else "insufficient_ranges"
        )
        return x.copy(),P.copy(),base

    # Recalcular S após qualquer descarte local: submatriz R mantém
    # eventuais correlações entre medições.
    def projected(selected):
        HH=H[selected]
        RR=covariance[np.ix_(selected,selected)]
        SS=HH@P@HH.T+RR
        SS=(SS+SS.T)*.5+np.eye(len(selected))*1e-10
        yy=z[selected]-h[selected]
        return HH,RR,SS,yy

    HH,RR,SS,yy=projected(idx)
    if range_gate_sigma is not None:
        standardized=np.abs(yy)/np.sqrt(np.maximum(np.diag(SS),1e-15))
        idx=idx[standardized<=float(range_gate_sigma)]
        if len(idx)<min_valid_ranges:
            base.update(
                status="range_gate_rejected",
                valid_ranges=valid_before,
                dropped_ranges=m,
            )
            return x.copy(),P.copy(),base
        HH,RR,SS,yy=projected(idx)

    try:
        solved=np.linalg.solve(SS,yy)
        gain=np.linalg.solve(SS,(P@HH.T).T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("BC-EKF: inovação possui covariância singular") from exc
    nis=max(0.,float(yy@solved))
    nis_per_dof=nis/len(idx)
    base.update(
        status="accepted",
        used_ranges=int(len(idx)),
        dropped_ranges=int(m-len(idx)),
        used_indices=idx.copy(),
        innovation=yy.copy(),
        S=SS.copy(),
        K=gain.copy(),
        h=h[idx].copy(),
        H=HH.copy(),
        nis=nis,
        nis_per_dof=nis_per_dof,
        innovation_rms_m=float(np.sqrt(np.mean(yy*yy))),
        max_abs_innovation_m=float(np.max(np.abs(yy))),
        max_standardized_innovation=float(
            np.max(np.abs(yy)/np.sqrt(np.maximum(np.diag(SS),1e-15)))
        ),
    )
    if nis_gate_per_dof is not None and nis_per_dof>nis_gate_per_dof:
        base["status"]="nis_gate_rejected"
        return x.copy(),P.copy(),base

    updated=x+gain@yy
    updated[2]=np.arctan2(np.sin(updated[2]),np.cos(updated[2]))
    # Forma de Joseph: reduz risco de covariância assimétrica/negativa.
    identity=np.eye(3)
    residual=identity-gain@HH
    updated_P=residual@P@residual.T+gain@RR@gain.T
    updated_P=(updated_P+updated_P.T)*.5
    return updated,updated_P,base


def compact_uwb_diagnostic(report, *, index, timestamp_s=None, dt_s=None):
    """Converte o diagnóstico para JSON sem matrizes de covariância."""
    return {
        "pcnt_index":int(index),
        "timestamp_s":float(timestamp_s) if timestamp_s is not None else None,
        "dt_s":float(dt_s) if dt_s is not None else None,
        "status":report["status"],
        "total_ranges":int(report["total_ranges"]),
        "valid_ranges":int(report["valid_ranges"]),
        "used_ranges":int(report["used_ranges"]),
        "dropped_ranges":int(report["dropped_ranges"]),
        "used_indices":[int(i) for i in report["used_indices"]],
        "nis":report["nis"],
        "nis_per_dof":report["nis_per_dof"],
        "innovation_rms_m":report["innovation_rms_m"],
        "max_abs_innovation_m":report["max_abs_innovation_m"],
        "max_standardized_innovation":report["max_standardized_innovation"],
    }
