"""Adaptador do EKF 1 tag ao Dataset PCNT+UWB, sem dependência de Pygame.

Conecta a odometria PCNT já normalizada e seus índices UWB ao motor
NumPy de fusão, sem alterar as séries originais ou reprocessar encoders.
Opcionalmente aplica a MESMA política off/flag/mask dos solvers estáticos.
"""
from __future__ import annotations

import numpy as np

from src.fusion.single_tag_range_ekf import (
    SingleTagEkfConfig, run_single_tag_range_ekf,
)
from src.uwb.preprocessing import (
    UwbPreprocessConfig, preprocess_uwb_batch,
)


def run_real_single_tag_fusion(
    *, odometry, uwb, anchors_xyz,
    distances, deviations,
    config: SingleTagEkfConfig,
    preprocess_config: UwbPreprocessConfig | dict | None = None,
):
    """Retorna resultado compatível com o Analyzer e histórico PCNT completo.

    Os algoritmos UWB puros produzem M posições; o EKF prediz N estados
    PCNT e extrai apenas M posições nos índices UWB para comparação
    justa. O histórico N é guardado separadamente para visualização
    futura e análises de covariância.
    """
    if odometry is None or uwb is None:
        raise ValueError("EKF 1 tag requer PCNT e UWB da mesma sessão")
    anchors=np.asarray(anchors_xyz,dtype=float)
    raw=np.asarray(distances,dtype=float)
    dev=np.asarray(deviations,dtype=float) if deviations is not None else None
    indices=np.asarray(uwb.sample_indices)
    timestamps=np.asarray(odometry.timestamps_s)
    if (raw.ndim!=2 or raw.shape!=(len(indices),len(anchors))
            or (dev is not None and dev.shape!=raw.shape)):
        raise ValueError("Dimensões UWB não correspondem ao PCNT/âncoras")
    if (len(timestamps)!=len(odometry.poses_xytheta)
            or len(odometry.v_m_s)!=len(timestamps)
            or len(odometry.omega_rad_s)!=len(timestamps)):
        raise ValueError("Odometria não está alinhada aos timestamps PCNT")
    if (len(indices)!=len(uwb.timestamps_s)
            or np.any(indices<0) or np.any(indices>=len(timestamps))
            or not np.allclose(
                timestamps[indices],uwb.timestamps_s,rtol=0,atol=1e-6
            )):
        raise ValueError("Snapshots UWB não correspondem aos índices PCNT")

    report=None
    if preprocess_config is not None:
        cfg=(UwbPreprocessConfig.from_dict(preprocess_config)
             if isinstance(preprocess_config,dict) else preprocess_config)
        prepared=preprocess_uwb_batch(
            anchors,raw,dev,config=cfg,
            sample_indices=indices,timestamps_s=uwb.timestamps_s,
        )
        raw,dev=prepared.distances,prepared.deviations
        report=prepared.summary

    output=run_single_tag_range_ekf(
        timestamps_s=timestamps,
        v_m_s=odometry.v_m_s,
        omega_rad_s=odometry.omega_rad_s,
        anchors_xyz=anchors,
        ranges_m=raw,
        deviations_m=dev,
        sample_indices=indices,
        sample_timestamps_s=uwb.timestamps_s,
        initial_pose=np.asarray(odometry.poses_xytheta[0],dtype=float),
        config=config,
    )
    return {
        "posicoes":output.sampled_xytheta,
        "full_pcnt_poses":output.poses_xytheta,
        "full_pcnt_covariances":output.covariances,
        "fusion_updates_applied":output.updates_applied,
        "fusion_used_ranges":output.used_ranges_per_epoch,
        "fusion_innovation_rms_m":output.innovation_rms_per_epoch_m,
        "fusion_diagnostics":output.summary,
        "algo_key_actual":"single_tag_range_ekf",
        "algo_label":"EKF 1 tag (odometria + ranges)",
        **({"preprocessing":report} if report is not None else {}),
    }
