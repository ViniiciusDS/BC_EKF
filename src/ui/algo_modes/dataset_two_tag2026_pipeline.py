"""Leitor de aquisições PCNT + DUAS tags UWB com cabeçalho explícito.

Formato suportado (e detectado, nunca confundido com PCNT antigo):
  time_ms,left,right,t1_a1_avg,t1_a1_std,...,t2_a9_avg,t2_a9_std

Ranges e desvios em cm conforme configuração `uwb_dual_2026` do
JSON de âncoras. Cada pacote inclui valores reutilizados por várias
linhas PCNT: só gera novo snapshot quando as duas tags recebem dados
novos. NÃO interpola, não pressupõe independência entre pacotes,
não troca ID das tags nem reescreve CSV bruto.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import re
from pathlib import Path

import numpy as np

from src.odometry.pcntlog_adapter import RawPcntRow, convert_session, Robot2026Profile
from src.ui.algo_modes.dataset_odometry_pipeline import Odometry2026Data

_FIELD = re.compile(r"^t([12])_a([0-9]+)_(avg|std)$")


@dataclass(frozen=True)
class DualTagPcnt2026:
    """Log completo + M observações novas, intercaladas por âncora/canal."""
    odometry: Odometry2026Data
    sample_indices: np.ndarray
    timestamps_s: np.ndarray
    distances_by_tag: np.ndarray    # M x 2 x N
    deviations_by_tag: np.ndarray   # M x 2 x N
    z_hist: np.ndarray              # 2N x K PCNT (NaN entre snapshots)
    repeated_rows: int
    partial_refresh_rows: int
    rejected_rows: int
    header_anchor_ids: tuple[int, ...]
    anchor_ids: tuple[int, ...]
    max_pair_gap_m: float
    pairing_anomaly_count: int

    @property
    def sample_count(self):
        return int(len(self.sample_indices))

    @property
    def repeated_readings(self):
        """Compatibilidade com painel PCNT 2026 (unidade: linhas repetidas)."""
        return self.repeated_rows

    @property
    def interleaved_distances(self):
        return self.distances_by_tag.transpose(0,2,1).reshape(self.sample_count,-1)

    @property
    def interleaved_deviations(self):
        return self.deviations_by_tag.transpose(0,2,1).reshape(self.sample_count,-1)


def load_dual_tag_settings(anchor_json):
    """Valida IDs de âncora e unidades: não depende da ordem das colunas."""
    with Path(anchor_json).open("r",encoding="utf-8") as fh:
        doc=json.load(fh)
    if not isinstance(doc,dict) or not isinstance(doc.get("uwb_dual_2026"),dict):
        raise ValueError(
            "JSON de âncoras requer bloco uwb_dual_2026 "
            "com anchor_ids_uwb e units explícitas"
        )
    cfg=doc["uwb_dual_2026"]
    allowed={"anchor_ids_uwb","range_unit","std_unit",
             "min_valid_anchors_per_tag","min_sigma_m",
             "pair_gap_tolerance_m"}
    unknown=set(cfg)-allowed
    if unknown:
        raise ValueError("Campos inesperados em uwb_dual_2026: "+str(sorted(unknown)))
    ids=cfg.get("anchor_ids_uwb")
    if (not isinstance(ids,list) or len(ids)<3
            or any(type(i) is not int or i<1 for i in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError("anchor_ids_uwb deve conter IDs inteiros únicos")
    if cfg.get("range_unit")!="cm" or cfg.get("std_unit")!="cm":
        raise ValueError("Este formato exige range_unit=cm e std_unit=cm")
    min_anchors=cfg.get("min_valid_anchors_per_tag",4)
    if type(min_anchors) is not int or not 3<=min_anchors<=len(ids):
        raise ValueError("min_valid_anchors_per_tag inválido")
    floor=cfg.get("min_sigma_m",.02)
    tolerance=cfg.get("pair_gap_tolerance_m",.05)
    for name,value in (("min_sigma_m",floor),("pair_gap_tolerance_m",tolerance)):
        if (type(value) not in (float,int) or not np.isfinite(value)
                or value<0 or (name=="min_sigma_m" and value<=0)):
            raise ValueError(f"{name} deve ser número válido")
    return tuple(ids),int(min_anchors),float(floor),float(tolerance)


def _validate_columns(header, selected_ids):
    names=[x.strip() for x in header]
    if len(set(names))!=len(names):
        raise ValueError("CSV possui nomes de colunas duplicados")
    if len(names)<3 or names[:3]!=["time_ms","left","right"]:
        raise ValueError("Novo log exige time_ms,left,right nas 3 primeiras colunas")
    observed={}
    for name in names[3:]:
        m=_FIELD.fullmatch(name)
        if m is None:
            raise ValueError(f"Campo não reconhecido no PCNT de duas tags: {name}")
        tag,anchor,kind=int(m[1]),int(m[2]),m[3]
        observed.setdefault(anchor,set()).add((tag,kind))
    required={(tag,kind) for tag in (1,2) for kind in ("avg","std")}
    if any(parts!=required for parts in observed.values()):
        raise ValueError("Todo ID de âncora deve ter t1/t2 avg e std")
    if not set(selected_ids)<=set(observed):
        missing=sorted(set(selected_ids)-set(observed))
        raise ValueError(f"Âncoras faltando no cabeçalho: {missing}")
    # Colunas extras (como A1/A2 inativas) são aceitas mas ignoradas,
    # desde que o JSON declare a lista/ordem corretas das âncoras ativas.
    return tuple(sorted(observed))


def _read_sessions(path):
    sessions=[]
    current=[]
    previous=None
    with Path(path).open("r",encoding="utf-8-sig",newline="") as file:
        reader=csv.DictReader(file,skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError("CSV de duas tags sem cabeçalho")
        header=[c.strip() for c in reader.fieldnames]
        for line_no,row in enumerate(reader,2):
            if not row or not any(v is not None and v.strip() for v in row.values()):
                continue
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Linha {line_no}: colunas incompletas")
            try:
                t=int(row["time_ms"])
                left=int(row["left"])
                right=int(row["right"])
                readings={key.strip():float(value) for key,value in row.items()
                          if key not in ("time_ms","left","right")}
            except (ValueError,TypeError,OverflowError) as exc:
                raise ValueError(f"Linha {line_no}: formato PCNT/UWB inválido") from exc
            if not np.isfinite(list(readings.values())).all():
                raise ValueError(f"Linha {line_no}: medições não finitas")
            item=(RawPcntRow(t,left,right,0,0),readings)
            if previous is not None and t<=previous:
                if current:
                    sessions.append(current)
                current=[]
            current.append(item)
            previous=t
        if current:
            sessions.append(current)
    if not sessions:
        raise ValueError("Nenhuma amostra PCNT de duas tags encontrada")
    return header,sessions


def load_dual_tag_pcnt(
    pcnt_path: str | Path, anchor_json: str | Path, *,
    profile: Robot2026Profile,
    session_id: int=1,
    baseline_m: float,
) -> DualTagPcnt2026:
    """Converte uma sessão sem modificar ou reduzir amostras de encoder.

    Requer as duas tags com mudanças de valores na mesma linha.
    Enquanto firmware não registrar timestamp individual de ranging,
    timestamps aqui representam o registro PCNT, não sincronismo
    físico de UWB. Um só canal renovado não é tratado como conjunto
    de medidas simultâneas.
    """
    path=Path(pcnt_path)
    ids,min_anchors,floor,tolerance=load_dual_tag_settings(anchor_json)
    header,sessions=_read_sessions(path)
    header_ids=_validate_columns(header,ids)
    if not (1<=session_id<=len(sessions)):
        raise ValueError(f"Sessão {session_id} inexistente; 1..{len(sessions)}")
    group=sessions[session_id-1]
    raw=[pair[0] for pair in group]
    odom_rows=convert_session(raw,profile)
    stamp=np.asarray([r["timestamp_s"] for r in odom_rows],dtype=float)
    poses=np.asarray([[r["x_m"],r["y_m"],r["theta_rad"]]
                      for r in odom_rows],dtype=float)
    v=np.asarray([r["v_m_s"] for r in odom_rows],dtype=float)
    w=np.asarray([r["omega_rad_s"] for r in odom_rows],dtype=float)
    odom=Odometry2026Data(
        path,session_id,len(sessions),stamp,poses,v,w,
        np.zeros(len(stamp),dtype=int),profile,
    )

    m=len(ids)
    full=np.full((2*m,len(raw)),np.nan)
    indices=[]
    distance_rows=[]
    sigma_rows=[]
    previous=None
    repeated=partial=invalid=pairing=0
    largest_gap=0.
    for k,(_,record) in enumerate(group):
        ranges=np.asarray([
            [record[f"t{t}_a{a}_avg"] for a in ids] for t in (1,2)
        ],dtype=float)
        devs=np.asarray([
            [record[f"t{t}_a{a}_std"] for a in ids] for t in (1,2)
        ],dtype=float)
        if previous is None:
            new=(True,True)
        else:
            new=tuple(bool(np.any(ranges[t]!=previous[0][t])
                            or np.any(devs[t]!=previous[1][t]))
                      for t in (0,1))
        previous=(ranges.copy(),devs.copy())
        if not any(new):
            repeated+=1
            continue
        if not all(new):
            partial+=1
            continue
        valid=np.isfinite(ranges)&(ranges>0)
        if np.any(np.sum(valid,axis=1)<min_anchors):
            invalid+=1
            continue
        range_m=np.where(valid,ranges/100.,np.nan)
        sigma_m=np.where(valid,np.maximum(floor,devs/100.),np.nan)
        sigma_m=np.where(np.isfinite(sigma_m),sigma_m,floor)
        # Checagem geométrica de pares (desigualdade triangular):
        # ||d_i(tag1)-d_i(tag2)|| <= baseline, salvo ruído/bias.
        gaps=np.abs(range_m[0]-range_m[1])
        if np.any(np.isfinite(gaps)):
            largest_gap=max(largest_gap,float(np.nanmax(gaps)))
            pairing+=int(np.count_nonzero(gaps>baseline_m+tolerance))
        indices.append(k)
        distance_rows.append(range_m)
        sigma_rows.append(sigma_m)
        full[:,k]=range_m.T.reshape(-1)

    if not indices:
        raise ValueError(
            "Nenhum snapshot novo e pareado com ambas as tags; "
            "confira IDs, unidades e atualização do firmware"
        )
    idx=np.asarray(indices,dtype=int)
    return DualTagPcnt2026(
        odometry=odom,
        sample_indices=idx,
        timestamps_s=stamp[idx],
        distances_by_tag=np.asarray(distance_rows,dtype=float),
        deviations_by_tag=np.asarray(sigma_rows,dtype=float),
        z_hist=full,
        repeated_rows=repeated,
        partial_refresh_rows=partial,
        rejected_rows=invalid,
        header_anchor_ids=header_ids,
        anchor_ids=ids,
        max_pair_gap_m=largest_gap,
        pairing_anomaly_count=pairing,
    )
