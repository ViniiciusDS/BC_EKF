"""Relatório reprodutível de correções BC-EKF de duas tags em PCNT real.

Este comando não calibra rodas, não altera arquivos brutos e NÃO
interpreta NIS como erro versus ground truth. O gate de NIS ou resíduos
individuais é opt-in; sem flags, apenas mede e exporta diagnósticos.

Exemplo (PowerShell, na raiz do repositório):
python tools/audit_bc_ekf_updates_2026.py --pcnt dados_reais/robot2026/pcntLog_14_11_26_1o.csv --anchors anchor_sets/lab_duas_tags_A3_A9_14_11_2026.json --tags tag_profiles/lab_14_11_2026_t1_direita_t2_esquerda_xyz_custom.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.bc_ekf import run_bc_ekf_from_data
from src.fusion.tag_geometry import load_tag_geometry
from src.ui.algo_modes.dataset_two_tag2026_pipeline import load_dual_tag_pcnt
from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    read_robot2026_profile_from_anchor_json,
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_csv(path,rows):
    if not rows:
        raise ValueError(f"Sem registros para exportar: {path}")
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(report,poses,output_dir):
    """Não sobrescreve ensaios anteriores (sufixos _001, _002...)."""
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=True)
    base="bc_ekf_updates"
    number=1
    while (folder/f"{base}_{number:03d}.json").exists():
        number+=1
    stem=f"{base}_{number:03d}"
    json_file=folder/(stem+".json")
    diag_file=folder/(stem+"_steps.csv")
    poses_file=folder/(stem+"_poses.csv")
    # O JSON guarda resumo/proveniência. O CSV guarda TODOS os instantes
    # PCNT, sem repetir 14 matrizes UWB a cada amostra.
    content={k:v for k,v in report.items() if k!="steps"}
    content["reports"]={
        "steps_csv":diag_file.name,
        "poses_csv":poses_file.name,
    }
    with json_file.open("w",encoding="utf-8") as fh:
        json.dump(content,fh,ensure_ascii=False,indent=2,allow_nan=False)
    rows=[]
    for step in report["steps"]:
        row={**step,"used_indices":";".join(str(i) for i in step["used_indices"])}
        rows.append(row)
    _save_csv(diag_file,rows)
    pose_rows=[
        {
            "pcnt_index":i,
            "timestamp_s":float(report["timestamps_s"][i]),
            "x_m":float(p[0]),"y_m":float(p[1]),
            "theta_rad":float(p[2]),
            "theta_deg":float(np.rad2deg(p[2])),
        } for i,p in enumerate(poses.T)
    ]
    _save_csv(poses_file,pose_rows)
    return {"json":json_file,"steps_csv":diag_file,"poses_csv":poses_file}


def create_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pcnt",required=True,type=Path)
    p.add_argument("--anchors",required=True,type=Path)
    p.add_argument("--tags",required=True,type=Path)
    p.add_argument("--session",type=int,default=1)
    p.add_argument("--output-dir",type=Path,
                   default=Path("resultados/robot2026/auditoria_bc_ekf"))
    p.add_argument("--sigma-uwb",type=float,default=None,
                   help="Desvio nominal de range (m); default igual ao Dataset Mode")
    p.add_argument("--min-valid-ranges",type=int,default=1)
    p.add_argument("--nis-gate-per-dof",type=float,default=None,
                   help="Opcional: rejeitar lote se NIS/dof superar este valor")
    p.add_argument("--range-gate-sigma",type=float,default=None,
                   help="Opcional: rejeitar individualmente resíduos padronizados")
    return p


def main(argv=None):
    a=create_parser().parse_args(argv)
    for file in (a.pcnt,a.anchors,a.tags):
        if not file.is_file():
            raise FileNotFoundError(file)
    anchor_doc=json.loads(a.anchors.read_text(encoding="utf-8"))
    anchors=np.asarray(anchor_doc["anchors_xy"],dtype=float)
    profile=read_robot2026_profile_from_anchor_json(a.anchors)
    geometry=load_tag_geometry(
        a.anchors,default_baseline_m=.25,default_height_m=.2,
        tag_profile_json=a.tags
    )
    dual=load_dual_tag_pcnt(
        a.pcnt,a.anchors,profile=profile,session_id=a.session,
        baseline_m=geometry.baseline_m
    )
    odom=dual.odometry
    nominal_T=float(np.median(np.diff(odom.timestamps_s)))
    sigma=float(a.sigma_uwb) if a.sigma_uwb is not None else max(
        .15,float(np.nanmedian(dual.deviations_by_tag))
    )
    poses,diag=run_bc_ekf_from_data(
        T=nominal_T,anchors=anchors.T,
        odometry_noisy=np.vstack((odom.v_m_s,odom.omega_rad_s)),
        z_hist=dual.z_hist,l=geometry.baseline_m/2.,
        z_c=float(np.mean(geometry.offsets[:,2])),
        sigma_uwb=sigma,x0=odom.poses_xytheta[0].copy(),
        tag_offsets_xyz=geometry.offsets,
        timestamps_s=odom.timestamps_s,
        min_valid_ranges=a.min_valid_ranges,
        nis_gate_per_dof=a.nis_gate_per_dof,
        range_gate_sigma=a.range_gate_sigma,
        return_diagnostics=True,
    )
    diag["provenance"]={
        "pcnt":a.pcnt.name,"pcnt_sha256":_hash(a.pcnt),
        "anchors":a.anchors.name,"anchors_sha256":_hash(a.anchors),
        "tag_profile":a.tags.name,"tag_profile_sha256":_hash(a.tags),
        "session":int(a.session),
        "uwb_new_epochs":dual.sample_count,
        "uwb_repeated_rows":dual.repeated_rows,
        "tag_geometry":geometry.as_dict(),
        "anchor_ids":list(dual.anchor_ids),
        "range_order": [
            f"A{anchor}_T{tag}"
            for anchor in dual.anchor_ids for tag in (1,2)
        ],
        "note":"PCNT timestamps são de registro; tempo físico de ranging não confirmado.",
    }
    diag["timestamps_s"]=odom.timestamps_s.tolist()
    valid_nis=[r for r in diag["steps"] if r["nis_per_dof"] is not None]
    diag["largest_nis_events"]=sorted(
        valid_nis,key=lambda e:e["nis_per_dof"],reverse=True
    )[:20]
    diag["warnings"]=[
        "NIS normalizado por dof NÃO é RMSE, nem ground truth.",
        "Gates opcionais mudam a estimação; sem flags ficam DESATIVADOS.",
        "Comparar diagnósticos entre hipóteses de Q/R, delays e medidas independentes.",
        "Atribuição de T1/T2, configuração das âncoras e timesteps afetam a inovação.",
    ]
    paths=_write_reports(diag,poses,a.output_dir)
    print("BC-EKF 2 tags — auditoria de correções UWB (sem GT)")
    for key,val in diag["summary"].items():
        print(f"{key}: {val}")
    for name,path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
