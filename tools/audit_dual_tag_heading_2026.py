"""CLI independente para auditoria angular PCNT × UWB de duas tags.

Exemplo:
python tools/audit_dual_tag_heading_2026.py
  --pcnt dados_reais/robot2026/pcntLog_14_11_26_1o.csv
  --anchors anchor_sets/lab_duas_tags_A3_A9_14_11_2026.json
  --tags tag_profiles/lab_14_11_2026_t1_direita_t2_esquerda_xyz_custom.json
  --session 1 --output-dir resultados/robot2026/auditoria_angular

Não grava no CSV bruto e não altera perfil de roda/âncora/tag. NumPy
é obrigatório; Matplotlib é opcional para PNGs (relatório sempre salvo).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Permite chamar diretamente do diretório raíz com 'python tools/...'.
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

from src.analysis.dual_tag_heading_audit import (
    HeadingAuditConfig,audit_dual_tag_heading,
)
from src.fusion.tag_geometry import load_tag_geometry
from src.ui.algo_modes.dataset_two_tag2026_pipeline import load_dual_tag_pcnt
from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    read_robot2026_profile_from_anchor_json,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path,items):
    if not items:
        raise ValueError(f"Relatório sem linhas: {path}")
    keys=list(items[0].keys())
    with Path(path).open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=keys)
        writer.writeheader()
        writer.writerows(items)


def display_uwb_heading(epochs):
    """UWB em ramo contínuo consistente com o painel de discordância.

    Cada observação UWB é estimada independentemente do PCNT.
    Somente a representação MULTIVOLTAS é ambígua sem referência
    externa; escolhe-se o mesmo ramo da diferença circular já
    desenrolada no relatório, para não exibir discrepância artificial
    de 360° entre gráficos após lacunas longas.
    """
    odom=np.asarray([r["theta_odom_deg"] for r in epochs],dtype=float)
    delta=np.asarray([
        float(r["difference_unwrapped_deg"])
        if r["difference_unwrapped_deg"] is not None else np.nan
        for r in epochs
    ],dtype=float)
    return odom+delta


def save_heading_audit(report,output_dir,*,plots=True):
    """Exporta JSON/CSVs e, quando matplotlib existe, 4 PNGs separados."""
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    stem="dual_tag_heading_audit"
    json_path=out/(stem+".json")
    with json_path.open("w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    exports={"json":str(json_path)}
    for label,key in (("snapshots","uwb_snapshots"),
                      ("pcnt","pcnt_steps"),
                      ("windows","windows")):
        dst=out/(stem+"_"+label+".csv")
        _write_csv(dst,report[key])
        exports[label]=str(dst)
    if plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            exports["plot_warning"]="Matplotlib indisponível; CSV/JSON salvos."
        else:
            epochs=report["uwb_snapshots"]
            pcnt=report["pcnt_steps"]
            t=np.array([r["timestamp_s"] for r in epochs])
            valid=np.array([r["valid"] for r in epochs],dtype=bool)
            def vals(key):
                return np.array([
                    np.nan if r[key] is None else r[key] for r in epochs
                ],dtype=float)
            pt=np.array([r["timestamp_s"] for r in pcnt])
            odom=np.array([r["theta_odom_deg"] for r in pcnt])
            # O formato MULTIVOLTAS da orientação UWB é ambíguo
            # após gaps longos. Não desenrolar raw UWB de modo
            # diferente do painel de discordância: isso gerava erro
            # VISUAL de 360° após ~631s no ensaio real.
            uwb_cont=display_uwb_heading(epochs)
            start=report["config"]["focus_start_s"]
            end=report["config"]["focus_end_s"]
            plots_spec=[
                ("angulos","Heading PCNT e UWB (ramo multivoltas do desvio)",
                 "Orientação (graus)"),
                ("divergencia","Discordância UWB - PCNT (não é erro GT)",
                 "Graus (desenrolado)"),
                ("encoders","Incrementos brutos normalizados dos encoders",
                 "Ticks por amostra PCNT"),
                ("qualidade","Distância entre tags inferida pelo UWB",
                 "Separação estimada (m)"),
            ]
            for label,title,ylabel in plots_spec:
              for focus_only in (False,True):
                fig,ax=plt.subplots(figsize=(12,4.3))
                def plotted(x,y,**kwargs):
                    x=np.asarray(x,dtype=float)
                    y=np.asarray(y,dtype=float)
                    if focus_only:
                        # Desenhar APENAS o trecho com margem para
                        # autoscale Y real do foco; não esconder a curva.
                        selection=((x>=max(0.,start-10.))&
                                   (x<=end+10.))
                        x=x[selection]
                        y=y[selection]
                    ax.plot(x,y,**kwargs)
                if label=="angulos":
                    plotted(pt,odom,label="PCNT: theta contínuo",lw=1)
                    plotted(t,uwb_cont,
                            label="UWB (ramo consistente com desvio)",lw=1,
                            marker=".",ms=3)
                elif label=="divergencia":
                    plotted(t,vals("difference_unwrapped_deg"),
                            label="UWB - PCNT",marker=".",ms=3,lw=1)
                    plotted(t,vals("relative_drift_deg"),
                            label="Relativo à 1ª época válida",alpha=.7,lw=1)
                elif label=="encoders":
                    plotted(pt,[r["delta_left_ticks"] for r in pcnt],
                            lw=.8,label="Roda esquerda")
                    plotted(pt,[r["delta_right_ticks"] for r in pcnt],
                            lw=.8,label="Roda direita")
                else:
                    plotted(t,vals("measured_baseline_m"),
                            label="UWB estimado",lw=1)
                    ax.axhline(
                        report["geometry"]["baseline_m"],ls="--",
                        label="Baseline físico configurado"
                    )
                    ax.axhline(
                        report["geometry"]["baseline_m"]+
                        report["config"]["max_baseline_error_m"],ls=":",
                        label="Limite superior de qualidade"
                    )
                ax.axvspan(start,end,alpha=.1,label="Foco diagnóstico")
                if focus_only:
                    ax.set_xlim(max(0.,start-10.),end+10.)
                ax.set(
                    title=title+(" — foco" if focus_only else ""),
                    xlabel="Tempo PCNT (s)",ylabel=ylabel,
                )
                ax.grid(True,alpha=.3)
                ax.legend(fontsize=8)
                fig.tight_layout()
                suffix="_foco" if focus_only else ""
                dest=out/(stem+"_"+label+suffix+".png")
                fig.savefig(dest,dpi=150)
                plt.close(fig)
                exports[label+suffix]=str(dest)
    return exports


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcnt",type=Path,required=True,
                        help="CSV bruto do firmware com duas tags e cabeçalho")
    parser.add_argument("--anchors",type=Path,required=True,
                        help="JSON de âncoras em XYZ, com uwb_dual_2026")
    parser.add_argument("--tags",type=Path,required=True,
                        help="Perfil separado que mapeia T1/T2 para XYZ")
    parser.add_argument("--session",type=int,default=1)
    parser.add_argument("--output-dir",type=Path,
                        default=Path("resultados/robot2026/auditoria_angular"))
    parser.add_argument("--window-s",type=float,default=10)
    parser.add_argument("--focus-start",type=float,default=80)
    parser.add_argument("--focus-end",type=float,default=120)
    parser.add_argument("--sigma-floor",type=float,default=.10)
    parser.add_argument("--huber-delta",type=float,default=.20)
    parser.add_argument("--max-fit-rms",type=float,default=.20)
    parser.add_argument("--max-baseline-error",type=float,default=.12)
    parser.add_argument("--max-heading-sigma",type=float,default=40)
    parser.add_argument("--no-plots",action="store_true")
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    if not args.pcnt.is_file():
        raise FileNotFoundError(args.pcnt)
    with args.anchors.open("r",encoding="utf-8") as fh:
        anchor_data=json.load(fh)
    xyz=np.asarray(anchor_data["anchors_xy"],dtype=float)
    geometry=load_tag_geometry(
        args.anchors,default_baseline_m=.25,default_height_m=.20,
        tag_profile_json=args.tags,
    )
    profile=read_robot2026_profile_from_anchor_json(args.anchors)
    dual=load_dual_tag_pcnt(
        args.pcnt,args.anchors,profile=profile,session_id=args.session,
        baseline_m=geometry.baseline_m,
    )
    settings=HeadingAuditConfig(
        window_s=args.window_s,
        focus_start_s=args.focus_start,focus_end_s=args.focus_end,
        sigma_floor_m=args.sigma_floor,
        huber_delta_m=args.huber_delta,
        max_fit_rms_m=args.max_fit_rms,
        max_baseline_error_m=args.max_baseline_error,
        max_heading_sigma_deg=args.max_heading_sigma,
    )
    report=audit_dual_tag_heading(
        timestamps_s=dual.odometry.timestamps_s,
        odometry_xytheta=dual.odometry.poses_xytheta,
        pcnt_delta_left_ticks=dual.pcnt_delta_left_ticks,
        pcnt_delta_right_ticks=dual.pcnt_delta_right_ticks,
        sample_indices=dual.sample_indices,
        distances_by_tag=dual.distances_by_tag,
        deviations_by_tag=dual.deviations_by_tag,
        anchors_xyz=xyz,
        tag_offsets_xyz=geometry.offsets,
        config=settings,
    )
    report["provenance"]={
        "pcnt_file":args.pcnt.name,"pcnt_sha256":_sha(args.pcnt),
        "anchors_file":args.anchors.name,
        "anchors_sha256":_sha(args.anchors),
        "tag_profile_file":args.tags.name,
        "tag_profile_sha256":_sha(args.tags),
        "session_id":args.session,
        "geometry":geometry.as_dict(),
        "raw_pcnt_count":dual.odometry.sample_count,
        "source_uwb_epochs":dual.sample_count,
        "repeated_uwb_rows":dual.repeated_rows,
        "partial_uwb_refresh_rows":dual.partial_refresh_rows,
        "range_pair_anomalies":dual.pairing_anomaly_count,
        "robot_profile":{
            "wheel_radius_m":profile.wheel_radius_m,
            "wheel_base_m":profile.wheel_base_m,
            "ticks_per_wheel_rev":profile.ticks_per_wheel_rev,
            "swap_encoder_columns":profile.swap_encoder_columns,
            "invert_left":profile.invert_left,
            "invert_right":profile.invert_right,
            "left_scale":profile.left_scale,
            "right_scale":profile.right_scale,
            "initial_x_m":profile.initial_x_m,
            "initial_y_m":profile.initial_y_m,
            "initial_theta_rad":profile.initial_theta_rad,
        },
    }
    exports=save_heading_audit(
        report,args.output_dir,plots=not args.no_plots
    )
    print("=== AUDITORIA ANGULAR (SEM GROUND TRUTH) ===")
    for name,val in report["summary"].items():
        if name not in ("top_change_windows","focus_window","quality_reasons"):
            print(f"{name}: {val}")
    print("Janelas de foco:")
    for window in report["summary"]["focus_window"]:
        print(
            f" {window['start_s']:.1f}–{window['end_s']:.1f}s: "
            f"UWB válido={window['uwb_good_count']}, "
            f"Δang(UWB-PCNT)={window['angular_difference_change_deg']}, "
            f"giro PCNT={window['pcnt_turn_deg']:.1f}°"
        )
    for key,path in exports.items():
        print(f"{key}: {path}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
