"""CLI de diagnóstico de campo: compara hipóteses sem recalibrar o robô.

Uso: python -m tools.compare_field_odometry_2026 --help
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.odometry.field_trajectory_diagnostic import (
    compare_odometry_hypotheses,
    load_nominal_route,
    load_straight_calibration_candidates,
)
from src.ui.algo_modes.dataset_uwb2026_pipeline import read_robot2026_profile_from_anchor_json


def main(argv=None):
    """Reintegra o mesmo log em cenários paralelos e exporta resultados."""
    parser = argparse.ArgumentParser(
        description="Comparar hipóteses odométricas sem alterar os dados originais"
    )
    parser.add_argument("--log", required=True, help="CSV PCNT bruto sem cabeçalho")
    parser.add_argument("--anchors", required=True, help="JSON de âncoras/perfil robot_2026")
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--route", help="Rota nominal JSON (distância geométrica somente)")
    parser.add_argument(
        "--calibration-report", action="append", default=[],
        help="Relatório de reta PR #10; repetir para agregar tentativas",
    )
    parser.add_argument("--pose-x", type=float, help="X inicial medido, metros")
    parser.add_argument("--pose-y", type=float, help="Y inicial medido, metros")
    parser.add_argument("--pose-theta-deg", type=float, help="Heading inicial medido, graus")
    parser.add_argument("--out", help="JSON resumo; falha se já existir")
    parser.add_argument("--traces-csv", help="CSV de trajetórias por hipótese")
    args = parser.parse_args(argv)
    fields = (args.pose_x, args.pose_y, args.pose_theta_deg)
    if any(v is not None for v in fields) and not all(v is not None for v in fields):
        parser.error("Informe a pose completa: --pose-x, --pose-y e --pose-theta-deg")
    if args.out is None and args.traces_csv is None:
        parser.error("Escolha --out e/ou --traces-csv para guardar a comparação")

    # Verifique todos os destinos ANTES de gerar qualquer saída.
    sources = {Path(p).resolve() for p in
               [args.log, args.anchors, args.route, *args.calibration_report] if p}
    targets = [Path(p).resolve() for p in (args.out, args.traces_csv) if p]
    if len(set(targets)) != len(targets):
        parser.error("--out e --traces-csv não podem ter mesmo caminho")
    for dest in targets:
        if dest in sources:
            parser.error("Saída não pode substituir arquivo de entrada")
        if dest.exists():
            parser.error(f"Saída já existe, escolha outro nome: {dest}")

    profile = read_robot2026_profile_from_anchor_json(args.anchors)
    straight = load_straight_calibration_candidates(args.calibration_report, profile)
    scales = (
        (straight["left_scale"], straight["right_scale"])
        if straight else None
    )
    report, trace = compare_odometry_hypotheses(
        args.log, profile, session_id=args.session,
        nominal_route=load_nominal_route(args.route),
        measured_pose=fields if all(v is not None for v in fields) else None,
        scale_candidates=scales,
    )
    report["anchor_config_filename"] = Path(args.anchors).name
    report["route_filename"] = Path(args.route).name if args.route else None
    report["straight_calibration_provenance"] = straight
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(payload + "\n", encoding="utf-8")
        print(f"Resumo: {dest}")
    if args.traces_csv:
        dest = Path(args.traces_csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("hypothesis", "timestamp_s", "x_m", "y_m", "theta_deg"),
            )
            writer.writeheader()
            writer.writerows(trace)
        print(f"Trajetórias: {dest}")
    print(f"Analisada sessão {args.session}: {len(report['cases'])} hipóteses")
    return report


if __name__ == "__main__":
    main()
