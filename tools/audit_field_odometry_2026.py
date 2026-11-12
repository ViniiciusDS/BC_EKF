"""CLI para auditoria de deriva por volta, sem alterar perfis.

Execute python -m tools.audit_field_odometry_2026 --help
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.odometry.field_calibration_audit import audit_field_odometry
from src.ui.algo_modes.dataset_uwb2026_pipeline import (
    read_robot2026_profile_from_anchor_json,
)


CSV_FIELDS = (
    "scenario", "segment_number", "lap_duration_s", "odom_path_m",
    "gate_to_gate_closure_m", "cumulative_offset_from_first_gate_m",
    "odom_net_heading_deg_unwrapped", "heading_residual_deg_assumed",
    "gate_start_index_pcnt", "gate_end_index_pcnt",
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Auditoria não destrutiva da odometria por voltas físicas F7"
    )
    parser.add_argument("--log", required=True, help="CSV PCNT bruto LAB7A")
    parser.add_argument("--anchors", required=True, help="Mesmo JSON de âncoras usado no F7")
    parser.add_argument("--laps", required=True, help="Relatório JSON F7 do PR #13")
    parser.add_argument("--algorithm", default="lms", help="Chave exata do algoritmo F7: lms, lmsp, gauss_newton etc")
    parser.add_argument(
        "--expected-lap-turn-deg", type=float,
        help="Hipótese externa de giro por circuito (LAB7A horário: -360); não é medição angular",
    )
    parser.add_argument(
        "--candidate-scales", nargs=2, metavar=("LEFT", "RIGHT"), type=float,
        help="Escalas absolutas hipotéticas para comparação (não aplicar automaticamente)",
    )
    parser.add_argument("--out", required=True, help="Novo relatório JSON, não sobrescreve")
    parser.add_argument("--csv", help="Novo relatório CSV por volta e hipótese")
    args = parser.parse_args(argv)

    sources = {Path(p).resolve() for p in (args.log, args.anchors, args.laps)}
    outputs = [Path(p).resolve() for p in (args.out, args.csv) if p]
    if len(set(outputs)) != len(outputs):
        parser.error("Saídas JSON e CSV não podem ter o mesmo nome")
    for dest in outputs:
        if dest in sources:
            parser.error("Não é permitido substituir um arquivo de entrada")
        if dest.exists():
            parser.error(f"Relatório já existe: {dest}. Escolha outro nome")

    try:
        source = json.loads(Path(args.laps).read_text(encoding="utf-8"))
        profile = read_robot2026_profile_from_anchor_json(args.anchors)
        report = audit_field_odometry(
            source, args.log, args.anchors, profile,
            algorithm=args.algorithm,
            expected_lap_turn_deg=args.expected_lap_turn_deg,
            scale_candidate=args.candidate_scales,
        )
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.error(str(exc))

    # Exclusividade no momento da escrita evita sobrescrever resultados
    # anteriores mesmo se outro processo criar arquivo após pré-checagem.
    created = []
    try:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as stream:
            created.append(out)
            stream.write(payload+"\n")
        if args.csv:
            dest = Path(args.csv)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", newline="", encoding="utf-8") as stream:
                created.append(dest)
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for scenario in report["scenarios"]:
                    for segment in scenario["complete_laps"]:
                        writer.writerow({
                            "scenario": scenario["name"],
                            **{k:segment[k] for k in CSV_FIELDS if k != "scenario"},
                        })
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise

    print(
        f"Auditoria: {report['complete_lap_count']} voltas completas, "
        f"{len(report['scenarios'])} configurações. JSON: {args.out}"
    )
    if args.csv:
        print(f"Comparação por volta: {args.csv}")
    return report


if __name__ == "__main__":
    main()
