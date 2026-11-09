"""CLI para calibração assistida sem escrever no JSON de hardware.

Uso: python -m tools.check_encoder_calibration_2026 --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.odometry.assisted_calibration import CalibrationReference, analyze_encoder_calibration
from src.ui.algo_modes.dataset_uwb2026_pipeline import read_robot2026_profile_from_anchor_json
from src.ui.algo_modes.dataset_odometry_pipeline import ROBOT2026_PROFILE


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pré-calibração assistida (somente diagnóstico)")
    parser.add_argument("--log", required=True, help="CSV bruto PCNT com 5+3*N colunas, sem cabeçalho")
    parser.add_argument("--anchors", help="JSON com parâmetros robot_2026; opcional para PCNT puro")
    parser.add_argument("--session", type=int, default=1)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--straight-m", type=float, help="Distância real medida para frente, em metros")
    group.add_argument("--turn-deg", type=float, help="Ângulo de giro medido (+anti-horário), em graus")
    parser.add_argument("--start-s", type=float, help="Início do movimento em segundos relativos à sessão")
    parser.add_argument("--end-s", type=float, help="Fim do movimento em segundos relativos à sessão")
    parser.add_argument("--out", help="Arquivo JSON de saída; não pode ser o CSV ou JSON de entrada")
    args = parser.parse_args(argv)
    profile = (read_robot2026_profile_from_anchor_json(args.anchors)
               if args.anchors else ROBOT2026_PROFILE)
    ref = CalibrationReference(
        kind="straight" if args.straight_m is not None else "turn",
        value=args.straight_m if args.straight_m is not None else args.turn_deg,
        start_s=args.start_s, end_s=args.end_s,
    )
    report = analyze_encoder_calibration(args.log, profile, ref, args.session)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        dest = Path(args.out).resolve()
        inputs = {Path(args.log).resolve()}
        if args.anchors:
            inputs.add(Path(args.anchors).resolve())
        if dest in inputs:
            parser.error("Saída não pode sobrescrever arquivo de entrada")
        # Evita perder relatórios de ensaios anteriores quando o operador
        # reutiliza acidentalmente o mesmo nome de saída.
        if dest.exists():
            parser.error("Arquivo de saída já existe; escolha outro nome em --out")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialized + "\n", encoding="utf-8")
        print(f"Relatório salvo: {dest}")
    else:
        print(serialized)
    return report


if __name__ == "__main__":
    main()
