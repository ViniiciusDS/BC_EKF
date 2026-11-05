"""CLI: executar a partir da raiz do BC_EKF.

python tools/convert_pcntlog_2026.py pcntLog.csv --out resultados/robot2026
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.odometry.pcntlog_adapter import Robot2026Profile, export_pcnt_sessions


def main():
    parser = argparse.ArgumentParser(description="Converte pcntLog.csv (5/20 colunas)")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("resultados/robot2026"))
    parser.add_argument("--wheel-radius", type=float, default=0.03425)
    parser.add_argument("--wheel-base", type=float, default=0.229)
    parser.add_argument("--cpr", type=float, default=2280.0)
    parser.add_argument("--left-scale", type=float, default=1.0)
    parser.add_argument("--right-scale", type=float, default=1.0)
    parser.add_argument("--initial-theta-deg", type=float, default=0.0)
    args = parser.parse_args()
    import math
    cfg = Robot2026Profile(
        wheel_radius_m=args.wheel_radius,
        wheel_base_m=args.wheel_base,
        ticks_per_wheel_rev=args.cpr,
        left_scale=args.left_scale,
        right_scale=args.right_scale,
        initial_theta_rad=math.radians(args.initial_theta_deg),
    )
    result = export_pcnt_sessions(args.input, args.out, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
