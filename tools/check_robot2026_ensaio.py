"""Inspeciona um ensaio do robô 2026 antes de rodar métodos UWB.

Exemplo (PowerShell):
  python -m tools.check_robot2026_ensaio --log dados_reais/robot2026/ensaio01.csv --anchors anchor_sets/ensaio01.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ui.algo_modes.dataset_uwb2026_pipeline import inspect_robot2026_capture


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Log bruto PCNT CSV/TXT")
    parser.add_argument("--anchors", type=Path, help="JSON físico das âncoras do ensaio")
    parser.add_argument("--session", type=int, default=1, help="Sessão após resets (padrão: 1)")
    parser.add_argument("--out", type=Path, help="Gravar relatório JSON opcional")
    args = parser.parse_args(argv)

    try:
        result = inspect_robot2026_capture(
            args.log, session_id=args.session, anchor_path=args.anchors
        )
    except (OSError, ValueError) as exc:
        result = {"status": "blocked", "file": str(args.log), "warnings": [str(exc)]}

    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if result["status"] in ("ready_for_static_batch", "odometry_only") else 2


if __name__ == "__main__":
    raise SystemExit(main())
