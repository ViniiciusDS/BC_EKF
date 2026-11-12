"""CLI da auditoria UWB por enlace (somente diagnóstico).

Exemplo:
 python -m tools.audit_uwb_links_2026 --log pcnt.csv --anchors layout.json \
     --session 1 --tag-height-m 0.215 --out diagnostico.json --csv diagnostico.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.analysis.uwb_link_quality import build_uwb_link_quality_report


FIELDS = [
    "slot_spi", "anchor_index_0_based", "mapped", "anchor_xyz_m",
    "height_separation_m", "new_valid", "repeated_valid", "invalid",
    "fresh_fraction_all_rows", "physically_impossible_new",
    "loa_attempted", "loa_solved", "loa_residual_above_threshold",
    "loa_outlier_fraction_of_solved", "new_range_mean_m",
    "new_range_p95_m", "reported_sigma_mean_m",
    "loa_abs_residual_rms_m", "loa_abs_residual_p95_m",
    "loa_abs_residual_max_m",
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnóstico por slot SPI/âncora UWB: repetição, limite físico e LOAO"
    )
    parser.add_argument("--log", type=Path, required=True, help="CSV bruto PCNT 5+3*N sem cabeçalho")
    parser.add_argument("--anchors", type=Path, required=True, help="JSON Nx3 e mapa explícito SPI→âncora")
    parser.add_argument("--session", type=int, default=1, help="Sessão após resets (1 por padrão)")
    parser.add_argument(
        "--tag-height-m", type=float, required=True,
        help="Altura Z da tag em metros no referencial das âncoras (ex. 0.215)",
    )
    parser.add_argument(
        "--height-tolerance-m", type=float, default=0.,
        help="Margem para incerteza da cota Z (padrão 0 m; somente diagnóstico)",
    )
    parser.add_argument(
        "--loa-outlier-threshold-m", type=float, default=.30,
        help="Limiar descritivo de resíduo LOAO absoluto (padrão 0.30 m)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Novo relatório JSON")
    parser.add_argument("--csv", type=Path, help="Novo resumo CSV por slot")
    args = parser.parse_args(argv)
    sources = {p.resolve() for p in (args.log, args.anchors)}
    outputs = [p.resolve() for p in (args.out, args.csv) if p is not None]
    if len(outputs) != len(set(outputs)) or any(p in sources for p in outputs):
        parser.error("Saídas devem ser distintas e não podem substituir os arquivos de entrada")
    if any(p.exists() for p in outputs):
        parser.error("Arquivo de saída já existe; use outro nome para preservar o diagnóstico anterior")

    try:
        report = build_uwb_link_quality_report(
            args.log, args.anchors, session_id=args.session,
            tag_height_m=args.tag_height_m,
            height_tolerance_m=args.height_tolerance_m,
            loa_outlier_threshold_m=args.loa_outlier_threshold_m,
        )
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))

    created = []
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            created.append(args.out)
            stream.write(payload + "\n")
        if args.csv is not None:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("x", encoding="utf-8", newline="") as stream:
                created.append(args.csv)
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for slot in report["per_slot"]:
                    range_stats = slot["range_new_summary_m"] or {}
                    sigma_stats = slot["reported_sigma_new_summary_m"] or {}
                    loa_stats = slot["loa_abs_residual_summary_m"] or {}
                    writer.writerow({
                        **{key:slot.get(key) for key in FIELDS if key in slot},
                        "new_range_mean_m": range_stats.get("mean_m"),
                        "new_range_p95_m": range_stats.get("p95_m"),
                        "reported_sigma_mean_m": sigma_stats.get("mean_m"),
                        "loa_abs_residual_rms_m": loa_stats.get("rms_m"),
                        "loa_abs_residual_p95_m": loa_stats.get("p95_m"),
                        "loa_abs_residual_max_m": loa_stats.get("max_m"),
                    })
    except Exception:
        for name in reversed(created):
            name.unlink(missing_ok=True)
        raise

    print(
        f"UWB por enlace: {report['pcnt_rows']} linhas PCNT, "
        f"{report['fresh_rows_with_4plus_mapped_ranges']} linhas com 4+ ranges novos, "
        f"{report['physically_impossible_new_total']} medidas novas fisicamente incompatíveis"
    )
    print(f"JSON: {args.out}")
    if args.csv:
        print(f"CSV: {args.csv}")
    return report


if __name__ == "__main__":
    main()
