"""Gera variante EXPERIMENTAL de um perfil de campo 2026.

Nunca sobrescreve o JSON de entrada nem arquivos existentes. Preserva
coordenadas de âncoras, mapeamento dos slots e configurações UWB; troca
somente a hipótese de odometria e a visualização das voltas.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def build_variant(source_json, *, start_xy, direction_deg=0.0, gate_source="uwb_lms"):
    """Retorna novo dicionário de cenário; input nunca é mutado."""
    root = json.loads(Path(source_json).read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("Configuração de origem deve ser objeto JSON")
    if root.get("meta", {}).get("example_only") is True:
        raise ValueError("Não usar perfil de exemplo sem coordenadas reais")
    for name in ("anchors_xy", "pcnt_slot_to_anchor_index"):
        if not isinstance(root.get(name), list) or not root[name]:
            raise ValueError(f"Configuração de origem deve conter {name}")
    for v in (*start_xy, direction_deg):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("Posição e direção devem ser finitas")
    if gate_source not in ("odometry", "uwb_lms"):
        raise ValueError("gate_source inválido")

    # Copia pelo parser JSON: garante isolamento de objetos aninhados.
    cloned = json.loads(json.dumps(root, ensure_ascii=False))
    robot = dict(cloned.get("robot_2026", {}))
    robot.pop("initial_theta_rad", None)
    robot.update({
        "swap_encoder_columns": False,
        "invert_left": False,
        "invert_right": True,
        "left_scale": 1.0,
        "right_scale": 1.0,
        "initial_theta_deg": 0.0,
        "initial_x_m": float(start_xy[0]),
        "initial_y_m": float(start_xy[1]),
    })
    cloned["robot_2026"] = robot

    heading = math.radians(direction_deg)
    view = dict(cloned.get("view_2026", {}))
    view.update({
        "segmentation": "start_gate",
        "gate_source": gate_source,
        # LAB7A: o retorno ocorre pelo lado +Y (vindo da última
        # reta), e a nova volta parte em +X. Não é uma travessia
        # do portão perpendicular a +X; é uma reentrada num canto.
        "gate_mode": "corner_return",
        "gate_xy_m": [float(start_xy[0]), float(start_xy[1])],
        "gate_direction_xy": [math.cos(heading), math.sin(heading)],
        "gate_return_direction_xy": [-math.sin(heading), math.cos(heading)],
        "gate_return_radius_m": .5,
        "gate_smoothing_window": 5 if gate_source == "uwb_lms" else 1,
        "gate_approach_lookback_s": 2.5,
        "gate_approach_min_m": .06,
        "gate_max_gap_s": 8.0,
        "gate_half_width_m": .4,
        "gate_hysteresis_m": .12,
        "gate_min_lap_s": 30.0,
        "gate_min_lap_distance_m": 4.5,
        "gate_departure_m": .8,
        "gate_max_step_m": .8,
        "window_duration_s": 80.0,
    })
    cloned["view_2026"] = view
    cloned["meta"] = {
        **cloned.get("meta", {}),
        "experimental_profile": True,
        "based_on_json": Path(source_json).name,
        "odometry_hypothesis": "PR11 swap_channels_and_signs, scale=1; not physically calibrated",
        "start_gate_is_virtual": True,
        "lap_source": gate_source,
        "reference_note": "LAB7A: saída +X, primeira curva horário; requer validação adicional",
    }
    # Validar ANTES de salvar, utilizando os parsers da aplicação.
    return cloned


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Criar cópia experimental do perfil LAB7A para PR #12"
    )
    parser.add_argument("--source", required=True, help="JSON original com âncoras medidas")
    parser.add_argument("--out", required=True, help="Novo JSON de cenário (não sobrescreve)")
    parser.add_argument("--start-x", type=float, default=.15)
    parser.add_argument("--start-y", type=float, default=1.38)
    parser.add_argument("--direction-deg", type=float, default=0.0)
    parser.add_argument(
        "--gate-source", choices=("uwb_lms", "odometry"), default="uwb_lms",
    )
    args = parser.parse_args(argv)
    source, dest = Path(args.source).resolve(), Path(args.out).resolve()
    if source == dest or dest.exists():
        parser.error("Arquivo de saída não pode existir ou ser o arquivo de entrada")
    try:
        variant = build_variant(
            source, start_xy=(args.start_x, args.start_y),
            direction_deg=args.direction_deg,
            gate_source=args.gate_source,
        )
        # Verifica parâmetros cinemáticos antes de criar o arquivo.
        # O leitor do Dataset Mode também valida o JSON de portão
        # integralmente quando o operador selecionar esse perfil.
        cfg = variant["robot_2026"]
        if not all(isinstance(cfg[k], bool) for k in (
            "swap_encoder_columns", "invert_left", "invert_right"
        )):
            raise ValueError("Sinais de encoder inválidos")
        from src.odometry.pcntlog_adapter import Robot2026Profile
        profile = Robot2026Profile(
            wheel_radius_m=float(cfg.get("wheel_radius_m", .03425)),
            wheel_base_m=float(cfg.get("wheel_base_m", .229)),
            ticks_per_wheel_rev=float(cfg.get("ticks_per_wheel_rev", 4560)),
            invert_left=cfg["invert_left"],
            invert_right=cfg["invert_right"],
            swap_encoder_columns=cfg["swap_encoder_columns"],
            left_scale=cfg["left_scale"],
            right_scale=cfg["right_scale"],
            initial_x_m=cfg["initial_x_m"],
            initial_y_m=cfg["initial_y_m"],
            initial_theta_rad=math.radians(cfg["initial_theta_deg"]),
        )
        profile.validate()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusividade evita corrida e preserva eventual arquivo criado
    # depois do teste de existência acima.
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(variant, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Perfil experimental: {dest}")
    return variant


if __name__ == "__main__":
    main()
