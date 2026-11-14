"""Geometria rígida de duas tags UWB em coordenadas locais do robô.

Motor de domínio NumPy/JSON independente da fonte e do Pygame.
Referencial: +X frente, +Y esquerda, +Z para cima. Cada linha da
matriz 2x3 corresponde, em ordem, ao canal UWB 0 e ao canal UWB 1;
o nome é informativo, NÃO altera automaticamente o mapeamento do log.

Compatibilidade: arquivos sem "tag_geometry" preservam rigorosamente
as posições clássicas [±baseline/2, 0, height] do BC-EKF.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


def _finite(name, value, *, positive=False):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError(f"{name} deve ser número finito" +
                         (" e positivo" if positive else ""))
    return float(value)


@dataclass(frozen=True)
class TagGeometry:
    """Configuração imutável dos dois canais de ranging em ordem de aquisição."""
    preset: str
    channels: tuple[str, str]
    offsets_xyz_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    source: str = "legacy"

    def __post_init__(self):
        if self.preset not in ("front_rear", "left_right", "custom"):
            raise ValueError("preset de geometria desconhecido")
        if len(self.channels) != 2 or any(not isinstance(n,str) or not n.strip()
                                          for n in self.channels):
            raise ValueError("geometria requer dois nomes de tags")
        if self.channels[0] == self.channels[1]:
            raise ValueError("tags precisam ter identificadores distintos")
        offsets=np.asarray(self.offsets_xyz_m,dtype=float)
        if offsets.shape != (2,3) or not np.all(np.isfinite(offsets)):
            raise ValueError("geometria exige offsets XYZ 2x3 finitos")
        if np.linalg.norm(offsets[0]-offsets[1]) <= 1e-9:
            raise ValueError("duas tags não podem ocupar o mesmo ponto")

    @property
    def offsets(self):
        """Matriz de cópia, evitando mutações acidentais."""
        return np.asarray(self.offsets_xyz_m,dtype=float).copy()

    @property
    def baseline_m(self):
        return float(np.linalg.norm(self.offsets[0]-self.offsets[1]))

    def as_dict(self):
        """Sumário JSON para auditoria e status na interface."""
        return {
            "preset":self.preset, "source":self.source,
            "channel_order":list(self.channels),
            "tags":[
                {"name":name,"x_m":xyz[0],"y_m":xyz[1],"z_m":xyz[2]}
                for name,xyz in zip(self.channels,self.offsets_xyz_m)
            ],
            "baseline_m":self.baseline_m,
        }


def resolve_tag_geometry(
    settings=None, *, default_baseline_m: float, default_height_m: float,
    selection: str="auto",
) -> TagGeometry:
    """Monta presets ou coordenadas personalizadas em ordem dos canais.

    selection='auto' lê o preset do JSON ou aplica legado quando ausente.
    Os presets podem ser escolhidos pela interface sem mudar o JSON.
    Para 'custom' duas coordenadas completas (x,y,z) são obrigatórias.

    O canal 0 continua sendo a PRIMEIRA coluna UWB de cada âncora;
    o canal 1, a SEGUNDA. Ao mudar montagem, confirme qual tag
    física fornece cada canal (não há detecção automática de ID).
    """
    _finite("default_baseline_m",default_baseline_m,positive=True)
    _finite("default_height_m",default_height_m)
    if settings is None:
        settings={}
    if not isinstance(settings,dict):
        raise ValueError("tag_geometry deve ser objeto JSON")
    allowed={"preset","baseline_m","height_m","channel_order","tags"}
    unknown=set(settings)-allowed
    if unknown:
        raise ValueError("Campos desconhecidos em tag_geometry: " +
                         ", ".join(sorted(unknown)))
    if selection not in ("auto","front_rear","left_right","custom"):
        raise ValueError("Seleção de geometria inválida")
    preset=(settings.get("preset","front_rear") if selection=="auto"
            else selection)
    if preset not in ("front_rear","left_right","custom"):
        raise ValueError("Preset precisa ser front_rear, left_right ou custom")

    if preset=="custom":
        tags=settings.get("tags")
        if not isinstance(tags,list) or len(tags)!=2:
            raise ValueError("Preset custom exige tags: lista com duas coordenadas XYZ")
        names=[]
        offsets=[]
        for i,tag in enumerate(tags):
            if not isinstance(tag,dict):
                raise ValueError(f"Tag {i} deve ser objeto JSON")
            if set(tag) != {"name","x_m","y_m","z_m"}:
                raise ValueError(f"Tag {i} exige name, x_m, y_m e z_m")
            if not isinstance(tag["name"],str) or not tag["name"].strip():
                raise ValueError(f"Tag {i} exige nome não vazio")
            names.append(tag["name"].strip())
            offsets.append(tuple(
                _finite(f"tag {i}.{axis}",tag[axis])
                for axis in ("x_m","y_m","z_m")
            ))
        if "channel_order" in settings and list(names)!=settings["channel_order"]:
            raise ValueError("channel_order em custom deve refletir a ordem de tags")
    else:
        baseline=_finite(
            "baseline_m",settings.get("baseline_m",default_baseline_m),
            positive=True,
        )
        height=_finite(
            "height_m",settings.get("height_m",default_height_m),
        )
        names=("front","rear") if preset=="front_rear" else ("left","right")
        # A opção GUI força um NOVO preset. Os nomes de canais de
        # outro preset não são transferíveis (left/right != front/rear).
        # Se o preset é o mesmo, preserva a ordem física explícita.
        from_file = (
            selection=="auto" or
            settings.get("preset", "front_rear")==preset
        )
        provided=(settings.get("channel_order",list(names))
                  if from_file else list(names))
        if (not isinstance(provided,list) or len(provided)!=2
                or set(provided)!=set(names)):
            raise ValueError(
                f"channel_order deve conter exatamente {names[0]} e {names[1]}"
            )
        positions=(
            {"front":(baseline/2,0.,height),"rear":(-baseline/2,0.,height)}
            if preset=="front_rear" else
            {"left":(0.,baseline/2,height),"right":(0.,-baseline/2,height)}
        )
        names=tuple(provided)
        offsets=[positions[n] for n in names]
    source="legacy" if not settings and selection=="auto" else (
        "json" if selection=="auto" else "ui"
    )
    return TagGeometry(
        preset=preset,
        channels=tuple(names),
        offsets_xyz_m=tuple(tuple(float(v) for v in xyz) for xyz in offsets),
        source=source,
    )


def load_tag_geometry(
    anchor_json: str | Path | None, *,
    default_baseline_m: float, default_height_m: float,
    selection: str="auto",
) -> TagGeometry:
    """Configuração do experimento; sem arquivo, cai no legado de forma explícita."""
    settings=None
    if anchor_json is not None and str(anchor_json).lower().endswith(".json"):
        with Path(anchor_json).open("r",encoding="utf-8") as stream:
            root=json.load(stream)
        if not isinstance(root,dict):
            raise ValueError("JSON de âncoras deve conter objeto")
        settings=root.get("tag_geometry")
    if selection=="custom" and (not isinstance(settings,dict) or
                                not settings.get("tags")):
        raise ValueError(
            "Personalizado exige tag_geometry.tags no JSON de âncoras"
        )
    return resolve_tag_geometry(
        settings, default_baseline_m=default_baseline_m,
        default_height_m=default_height_m, selection=selection,
    )
