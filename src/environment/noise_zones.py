from __future__ import annotations

from typing import Iterable

from .noise_profiles import noise_profile_severity


def make_rect_noise_zone(
    x: float,
    y: float,
    w: float,
    h: float,
    profile: str,
    *,
    zone_id: str | None = None,
) -> dict:
    '''Cria um dicionário representando uma zona de ruído retangular com as coordenadas, dimensões e perfil especificados. 
    O campo "id" é opcional e pode ser fornecido para identificar a zona.'''
    return {
        "id": zone_id or "",
        "type": "rect",
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
        "profile": str(profile),
    }


def normalize_noise_zone(zone: dict) -> dict | None:
    '''Normaliza um dicionário representando uma zona de ruído, garantindo que ele tenha as chaves e valores corretos.'''
    if not isinstance(zone, dict):
        return None

    if zone.get("type") != "rect":
        return None

    try:
        x = float(zone.get("x", 0.0))
        y = float(zone.get("y", 0.0))
        w = float(zone.get("w", 0.0))
        h = float(zone.get("h", 0.0))
    except Exception:
        return None

    if w == 0 or h == 0:
        return None

    if w < 0:
        x = x + w
        w = abs(w)

    if h < 0:
        y = y + h
        h = abs(h)

    return {
        "id": str(zone.get("id", "")),
        "type": "rect",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "profile": str(zone.get("profile", "default")),
    }


def normalize_noise_zones(zones: Iterable[dict] | None) -> list[dict]:
    '''Normaliza uma lista de dicionários representando zonas de ruído, garantindo que cada zona tenha as chaves e valores corretos.'''
    out: list[dict] = []
    if not zones:
        return out

    for zone in zones:
        nz = normalize_noise_zone(zone)
        if nz is not None:
            out.append(nz)

    return out


def point_in_rect(px: float, py: float, rect: dict) -> bool:
    '''Verifica se um ponto (px, py) está dentro de um retângulo definido por um dicionário com as chaves "x", "y", "w" e "h".'''
    rx = float(rect["x"])
    ry = float(rect["y"])
    rw = float(rect["w"])
    rh = float(rect["h"])
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def point_in_noise_zone(px: float, py: float, zone: dict) -> bool:
    '''Verifica se um ponto (px, py) está dentro de uma zona de ruído.'''
    if zone.get("type") != "rect":
        return False
    return point_in_rect(px, py, zone)


def _ccw(ax, ay, bx, by, cx, cy) -> bool:
    '''Verifica se os pontos A(ax, ay), B(bx, by) e C(cx, cy) estão em sentido anti-horário.'''
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def _segments_intersect(a, b, c, d) -> bool:
    '''Verifica se os segmentos AB e CD se intersectam.'''
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    
    return (_ccw(ax, ay, cx, cy, dx, dy) != _ccw(bx, by, cx, cy, dx, dy)) and (
        _ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy)
    )


def segment_intersects_rect(p0, p1, rect: dict) -> bool:
    '''Verifica se o segmento definido pelos pontos p0 e p1 intersecta um retângulo definido por um dicionário com as chaves
    "x", "y", "w" e "h".'''
    x = float(rect["x"])
    y = float(rect["y"])
    w = float(rect["w"])
    h = float(rect["h"])

    corners = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
    ]

    if point_in_rect(p0[0], p0[1], rect) or point_in_rect(p1[0], p1[1], rect):
        return True

    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]

    for a, b in edges:
        if _segments_intersect(p0, p1, a, b):
            return True

    return False

def zones_containing_point(px: float, py: float, zones: list[dict] | None) -> list[dict]:
    if not zones:
        return []
    return [z for z in zones if point_in_noise_zone(px, py, z)]


def zones_intersecting_segment(p0, p1, zones: list[dict] | None) -> list[dict]:
    if not zones:
        return []
    out = []
    for z in zones:
        if z.get("type") == "rect" and segment_intersects_rect(p0, p1, z):
            out.append(z)
    return out


def worst_zone_from_list(zones: list[dict] | None) -> dict | None:
    if not zones:
        return None
    return max(zones, key=lambda z: noise_profile_severity(z.get("profile")))


def worst_zone_at_point(px: float, py: float, zones: list[dict] | None) -> dict | None:
    return worst_zone_from_list(zones_containing_point(px, py, zones))


def worst_zone_on_segment(p0, p1, zones: list[dict] | None) -> dict | None:
    return worst_zone_from_list(zones_intersecting_segment(p0, p1, zones))