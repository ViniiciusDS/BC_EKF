from __future__ import annotations

NOISE_PROFILES = {
    "default": {
        "label": "Padrão",
        "short_label": "DEF",
        "severity": 0,
        "color": None,
        "border_color": None,
        "sigma_scale": 1.0,
        "bias_mean": 0.0,
        "bias_std": 0.0,
        "dropout_prob": 0.0,
        "outlier_prob": 0.0,
    },
    "low_noise": {
        "label": "Ruído Baixo",
        "short_label": "LOW",
        "severity": 1,
        "color": (80, 140, 255, 70),
        "border_color": (120, 170, 255),
        "sigma_scale": 0.7,
        "bias_mean": 0.0,
        "bias_std": 0.0,
        "dropout_prob": 0.0,
        "outlier_prob": 0.0,
    },
    "medium_noise": {
        "label": "Ruído Médio",
        "short_label": "MED",
        "severity": 2,
        "color": (255, 215, 0, 80),
        "border_color": (255, 180, 0),
        "sigma_scale": 1.5,
        "bias_mean": 0.10,
        "bias_std": 0.03,
        "dropout_prob": 0.02,
        "outlier_prob": 0.01,
    },
    "high_noise": {
        "label": "Ruído Alto",
        "short_label": "HIGH",
        "severity": 3,
        "color": (255, 80, 80, 90),
        "border_color": (255, 120, 120),
        "sigma_scale": 2.5,
        "bias_mean": 0.30,
        "bias_std": 0.10,
        "dropout_prob": 0.08,
        "outlier_prob": 0.04,
    },
    "severe_nlos": {
        "label": "NLOS Severo",
        "short_label": "NLOS",
        "severity": 4,
        "color": (120, 0, 0, 110),
        "border_color": (170, 70, 70),
        "sigma_scale": 4.0,
        "bias_mean": 0.80,
        "bias_std": 0.25,
        "dropout_prob": 0.20,
        "outlier_prob": 0.10,
    },
}


def get_noise_profile(profile_name: str | None) -> dict:
    if not profile_name:
        return NOISE_PROFILES["default"]
    return NOISE_PROFILES.get(profile_name, NOISE_PROFILES["default"])


def noise_profile_label(profile_name: str | None) -> str:
    return get_noise_profile(profile_name)["label"]


def noise_profile_short_label(profile_name: str | None) -> str:
    return get_noise_profile(profile_name)["short_label"]


def noise_profile_color(profile_name: str | None):
    return get_noise_profile(profile_name)["color"]


def noise_profile_border_color(profile_name: str | None):
    return get_noise_profile(profile_name)["border_color"]


def noise_profile_severity(profile_name: str | None) -> int:
    return int(get_noise_profile(profile_name)["severity"])


def noise_profile_sigma_scale(profile_name: str | None) -> float:
    return float(get_noise_profile(profile_name)["sigma_scale"])


def noise_profile_bias_mean(profile_name: str | None) -> float:
    return float(get_noise_profile(profile_name)["bias_mean"])


def noise_profile_bias_std(profile_name: str | None) -> float:
    return float(get_noise_profile(profile_name)["bias_std"])


def list_noise_profiles() -> list[str]:
    return [k for k in NOISE_PROFILES.keys() if k != "default"]


def worse_profile_name(a: str | None, b: str | None) -> str:
    pa = get_noise_profile(a)
    pb = get_noise_profile(b)
    return a if int(pa["severity"]) >= int(pb["severity"]) else b