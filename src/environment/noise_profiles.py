from __future__ import annotations

NOISE_PROFILES = {
    "default": {
        "label": "Padrão",
        "severity": 0,
        "color": None,
        "sigma_scale": 1.0,
        "bias_mean": 0.0,
        "bias_std": 0.0,
        "dropout_prob": 0.0,
        "outlier_prob": 0.0,
    },
    "low_noise": {
        "label": "Ruído Baixo",
        "severity": 1,
        "color": (80, 140, 255, 70),
        "sigma_scale": 0.7,
        "bias_mean": 0.0,
        "bias_std": 0.0,
        "dropout_prob": 0.0,
        "outlier_prob": 0.0,
    },
    "medium_noise": {
        "label": "Ruído Médio",
        "severity": 2,
        "color": (255, 215, 0, 80),
        "sigma_scale": 1.5,
        "bias_mean": 0.10,
        "bias_std": 0.03,
        "dropout_prob": 0.02,
        "outlier_prob": 0.01,
    },
    "high_noise": {
        "label": "Ruído Alto",
        "severity": 3,
        "color": (255, 80, 80, 90),
        "sigma_scale": 2.5,
        "bias_mean": 0.30,
        "bias_std": 0.10,
        "dropout_prob": 0.08,
        "outlier_prob": 0.04,
    },
    "severe_nlos": {
        "label": "NLOS Severo",
        "severity": 4,
        "color": (120, 0, 0, 110),
        "sigma_scale": 4.0,
        "bias_mean": 0.80,
        "bias_std": 0.25,
        "dropout_prob": 0.20,
        "outlier_prob": 0.10,
    },
}


def get_noise_profile(profile_name: str | None) -> dict:
    '''Retorna o perfil de ruído correspondente ao nome fornecido. Se o nome for None ou inválido, retorna o perfil padrão.'''
    if not profile_name:
        return NOISE_PROFILES["default"]
    return NOISE_PROFILES.get(profile_name, NOISE_PROFILES["default"])


def noise_profile_label(profile_name: str | None) -> str:
    '''Retorna o rótulo do perfil de ruído correspondente ao nome fornecido. Se o nome for None ou inválido, retorna o rótulo do perfil padrão.'''
    return get_noise_profile(profile_name)["label"]


def noise_profile_color(profile_name: str | None):
    '''Retorna a cor do perfil de ruído correspondente ao nome fornecido. Se o nome for None ou inválido, retorna a cor do perfil padrão.'''
    return get_noise_profile(profile_name)["color"]


def noise_profile_severity(profile_name: str | None) -> int:
    '''Retorna a severidade do perfil de ruído correspondente ao nome fornecido. Se o nome for None ou inválido, retorna a severidade do perfil padrão.'''
    return int(get_noise_profile(profile_name)["severity"])


def list_noise_profiles() -> list[str]:
    ''''Retorna uma lista dos nomes dos perfis de ruído disponíveis, excluindo o perfil padrão.'''
    return [k for k in NOISE_PROFILES.keys() if k != "default"]