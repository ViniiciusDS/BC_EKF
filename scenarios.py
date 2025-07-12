import numpy as np

# ==========================
# Grupo 1: Layout e quantidade de âncoras
# ==========================
scenarios_group1 = [
    {
        "label": "3 Âncoras Distribuídas",
        "anchors": np.array([
            [0,0,1],
            [0,5,1],
            [5,0,1]
        ]).T,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.0025)
    },
    {
        "label": "3 Âncoras Agrupadas",
        "anchors": np.array([
            [0,0,1],
            [0,1,1],
            [1,0,1]
        ]).T,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.0025)
    },
    {
        "label": "4 Âncoras",
        "anchors": np.array([
            [0,0,1],
            [0,5,1],
            [5,0,1],
            [5,5,1]
        ]).T,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.0025)
    }
]

# ==========================
# Grupo 2: Baseline e Ruído
# ==========================
anchors_default = np.array([
    [0,0,1],
    [0,5,1],
    [5,0,1]
]).T

scenarios_group2 = [
    {
        "label": "Baseline 0.65 - Ruído Baixo",
        "anchors": anchors_default,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.0025)
    },
    {
        "label": "Baseline 0.85 - Ruído Baixo",
        "anchors": anchors_default,
        "baseline": 0.85/2,
        "sigma_uwb": np.sqrt(0.0025)
    },
    {
        "label": "Baseline 0.65 - Ruído Alto",
        "anchors": anchors_default,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.01)
    }
]

# ==========================
# Grupo 3: Diferentes trajetórias
# ==========================
scenarios_group3 = [
    {
        "label": "Trajetória Reta",
        "anchors": anchors_default,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.0025),
        "v_true": 0.3,
        "w_true": 0.0
    },
    {
        "label": "Trajetória Curva Lenta",
        "anchors": anchors_default,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.0025),
        "v_true": 0.3,
        "w_true": np.deg2rad(5)
    },
    {
        "label": "Trajetória Curva Rápida",
        "anchors": anchors_default,
        "baseline": 0.65/2,
        "sigma_uwb": np.sqrt(0.0025),
        "v_true": 0.3,
        "w_true": np.deg2rad(20)
    }
]

# ==========================
# Grupo 4: Todos combinados
# ==========================
scenarios_all = scenarios_group1 + scenarios_group2 + scenarios_group3
