import numpy as np

anchors_tectrol = np.array([
    [6.351, -4.966, 1.0],
    [5.947, -4.966, 1.0],
    [17.173, 10.477, 1.0],
    [14.876, 15.244, 1.0],
    [9.530, 17.474, 1.0],
    [5.115, 14.920, 1.0],
    [2.492, 10.007, 1.0],
    [11.865, 24.855, 1.0],
    [15.298, -1.511, 1.0]
]).T

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
# Grupo 4: Tectrol
# ==========================
scenarios_group4 = [
    {
        "label": "Tectrol - Curva Grande - Ruído Alto",
        "anchors": anchors_tectrol,
        "baseline": 0.65 / 2,
        "sigma_uwb": np.sqrt(0.05),
        "v_true": 0.3,
        "w_true": np.deg2rad(1.5)   # curva bem ampla
    },
    {
        "label": "Tectrol - Curva Moderada - Ruído Alto",
        "anchors": anchors_tectrol,
        "baseline": 0.65 / 2,
        "sigma_uwb": np.sqrt(0.05),
        "v_true": 0.3,
        "w_true": np.deg2rad(3)
    }
]

scenarios_rectangular = [
    {
        "label": "Tectrol - Trajetória Retangular",
        "anchors": anchors_tectrol,
        "baseline": 0.5/2,
        "sigma_uwb": np.sqrt(5)  # ruído alto, ambiente real
    }
]

# ==========================
# Grupo 5: Todos combinados
# ==========================
scenarios_all = scenarios_group1 + scenarios_group2 + scenarios_group3
