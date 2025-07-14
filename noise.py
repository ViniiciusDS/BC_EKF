# noise.py
import numpy as np

rng = np.random.default_rng()

def add_gaussian_noise(value, std_dev):
    """
    Retorna 'value' com ruído gaussiano de desvio padrão 'std_dev'.
    """
    return value + rng.normal(0, std_dev)
