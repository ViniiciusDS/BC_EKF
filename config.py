# config.py

# Mapa
MAP_WIDTH = 5.0  # metros
MAP_HEIGHT = 5.0  # metros

# Robô
WHEEL_RADIUS = 0.05        # m
WHEEL_BASE = 0.2           # m
MAX_LINEAR_VELOCITY = 0.8  # m/s
MAX_LINEAR_ACCEL = 0.2     # m/s^2
MAX_ANGULAR_VELOCITY = 1.0 # rad/s
MAX_ANGULAR_ACCEL = 0.5    # rad/s^2

# Simulação
TIME_STEP = 0.1            # s
SIM_DURATION = 30.0        # s

# Ruído
NOISE_STD_V = 0.0005         # desvio padrão do ruído linear (m/s)
NOISE_STD_W = 0.0002       # desvio padrão do ruído angular (rad/s)

# Parâmetros de salvamento dos dados
CSV_PRECISION = 5  # Quantidade de casas decimais
