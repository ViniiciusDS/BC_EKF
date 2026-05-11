    # config.py

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

# --- Mapa ---
# Observação: na interface PyGame atual usamos grid infinito com pan/zoom.
# MAP_WIDTH/HEIGHT ficam para um modo "mapa fixo" ou exportações offline.
MAP_WIDTH = 5.0   # m (não usado no PyGame infinito)
MAP_HEIGHT = 5.0  # m (não usado no PyGame infinito)
# Diretório e nome padrão do mapa
MAPS_DIR = "maps"
DEFAULT_MAP_NAME = "default_map.json"

# --- Robô (cinemática) ---
WHEEL_RADIUS = 0.035          # m
WHEEL_BASE = 0.15            # m  (distância entre rodas)
TAG_BASELINE = 0.25          # m  (distância entre tags = 2*l)
TAG_HEIGHT = 0.20            # m  (altura das tags em relação ao chão)


# --- Encoder real ---
ENCODER_TICKS_PER_REV = 1075.0

REAL_ENCODER_USE_DISTANCE_COLUMNS = True
REAL_ENCODER_DISTANCE_UNIT_SCALE = 0.01  # ESP32 calcula em cm, converte para m

REAL_ENCODER_SWAP_LR = True
REAL_ENCODER_INVERT_LEFT = False
REAL_ENCODER_INVERT_RIGHT = False

REAL_ODOM_INITIAL_X = 1.652
REAL_ODOM_INITIAL_Y = 1.977
REAL_ODOM_INITIAL_THETA_DEG = 00.0

# --- UWB (geometria das tags) ---
UWB_BASELINE = 0.25          # m  (distância entre tags = 2*l)

# Limites de velocidade/aceleração do robô
MAX_LINEAR_VELOCITY  = 0.8   # m/s
MAX_LINEAR_ACCEL     = 0.2   # m/s^2
MAX_ANGULAR_VELOCITY = 1.0   # rad/s
MAX_ANGULAR_ACCEL    = 0.5   # rad/s^2

# --- Simulação temporal ---
TIME_STEP   = 0.05   # s (alinhado ao EKF e simuladores)
SIM_DURATION = 30.0  # s (para cenários offline)

# --- Ruído de odometria (realista p/ debug) ---
NOISE_STD_V = 0.02   # m/s
NOISE_STD_W = 0.05   # rad/s

# --- UWB: ruído e efeitos sistemáticos ---
# (Desvios e eventos estocásticos aplicados em utils.apply_uwb_errors)
UWB_NOISE_STD = 0.05              # m (sigma do ruído branco)
UWB_BIAS_ENABLED = True
UWB_BIAS_VALUE = 0.20             # m (ex.: 20 cm)
UWB_BIAS_PROBABILITY = 0.10       # 10% das leituras terão viés
UWB_MISALIGNMENT_ENABLED = True
UWB_MISALIGNMENT_PROBABILITY = 0.10  # prob. de “desalinhamento”/outlier
UWB_MISALIGNMENT_FACTOR = 3.0        # multiplica o ruído quando ocorre

# --- Saída numérica ---
CSV_PRECISION = 5

# --- Logging ---
LOGGING_ENABLED   = False
LOG_DIR           = "resultados/logs"
LOG_FLUSH_EVERY_N = 50

