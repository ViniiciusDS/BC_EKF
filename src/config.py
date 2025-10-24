    # config.py

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

# --- Mapa ---
# Observação: na interface PyGame atual usamos grid infinito com pan/zoom.
# MAP_WIDTH/HEIGHT ficam para um modo "mapa fixo" ou exportações offline.
MAP_WIDTH = 5.0   # m (não usado no PyGame infinito)
MAP_HEIGHT = 5.0  # m (não usado no PyGame infinito)

# --- Robô (cinemática) ---
WHEEL_RADIUS = 0.05          # m
WHEEL_BASE = 0.20            # m  (distância entre rodas)

# --- UWB (geometria das tags) ---
UWB_BASELINE = 0.65          # m  (distância entre tags = 2*l)

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
