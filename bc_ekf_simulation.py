import numpy as np
import matplotlib.pyplot as plt
from utils import simulate_run

def generate_ground_truth(t, v_true, w_true):
    """
    Gera a trajetória real do robô diferencial.
    """
    x_hist = np.zeros((3, len(t)))
    x_hist[:,0] = [2.5, 0, 0]  # estado inicial
    for k in range(1, len(t)):
        theta_prev = x_hist[2, k-1]
        x_hist[0, k] = x_hist[0, k-1] + v_true * T * np.cos(theta_prev + w_true*T/2)
        x_hist[1, k] = x_hist[1, k-1] + v_true * T * np.sin(theta_prev + w_true*T/2)
        x_hist[2, k] = x_hist[2, k-1] + w_true*T
        x_hist[2, k] = np.arctan2(np.sin(x_hist[2,k]), np.cos(x_hist[2,k]))
    return x_hist

def generate_noisy_odometry(v_true, w_true, t, sigma_v, sigma_w):
    v_noisy = v_true + sigma_v * np.random.randn(len(t))
    w_noisy = w_true + sigma_w * np.random.randn(len(t))
    return v_noisy, w_noisy

def generate_uwb_measurements(x_hist, anchors, l, z_c, sigma_uwb):
    num_anchors = anchors.shape[1]
    z_hist = np.zeros((2*num_anchors, x_hist.shape[1]))
    for k in range(x_hist.shape[1]):
        theta = x_hist[2,k]
        xt = x_hist[0,k]
        yt = x_hist[1,k]
        pf = np.array([xt + l*np.cos(theta), yt + l*np.sin(theta), z_c])
        pr = np.array([xt - l*np.cos(theta), yt - l*np.sin(theta), z_c])
        for i in range(num_anchors):
            dist_f = np.linalg.norm(pf - anchors[:,i]) + sigma_uwb * np.random.randn()
            dist_r = np.linalg.norm(pr - anchors[:,i]) + sigma_uwb * np.random.randn()
            z_hist[2*i, k] = dist_f
            z_hist[2*i+1, k] = dist_r
    return z_hist

# ================================
# 1. Parâmetros de simulação
# ================================
T = 0.05
t_final = 50
anchors = np.array([
    [0, 0, 1],
    [0, 5, 1],
    [5, 0, 1]
]).T
num_anchors = anchors.shape[1]

l = 0.65 / 2
z_c = 0.5

v_true = 0.3
w_true = np.deg2rad(7.5)

sigma_v = 0.02
sigma_w = 0.05
sigma_uwb = np.sqrt(0.0025)

# ================================
# 2. Chamar simulador
# ================================
t, x_hist_true, v_noisy, w_noisy, z_hist = simulate_run(
    T, t_final, anchors, v_true, w_true, l, z_c, sigma_v, sigma_w, sigma_uwb
)

update_ratio = int((1/T) / 5)  # Correção UWB a cada 0.2s

print("Simulação concluída, dados prontos.")

# ========================
# 3. Ruído dos sensores
# ========================
sigma_v = 0.02
sigma_w = 0.05
sigma_uwb = np.sqrt(0.0025)

# Odometria ruidosa
v_noisy = v_true + sigma_v * np.random.randn(len(t))
w_noisy = w_true + sigma_w * np.random.randn(len(t))

# Medições UWB
z_hist = np.zeros((2*num_anchors, len(t)))

for k in range(len(t)):
    theta = x_hist_true[2,k]
    xt, yt = x_hist_true[0,k], x_hist_true[1,k]
    pf = np.array([xt + l*np.cos(theta), yt + l*np.sin(theta), z_c])
    pr = np.array([xt - l*np.cos(theta), yt - l*np.sin(theta), z_c])

    for i in range(num_anchors):
        dist_f = np.linalg.norm(pf - anchors[:,i]) + sigma_uwb*np.random.randn()
        dist_r = np.linalg.norm(pr - anchors[:,i]) + sigma_uwb*np.random.randn()
        z_hist[2*i,k] = dist_f
        z_hist[2*i+1,k] = dist_r

print("Medições simuladas geradas.")

# ==========================
# 4. Estado estimado e covariâncias
# ==========================
x_est = np.array([2.5, 0, 0])  # mesmo que o verdadeiro
P = np.diag([0.1, 0.1, 0.1])

# Ruído do processo (Q)
Q = np.diag([0.0001, 0.0001, 0.0001])

# Ruído da medição (R)
R = np.diag([sigma_uwb**2] * (2*num_anchors))

# Histórico de estimativas
x_hist_est = np.zeros((3, len(t)))
x_hist_est[:,0] = x_est

print("Iniciando BC-EKF...")

for k in range(1, len(t)):

    # ===================
    # Predição
    # ===================
    v_k = v_noisy[k]
    w_k = w_noisy[k]
    theta_prev = x_est[2]

    # Predição do estado
    x_pred = x_est + np.array([
        v_k*T*np.cos(theta_prev + w_k*T/2),
        v_k*T*np.sin(theta_prev + w_k*T/2),
        w_k*T
    ])
    x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

    # Jacobiano A_k
    A_k = np.array([
        [1, 0, -v_k*T*np.sin(theta_prev + w_k*T/2)],
        [0, 1,  v_k*T*np.cos(theta_prev + w_k*T/2)],
        [0, 0, 1]
    ])

    # Covariância predita
    P_pred = A_k @ P @ A_k.T + Q

    # ===================
    # Correção se for passo UWB
    # ===================
    if k % update_ratio == 0:

        # Posições preditas das tags
        xp, yp, theta_p = x_pred
        pf_pred = np.array([xp + l*np.cos(theta_p), yp + l*np.sin(theta_p), z_c])
        pr_pred = np.array([xp - l*np.cos(theta_p), yp - l*np.sin(theta_p), z_c])

        # Vetor de medição esperada
        h_pred = np.zeros(2*num_anchors)
        for i in range(num_anchors):
            h_pred[2*i]   = np.linalg.norm(pf_pred - anchors[:,i])
            h_pred[2*i+1] = np.linalg.norm(pr_pred - anchors[:,i])

        # Jacobiano H_k
        H_k = np.zeros((2*num_anchors, 3))
        for i in range(num_anchors):
            D_fi = h_pred[2*i]
            D_ri = h_pred[2*i+1]

            C_fi = -(pf_pred[0]-anchors[0,i])*l*np.sin(theta_p) + (pf_pred[1]-anchors[1,i])*l*np.cos(theta_p)
            C_ri = (pr_pred[0]-anchors[0,i])*l*np.sin(theta_p) - (pr_pred[1]-anchors[1,i])*l*np.cos(theta_p)

            H_k[2*i,:] = [
                (pf_pred[0]-anchors[0,i])/D_fi,
                (pf_pred[1]-anchors[1,i])/D_fi,
                C_fi/D_fi
            ]

            H_k[2*i+1,:] = [
                (pr_pred[0]-anchors[0,i])/D_ri,
                (pr_pred[1]-anchors[1,i])/D_ri,
                C_ri/D_ri
            ]

        # Ganho de Kalman
        S = H_k @ P_pred @ H_k.T + R
        K_k = P_pred @ H_k.T @ np.linalg.inv(S)

        # Medição real
        z_k = z_hist[:,k]

        # Correção do estado
        y_k = z_k - h_pred
        x_est = x_pred + K_k @ y_k
        x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))

        # Correção da covariância
        P = (np.eye(3) - K_k @ H_k) @ P_pred

    else:
        # Sem correção, só prediz
        x_est = x_pred
        P = P_pred

    # Armazena
    x_hist_est[:,k] = x_est

print("BC-EKF concluído.")

# ===========================
# 5. Cálculo dos erros
# ===========================
error = x_hist_true - x_hist_est
# Normaliza ângulo
error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))

# Erro de posição Euclidiano
pos_error = np.linalg.norm(error[0:2,:], axis=0)

# Erro de heading em graus
heading_error_deg = np.abs(error[2,:]) * (180/np.pi)

# RMSE
rmse_pos = np.sqrt(np.mean(pos_error**2))
rmse_heading = np.sqrt(np.mean(heading_error_deg**2))

# ===========================
# 6. Plot das trajetórias
# ===========================
plt.figure(figsize=(8,8))
plt.plot(x_hist_true[0,:], x_hist_true[1,:], 'k-', linewidth=2, label='Trajetória Real')
plt.plot(x_hist_est[0,:], x_hist_est[1,:], 'b--', linewidth=1.5, label='Trajetória Estimada (BC-EKF)')
plt.scatter(anchors[0,:], anchors[1,:], c='r', marker='*', s=100, label='Âncoras UWB')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajetória do Robô")
plt.legend()
plt.axis('equal')
plt.grid()
plt.tight_layout()

# ===========================
# 7. Plot dos erros
# ===========================
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, pos_error, label=f"RMSE: {rmse_pos:.4f} m")
plt.ylabel("Erro de posição (m)")
plt.title("Erro Euclidiano de posição ao longo do tempo")
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, heading_error_deg, label=f"RMSE: {rmse_heading:.4f}°")
plt.ylabel("Erro de heading (graus)")
plt.xlabel("Tempo (s)")
plt.title("Erro de orientação ao longo do tempo")
plt.grid()
plt.legend()

plt.tight_layout()

print(f"RMSE posição: {rmse_pos:.4f} m")
print(f"RMSE heading: {rmse_heading:.4f} graus")

plt.show()