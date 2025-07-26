# bc_ekf_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_ground_truth, generate_noisy_odometry, generate_uwb_measurements

# ================================
# 1. Parâmetros de simulação
# ================================
T = 0.05
t_final = 50
t = np.arange(0, t_final + T, T)

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

update_ratio = int((1/T) / 5)  # Correção UWB a cada 0.2s

# ================================
# 2. Geração de dados simulados
# ================================
x_hist_true = generate_ground_truth(t, v_true, w_true)
v_noisy, w_noisy = generate_noisy_odometry(v_true, w_true, t, sigma_v, sigma_w)
z_hist = generate_uwb_measurements(x_hist_true, anchors, l, z_c, sigma_uwb)

print("Simulação concluída. Iniciando EKF...")

# ================================
# 3. Inicialização do EKF
# ================================
x_est = np.array([2.5, 0, 0])
P = np.diag([0.1, 0.1, 0.1])
Q = np.diag([1e-4]*3)
R = np.diag([sigma_uwb**2]*(2*num_anchors))

x_hist_est = np.zeros((3, len(t)))
x_hist_est[:,0] = x_est

# ================================
# 4. Loop EKF
# ================================
for k in range(1, len(t)):
    v_k = v_noisy[k]
    w_k = w_noisy[k]
    theta_prev = x_est[2]

    # Predição
    dx = v_k*T*np.cos(theta_prev + w_k*T/2)
    dy = v_k*T*np.sin(theta_prev + w_k*T/2)
    dtheta = w_k*T
    x_pred = x_est + np.array([dx, dy, dtheta])
    x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

    A_k = np.array([
        [1,0,-v_k*T*np.sin(theta_prev + w_k*T/2)],
        [0,1, v_k*T*np.cos(theta_prev + w_k*T/2)],
        [0,0,1]
    ])

    P_pred = A_k @ P @ A_k.T + Q

    if k % update_ratio == 0:
        xp, yp, theta_p = x_pred
        pf = np.array([xp + l*np.cos(theta_p), yp + l*np.sin(theta_p), z_c])
        pr = np.array([xp - l*np.cos(theta_p), yp - l*np.sin(theta_p), z_c])

        h_pred = np.zeros(2*num_anchors)
        H_k = np.zeros((2*num_anchors,3))

        for i in range(num_anchors):
            D_fi = np.linalg.norm(pf - anchors[:,i])
            D_ri = np.linalg.norm(pr - anchors[:,i])

            C_fi = -(pf[0]-anchors[0,i])*l*np.sin(theta_p) + (pf[1]-anchors[1,i])*l*np.cos(theta_p)
            C_ri = (pr[0]-anchors[0,i])*l*np.sin(theta_p) - (pr[1]-anchors[1,i])*l*np.cos(theta_p)

            h_pred[2*i] = D_fi
            h_pred[2*i+1] = D_ri

            H_k[2*i,:] = [
                (pf[0]-anchors[0,i])/D_fi,
                (pf[1]-anchors[1,i])/D_fi,
                C_fi/D_fi
            ]
            H_k[2*i+1,:] = [
                (pr[0]-anchors[0,i])/D_ri,
                (pr[1]-anchors[1,i])/D_ri,
                C_ri/D_ri
            ]

        S = H_k @ P_pred @ H_k.T + R
        K_k = P_pred @ H_k.T @ np.linalg.inv(S)
        z_k = z_hist[:,k]
        y_k = z_k - h_pred
        x_est = x_pred + K_k @ y_k
        x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
        P = (np.eye(3) - K_k @ H_k) @ P_pred
    else:
        x_est = x_pred
        P = P_pred

    x_hist_est[:,k] = x_est

print("BC-EKF concluído.")

# ================================
# 5. Cálculo dos erros
# ================================
error = x_hist_true - x_hist_est
error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))
pos_error = np.linalg.norm(error[0:2,:], axis=0)
heading_error_deg = np.abs(error[2,:]) * (180/np.pi)

rmse_pos = np.sqrt(np.mean(pos_error**2))
rmse_heading = np.sqrt(np.mean(heading_error_deg**2))

# ================================
# 6. Gráficos
# ================================
plt.figure(figsize=(8,8))
plt.plot(x_hist_true[0,:], x_hist_true[1,:], 'k-', linewidth=2, label='Trajetória Real')
plt.plot(x_hist_est[0,:], x_hist_est[1,:], 'b--', linewidth=1.5, label='Trajetória Estimada')
plt.scatter(anchors[0,:], anchors[1,:], c='r', marker='*', s=100, label='Âncoras')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.axis('equal')
plt.grid()
plt.title("Trajetória do Robô")

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, pos_error)
plt.ylabel("Erro de posição (m)")
plt.title(f"Erro Euclidiano (RMSE: {rmse_pos:.4f} m)")
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, heading_error_deg)
plt.ylabel("Erro de heading (graus)")
plt.xlabel("Tempo (s)")
plt.title(f"Erro de Orientação (RMSE: {rmse_heading:.4f}°)")
plt.grid()

plt.tight_layout()
plt.show()
