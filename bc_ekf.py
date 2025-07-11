import numpy as np

def run_bc_ekf(
    T,
    t_final,
    anchors,
    v_true,
    w_true,
    l,
    z_c,
    sigma_v,
    sigma_w,
    sigma_uwb
):
    """
    Executa uma simulação única do BC-EKF.
    Retorna:
        rmse_pos: float
        rmse_heading: float
        t: array
        x_hist_true: array (3,N)
        x_hist_est: array (3,N)
    """
    num_anchors = anchors.shape[1]
    update_ratio = int((1/T)/5)

    t = np.arange(0, t_final+T, T)

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    x_hist_true[:,0] = [2.5,0,0]
    for k in range(1,len(t)):
        theta = x_hist_true[2,k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_true*T*np.cos(theta + w_true*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_true*T*np.sin(theta + w_true*T/2)
        x_hist_true[2,k] = x_hist_true[2,k-1] + w_true*T
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k]), np.cos(x_hist_true[2,k]))

    # Odometria ruidosa
    v_noisy = v_true + sigma_v*np.random.randn(len(t))
    w_noisy = w_true + sigma_w*np.random.randn(len(t))

    # Medições UWB
    z_hist = np.zeros((2*num_anchors, len(t)))
    for k in range(len(t)):
        theta = x_hist_true[2,k]
        xt,yt = x_hist_true[0,k], x_hist_true[1,k]
        pf = np.array([xt + l*np.cos(theta), yt + l*np.sin(theta), z_c])
        pr = np.array([xt - l*np.cos(theta), yt - l*np.sin(theta), z_c])
        for i in range(num_anchors):
            dist_f = np.linalg.norm(pf - anchors[:,i]) + sigma_uwb*np.random.randn()
            dist_r = np.linalg.norm(pr - anchors[:,i]) + sigma_uwb*np.random.randn()
            z_hist[2*i,k]   = dist_f
            z_hist[2*i+1,k] = dist_r

    # Estado inicial e covariâncias
    x_est = np.array([2.5,0,0])
    P = np.diag([0.1,0.1,0.1])
    Q = np.diag([0.0001,0.0001,0.0001])
    R = np.diag([sigma_uwb**2]*(2*num_anchors))

    x_hist_est = np.zeros((3,len(t)))
    x_hist_est[:,0] = x_est

    # Loop principal EKF
    for k in range(1,len(t)):
        v_k = v_noisy[k]
        w_k = w_noisy[k]
        theta_prev = x_est[2]

        x_pred = x_est + np.array([
            v_k*T*np.cos(theta_prev + w_k*T/2),
            v_k*T*np.sin(theta_prev + w_k*T/2),
            w_k*T
        ])
        x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

        A_k = np.array([
            [1,0,-v_k*T*np.sin(theta_prev + w_k*T/2)],
            [0,1, v_k*T*np.cos(theta_prev + w_k*T/2)],
            [0,0,1]
        ])
        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio ==0:
            xp,yp,theta_p = x_pred
            pf_pred = np.array([xp + l*np.cos(theta_p), yp + l*np.sin(theta_p), z_c])
            pr_pred = np.array([xp - l*np.cos(theta_p), yp - l*np.sin(theta_p), z_c])

            h_pred = np.zeros(2*num_anchors)
            H_k = np.zeros((2*num_anchors,3))
            for i in range(num_anchors):
                D_fi = np.linalg.norm(pf_pred - anchors[:,i])
                D_ri = np.linalg.norm(pr_pred - anchors[:,i])

                C_fi = -(pf_pred[0]-anchors[0,i])*l*np.sin(theta_p) + (pf_pred[1]-anchors[1,i])*l*np.cos(theta_p)
                C_ri = (pr_pred[0]-anchors[0,i])*l*np.sin(theta_p) - (pr_pred[1]-anchors[1,i])*l*np.cos(theta_p)

                h_pred[2*i]   = D_fi
                h_pred[2*i+1] = D_ri

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

    # Cálculo dos erros
    error = x_hist_true - x_hist_est
    error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))
    pos_error = np.linalg.norm(error[0:2,:], axis=0)
    heading_error_deg = np.abs(error[2,:])*(180/np.pi)

    rmse_pos = np.sqrt(np.mean(pos_error**2))
    rmse_heading = np.sqrt(np.mean(heading_error_deg**2))

    return rmse_pos, rmse_heading, t, x_hist_true, x_hist_est
