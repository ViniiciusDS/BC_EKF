import numpy as np

def run_bc_ekf(
    T,
    t_final,
    anchors,
    v_true=0.3,
    w_true=np.deg2rad(7.5),
    l=0.65/2,
    z_c=0.5,
    sigma_v=0.02,
    sigma_w=0.05,
    sigma_uwb=np.sqrt(0.0025)
):
    t = np.arange(0, t_final + T, T)
    num_anchors = anchors.shape[1]
    f_pred = 1/T
    f_corr = 5
    update_ratio = f_pred / f_corr

    x_true = np.array([2.5, 0, 0])
    x_hist_true = np.zeros((3, len(t)))
    x_hist_true[:,0] = x_true

    for k in range(1, len(t)):
        theta = x_hist_true[2, k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_true * T * np.cos(theta + w_true*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_true * T * np.sin(theta + w_true*T/2)
        x_hist_true[2,k] = x_hist_true[2,k-1] + w_true * T
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k]), np.cos(x_hist_true[2,k]))

    input_noisy = np.vstack([
        v_true + sigma_v * np.random.randn(len(t)),
        w_true + sigma_w * np.random.randn(len(t))
    ])

    z_hist = np.zeros((2*num_anchors, len(t)))
    for k in range(len(t)):
        xt, yt, thetat = x_hist_true[:,k]
        pf = [xt + l*np.cos(thetat), yt + l*np.sin(thetat), z_c]
        pr = [xt - l*np.cos(thetat), yt - l*np.sin(thetat), z_c]
        for i in range(num_anchors):
            dist_f = np.linalg.norm(np.array(pf) - anchors[:,i]) + sigma_uwb*np.random.randn()
            dist_r = np.linalg.norm(np.array(pr) - anchors[:,i]) + sigma_uwb*np.random.randn()
            z_hist[2*i,k] = dist_f
            z_hist[2*i+1,k] = dist_r

    x_est = np.array([2.5,0,0])
    P = np.diag([0.1,0.1,0.1])
    Q = np.diag([0.0001,0.0001,0.0001])
    R = np.diag([sigma_uwb**2]*(2*num_anchors))

    x_hist_est = np.zeros((3, len(t)))
    x_hist_est[:,0] = x_est

    for k in range(1, len(t)):
        v_k = input_noisy[0,k]
        w_k = input_noisy[1,k]
        theta_prev = x_est[2]

        x_pred = x_est + [
            v_k*T*np.cos(theta_prev + w_k*T/2),
            v_k*T*np.sin(theta_prev + w_k*T/2),
            w_k*T
        ]
        x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

        A_k = [
            [1,0,-v_k*T*np.sin(theta_prev + w_k*T/2)],
            [0,1, v_k*T*np.cos(theta_prev + w_k*T/2)],
            [0,0,1]
        ]
        A_k = np.array(A_k)

        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio == 0:
            xp, yp, theta_p = x_pred
            pf_pred = [xp + l*np.cos(theta_p), yp + l*np.sin(theta_p), z_c]
            pr_pred = [xp - l*np.cos(theta_p), yp - l*np.sin(theta_p), z_c]

            h_pred = []
            for i in range(num_anchors):
                h_pred.append(np.linalg.norm(np.array(pf_pred)-anchors[:,i]))
                h_pred.append(np.linalg.norm(np.array(pr_pred)-anchors[:,i]))
            h_pred = np.array(h_pred)

            H_k = []
            for i in range(num_anchors):
                D_f = h_pred[2*i]
                D_r = h_pred[2*i+1]
                C_f = -(pf_pred[0]-anchors[0,i])*l*np.sin(theta_p) + (pf_pred[1]-anchors[1,i])*l*np.cos(theta_p)
                C_r = (pr_pred[0]-anchors[0,i])*l*np.sin(theta_p) - (pr_pred[1]-anchors[1,i])*l*np.cos(theta_p)
                H_k.append([
                    (pf_pred[0]-anchors[0,i])/D_f,
                    (pf_pred[1]-anchors[1,i])/D_f,
                    C_f/D_f
                ])
                H_k.append([
                    (pr_pred[0]-anchors[0,i])/D_r,
                    (pr_pred[1]-anchors[1,i])/D_r,
                    C_r/D_r
                ])
            H_k = np.array(H_k)

            K_k = P_pred @ H_k.T @ np.linalg.inv(H_k @ P_pred @ H_k.T + R)
            z_k = z_hist[:,k]
            x_est = x_pred + K_k @ (z_k - h_pred)
            x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
            P = (np.eye(3) - K_k @ H_k) @ P_pred
        else:
            x_est = x_pred
            P = P_pred

        x_hist_est[:,k] = x_est

    error = x_hist_true - x_hist_est
    error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))
    pos_error = np.linalg.norm(error[0:2,:], axis=0)
    heading_error_deg = np.abs(error[2,:])*(180/np.pi)
    rmse_pos = np.sqrt(np.mean(pos_error**2))
    rmse_heading = np.sqrt(np.mean(heading_error_deg**2))

    return rmse_pos, rmse_heading, t, x_hist_true, x_hist_est

def run_bc_ekf_custom_commands(
    T,
    t_final,
    anchors,
    v_commands,
    w_commands,
    l=0.65/2,
    z_c=0.5,
    sigma_v=0.02,
    sigma_w=0.05,
    sigma_uwb=np.sqrt(0.0025)
):
    t = np.arange(0, t_final, T)
    num_anchors = anchors.shape[1]
    f_pred = 1/T
    f_corr = 5
    update_ratio = f_pred / f_corr

    x_true = np.array([2.5, 0, 0])
    x_hist_true = np.zeros((3, len(t)))
    x_hist_true[:,0] = x_true

    # Trajetória real baseada em comandos variáveis
    for k in range(1, len(t)):
        v_k = v_commands[k]
        w_k = w_commands[k]
        theta = x_hist_true[2, k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_k * T * np.cos(theta + w_k*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_k * T * np.sin(theta + w_k*T/2)
        x_hist_true[2,k] = x_hist_true[2,k-1] + w_k * T
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k]), np.cos(x_hist_true[2,k]))

    # Odometria ruidosa
    input_noisy = np.vstack([
        v_commands + sigma_v * np.random.randn(len(t)),
        w_commands + sigma_w * np.random.randn(len(t))
    ])

    # Medidas UWB
    z_hist = np.zeros((2*num_anchors, len(t)))
    for k in range(len(t)):
        xt, yt, thetat = x_hist_true[:,k]
        pf = [xt + l*np.cos(thetat), yt + l*np.sin(thetat), z_c]
        pr = [xt - l*np.cos(thetat), yt - l*np.sin(thetat), z_c]
        for i in range(num_anchors):
            dist_f = np.linalg.norm(np.array(pf) - anchors[:,i]) + sigma_uwb*np.random.randn()
            dist_r = np.linalg.norm(np.array(pr) - anchors[:,i]) + sigma_uwb*np.random.randn()
            z_hist[2*i,k] = dist_f
            z_hist[2*i+1,k] = dist_r

    # Inicialização do EKF
    x_est = np.array([2.5,0,0])
    P = np.diag([0.1,0.1,0.1])
    Q = np.diag([0.0001,0.0001,0.0001])
    R = np.diag([sigma_uwb**2]*(2*num_anchors))
    x_hist_est = np.zeros((3, len(t)))
    x_hist_est[:,0] = x_est

    for k in range(1, len(t)):
        v_k = input_noisy[0,k]
        w_k = input_noisy[1,k]
        theta_prev = x_est[2]

        x_pred = x_est + [
            v_k*T*np.cos(theta_prev + w_k*T/2),
            v_k*T*np.sin(theta_prev + w_k*T/2),
            w_k*T
        ]
        x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

        A_k = [
            [1,0,-v_k*T*np.sin(theta_prev + w_k*T/2)],
            [0,1, v_k*T*np.cos(theta_prev + w_k*T/2)],
            [0,0,1]
        ]
        A_k = np.array(A_k)
        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio == 0:
            xp, yp, theta_p = x_pred
            pf_pred = [xp + l*np.cos(theta_p), yp + l*np.sin(theta_p), z_c]
            pr_pred = [xp - l*np.cos(theta_p), yp - l*np.sin(theta_p), z_c]

            h_pred = []
            for i in range(num_anchors):
                h_pred.append(np.linalg.norm(np.array(pf_pred)-anchors[:,i]))
                h_pred.append(np.linalg.norm(np.array(pr_pred)-anchors[:,i]))
            h_pred = np.array(h_pred)

            H_k = []
            for i in range(num_anchors):
                D_f = h_pred[2*i]
                D_r = h_pred[2*i+1]
                C_f = -(pf_pred[0]-anchors[0,i])*l*np.sin(theta_p) + (pf_pred[1]-anchors[1,i])*l*np.cos(theta_p)
                C_r = (pr_pred[0]-anchors[0,i])*l*np.sin(theta_p) - (pr_pred[1]-anchors[1,i])*l*np.cos(theta_p)
                H_k.append([
                    (pf_pred[0]-anchors[0,i])/D_f,
                    (pf_pred[1]-anchors[1,i])/D_f,
                    C_f/D_f
                ])
                H_k.append([
                    (pr_pred[0]-anchors[0,i])/D_r,
                    (pr_pred[1]-anchors[1,i])/D_r,
                    C_r/D_r
                ])
            H_k = np.array(H_k)

            K_k = P_pred @ H_k.T @ np.linalg.inv(H_k @ P_pred @ H_k.T + R)
            z_k = z_hist[:,k]
            x_est = x_pred + K_k @ (z_k - h_pred)
            x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
            P = (np.eye(3) - K_k @ H_k) @ P_pred
        else:
            x_est = x_pred
            P = P_pred

        x_hist_est[:,k] = x_est

    error = x_hist_true - x_hist_est
    error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))
    pos_error = np.linalg.norm(error[0:2,:], axis=0)
    heading_error_deg = np.abs(error[2,:])*(180/np.pi)
    rmse_pos = np.sqrt(np.mean(pos_error**2))
    rmse_heading = np.sqrt(np.mean(heading_error_deg**2))

    return rmse_pos, rmse_heading, t, x_hist_true, x_hist_est