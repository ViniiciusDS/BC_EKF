import math

# ==========================
# Autopilot simples
# ==========================
def waypoint_controller(state_xyz, waypoints, idx, v_max=0.25, w_max=0.8, threshold=0.35):
    if waypoints is None or len(waypoints) == 0 or idx >= len(waypoints):
        return 0.0, 0.0, idx

    x, y, th = state_xyz
    tx, ty = waypoints[idx]
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    target_th = math.atan2(dy, dx)
    angle_err = math.atan2(math.sin(target_th - th), math.cos(target_th - th))

    # ganho angular um pouco menor + limitação
    kp_ang = 1.2
    w = max(-w_max, min(w_max, kp_ang * angle_err))

    # reduz v quando erro angular é grande (mais aderência na curva)
    ang_scale = max(0.2, math.cos(angle_err))  # [0.2..1]
    v_ref = v_max * max(0.0, min(1.0, dist))
    v = max(-v_max, min(v_max, v_ref * ang_scale))

    if dist < threshold:
        idx += 1

    return v, w, idx