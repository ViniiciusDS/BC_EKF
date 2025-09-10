# main_interactive.py
import math
import time
import numpy as np
import pygame as pg

from simulator import Simulator
from scenarios import anchors_tectrol
from trajectory import Trajectory
import config

# ==========================
# Mundo
# ==========================
WORLD_W, WORLD_H = 30.0, 30.0  # m

# ==========================
# Layout
# ==========================
SIDE_W = 320  # coluna de HUD
BASE_SCALE = 20  # px/m (padrão)
MIN_SCALE, MAX_SCALE = 5, 80

# ==========================
# Cores
# ==========================
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRID = (210, 210, 210)
GRID5 = (170, 170, 170)
RED = (220, 60, 60)
BLUE = (55, 120, 220)
ORANGE = (250, 160, 60)
PURPLE = (185, 60, 200)

# ==========================
# Câmera
# ==========================
class Camera:
    def __init__(self, w_m, h_m, scale_px_m=BASE_SCALE, offset_px=(0, 0)):
        self.world_w = w_m
        self.world_h = h_m
        self.scale = scale_px_m
        self.offset = list(offset_px)  # deslocamento em px (pan)
        self.viewport = (int(w_m * scale_px_m), int(h_m * scale_px_m))

    def set_viewport(self, width_px, height_px):
        self.viewport = (width_px, height_px)

    def world_to_screen(self, x, y):
        # origem do mundo no canto inferior esquerdo
        sx = int(x * self.scale + self.offset[0])
        sy = int((self.world_h - y) * self.scale + self.offset[1])
        return sx, sy

    def screen_to_world(self, sx, sy):
        x = (sx - self.offset[0]) / self.scale
        y = self.world_h - (sy - self.offset[1]) / self.scale
        return x, y

    def zoom_at(self, mouse_px, factor):
        """
        Zoom centrado no ponto do mouse.
        factor > 1 aumenta, < 1 diminui.
        """
        mx, my = mouse_px
        before = self.screen_to_world(mx, my)
        # aplica zoom
        self.scale = float(np.clip(self.scale * factor, MIN_SCALE, MAX_SCALE))
        after = self.screen_to_world(mx, my)
        # ajusta pan p/ manter o mesmo ponto sob o cursor
        bx, by = before
        ax, ay = after
        dx, dy = (ax - bx), (ay - by)
        self.offset[0] += dx * self.scale
        self.offset[1] -= dy * self.scale

    def pan(self, dx_px, dy_px):
        self.offset[0] += dx_px
        self.offset[1] += dy_px

    def reset(self):
        self.scale = BASE_SCALE
        self.offset = [0, 0]

# ==========================
# Desenho
# ==========================
def draw_grid(surface, cam: Camera):
    w_px, h_px = cam.viewport
    # passo em m
    step = 1.0
    # vertical
    m = 0.0
    while m <= cam.world_w + 1e-6:
        x1, y1 = cam.world_to_screen(m, 0)
        x2, y2 = cam.world_to_screen(m, cam.world_h)
        col = GRID if (int(round(m)) % 5) else GRID5
        pg.draw.line(surface, col, (x1, y1), (x2, y2), 1)
        m += step
    # horizontal
    m = 0.0
    while m <= cam.world_h + 1e-6:
        x1, y1 = cam.world_to_screen(0, m)
        x2, y2 = cam.world_to_screen(cam.world_w, m)
        col = GRID if (int(round(m)) % 5) else GRID5
        pg.draw.line(surface, col, (x1, y1), (x2, y2), 1)
        m += step

def draw_anchors(surface, cam: Camera, anchors3xN):
    if anchors3xN is None or anchors3xN.shape[1] == 0:
        return
    for ax, ay in zip(anchors3xN[0], anchors3xN[1]):
        sx, sy = cam.world_to_screen(ax, ay)
        pg.draw.circle(surface, RED, (sx, sy), 6)
        pg.draw.circle(surface, BLACK, (sx, sy), 6, 1)

def draw_path(surface, cam: Camera, path_xy, color, width=2, dashed=False):
    if len(path_xy) < 2:
        return
    pts = [cam.world_to_screen(x, y) for x, y in path_xy]
    if dashed:
        for i in range(0, len(pts) - 1, 2):
            pg.draw.line(surface, color, pts[i], pts[i + 1], width)
    else:
        pg.draw.lines(surface, color, False, pts, width)

def draw_robot(surface, cam: Camera, x, y, theta, color=BLACK):
    # triângulo + “nariz”
    L = 0.45  # “comprimento visual” em m
    W = 0.28
    p_front = (x + L * math.cos(theta), y + L * math.sin(theta))
    p_l = (x + W * math.cos(theta + 2.5), y + W * math.sin(theta + 2.5))
    p_r = (x + W * math.cos(theta - 2.5), y + W * math.sin(theta - 2.5))
    pts = [cam.world_to_screen(*p) for p in (p_front, p_l, p_r)]
    pg.draw.polygon(surface, color, pts)
    # nariz
    a = cam.world_to_screen(x, y)
    b = cam.world_to_screen(x + (L + 0.2) * math.cos(theta), y + (L + 0.2) * math.sin(theta))
    pg.draw.line(surface, WHITE, a, b, 2)

def draw_text(surface, txt, x, y, font, color=BLACK):
    img = font.render(txt, True, color)
    surface.blit(img, (x, y))

# ==========================
# Autopilot (segue Trajectory)
# ==========================
def waypoint_controller(current_state, waypoints, idx, v_max=0.25, w_max=0.8, threshold=0.3):
    if idx >= len(waypoints):
        return 0.0, 0.0, idx
    x, y, th = current_state
    tx, ty = waypoints[idx]
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    target_th = math.atan2(dy, dx)
    angle_err = math.atan2(math.sin(target_th - th), math.cos(target_th - th))
    v = v_max * max(0.0, min(1.0, dist))
    w = 1.5 * angle_err
    v = max(-v_max, min(v_max, v))
    w = max(-w_max, min(w_max, w))
    if dist < threshold:
        idx += 1
    return v, w, idx

# ==========================
# Programa principal
# ==========================
def main():
    pg.init()
    # tela inicial baseada no mundo
    map_w_px = int(WORLD_W * BASE_SCALE)
    map_h_px = int(WORLD_H * BASE_SCALE)
    screen = pg.display.set_mode((map_w_px + SIDE_W, map_h_px))
    pg.display.set_caption("BC-EKF — Simulador Interativo 2D (PyGame)")

    font = pg.font.SysFont("arial", 16)
    bigfont = pg.font.SysFont("arial", 20, bold=True)

    # Câmera
    cam = Camera(WORLD_W, WORLD_H, BASE_SCALE, (0, 0))
    cam.set_viewport(map_w_px, map_h_px)

    # Simulador / EKF
    DT = getattr(config, "TIME_STEP", 0.05)
    sim = Simulator(
        anchors=anchors_tectrol.copy(),
        baseline=getattr(config, "WHEEL_BASE", 0.65),
        z_c=getattr(config, "TAG_HEIGHT", 0.5),
        Q=np.diag([1e-4, 1e-4, 1e-4]),
        R=np.eye(2 * anchors_tectrol.shape[1]) * 0.0025,
        dt=DT,
        config=config,
    )

    # Rota planejada
    route_name = "square"  # "square" | "circle" | "figure_eight"
    if route_name == "square":
        traj = Trajectory.square(size=10, start=(0, 0))
    elif route_name == "circle":
        traj = Trajectory.circle(radius=8, points=72, center=(6, 6))
    else:
        traj = Trajectory.figure_eight(radius=4, points=72, center=(6, 6))
    waypoints = np.array(traj.waypoints)

    autopilot = True  # começa ligado para o robô já andar
    wp_idx = 0

    # Trilhas
    path_true, path_pred, path_est = [], [], []
    MAX_TRACE = 3000

    # Controle manual
    v_cmd = 0.0
    w_cmd = 0.0
    V_MAX = 0.35
    W_MAX = 1.2
    accel_lin = 0.02
    accel_ang = 0.08

    # Velocidade da simulação (passos de física por frame)
    speed_factor = 1

    # Âncoras dinâmicas
    anchors_dyn = sim.anchors.copy()

    clock = pg.time.Clock()
    running = True

    # Pan com botão do meio
    panning = False
    pan_last = (0, 0)

    while running:
        dt_frame = clock.tick(60) / 1000.0
        # redimensionamento (caso mude no futuro)
        cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

        # Eventos
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_SPACE:
                    autopilot = not autopilot
                elif event.key == pg.K_LEFTBRACKET:   # [
                    speed_factor = max(1, speed_factor - 1)
                elif event.key == pg.K_RIGHTBRACKET:  # ]
                    speed_factor = min(20, speed_factor + 1)
                elif event.key == pg.K_c:
                    anchors_dyn = np.zeros((3, 0))
                    sim.anchors = anchors_dyn
                    sim.R = np.eye(2 * anchors_dyn.shape[1]) * 0.0025
                elif event.key == pg.K_b:
                    anchors_dyn = anchors_tectrol.copy()
                    sim.anchors = anchors_dyn
                    sim.R = np.eye(2 * anchors_dyn.shape[1]) * 0.0025
                elif event.key == pg.K_r:
                    cam.reset()
            elif event.type == pg.MOUSEBUTTONDOWN:
                mx, my = pg.mouse.get_pos()
                # scroll wheel -> zoom
                if event.button == 4:  # wheel up
                    cam.zoom_at((mx, my), 1.15)
                elif event.button == 5:  # wheel down
                    cam.zoom_at((mx, my), 1/1.15)
                elif mx < cam.viewport[0]:
                    # dentro do mapa
                    if event.button == 1:
                        wx, wy = cam.screen_to_world(mx, my)
                        z = getattr(config, "TAG_HEIGHT", 0.5)
                        anchors_dyn = np.hstack([anchors_dyn, np.array([[wx], [wy], [z]])])
                        sim.anchors = anchors_dyn
                        sim.R = np.eye(2 * anchors_dyn.shape[1]) * 0.0025
                    elif event.button == 3:
                        wx, wy = cam.screen_to_world(mx, my)
                        if anchors_dyn.shape[1] > 0:
                            dif = anchors_dyn[:2, :].T - np.array([wx, wy])[None, :]
                            j = int(np.argmin(np.sum(dif**2, axis=1)))
                            anchors_dyn = np.delete(anchors_dyn, j, axis=1)
                            sim.anchors = anchors_dyn
                            sim.R = np.eye(2 * anchors_dyn.shape[1]) * 0.0025
                    elif event.button == 2:
                        panning = True
                        pan_last = (mx, my)
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 2:
                    panning = False
            elif event.type == pg.MOUSEMOTION and panning:
                mx, my = event.pos
                dx = mx - pan_last[0]
                dy = my - pan_last[1]
                cam.pan(dx, dy)
                pan_last = (mx, my)

        # Teclas de direção (manual)
        keys = pg.key.get_pressed()
        if not autopilot:
            if keys[pg.K_UP]:
                v_cmd = min(V_MAX, v_cmd + accel_lin)
            elif keys[pg.K_DOWN]:
                v_cmd = max(-V_MAX, v_cmd - accel_lin)
            else:
                v_cmd *= 0.90
            if keys[pg.K_LEFT]:
                w_cmd = max(-W_MAX, w_cmd - accel_ang)
            elif keys[pg.K_RIGHT]:
                w_cmd = min(W_MAX, w_cmd + accel_ang)
            else:
                w_cmd *= 0.85
        else:
            v_cmd, w_cmd, wp_idx = waypoint_controller(sim.x_est, waypoints, wp_idx, v_max=0.25, w_max=0.8)

        # Integra N passos por frame
        for _ in range(speed_factor):
            sim.step(v_cmd, w_cmd, noisy=True)

        # Coleta logs
        true_traj, est_traj = sim.get_logs()
        pred_traj = np.array(sim.history_pred)

        if len(true_traj) > 0:
            path_true.append((true_traj[-1, 0], true_traj[-1, 1]))
        if len(pred_traj) > 0:
            path_pred.append((pred_traj[-1, 0], pred_traj[-1, 1]))
        if len(est_traj) > 0:
            path_est.append((est_traj[-1, 0], est_traj[-1, 1]))
        path_true = path_true[-MAX_TRACE:]
        path_pred = path_pred[-MAX_TRACE:]
        path_est = path_est[-MAX_TRACE:]

        # Erros atuais
        pos_err = 0.0
        head_err_deg = 0.0
        if len(true_traj) > 0 and len(est_traj) > 0:
            pos_err = float(np.linalg.norm(true_traj[-1, :2] - est_traj[-1, :2]))
            dth = (true_traj[-1, 2] - est_traj[-1, 2])
            dth = math.atan2(math.sin(dth), math.cos(dth))
            head_err_deg = abs(dth * 180.0 / math.pi)

        # ======= Desenho =======
        screen.fill(WHITE)

        # área do mapa
        map_rect = pg.Rect(0, 0, cam.viewport[0], cam.viewport[1])
        pg.draw.rect(screen, WHITE, map_rect)

        draw_grid(screen, cam)
        draw_anchors(screen, cam, anchors_dyn)
        # rota planejada
        draw_path(screen, cam, waypoints, BLACK, 2, dashed=True)
        # trilhas
        draw_path(screen, cam, path_true, BLACK, 2)
        draw_path(screen, cam, path_pred, BLUE, 2, dashed=True)
        draw_path(screen, cam, path_est, ORANGE, 2, dashed=True)
        # robôs (real + estimado)
        if len(true_traj) > 0:
            xr, yr, tr = true_traj[-1]
            draw_robot(screen, cam, xr, yr, tr, BLACK)
        if len(est_traj) > 0:
            xe, ye, te = est_traj[-1]
            draw_robot(screen, cam, xe, ye, te, ORANGE)

        # HUD lateral
        pg.draw.rect(screen, (245, 245, 245), (cam.viewport[0], 0, SIDE_W, cam.viewport[1]))
        draw_text(screen, "BC-EKF — Simulador Interativo", cam.viewport[0] + 16, 12, bigfont)
        draw_text(screen, f"FPS:  {clock.get_fps():5.1f}", cam.viewport[0] + 16, 44, font)
        draw_text(screen, f"Speed x: {speed_factor}", cam.viewport[0] + 16, 66, font)
        draw_text(screen, f"Autopilot: {'ON' if autopilot else 'OFF'} (SPACE)", cam.viewport[0] + 16, 88, font)
        draw_text(screen, f"Âncoras: {anchors_dyn.shape[1]}  (LMB add, RMB remove, C clear, B base)", cam.viewport[0] + 16, 110, font)
        draw_text(screen, f"Zoom: {cam.scale:.1f} px/m   (R para reset)", cam.viewport[0] + 16, 132, font)
        draw_text(screen, f"Erro Pos (m): {pos_err:.3f}", cam.viewport[0] + 16, 160, font)
        draw_text(screen, f"Erro Heading (°): {head_err_deg:.2f}", cam.viewport[0] + 16, 182, font)

        draw_text(screen, "Controles:", cam.viewport[0] + 16, 220, bigfont)
        draw_text(screen, "↑/↓ acel. linear   ←/→ acel. angular", cam.viewport[0] + 16, 246, font)
        draw_text(screen, "[ / ]  velocidade da simulação", cam.viewport[0] + 16, 266, font)
        draw_text(screen, "SPACE ativa/desativa Autopilot", cam.viewport[0] + 16, 286, font)
        draw_text(screen, "Clique Esq: adiciona âncora", cam.viewport[0] + 16, 306, font)
        draw_text(screen, "Clique Dir: remove âncora", cam.viewport[0] + 16, 326, font)
        draw_text(screen, "Scroll: zoom  |  Botão do meio: pan", cam.viewport[0] + 16, 346, font)
        draw_text(screen, "C: limpar  |  B: âncoras padrão | R: reset view", cam.viewport[0] + 16, 366, font)
        draw_text(screen, "ESC: sair", cam.viewport[0] + 16, 386, font)

        pg.display.flip()

    pg.quit()

if __name__ == "__main__":
    main()
