# main_interactive.py
import os
import sys
import math
import time
import platform
import subprocess
import numpy as np
import pygame as pg

from src.simulator import Simulator
from src.trajectory import Trajectory
from src.scenarios import anchors_tectrol
import src.config as config

# ==========================
# UI helpers
# ==========================
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRID = (210, 210, 210)
GRID5 = (170, 170, 170)
RED = (220, 60, 60)
BLUE = (55, 120, 220)
ORANGE = (250, 160, 60)
PURPLE = (185, 60, 200)
GRAY = (230, 230, 230)
LBL = (40, 40, 40)
GREEN = (50, 170, 80)

SIDE_W = 340

BASE_SCALE = 22    # px/m
MIN_SCALE = 5
MAX_SCALE = 100

# ==========================
# Camera (origem no centro)
# ==========================
class Camera:
    def __init__(self, scale=BASE_SCALE, pan=(0, 0)):
        self.scale = scale
        self.pan = list(pan)  # pixels
        self.viewport = (1200, 700)

    def set_viewport(self, w, h):
        self.viewport = (w, h)

    @property
    def cx(self):
        return self.viewport[0] // 2

    @property
    def cy(self):
        return self.viewport[1] // 2

    def world_to_screen(self, x, y):
        sx = int(self.cx + x * self.scale + self.pan[0])
        sy = int(self.cy - y * self.scale + self.pan[1])
        return sx, sy

    def screen_to_world(self, sx, sy):
        x = (sx - self.cx - self.pan[0]) / self.scale
        y = -(sy - self.cy - self.pan[1]) / self.scale
        return x, y

    def zoom_at(self, mouse_px, factor):
        mx, my = mouse_px
        bx, by = self.screen_to_world(mx, my)
        self.scale = float(np.clip(self.scale * factor, MIN_SCALE, MAX_SCALE))
        ax, ay = self.screen_to_world(mx, my)
        dx, dy = (ax - bx), (ay - by)
        # ajustar pan para manter o ponto sob o cursor
        self.pan[0] += dx * self.scale
        self.pan[1] -= dy * self.scale

    def pan_pixels(self, dx, dy):
        self.pan[0] += dx
        self.pan[1] += dy

    def reset_view(self):
        self.scale = BASE_SCALE
        self.pan = [0, 0]

# ==========================
# Desenho mapa infinito
# ==========================
def draw_grid(surface, cam: Camera):
    w, h = cam.viewport
    # calcula os limites de mundo visíveis
    x0, y0 = cam.screen_to_world(0, h)
    x1, y1 = cam.screen_to_world(w, 0)
    # linhas verticais a cada 1m
    x_start = math.floor(min(x0, x1))
    x_end = math.ceil(max(x0, x1))
    for xm in range(x_start, x_end + 1):
        col = GRID if (xm % 5) else GRID5
        sx0, sy0 = cam.world_to_screen(xm, y0)
        sx1, sy1 = cam.world_to_screen(xm, y1)
        pg.draw.line(surface, col, (sx0, sy0), (sx1, sy1), 1)
    # horizontais
    y_start = math.floor(min(y0, y1))
    y_end = math.ceil(max(y0, y1))
    for ym in range(y_start, y_end + 1):
        col = GRID if (ym % 5) else GRID5
        sx0, sy0 = cam.world_to_screen(x0, ym)
        sx1, sy1 = cam.world_to_screen(x1, ym)
        pg.draw.line(surface, col, (sx0, sy0), (sx1, sy1), 1)

def draw_anchors(surface, cam: Camera, anchors3xN):
    if anchors3xN is None or anchors3xN.shape[1] == 0:
        return
    for ax, ay in zip(anchors3xN[0], anchors3xN[1]):
        sx, sy = cam.world_to_screen(ax, ay)
        pg.draw.circle(surface, RED, (sx, sy), 6)
        pg.draw.circle(surface, BLACK, (sx, sy), 6, 1)

def draw_path(surface, cam: Camera, path_xy, color, width=2, dashed=False):
    """
    Desenha uma polyline no plano do mundo.
    - Se dashed=False, usa pg.draw.lines direto.
    - Se dashed=True, desenha traço-pausa ao longo de CADA segmento consecutivo,
      garantindo que rotas fechadas (ex.: quadrado) apareçam corretamente.
    """
    if path_xy is None or len(path_xy) < 2:
        return

    pts = [cam.world_to_screen(x, y) for x, y in path_xy]

    if not dashed:
        pg.draw.lines(surface, color, False, pts, width)
        return

    # --- Desenho pontilhado contínuo ao longo de cada segmento (p[i] -> p[i+1]) ---
    # parâmetros do tracejado em pixels
    dash_len = 12
    gap_len = 6
    period = dash_len + gap_len

    for i in range(len(pts) - 1):
        (x0, y0) = pts[i]
        (x1, y1) = pts[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue

        # vetor unitário na direção do segmento
        ux = dx / seg_len
        uy = dy / seg_len

        # avança ao longo do segmento alternando traço e espaço
        dist = 0.0
        while dist < seg_len:
            # início do traço
            sx = x0 + ux * dist
            sy = y0 + uy * dist
            # fim do traço (clamp no fim do segmento)
            dist_end = min(dist + dash_len, seg_len)
            ex = x0 + ux * dist_end
            ey = y0 + uy * dist_end
            # desenha o "dash"
            pg.draw.line(surface, color, (int(sx), int(sy)), (int(ex), int(ey)), width)
            # pula o "gap"
            dist = dist_end + gap_len

def draw_robot(surface, cam: Camera, x, y, theta, color=BLACK):
    L = 0.5
    W = 0.32
    p_front = (x + L * math.cos(theta), y + L * math.sin(theta))
    p_l = (x + W * math.cos(theta + 2.5), y + W * math.sin(theta + 2.5))
    p_r = (x + W * math.cos(theta - 2.5), y + W * math.sin(theta - 2.5))
    pts = [cam.world_to_screen(*p) for p in (p_front, p_l, p_r)]
    pg.draw.polygon(surface, color, pts)
    # nariz
    a = cam.world_to_screen(x, y)
    b = cam.world_to_screen(x + (L + 0.25) * math.cos(theta), y + (L + 0.25) * math.sin(theta))
    pg.draw.line(surface, WHITE, a, b, 2)

def draw_text(surface, txt, x, y, font, color=LBL):
    img = font.render(txt, True, color)
    surface.blit(img, (x, y))

class Button:
    def __init__(self, rect, text, font, bg=(245,245,245), fg=LBL, border=BLACK):
        self.rect = pg.Rect(rect)
        self.text = text
        self.font = font
        self.bg = bg
        self.fg = fg
        self.border = border

    def draw(self, surface):
        pg.draw.rect(surface, self.bg, self.rect, border_radius=6)
        pg.draw.rect(surface, self.border, self.rect, 1, border_radius=6)
        img = self.font.render(self.text, True, self.fg)
        surface.blit(img, (self.rect.x + (self.rect.w - img.get_width())//2,
                           self.rect.y + (self.rect.h - img.get_height())//2))

    def hit(self, pos):
        return self.rect.collidepoint(pos)

# ==========================
# Util
# ==========================
def open_folder(path):
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception as e:
        print("Não foi possível abrir a pasta:", e)

# ==========================
# Autopilot simples
# ==========================
def waypoint_controller(state_xyz, waypoints, idx, v_max=0.25, w_max=0.8, threshold=0.35):
    if idx >= len(waypoints):
        return 0.0, 0.0, idx
    x, y, th = state_xyz
    tx, ty = waypoints[idx]
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    target_th = math.atan2(dy, dx)
    angle_err = math.atan2(math.sin(target_th - th), math.cos(target_th - th))
    v = v_max * max(0.0, min(1.0, dist))
    w = 1.6 * angle_err
    v = max(-v_max, min(v_max, v))
    w = max(-w_max, min(w_max, w))
    if dist < threshold:
        idx += 1
    return v, w, idx

# ==========================
# Estados
# ==========================
STATE_MENU = 0
STATE_SIM = 1

def main():
    pg.init()
    screen = pg.display.set_mode((1280, 760))
    pg.display.set_caption("BC-EKF — Simulador Interativo 2D (PyGame)")
    clock = pg.time.Clock()
    font = pg.font.SysFont("arial", 18)
    bigfont = pg.font.SysFont("arial", 22, bold=True)

    # -------- MENU --------
    state = STATE_MENU

    # presets
    anchor_presets = ["Tectrol", "Nenhuma"]
    route_presets = ["Quadrado", "Círculo", "Oito", "Nenhuma"]
    sel_anchor = 0
    sel_route = 0

    btn_start = Button((440, 520, 180, 44), "Iniciar simulação", bigfont, bg=(240,250,240), fg=GREEN, border=GREEN)
    btn_logs  = Button((720, 520, 180, 44), "Abrir pasta de logs", bigfont)
    btn_anchor = Button((440, 420, 220, 40), f"Âncoras: {anchor_presets[sel_anchor]}", font)
    btn_route  = Button((720, 420, 220, 40), f"Rota: {route_presets[sel_route]}", font)

    # -------- SIM --------
    cam = Camera()
    cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

    DT = getattr(config, "TIME_STEP", 0.05)

    def make_anchors():
        if anchor_presets[sel_anchor] == "Tectrol":
            return anchors_tectrol.copy()
        else:
            return np.zeros((3, 0))

    def make_route():
        if route_presets[sel_route] == "Quadrado":
            return np.array(Trajectory.square(size=10, start=(-5,-5)).waypoints)
        elif route_presets[sel_route] == "Círculo":
            return np.array(Trajectory.circle(radius=7, points=90, center=(0,0)).waypoints)
        elif route_presets[sel_route] == "Oito":
            return np.array(Trajectory.figure_eight(radius=5, points=120, center=(0,0)).waypoints)
        else:
            return np.zeros((0,2))

    # placeholders
    sim = None
    anchors_dyn = None
    waypoints = None
    autopilot = True
    wp_idx = 0
    path_true, path_pred, path_est = [], [], []
    speed_factor = 1
    show_debug = False

    # pan/zoom
    panning = False
    pan_last = (0, 0)

    # controle manual
    v_cmd = 0.0
    w_cmd = 0.0
    V_MAX = 0.35
    W_MAX = 1.2
    accel_lin = 0.02
    accel_ang = 0.08

    # gravação de rota
    recording = False
    recorded_points = []

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if state == STATE_MENU:
            # eventos
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if btn_start.hit(event.pos):
                        # cria sim conforme seleções
                        anchors_dyn = make_anchors()
                        sim = Simulator(
                            anchors=anchors_dyn.copy(),
                            baseline=getattr(config, "WHEEL_BASE", 0.65),
                            z_c=getattr(config, "TAG_HEIGHT", 0.5),
                            Q=np.diag([1e-4, 1e-4, 1e-4]),
                            R=np.eye(max(1, 2 * anchors_dyn.shape[1])) * 0.0025 if anchors_dyn.shape[1] > 0 else np.eye(2) * 1e6,
                            dt=DT,
                            config=config,
                        )
                        waypoints = make_route()
                        autopilot = len(waypoints) > 0
                        wp_idx = 0
                        path_true, path_pred, path_est = [], [], []
                        recording = False
                        recorded_points = []
                        cam.reset_view()
                        state = STATE_SIM
                    elif btn_logs.hit(event.pos):
                        open_folder(getattr(config, "LOG_DIR", "logs"))
                    elif btn_anchor.hit(event.pos):
                        sel_anchor = (sel_anchor + 1) % len(anchor_presets)
                        btn_anchor.text = f"Âncoras: {anchor_presets[sel_anchor]}"
                    elif btn_route.hit(event.pos):
                        sel_route = (sel_route + 1) % len(route_presets)
                        btn_route.text = f"Rota: {route_presets[sel_route]}"

                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    running = False

            # desenhar menu
            screen.fill(WHITE)
            title = pg.font.SysFont("arial", 34, bold=True).render("BC-EKF — Simulador 2D (Menu)", True, BLACK)
            screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 120))
            draw_text(screen, "Selecione presets e inicie a simulação:", 440, 240, bigfont)
            btn_anchor.draw(screen)
            btn_route.draw(screen)
            btn_start.draw(screen)
            btn_logs.draw(screen)
            draw_text(screen, "ESC para sair", 20, screen.get_height()-30, font)
            pg.display.flip()
            continue

        # ======= ESTADO SIM =======

        cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

        # eventos sim
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    state = STATE_MENU
                elif event.key == pg.K_d:
                    show_debug = not show_debug
                elif event.key == pg.K_SPACE:
                    autopilot = not autopilot
                elif event.key == pg.K_LEFTBRACKET:
                    speed_factor = max(1, speed_factor - 1)
                elif event.key == pg.K_RIGHTBRACKET:
                    speed_factor = min(20, speed_factor + 1)
                elif event.key == pg.K_c:
                    anchors_dyn = np.zeros((3, 0))
                    sim.anchors = anchors_dyn
                    sim.R = np.eye(2) * 1e6
                elif event.key == pg.K_b:
                    anchors_dyn = anchors_tectrol.copy()
                    sim.anchors = anchors_dyn
                    sim.R = np.eye(2 * anchors_dyn.shape[1]) * 0.0025
                elif event.key == pg.K_r:
                    cam.reset_view()
                elif event.key == pg.K_g:
                    recording = not recording
                    if not recording and len(recorded_points) > 1:
                        waypoints = np.array(recorded_points, dtype=float)
                        wp_idx = 0
                        autopilot = True
                elif event.key == pg.K_RETURN:
                    if recording and len(recorded_points) > 1:
                        waypoints = np.array(recorded_points, dtype=float)
                        wp_idx = 0
                        autopilot = True
                        recording = False
                elif event.key == pg.K_DELETE:
                    recorded_points = []

            elif event.type == pg.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # zoom
                if event.button == 4:  # up
                    cam.zoom_at((mx, my), 1.15)
                elif event.button == 5:  # down
                    cam.zoom_at((mx, my), 1/1.15)
                # dentro do mapa?
                elif mx < cam.viewport[0]:
                    if event.button == 1:  # LMB
                        wx, wy = cam.screen_to_world(mx, my)
                        if recording:
                            recorded_points.append((wx, wy))
                        else:
                            # adicionar âncora
                            z = getattr(config, "TAG_HEIGHT", 0.5)
                            anchors_dyn = np.hstack([anchors_dyn, np.array([[wx], [wy], [z]])])
                            sim.anchors = anchors_dyn
                            sim.R = np.eye(2 * anchors_dyn.shape[1]) * 0.0025
                    elif event.button == 3:  # RMB
                        if recording and recorded_points:
                            recorded_points.pop()
                        else:
                            wx, wy = cam.screen_to_world(mx, my)
                            if anchors_dyn.shape[1] > 0:
                                dif = anchors_dyn[:2, :].T - np.array([wx, wy])[None, :]
                                j = int(np.argmin(np.sum(dif**2, axis=1)))
                                anchors_dyn = np.delete(anchors_dyn, j, axis=1)
                                sim.anchors = anchors_dyn
                                sim.R = (np.eye(2 * anchors_dyn.shape[1]) * 0.0025) if anchors_dyn.shape[1] > 0 else np.eye(2) * 1e6
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
                cam.pan_pixels(dx, dy)
                pan_last = (mx, my)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_d:
                    show_debug = not show_debug


        # teclas contínuas (manual)
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
                w_cmd *= 0.86
        else:
            v_cmd, w_cmd, wp_idx = waypoint_controller(sim.x_est, waypoints, wp_idx, v_max=0.25, w_max=0.9)

        # física
        for _ in range(speed_factor):
            sim.step(v_cmd, w_cmd, noisy=True)

        innov_norm = None
        nis = None
        if getattr(sim, 'last_debug', None) and sim.last_debug['innov'] is not None:
            y = sim.last_debug['innov']
            S = sim.last_debug['S']
            try:
                nis = float(y.T @ np.linalg.inv(S) @ y)   # NIS: consistência da medição
            except:
                nis = None
            innov_norm = float(np.linalg.norm(y))
            
        # logs
        true_traj, est_traj = sim.get_logs()
        pred_traj = np.array(sim.history_pred)

        if len(true_traj) > 0:
            p = (true_traj[-1, 0], true_traj[-1, 1])
            if not path_true or p != path_true[-1]:
                path_true.append(p)
        if len(pred_traj) > 0:
            path_pred.append((pred_traj[-1, 0], pred_traj[-1, 1]))
        if len(est_traj) > 0:
            path_est.append((est_traj[-1, 0], est_traj[-1, 1]))
        path_true = path_true[-2500:]
        path_pred = path_pred[-2500:]
        path_est = path_est[-2500:]

        # erros atuais
        pos_err = 0.0
        head_err = 0.0
        if len(true_traj) > 0 and len(est_traj) > 0:
            pos_err = float(np.linalg.norm(true_traj[-1, :2] - est_traj[-1, :2]))
            dth = (true_traj[-1, 2] - est_traj[-1, 2])
            dth = math.atan2(math.sin(dth), math.cos(dth))
            head_err = abs(dth * 180.0 / math.pi)

        # desenho
        screen.fill(WHITE)
        # mapa (esquerda)
        map_rect = pg.Rect(0, 0, cam.viewport[0], cam.viewport[1])
        pg.draw.rect(screen, WHITE, map_rect)
        draw_grid(screen, cam)
        draw_anchors(screen, cam, anchors_dyn)

        # rota planejada
        if waypoints is not None and len(waypoints) > 1:
            draw_path(screen, cam, waypoints, BLACK, 2, dashed=True)
        # rota gravada em curso (pontos)
        if recording and len(recorded_points) > 0:
            for pt in recorded_points:
                sx, sy = cam.world_to_screen(*pt)
                pg.draw.circle(screen, PURPLE, (sx, sy), 4)

        # trilhas
        draw_path(screen, cam, path_true, BLACK, 2)
        draw_path(screen, cam, path_pred, BLUE, 2, dashed=True)
        draw_path(screen, cam, path_est, ORANGE, 2, dashed=True)

        # robôs
        if len(true_traj) > 0:
            xr, yr, tr = true_traj[-1]
            draw_robot(screen, cam, xr, yr, tr, BLACK)
        if len(est_traj) > 0:
            xe, ye, te = est_traj[-1]
            draw_robot(screen, cam, xe, ye, te, ORANGE)

        # HUD lateral
        pg.draw.rect(screen, (245, 245, 245), (cam.viewport[0], 0, SIDE_W, cam.viewport[1]))
        draw_text(screen, "BC-EKF — Simulador", cam.viewport[0] + 16, 18, bigfont)
        draw_text(screen, f"FPS: {clock.get_fps():5.1f}", cam.viewport[0] + 16, 52, font)
        draw_text(screen, f"Speed x: {speed_factor}", cam.viewport[0] + 16, 74, font)
        draw_text(screen, f"Autopilot: {'ON' if autopilot else 'OFF'} (SPACE)", cam.viewport[0] + 16, 96, font)
        draw_text(screen, f"Âncoras: {anchors_dyn.shape[1]}  (LMB add, RMB rem)", cam.viewport[0] + 16, 118, font)
        draw_text(screen, f"Zoom: {cam.scale:.1f} px/m  (R reset view)", cam.viewport[0] + 16, 140, font)
        draw_text(screen, f"Erro Pos (m): {pos_err:.3f}", cam.viewport[0] + 16, 168, font)
        draw_text(screen, f"Erro Heading (°): {head_err:.2f}", cam.viewport[0] + 16, 190, font)
        draw_text(screen, f"Gravar rota (G): {'ON' if recording else 'OFF'}", cam.viewport[0] + 16, 212, font)
        draw_text(screen, "ENTER finaliza rota  |  DEL limpa", cam.viewport[0] + 16, 232, font)

        draw_text(screen, "Controles:", cam.viewport[0] + 16, 270, bigfont)
        draw_text(screen, "↑/↓ acel. linear   ←/→ acel. angular", cam.viewport[0] + 16, 296, font)
        draw_text(screen, "[ / ] velocidade simulação", cam.viewport[0] + 16, 316, font)
        draw_text(screen, "Scroll: zoom  |  Botão do meio: pan", cam.viewport[0] + 16, 336, font)
        draw_text(screen, "C: limpar âncoras  |  B: âncoras padrão", cam.viewport[0] + 16, 356, font)
        draw_text(screen, "ESC: voltar ao menu", cam.viewport[0] + 16, 376, font)
        if show_debug:
            draw_text(screen, "DEBUG EKF", cam.viewport[0]+16, 410, bigfont)
            draw_text(screen, f"||innov||: {innov_norm:.3f}" if innov_norm is not None else "||innov||: n/a",
                    cam.viewport[0]+16, 436, font)
            draw_text(screen, f"NIS: {nis:.3f}" if nis is not None else "NIS: n/a",
                    cam.viewport[0]+16, 456, font)
            draw_text(screen, "D: mostra/oculta debug", cam.viewport[0]+16, 476, font)

        pg.display.flip()

    pg.quit()

if __name__ == "__main__":
    main()
