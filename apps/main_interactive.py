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
from src.environment import Environment, Obstacle, draw_environment
from src.utils import start_plot_process, stop_plot_process, push_plot_data, point_segment_distance, list_map_files, map_file_path
from src.ui_elements import TextBoxDropdown

# ==========================
# Environment setup
# ==========================
def make_environment():
    env = Environment()
    env.add(Obstacle(np.array([-6.0, -6.0]), np.array([6.0, -6.0]), material="metal"))
    env.add(Obstacle(np.array([-6.0, 6.0]),  np.array([6.0, 6.0]),  material="metal"))
    return env

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

MAPS_DIR = getattr(config, "MAPS_DIR", "maps")
DEFAULT_MAP_NAME = getattr(config, "DEFAULT_MAP_NAME", "default_map.json")

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

# ==========================
# Estados
# ==========================
STATE_MENU = 0  # menu inicial
STATE_SIM = 1   # simulação rodando
STATE_MAPEDITOR = 2     # editor de mapa

# =========================
# Main loop
# =========================
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
    btn_mapedit = Button((440, 580, 220, 44), "Editor de mapa", bigfont)
    
    # holders p/ botões do editor de mapas
    btn_save_as = None
    btn_load_by_name = None

    # -------- SIM --------
    # --- botões no HUD (estado SIM) ---
    btn_filelog = None
    btn_graphs  = None

    # estado do file logging
    filelog_on = False
    file_logger = None

    # buffers p/ gráficos em tempo real
    ts_hist, pos_err_hist, head_err_hist = [], [], []
    # estado do processo de plot
    plot_state = {"plot_proc": None, "plot_q": None}

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
    anchors_dyn = make_anchors()
    env = make_environment()
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

    # editor de mapas
    editor_first_pt = None      # ponto inicial da parede (mundo)
    editor_preview_pt = None    # ponto “do mouse” para pré-visualização
    editor_material = "metal"   # material padrão
    
    # mensagens temporárias (popups) no editor de mapas
    editor_msg = ""
    editor_msg_t = 0.0 # tempo restante em segundos

    # Campo de texto do nome do mapa
    editor_filename = "map_name"
    name_box = None       

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
                            env=env
                        )
                        waypoints = make_route()
                        autopilot = len(waypoints) > 0
                        wp_idx = 0
                        path_true, path_pred, path_est = [], [], []
                        recording = False
                        recorded_points = []
                        ts_hist.clear(); pos_err_hist.clear(); head_err_hist.clear()
                        cam.reset_view()
                        state = STATE_SIM
                        sidebar_x = cam.viewport[0]
                        # posição vertical de base para botões HUD
                        hud_y0 = 260  # ajuste fino conforme botões adicionados
                        btn_filelog = Button((sidebar_x + 16, hud_y0 + 160, 180, 32), "Log arquivo: OFF", font, bg=(240,240,255))
                        btn_graphs  = Button((sidebar_x + 210, hud_y0 + 160, 180, 32), "Abrir gráficos", font, bg=(240,255,240))
                        filelog_on = False
                        file_logger = None
                
                    elif btn_logs.hit(event.pos):
                        # abre pasta de logs
                        open_folder(getattr(config, "LOG_DIR", "logs"))
                    elif btn_anchor.hit(event.pos):
                        # muda preset de âncoras
                        sel_anchor = (sel_anchor + 1) % len(anchor_presets)
                        btn_anchor.text = f"Âncoras: {anchor_presets[sel_anchor]}"
                    elif btn_route.hit(event.pos):
                        # muda preset de rota
                        sel_route = (sel_route + 1) % len(route_presets)
                        btn_route.text = f"Rota: {route_presets[sel_route]}"
                    elif btn_mapedit.hit(event.pos):
                        # muda para o editor de mapas
                        state = STATE_MAPEDITOR
                        cam.reset_view()

                        env = Environment() # novo ambiente vazio
                        
                        editor_first_pt = None
                        editor_preview_pt = None

                        # botões salvar/carregar
                        sidebar_x = screen.get_width() - SIDE_W + 16
                        y0 = 260
                        btn_save_as = Button((sidebar_x, y0, 160, 32), "Salvar como", font, bg=(240,255,240))
                        btn_load_by_name = Button((sidebar_x, y0+40, 160, 32), "Carregar (nome)", font, bg=(240,240,255))

                        # ==== cria TextBoxDropdown uma única vez ====
                        map_choices = list_map_files(MAPS_DIR)
                        editor_filename = DEFAULT_MAP_NAME if DEFAULT_MAP_NAME else "map_name"

                        # posição inicial aproximada do textbox; 
                        name_box = TextBoxDropdown(
                            rect=(sidebar_x, 200, 220, 28),
                            font=font,
                            options=map_choices
                        )
                        name_box.text = editor_filename
                        name_box.cursor_pos = len(editor_filename)

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
            btn_mapedit.draw(screen)
            draw_text(screen, "ESC para sair", 20, screen.get_height()-30, font)
            pg.display.flip()
            continue
        
        # ======= ESTADO MAPEDITOR =======
        if state == STATE_MAPEDITOR:
            cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

            if editor_msg_t > 0:
                editor_msg_t -= dt
                if editor_msg_t <= 0:
                    editor_msg_t = 0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        # Limpa qualquer preview e volta pro menu
                        editor_first_pt = None
                        editor_preview_pt = None
                        state = STATE_MENU
                    
                    # Edição de Material
                    elif event.key == pg.K_m:
                        # Cicla entre materiais pré-definidos
                        material_list = ["metal", "wall", "glass", "human"]
                        try:
                            idx = material_list.index(editor_material)
                        except ValueError:
                            idx = 0
                        editor_material = material_list[(idx + 1) % len(material_list)]
                        print("Material atual do editor:", editor_material)
                    
                    # Teclado para o TextBoxDropdown
                    if name_box and name_box.handle_event(event):
                        editor_filename = name_box.text
                        continue

                # Eventos do mouse
                elif event.type == pg.MOUSEBUTTONDOWN:
                    mx, my = event.pos

                    # zoom (scroll)
                    if event.button == 4:
                        cam.zoom_at((mx, my), 1.15)
                        continue
                    elif event.button == 5:
                        cam.zoom_at((mx, my), 1/1.15)
                        continue

                    # pan com botão do meio
                    if event.button == 2:
                        panning = True
                        pan_last = (mx, my)
                        continue

                    # ==== clique na área do mapa ====
                    if mx < cam.viewport[0]:
                        wx, wy = cam.screen_to_world(mx, my)

                        # LMB: desenhar paredes (2 cliques = 1 parede)
                        if event.button == 1:
                            if editor_first_pt is None:
                                # começa uma nova parede
                                editor_first_pt = (wx, wy)
                                editor_preview_pt = (wx, wy)
                            else:
                                # finaliza parede: cria obstáculo
                                p0 = np.array(editor_first_pt, dtype=float)
                                p1 = np.array([wx, wy], dtype=float)
                                # evita segmento zerado
                                if np.linalg.norm(p1 - p0) > 1e-3:
                                    env.add(Obstacle(p0, p1, material=editor_material))
                                editor_first_pt = None
                                editor_preview_pt = None

                        # RMB: remover parede mais próxima
                        elif event.button == 3 and env is not None and env.obstacles:
                            p_click = np.array([wx, wy], dtype=float)
                            min_dist = float("inf")
                            min_idx = None
                            for idx, obs in enumerate(env.obstacles):
                                d = point_segment_distance(p_click, obs.p0, obs.p1)
                                if d < min_dist:
                                    min_dist = d
                                    min_idx = idx
                            # limiar em metros para “pegar” uma parede
                            if min_idx is not None and min_dist < 0.4:
                                env.obstacles.pop(min_idx)

                    # ============================
                    # HUD LATERAL (direita)
                    # ============================
                    else:
                        # 1) textbox/dropdown
                        if name_box and name_box.handle_event(event):
                            editor_filename = name_box.text
                            continue

                        # 2) botões
                        if btn_save_as and btn_save_as.hit(event.pos):
                            path = map_file_path(MAPS_DIR, editor_filename)
                            env.save_json(path)
                            editor_msg = f"Mapa salvo como: {os.path.basename(path)}"
                            editor_msg_t = 2.0

                            # atualiza lista de mapas no dropdown
                            name_box.options_all = list_map_files(MAPS_DIR)
                            name_box.update_filter()
                            continue
                        
                        if btn_load_by_name and btn_load_by_name.hit(event.pos):
                            path = map_file_path(MAPS_DIR, editor_filename)
                            if os.path.exists(path):
                                env = Environment.load_json(path)
                                editor_msg = "Mapa carregado!"
                            else:
                                editor_msg = "Arquivo não encontrado!"
                            editor_msg_t = 2.0
                            continue

                elif event.type == pg.MOUSEBUTTONUP:
                    if event.button == 2:
                        panning = False

                elif event.type == pg.MOUSEMOTION:
                    mx, my = event.pos
                    if panning:
                        dx = mx - pan_last[0]
                        dy = my - pan_last[1]
                        cam.pan_pixels(dx, dy)
                        pan_last = (mx, my)
                    # atualiza preview da parede se já temos o primeiro ponto
                    if editor_first_pt is not None and mx < cam.viewport[0]:
                        editor_preview_pt = cam.screen_to_world(mx, my)

            # --- desenho do editor ---
            screen.fill(WHITE)
            cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

            # área do mapa
            map_rect = pg.Rect(0, 0, cam.viewport[0], cam.viewport[1])
            pg.draw.rect(screen, WHITE, map_rect)
            draw_grid(screen, cam)
            draw_environment(screen, cam, env)  # desenha paredes existentes

            # preview da parede (linha “fantasma”)
            if editor_first_pt is not None and editor_preview_pt is not None:
                p0s = cam.world_to_screen(*editor_first_pt)
                p1s = cam.world_to_screen(*editor_preview_pt)
                pg.draw.line(screen, (0, 160, 0), p0s, p1s, 2)
                pg.draw.circle(screen, (0, 200, 0), p0s, 4)
                pg.draw.circle(screen, (0, 200, 0), p1s, 4)

            # HUD lateral do editor
            pg.draw.rect(screen, (245, 245, 245), (cam.viewport[0], 0, SIDE_W, cam.viewport[1]))
            sidebar_x = cam.viewport[0] + 16
            y = 18
            LINE_H = 22

            draw_text(screen, "Editor de mapa (beta)", sidebar_x, y, bigfont); y += 34
            draw_text(screen, "ESC: voltar ao menu", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "Scroll: zoom  |  Botão do meio: pan", sidebar_x, y, font); y += LINE_H
            y += 8
            draw_text(screen, "Desenho de paredes:", sidebar_x, y, bigfont); y += LINE_H
            draw_text(screen, "LMB: 1º clique = inicio parede", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "LMB: 2º clique = fim parede", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "RMB: remover parede mais próxima", sidebar_x, y, font); y += LINE_H

            y += 8
            draw_text(screen, f"Material atual: {editor_material}", sidebar_x, y, font); y += LINE_H
            draw_text(screen, "M: trocar material", sidebar_x, y, font); y += LINE_H

            draw_text(screen, "Nome do mapa:", sidebar_x, y, font); y += 25

            if name_box:
                name_box.rect.topleft = (sidebar_x, y)
                name_box.update(dt)
                name_box.draw(screen)
                
                y = name_box.rect.bottom + 10

                if name_box.dropdown_open and name_box.options_filtered:
                    num = min(name_box.max_visible, len(name_box.options_filtered))
                    y += num * name_box.line_h + 10
            else:
                # fallback
                y += 60

            # botões salvar/carregar
            if btn_save_as:
                btn_save_as.rect.topleft = (sidebar_x, y)
                btn_save_as.draw(screen)
                y += 40
            if btn_load_by_name:
                btn_load_by_name.rect.topleft = (sidebar_x, y)
                btn_load_by_name.draw(screen)
                y += 40

            if editor_first_pt is not None:
                y += 10
                draw_text(screen, "Status: definindo fim da parede...", sidebar_x, y, font); y += LINE_H
            else:
                y += 10
                draw_text(screen, "Status: pronto para novo segmento", sidebar_x, y, font); y += LINE_H
                
            # popup flutuante de feedback (salvo / carregado / erro)
            if editor_msg_t > 0 and editor_msg:
                msg_img = font.render(editor_msg, True, BLACK)
                pad = 10
                box_w = msg_img.get_width() + 2 * pad
                box_h = msg_img.get_height() + 2 * pad

                # centraliza o popup na coluna lateral
                box_x = cam.viewport[0] + (SIDE_W - box_w) // 2
                # um pouco acima da borda inferior
                box_y = cam.viewport[1] - box_h - 20

                # fundo amarelo claro
                pg.draw.rect(screen, (255, 255, 210), (box_x, box_y, box_w, box_h), border_radius=6)
                pg.draw.rect(screen, BLACK, (box_x, box_y, box_w, box_h), 1, border_radius=6)
                screen.blit(msg_img, (box_x + pad, box_y + pad))
            
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
                    # quando voltar para o MENU (em KEYDOWN K_ESCAPE que troca o state) OU ao encerrar o programa:
                    if file_logger:
                        try: file_logger.close()
                        except: pass
                        file_logger = None
                        filelog_on = False
                    stop_plot_process(plot_state)
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

                # zoom sempre
                if event.button == 4:
                    cam.zoom_at((mx, my), 1.15)
                    continue
                elif event.button == 5:
                    cam.zoom_at((mx, my), 1/1.15)
                    continue

                # pan (botão do meio) – independe de mapa ou HUD
                if event.button == 2:
                    panning = True
                    pan_last = (mx, my)
                    continue

                # -----------------------------------------
                # CLIQUE ESQUERDO (mapa OU HUD)
                # -----------------------------------------
                if event.button == 1:
                    if mx < cam.viewport[0]:
                        # >>> MAPA (LMB)
                        wx, wy = cam.screen_to_world(mx, my)
                        if recording:
                            recorded_points.append((wx, wy))
                        else:
                            z = getattr(config, "TAG_HEIGHT", 0.5)
                            anchors_dyn = np.hstack([anchors_dyn, np.array([[wx], [wy], [z]])])
                            sim.anchors = anchors_dyn
                            sim.R = np.eye(2 * anchors_dyn.shape[1]) * 0.0025
                    else:
                        # >>> HUD (LMB)
                        if btn_filelog and btn_filelog.hit((mx, my)):
                            if not filelog_on:
                                from src.utils import RunLogger
                                meta = {
                                    "dt": DT,
                                    "anchors": anchors_dyn[:2,:].T.tolist() if anchors_dyn is not None else [],
                                    "z_c": float(getattr(config, "TAG_HEIGHT", 0.5)),
                                    "baseline": float(getattr(config, "WHEEL_BASE", 0.65)),
                                    "route_waypoints": waypoints.tolist() if waypoints is not None else [],
                                    "config": {
                                        "TIME_STEP": getattr(config, "TIME_STEP", None),
                                        "NOISE_STD_V": getattr(config, "NOISE_STD_V", None),
                                        "NOISE_STD_W": getattr(config, "NOISE_STD_W", None),
                                        "UWB_BIAS_ENABLED": getattr(config, "UWB_BIAS_ENABLED", None),
                                        "UWB_MISALIGNMENT_ENABLED": getattr(config, "UWB_MISALIGNMENT_ENABLED", None),
                                    }
                                }
                                file_logger = RunLogger(
                                    out_dir=getattr(config, "LOG_DIR", "resultados/logs"),
                                    run_name=None,
                                    meta=meta,
                                    flush_every_n=getattr(config, "LOG_FLUSH_EVERY_N", 50),
                                )
                                filelog_on = True
                                btn_filelog.text = "Log arquivo: ON"
                            else:
                                try:
                                    if file_logger: file_logger.close()
                                finally:
                                    file_logger = None
                                    filelog_on = False
                                    btn_filelog.text = "Log arquivo: OFF"

                        elif btn_graphs and btn_graphs.hit((mx, my)):
                            # inicia/reativa a janela de gráficos (processo separado)
                            start_plot_process(plot_state)

                # -----------------------------------------
                # CLIQUE DIREITO (mapa)
                # -----------------------------------------
                elif event.button == 3 and mx < cam.viewport[0]:
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

            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 2:
                    panning = False
            elif event.type == pg.MOUSEMOTION and panning:
                mx, my = event.pos
                dx = mx - pan_last[0]
                dy = my - pan_last[1]
                cam.pan_pixels(dx, dy)
                pan_last = (mx, my)

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
            v_cmd, w_cmd, wp_idx = waypoint_controller(sim.x_est, waypoints, wp_idx, v_max=0.25, w_max=0.8)

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

        # atualiza séries e envia ao processo de plot
        if len(true_traj) > 0 and len(est_traj) > 0:
            t_now = (len(ts_hist)) * sim.dt
            ts_hist.append(t_now); pos_err_hist.append(pos_err); head_err_hist.append(head_err)
            push_plot_data(plot_state, ts_hist, pos_err_hist, head_err_hist)

        # logging em arquivo (se ativo)
        if filelog_on and file_logger and len(true_traj) > 0 and len(est_traj) > 0:
            true_state = true_traj[-1, :]
            est_state  = est_traj[-1, :]
            pred_state = sim.last_debug.get('x_pred', None) if getattr(sim, 'last_debug', None) else None
            v_meas, w_meas = getattr(sim, 'last_meas', (float('nan'), float('nan')))
            try:
                file_logger.log_step(
                    true_state=true_state,
                    pred_state=pred_state,   # pode ser None; o RunLogger já trata
                    est_state=est_state,
                    v_cmd=v_cmd, w_cmd=w_cmd,
                    v_meas=v_meas, w_meas=w_meas,
                    pos_err=pos_err, heading_err_deg=head_err
                )
            except Exception as e:
                # evita quebrar a simulação por erro de I/O
                pass

        # desenho
        screen.fill(WHITE)
        # mapa (esquerda)
        map_rect = pg.Rect(0, 0, cam.viewport[0], cam.viewport[1])
        pg.draw.rect(screen, WHITE, map_rect)
        draw_grid(screen, cam)
        draw_environment(screen, cam, sim.env)
        draw_anchors(screen, cam, anchors_dyn)

        # rota planejada
        if waypoints is not None and len(waypoints) > 1:
            draw_path(screen, cam, waypoints, BLACK, 2, dashed=True)
            # destaque do waypoint atual
            if autopilot and 0 <= wp_idx < len(waypoints):
                sx, sy = cam.world_to_screen(*waypoints[wp_idx])
                pg.draw.circle(screen, GREEN, (sx, sy), 6)
                pg.draw.circle(screen, BLACK, (sx, sy), 6, 1)
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

        # --- Overlay: indicador de waypoint no canto inferior-direito do MAPA ---
        if waypoints is not None and len(waypoints) > 0:
            wp_current = min(wp_idx + 1, len(waypoints))
            label = f"WP: {wp_current}/{len(waypoints)}"
            img = font.render(label, True, BLACK)

            pad = 6
            tx = cam.viewport[0] - img.get_width() - pad - 8
            ty = cam.viewport[1] - img.get_height() - pad - 8

            card = pg.Surface((img.get_width() + 2*pad, img.get_height() + 2*pad), pg.SRCALPHA)
            card.fill((255, 255, 255, 210))  # branco com alpha
            card.blit(img, (pad, pad))
            screen.blit(card, (tx - pad, ty - pad))
            pg.draw.rect(screen, BLACK, (tx - pad, ty - pad, card.get_width(), card.get_height()), 1)

        # HUD lateral
        pg.draw.rect(screen, (245, 245, 245), (cam.viewport[0], 0, SIDE_W, cam.viewport[1]))
        sidebar_x = cam.viewport[0] + 16
        LINE_H = 22
        y = 18
        # título
        draw_text(screen, "BC-EKF — Simulador", sidebar_x, y, bigfont); y += 34

        # métricas gerais
        draw_text(screen, f"FPS: {clock.get_fps():5.1f}", sidebar_x, y, font); y += LINE_H
        draw_text(screen, f"Speed x: {speed_factor}", sidebar_x, y, font); y += LINE_H
        draw_text(screen, f"Autopilot: {'ON' if autopilot else 'OFF'} (SPACE)", sidebar_x, y, font); y += LINE_H
        draw_text(screen, f"Âncoras: {anchors_dyn.shape[1]}  (LMB add, RMB rem)", sidebar_x, y, font); y += LINE_H
        draw_text(screen, f"Zoom: {cam.scale:.1f} px/m  (R reset view)", sidebar_x, y, font); y += LINE_H
        draw_text(screen, f"Erro Pos (m): {pos_err:.3f}", sidebar_x, y, font); y += LINE_H
        draw_text(screen, f"Erro Heading (°): {head_err:.2f}", sidebar_x, y, font); y += LINE_H
        draw_text(screen, f"Gravar rota (G): {'ON' if recording else 'OFF'}", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "ENTER finaliza rota  |  DEL limpa", sidebar_x, y, font); y += LINE_H

        # separador
        y += 10
        draw_text(screen, "Controles:", sidebar_x, y, bigfont); y += LINE_H
        draw_text(screen, "↑/↓ acel. linear   ←/→ acel. angular", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "[ / ] velocidade simulação", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "Scroll: zoom  |  Botão do meio: pan", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "C: limpar âncoras  |  B: âncoras padrão", sidebar_x, y, font); y += LINE_H
        draw_text(screen, "ESC: voltar ao menu", sidebar_x, y, font); y += LINE_H

        # --- MÉTRICAS DE CANAL / NLOS ---
        if sim.last_meas_meta:
            y += 10
            nlos = sum(1 for m in sim.last_meas_meta if m["mode"] == "NLOS")
            total = len(sim.last_meas_meta)
            draw_text(screen, f"Medições NLOS: {nlos}/{total}", sidebar_x, y, font); y += LINE_H
        
        # debug EKF
        if show_debug:
            y += 14
            draw_text(screen, "DEBUG EKF", sidebar_x, y, bigfont); y += LINE_H
            if innov_norm is not None:
                draw_text(screen, f"||innov||: {innov_norm:.3f}", sidebar_x, y, font); y += LINE_H
            else:
                draw_text(screen, "||innov||: n/a", sidebar_x, y, font); y += LINE_H

            if nis is not None:
                draw_text(screen, f"NIS: {nis:.3f}", sidebar_x, y, font); y += LINE_H
            else:
                draw_text(screen, "NIS: n/a", sidebar_x, y, font); y += LINE_H

            draw_text(screen, "D: mostra/oculta debug", sidebar_x, y, font); y += LINE_H

        y += 10
        draw_text(screen, "Ferramentas:", cam.viewport[0] + 16, y, bigfont); y += 26
        # dicas
        draw_text(screen, "Log em arquivo (meta.json + data.csv):", cam.viewport[0] + 16, y, font); y += 24
        if btn_filelog:
            btn_filelog.rect.topleft = (cam.viewport[0] + 16, y)  # manter alinhado se janela redimensionar
            btn_filelog.draw(screen)
        draw_text(screen, "Gráficos de erro em tempo real:", cam.viewport[0] + 16, y + 40, font); y += 64
        if btn_graphs:
            btn_graphs.rect.topleft = (cam.viewport[0] + 16, y)   # idem
            btn_graphs.draw(screen)

        pg.display.flip()

    # sair: fecha logger e processo de plot
    if file_logger:
        try: file_logger.close()
        except: pass
    stop_plot_process(plot_state)
    pg.quit()

if __name__ == "__main__":
    main()




