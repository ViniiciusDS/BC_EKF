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
from src.ui.ui_elements import TextBoxDropdown
from src.ui.map_editor import MapEditorScreen
from src.ui.simulation_screen import SimulationScreen 
from src.ui.uwb_test_screen import UwbTestScreen
from src.ui.drawing import draw_grid, draw_axes, draw_anchors, draw_path, draw_robot, draw_text
from src.ui.botton import Button


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
# Estados
# ==========================
STATE_MENU = 0          # menu inicial
STATE_SIM = 1           # simulação rodando
STATE_MAPEDITOR = 2     # editor de mapa
STATE_UWB_TEST = 3      # tela de testes UWB

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
    FPS = 60

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
    btn_uwbtest = Button((720, 580, 220, 44), "Testes UWB", bigfont)
    

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

    # Campo de posicionamento das âncoras 
    font = pg.font.SysFont("arial", 18)
    bigfont = pg.font.SysFont("arial", 22, bold=True)
    textbox_x = None
    textbox_y = None
    btn_place_anchor = None

    # instancia do menu
    editor_screen = MapEditorScreen(env, font, bigfont, SIDE_W)     # instancia do map editor
    sim_screen = None                 # instancia do sim screen
    uwb_screen = None                 # instancia do uwb test screen


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

                        anchors_dyn = make_anchors()

                        sim = Simulator(
                            anchors=anchors_dyn.copy(),
                            baseline=getattr(config, "WHEEL_BASE", 0.65),
                            z_c=getattr(config, "TAG_HEIGHT", 0.5),
                            Q=np.diag([1e-4, 1e-4, 1e-4]),
                            R=np.eye(max(1, 2 * anchors_dyn.shape[1])) * 0.0025
                            if anchors_dyn.shape[1] > 0 else np.eye(2) * 1e6,
                            dt=DT,
                            config=config,
                            env=env
                        )

                        sim_screen = SimulationScreen(
                            screen=screen,
                            cam=cam,
                            clock=clock,
                            sim=sim,
                            font=font,
                            bigfont=bigfont,
                            side_width=SIDE_W,
                            plot_state=plot_state,
                        )

                        # inicializações específicas do SIM (rotas, âncoras etc)
                        sim_screen.anchors_dyn = anchors_dyn.copy()
                        sim_screen.waypoints = make_route()
                        sim_screen.autopilot = len(sim_screen.waypoints) > 0

                        state = STATE_SIM

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
                    
                    elif btn_uwbtest.hit(event.pos):
                        # muda para a aba de testes UWB
                        state = STATE_UWB_TEST
                        cam.reset_view()
                        cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

                        uwb_screen = UwbTestScreen(
                            screen=screen,
                            cam=cam,
                            clock=clock,
                            font=font,
                            bigfont=bigfont,
                            side_width=SIDE_W,
                        )

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
            btn_uwbtest.draw(screen)
            draw_text(screen, "ESC para sair", 20, screen.get_height()-30, font)
            pg.display.flip()
            continue
        
        # ======= ESTADO MAPEDITOR =======
        if state == STATE_MAPEDITOR:
            cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

            dt = clock.tick(FPS) / 1000.0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        # sai do editor, limpa preview e volta pro menu
                        editor_screen.reset_preview()
                        # sincroniza env global com o env do editor
                        env = editor_screen.env
                        # Volta para o menu
                        state = STATE_MENU
                        continue

                    # passa resto do teclado pro editor (M, textbox etc.)
                    editor_screen.handle_event(event, cam, cam.viewport[0])
                    continue

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

                    # demais cliques vão pro editor (LMB/RMB, HUD)
                    editor_screen.handle_event(event, cam, cam.viewport[0])

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

                    # mover mouse para atualizar preview
                    editor_screen.handle_event(event, cam, cam.viewport[0])

            # --- atualização do editor ---
            editor_screen.update(dt)

            # --- desenho do editor ---
            screen.fill(WHITE)
            cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

            # área do mapa (fundo, grid, paredes existentes, eixos)
            map_rect = pg.Rect(0, 0, cam.viewport[0], cam.viewport[1])
            pg.draw.rect(screen, WHITE, map_rect)
            draw_grid(screen, cam)
            draw_environment(screen, cam, editor_screen.env)
            draw_axes(screen, cam, font)

            # preview da parede (linha “fantasma”)
            editor_screen.draw_preview(screen, cam)

            # HUD lateral do editor
            editor_screen.draw_sidebar(screen, cam, font, bigfont)

            pg.display.flip()
            continue
                
        # ======= ESTADO SIM =======
        if state == STATE_SIM:
            cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

            sim_screen.layout_hud()
            
            events = pg.event.get()

            actions = sim_screen.handle_events(events)

            if actions.go_to_menu:
                sim_screen.close()
                sim_screen = None
                state = STATE_MENU
                continue

            if actions.quite_app:
                sim_screen.close()
                running = False
                continue

            sim_screen.update(dt)
            sim_screen.draw()

            pg.display.flip()

        # ======= ESTADO UWB_TEST =======
        if state == STATE_UWB_TEST:
            cam.set_viewport(screen.get_width() - SIDE_W, screen.get_height())

            events = pg.event.get()
            actions = uwb_screen.handle_events(events)

            if actions.go_to_menu:
                uwb_screen.close()
                uwb_screen = None
                state = STATE_MENU
                continue

            if actions.quite_app:
                uwb_screen.close()
                running = False
                continue

            uwb_screen.update(dt)
            uwb_screen.draw()
            pg.display.flip()
            continue



if __name__ == "__main__":
    import multiprocessing as mp    # Evita problemas no Windows multiprocessing
    mp.freeze_support()

    # set 'spawn' como método de start (mais seguro) evita erro se já estiver setado
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    main()




