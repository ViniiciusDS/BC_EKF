from __future__ import annotations

import numpy as np
import pygame as pg

from src.ui.drawing import draw_grid, draw_axes
from src.environment.environment import draw_environment
from src.ui.legend_overlay import draw_legend_overlay
from src.ui.algo_modes.shared import (
    ALGO_ORDER,
    ALGO_COLORS,
    draw_analyzer_panel,
)


BLACK = (0, 0, 0)
GRAY_D = (90, 90, 90)

def _safe_anchor_xy_array(data):
    """
    Normaliza âncoras para matriz N x 2.

    Aceita:
    - N x 2
    - N x 3
    - 2 x N
    - 3 x N
    """
    if data is None:
        return None

    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2:
        return None

    # Caso venha transposto: 2xN ou 3xN
    if arr.shape[0] in (2, 3) and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    if arr.shape[1] < 2:
        return None

    arr = arr[:, :2]

    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[valid]

    return arr if len(arr) > 0 else None


def draw_dataset_anchors(mode):
    """
    Desenha âncoras com estilo consistente, independentemente do formato interno.
    """
    anchors = _safe_anchor_xy_array(getattr(mode, "_dataset_anchors", None))

    if anchors is None:
        return

    screen = mode.host.screen
    cam = mode.host.cam

    for x, y in anchors:
        sx, sy = cam.world_to_screen(float(x), float(y))
        p = (int(sx), int(sy))

        # estilo parecido com o anterior: vermelho com borda preta
        pg.draw.circle(screen, (0, 0, 0), p, 7)
        pg.draw.circle(screen, (210, 55, 55), p, 6)
        
def _safe_xy_array(data):
    if data is None:
        return None

    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return None

    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[valid]

    if len(arr) == 0:
        return None

    return arr[:, :2]


def _draw_polyline_world(screen, cam, points, color, width=2, dashed=False, dash_px=14, gap_px=8):
    pts = _safe_xy_array(points)
    if pts is None or len(pts) < 2:
        return

    screen_pts = [cam.world_to_screen(float(x), float(y)) for x, y in pts]

    if not dashed:
        pg.draw.lines(screen, color, False, screen_pts, width)
        return

    for p0, p1 in zip(screen_pts[:-1], screen_pts[1:]):
        x0, y0 = p0
        x1, y1 = p1

        dx = x1 - x0
        dy = y1 - y0
        length = float(np.hypot(dx, dy))

        if length <= 1e-9:
            continue

        ux = dx / length
        uy = dy / length

        s = 0.0
        while s < length:
            e = min(s + dash_px, length)
            a = (x0 + ux * s, y0 + uy * s)
            b = (x0 + ux * e, y0 + uy * e)
            pg.draw.line(screen, color, a, b, width)
            s += dash_px + gap_px


def _draw_estimated_track(mode, algo, result):
    """
    Desenha a trajetória estimada de um algoritmo.

    Aceita tanto:
    - result como dict: {"posicoes": array}, {"positions": array}, {"trajectory": array}
    - result diretamente como array Nx2/Nx3

    A função também ignora pontos inválidos para evitar crash no pygame.
    """
    import numpy as np
    import pygame as pg

    if result is None:
        return

    screen = mode.host.screen
    cam = mode.host.cam

    # -------------------------------------------------
    # 1) Extrair pontos sem usar "or" com numpy array
    # -------------------------------------------------
    pts = None

    if isinstance(result, dict):
        if "posicoes" in result and result["posicoes"] is not None:
            pts = result["posicoes"]
        elif "positions" in result and result["positions"] is not None:
            pts = result["positions"]
        elif "trajectory" in result and result["trajectory"] is not None:
            pts = result["trajectory"]
        elif "traj" in result and result["traj"] is not None:
            pts = result["traj"]
    else:
        pts = result

    if pts is None:
        return

    # -------------------------------------------------
    # 2) Converter para array numérico
    # -------------------------------------------------
    try:
        pts = np.asarray(pts, dtype=float)
    except Exception:
        return

    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return

    if getattr(mode, "dataset_source_type", "") == "real_odom_uwb_2026":
        sample_indices = getattr(mode, "_robot2026_uwb_indices", np.array([]))
        if len(sample_indices) == len(pts):
            selected = mode._selected_odom_laps()
            visible = np.zeros(len(sample_indices), dtype=bool)
            for lap in selected:
                visible |= (sample_indices >= lap.start) & (sample_indices < lap.stop)
            pts = pts[visible]
            if len(pts) < 2:
                return

    # Pygame precisa de pares x,y.
    pts = pts[:, :2]

    # Remove NaN/Inf.
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]

    if pts.shape[0] < 2:
        return

    # -------------------------------------------------
    # 3) Quebrar a trajetória em segmentos válidos
    # -------------------------------------------------
    color = getattr(mode, "algo_colors", {}).get(algo, None)

    if color is None:
        try:
            from src.ui.algo_modes.shared import ALGO_COLORS
            color = ALGO_COLORS.get(algo, (0, 0, 0))
        except Exception:
            color = (0, 0, 0)

    line_width = 2

    try:
        # -------------------------------------------------
        # Converte pontos para coordenadas de tela
        # -------------------------------------------------
        screen_pts = []
        valid_pts = []

        for x, y in pts:
            screen_pt = cam.world_to_screen(float(x), float(y))
            
            # Valida o ponto de tela
            if isinstance(screen_pt, (tuple, list)) and len(screen_pt) >= 2:
                if np.isfinite(screen_pt[0]) and np.isfinite(screen_pt[1]):
                    screen_pts.append((int(screen_pt[0]), int(screen_pt[1])))
                    valid_pts.append([x, y])

        if len(screen_pts) < 2:
            return

        valid_pts = np.array(valid_pts, dtype=float)

        # -------------------------------------------------
        # 4) Desenhar segmentos, evitando conectar outliers
        # -------------------------------------------------
        max_world_jump = 1.5  # metros; ajuste se necessário

        segments = []
        current = []

        for i, (sx, sy) in enumerate(screen_pts):
            if i == 0:
                current = [(sx, sy)]
                continue

            jump = float(np.linalg.norm(valid_pts[i] - valid_pts[i - 1]))

            if jump > max_world_jump:
                if len(current) >= 2:
                    segments.append(current)
                current = [(sx, sy)]
            else:
                current.append((sx, sy))

        if len(current) >= 2:
            segments.append(current)

        for seg in segments:
            if len(seg) >= 2:
                pg.draw.lines(screen, color, False, seg, line_width)

        # Desenha os pontos por cima.
        for sx, sy in screen_pts:
            pg.draw.circle(screen, color, (sx, sy), 3)
            pg.draw.circle(screen, (0, 0, 0), (sx, sy), 3, 1)

    except Exception:
        return


def draw_reference_routes(mode):
    """
    Desenha rotas de referência/odometria do dataset.

    Convenção:
    - rota planejada/arquivo JSON: cinza tracejado;
    - odometria reconstruída/dataset_route: preto contínuo fino.
    """
    screen = mode.host.screen
    cam = mode.host.cam

    # Rota de referência vinda do JSON/PDF.
    ref = (
        getattr(mode, "_reference_route_display", None)
        if getattr(mode, "_reference_route_display", None) is not None
        else getattr(mode, "_route_waypoints", None)
    )

    if ref is not None:
        _draw_polyline_world(
            screen,
            cam,
            ref,
            color=(95, 95, 95),
            width=2,
            dashed=True,
            dash_px=14,
            gap_px=8,
        )

    # Portão virtual da pista: desenhado explicitamente, sem confundir
    # marcação visual com obstáculo físico/oclusão de rádio.
    gate = getattr(mode, "_robot2026_view_settings", None)
    if (getattr(mode, "dataset_source_type", "") == "real_odom_uwb_2026"
            and gate and gate.get("segmentation") == "start_gate"):
        center = np.asarray(gate["gate_xy_m"], dtype=float)
        normal = np.asarray(gate["gate_direction_xy"], dtype=float)
        normal = normal / np.linalg.norm(normal)
        tangent = np.array([-normal[1], normal[0]], dtype=float)
        half_width = gate["gate_half_width_m"]
        start = center - half_width * tangent
        finish = center + half_width * tangent
        p0 = cam.world_to_screen(float(start[0]), float(start[1]))
        p1 = cam.world_to_screen(float(finish[0]), float(finish[1]))
        pg.draw.line(screen, (20, 155, 82), p0, p1, 4)
        middle = cam.world_to_screen(float(center[0]), float(center[1]))
        tip = center + 0.25 * normal
        ptip = cam.world_to_screen(float(tip[0]), float(tip[1]))
        pg.draw.line(screen, (20, 155, 82), middle, ptip, 2)
        # O início do LAB7A é um canto: a detecção usa reentrada no
        # círculo verde pelo +Y, não cruzamento x<0 -> x>0.
        if gate.get("gate_mode") == "corner_return":
            radius = float(gate["gate_return_radius_m"])
            radial = cam.world_to_screen(float(center[0]+radius), float(center[1]))
            px_radius = round(float(np.hypot(
                radial[0]-middle[0], radial[1]-middle[1]
            )))
            if px_radius >= 2:
                pg.draw.circle(screen, (40, 160, 85), middle, px_radius, 1)
            incoming = np.asarray(gate["gate_return_direction_xy"], dtype=float)
            incoming = incoming / np.linalg.norm(incoming)
            approach = center - .3*incoming
            p_approach = cam.world_to_screen(
                float(approach[0]), float(approach[1])
            )
            pg.draw.line(screen, (40, 160, 85), p_approach, middle, 2)
        caption = mode.host.font.render("Portao", True, (20, 120, 70))
        screen.blit(caption, (int(middle[0])+5, int(middle[1])-25))

    # Rota odométrica/reconstruída. Evita duplicar se for igual à referência.
    odom = (
        getattr(mode, "_dataset_route", None)
        if getattr(mode, "dataset_source_type", "") in ("real_encoder_uwb", "real_odometry_2026", "real_odom_uwb_2026")
        else None
    )

    # O modo 2026 desenha as voltas independentemente, sem ligar o fim
    # de uma volta ao início da seguinte com um segmento artificial.
    if getattr(mode, "dataset_source_type", "") in ("real_odometry_2026", "real_odom_uwb_2026"):
        data = getattr(mode, "_robot2026_data", None)
        if data is not None:
            # Referência aproximada extraída da primeira volta odométrica.
            # Não é ground truth nem altera _reference_route_display ou métricas.
            approx = getattr(mode, "_odom_track_template", None)
            if approx is not None and mode.show_odom_track_template:
                _draw_polyline_world(
                    screen, cam, approx.xy,
                    color=(35, 122, 180), width=2, dashed=True,
                    dash_px=9, gap_px=6,
                )
            selected = mode._selected_odom_laps()
            for lap in selected:
                pts = data.poses_xytheta[lap.start:lap.stop, :2]
                _draw_polyline_world(
                    screen, cam, pts,
                    color=(65, 65, 65) if mode.odom_view_mode == "all" else (10, 10, 10),
                    width=1 if mode.odom_view_mode == "all" else 2,
                )
            # Início e fim da seleção visível, não o estado global da sessão.
            if selected:
                for point, color, caption in (
                    (data.poses_xytheta[selected[0].start, :2], (30, 150, 70), "I"),
                    (data.poses_xytheta[selected[-1].stop - 1, :2], (210, 50, 50), "F"),
                ):
                    px, py = cam.world_to_screen(float(point[0]), float(point[1]))
                    if np.isfinite(px) and np.isfinite(py):
                        marker = (int(px), int(py))
                        pg.draw.circle(screen, (20, 20, 20), marker, 6)
                        pg.draw.circle(screen, color, marker, 5)
                        label = mode.host.font.render(caption, True, color)
                        screen.blit(label, (marker[0] + 7, marker[1] - 13))
        return

    if odom is not None:
        arr = np.asarray(odom, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
            _draw_polyline_world(
                screen,
                cam,
                arr[:, :2],
                color=(0, 0, 0),
                width=2,
                dashed=False,
            )


def draw_dataset_world(mode):
    """
    Desenha o mundo principal: grade, eixos, mapa, âncoras, rotas e resultados.
    """
    screen = mode.host.screen
    cam = mode.host.cam

    # Limpa a área do mapa antes de redesenhar.
    # Sem isso, a tela anterior/menu fica aparecendo por baixo da grade.
    world_w = cam.viewport[0] if hasattr(cam, "viewport") else screen.get_width()
    world_rect = pg.Rect(0, 0, world_w, screen.get_height())
    pg.draw.rect(screen, (255, 255, 255), world_rect)

    draw_grid(screen, cam)
    draw_axes(screen, cam, mode.host.font)

    if getattr(mode, "_map_env", None) is not None:
        draw_environment(screen, cam, mode._map_env)

    draw_reference_routes(mode)

    draw_dataset_anchors(mode)

    if getattr(mode, "_batch_results", None) is not None:
        for algo in ALGO_ORDER:
            res = mode._batch_results.get(algo)
            _draw_estimated_track(mode, algo, res)


def draw_dataset_sidebar_controls(mode):
    """
    Desenha botões inferiores do painel lateral do Dataset Mode.
    Não desenha botões principais do host; apenas controles próprios do modo.
    """
    screen = mode.host.screen
    sidebar_x = mode.host.cam.viewport[0] + 16
    screen_h = screen.get_height()

    panel_x = mode.host.cam.viewport[0]
    panel_rect = pg.Rect(panel_x, 0, screen.get_width() - panel_x, screen_h)
    pg.draw.rect(screen, (245, 247, 252), panel_rect)

    legend_y = screen_h - 44
    toggle_y = legend_y - 40
    metric_y = toggle_y - 40

    if hasattr(mode, "btn_metric_mode"):
        mode.btn_metric_mode.rect.topleft = (sidebar_x, metric_y)
        mode.btn_metric_mode.rect.size = (190, 32)
        mode.btn_metric_mode.text = f"Métrica: {mode._metric_mode_label()}"
        mode.btn_metric_mode.draw(screen)

    mode.btn_toggle_analyzer.rect.topleft = (sidebar_x, toggle_y)
    mode.btn_toggle_analyzer.rect.size = (190, 32)
    mode.btn_toggle_analyzer.text = (
        "Ocultar Analyzer" if mode.show_analyzer else "Mostrar Analyzer"
    )
    mode.btn_toggle_analyzer.draw(screen)

    mode.btn_toggle_legend.rect.topleft = (sidebar_x, legend_y)
    mode.btn_toggle_legend.rect.size = (190, 32)
    mode.btn_toggle_legend.text = (
        "Ocultar Legenda" if mode.show_legend_overlay else "Mostrar Legenda"
    )
    mode.btn_toggle_legend.draw(screen)

    # Controles de voltas ocupam o espaço entre o status e os botões inferiores.
    if (
        getattr(mode, "dataset_source_type", "") in ("real_odometry_2026", "real_odom_uwb_2026")
        and getattr(mode, "_robot2026_data", None) is not None
    ):
        font = mode.host.font
        # Reservar intervalo entre "Rodar batch" e os controles inferiores.
        # Antes y-72 sobrepunha o botão Rodar batch.
        y = metric_y - 112
        if getattr(mode, "_odom_track_template", None) is not None:
            mode.btn_odom_track.rect.topleft = (sidebar_x, y - 55)
            mode.btn_odom_track.text = (
                "Pista aprox.: ON" if mode.show_odom_track_template
                else "Pista aprox.: OFF"
            )
            mode.btn_odom_track.draw(screen)
        status = getattr(mode, "_odom_gate_status", "")
        if status:
            stamp = font.render(status, True, (27, 117, 75))
            screen.blit(stamp, (sidebar_x, y - 39))
        if (not getattr(mode, "_odom_is_time_window", True)
                and any(lap.complete for lap in getattr(mode, "_robot2026_laps", ()))
                and getattr(mode, "_batch_results", None)):
            # Atalho apenas quando há volta física realmente confirmada.
            hint = font.render("F7: exportar indicadores por volta", True, (55, 85, 120))
            screen.blit(hint, (sidebar_x, y - 59))
        label = font.render(
            f"Visualizar: {mode._odometry_view_label()}", True, (65, 65, 72)
        )
        screen.blit(label, (sidebar_x, y - 17))

        mode.btn_lap_mode.rect.topleft = (sidebar_x, y)
        is_time = getattr(mode, "_odom_is_time_window", False)
        mode.btn_lap_mode.text = (
            {"single": "Modo: Uma janela", "range": "Modo: Intervalo",
             "all": "Modo: Todas"}.get(mode.odom_view_mode, "Modo: Uma janela")
            if is_time else
            {"single": "Modo: Uma volta", "range": "Modo: Intervalo",
             "all": "Modo: Todas"}.get(mode.odom_view_mode, "Modo: Uma volta")
        )
        mode.btn_lap_mode.draw(screen)

        if mode.odom_view_mode != "all":
            mode.btn_lap_prev.rect.topleft = (sidebar_x, y + 37)
            mode.btn_lap_next.rect.topleft = (sidebar_x + 98, y + 37)
            mode.btn_lap_prev.draw(screen)
            mode.btn_lap_next.draw(screen)

        if mode.odom_view_mode == "range":
            mode.btn_lap_less.rect.topleft = (sidebar_x, y + 74)
            mode.btn_lap_more.rect.topleft = (sidebar_x + 98, y + 74)
            mode.btn_lap_less.draw(screen)
            mode.btn_lap_more.draw(screen)


def draw_dataset_status(mode):
    """
    Desenha textos pequenos de status do dataset no painel lateral.
    """
    screen = mode.host.screen
    font = mode.host.font

    # O painel sobre o mapa já mostra fonte e número de amostras.
    # Evita colisão com os controles de voltas e de pista aproximada.
    if getattr(mode, "dataset_source_type", "") in ("real_odometry_2026", "real_odom_uwb_2026"):
        return

    sidebar_x = mode.host.cam.viewport[0] + 16
    y = 470

    label = getattr(mode, "_dataset_label", "")
    if label:
        txt = font.render(str(label), True, (150, 150, 150))
        screen.blit(txt, (sidebar_x, y))
        y += 28

    if getattr(mode, "_batch_dists", None) is not None:
        n = int(mode._batch_dists.shape[0])
        txt = font.render(f"Amostras: {n}", True, (150, 150, 150))
        screen.blit(txt, (sidebar_x, y))
    elif getattr(mode, "dataset_source_type", "") == "real_odometry_2026":
        odom = getattr(mode, "_robot2026_data", None)
        if odom is not None:
            txt = font.render(f"Odometria: {odom.sample_count} pontos", True, (110, 110, 110))
            screen.blit(txt, (sidebar_x, y))


def draw_dataset_analyzer(mode):
    """Mostra métricas adequadas à referência; nunca rotula a pista como GT."""
    if not getattr(mode, "show_analyzer", True):
        return

    is_single_tag = getattr(mode, "dataset_source_type", "") == "real_odom_uwb_2026"
    stats = getattr(mode, "_dataset_stats", None)
    if not stats:
        # Ao clicar em "Mostrar", exibir informação em vez de botão inerte.
        if is_single_tag and getattr(mode, "_batch_results", None) is not None:
            screen = mode.host.screen
            panel = pg.Surface((500, 90), pg.SRCALPHA)
            panel.fill((250, 250, 252, 240))
            screen.blit(panel, (10, 40))
            font = mode.host.font
            screen.blit(font.render("Analyzer - Sem comparacao disponivel", True, (30, 30, 30)), (20, 52))
            screen.blit(font.render("Confira a rota nominal ou a serie UWB do batch.", True, (60, 60, 70)), (20, 82))
        return

    title = f"Dataset Analyzer - {mode._metric_mode_label()}"
    column_titles = ("RMSE", "MAE", "Max")
    boxplot_title = "Boxplot de erro"

    if is_single_tag:
        # São distâncias geométricas / diferenças odométricas, NÃO
        # estimativas do erro de posição verdadeiro.
        captions = {
            "reference_route": ("Analyzer - Rota nominal (nao GT)", "Distancia a pista nominal (m)"),
            "encoder_route": ("Analyzer - UWB x odometria (nao GT)", "Distancia entre trajetorias (m)"),
            "cluster": ("Analyzer - Dispersao (nao erro)", "Distancia ao centroide (m)"),
        }
        title, boxplot_title = captions.get(
            mode.metric_mode, ("Analyzer - Sem ground truth", "Distancia (m)")
        )
        column_titles = ("RMS", "Media", "Max")

    draw_analyzer_panel(
        screen=mode.host.screen,
        font=mode.host.font,
        bigfont=mode.host.bigfont,
        title=title,
        stats=stats,
        selected=mode.selected,
        column_titles=column_titles,
        boxplot_title=boxplot_title,
        display_names={"bc_ekf": "EKF 1 tag"} if is_single_tag else None,
        x=10,
        y=40,
        w=500,
        h=380,
    )


def draw_robot2026_odometry_info(mode):
    """Resumo da odometria isolada, sem apresentar RMSE como ground truth."""
    if getattr(mode, "dataset_source_type", "") not in ("real_odometry_2026", "real_odom_uwb_2026"):
        return
    odom = getattr(mode, "_robot2026_data", None)
    if odom is None or odom.sample_count == 0:
        return
    screen = mode.host.screen
    font = mode.host.font
    # Quando o Analyzer está visível, o painel PCNT vai para a direita.
    # Evita sobrepor duas superfícies em (10,40), preservando o status.
    show_summary_compact = (
        getattr(mode, "dataset_source_type", "") == "real_odom_uwb_2026"
        and getattr(mode, "show_analyzer", False)
        and bool(getattr(mode, "_dataset_stats", None))
    )
    world_width = int(getattr(mode.host.cam, "viewport", (screen.get_width(),))[0])
    x, y, w, h = 10, 40, 485, 150
    if show_summary_compact:
        if world_width <= 935:
            return  # largura insuficiente: prioriza Analyzer
        w = min(405, world_width - 530)
        x = world_width - w - 8
    panel = pg.Surface((w, h), pg.SRCALPHA)
    panel.fill((250, 250, 252, 224))
    screen.blit(panel, (x, y))
    pg.draw.rect(screen, (125, 125, 135), pg.Rect(x, y, w, h), 1)
    selected = mode._selected_odom_laps()
    end_idx = selected[-1].stop - 1 if selected else odom.sample_count - 1
    visible_count = sum(lap.sample_count for lap in selected)
    xf, yf, heading = odom.poses_xytheta[end_idx]
    combined = getattr(mode, "_robot2026_uwb_data", None)
    lines = [
        ("PCNT + UWB tag unica (sem ground truth)" if combined is not None
         else "Odometria real - sem UWB / sem ground truth"),
        f"Sessao {odom.session_id} | dataset completo: {odom.sample_count} amostras | {odom.duration_s:.1f} s",
        f"Exibindo {mode._odometry_view_label()} | {visible_count} pontos desenhados",
        (
            f"Pista aprox. da volta {mode._odom_track_template.source_lap}: "
            f"{mode._odom_track_template.span_x_m:.2f} x "
            f"{mode._odom_track_template.span_y_m:.2f} m (NAO e ground truth)"
            if getattr(mode, "_odom_track_template", None) is not None
            else "Pista aprox.: indisponivel (sem volta completa valida)"
        ),
        f"Raio {odom.profile.wheel_radius_m:.5f} m | Base {odom.profile.wheel_base_m:.3f} m | CPR {odom.profile.ticks_per_wheel_rev:.0f}",
        f"Fim exibido: x={xf:.3f} m, y={yf:.3f} m, heading={np.degrees(heading) % 360:.1f} graus",
    ]
    if combined is not None:
        lines[2] = (
            f"PCNT: {odom.sample_count} | UWB batch: {combined.sample_count} "
            f"| repetidas: {combined.repeated_readings}"
        )
    if show_summary_compact:
        lines = [
            f"PCNT + UWB | Sessao {odom.session_id}",
            f"PCNT: {odom.sample_count} | UWB: {combined.sample_count}",
            f"Repetidas: {combined.repeated_readings}",
            f"Exibindo: {mode._odometry_view_label()}",
            f"CPR: {odom.profile.ticks_per_wheel_rev:.0f} | Base: {odom.profile.wheel_base_m:.3f}m",
            "Sem ground truth sincronizado",
        ]
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, (30, 30, 40))
        screen.blit(text_surface, (x + 8, y + 7 + 23 * i))


def draw_dataset_legend(mode):
    if getattr(mode, "show_legend_overlay", False):
        mode._legend_close_rect = draw_legend_overlay(
            mode.host.screen,
            mode.host.font,
            mode.host.bigfont,
            selected=mode.selected,
            labels=(
                {"bc_ekf":"EKF 1 tag (odometria + UWB)"}
                if getattr(mode,"dataset_source_type","")=="real_odom_uwb_2026"
                else None
            ),
        )
    else:
        mode._legend_close_rect = None


def draw_field_quality_panel(mode):
    """F8: sobreposição informativa, sem alterar o mundo nem o Analyzer."""
    if not getattr(mode, "show_field_quality_panel", False):
        return
    report = getattr(mode, "_field_quality_report", None)
    if not report:
        return
    screen = mode.host.screen
    font = mode.host.font
    viewport = int(mode.host.cam.viewport[0])
    x, y = 12, 222
    w = min(470, max(260, viewport - 24))
    h = min(380, max(140, screen.get_height() - y - 14))
    if h < 200:
        y = 40
        h = min(380, screen.get_height() - 54)
    if h < 100:
        return
    panel = pg.Surface((w, h), pg.SRCALPHA)
    panel.fill((248, 250, 253, 245))
    screen.blit(panel, (x, y))
    pg.draw.rect(screen, (85, 102, 132), (x, y, w, h), 1)
    heading = mode.host.bigfont.render("Qualidade PCNT/UWB (F8)", True, (27, 45, 73))
    screen.blit(heading, (x + 12, y + 9))
    summary = report.get("summary", {})
    enc = report.get("encoder_quality", {})
    slots = report.get("slot_quality", [])
    count = max(1, int(summary.get("pcnt_samples", 0)))
    lines = [
        f"Sessao {report.get('session_id')} | PCNT {summary.get('pcnt_samples', 0)} | {summary.get('duration_s', 0):.1f} s",
        f"UWB: {summary.get('uwb_snapshots', 0)} snapshots novos completos",
        f"Disponibilidade por linha: {100 * summary.get('uwb_snapshots', 0) / count:.1f}%",
        f"Encoder: {enc.get('estimated_travel_m', 0):.2f} m percorridos (estimativa)",
        f"Velocidade max.: {enc.get('max_abs_linear_speed_m_s', 0):.2f} m/s",
        "Slots: novos | repetidos | invalidos",
    ]
    for s in slots:
        lines.append(
            f"  SPI {s['slot']}: {s['new_valid']} | {s['repeated_valid']} | {s['invalid']}"
        )
    if summary.get("warnings"):
        lines.append(f"Avisos: {len(summary['warnings'])} (detalhes no F9)")
    lines.append("Sem GT; sinais WR/WL exigem teste fisico")
    max_lines = max(0, (h - 58) // 22)
    for i, line in enumerate(lines[:max_lines]):
        surf = font.render(line, True, (41, 47, 62))
        screen.blit(surf, (x + 12, y + 48 + i * 22))


def draw_dataset_mode(mode):
    """
    Função principal de renderização do DatasetMode.
    O modal ainda fica no dataset_mode.py por enquanto.
    """
    draw_dataset_world(mode)
    draw_robot2026_odometry_info(mode)
    draw_dataset_status(mode)
    draw_dataset_sidebar_controls(mode)

    if getattr(mode, "_batch_results", None) is not None:
        draw_dataset_analyzer(mode)

    draw_dataset_legend(mode)
    # Renderizacao do painel F8 depois do Analyzer e da legenda.
    draw_field_quality_panel(mode)
