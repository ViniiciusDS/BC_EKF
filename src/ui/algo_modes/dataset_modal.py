from __future__ import annotations

import pygame as pg

from src.ui.ui_elements import (
    TextBoxDropdown,
    ModalFrame,
    FormDropdownRow,
    ModalButtonBar,
)
from src.ui.algo_modes.shared import list_files_by_extension
from src.ui.algo_modes.dataset_odometry_pipeline import list_robot2026_sources

class DatasetConfigModal:
    """
    Modal específico do Dataset Mode.

    Usa componentes genéricos de ui_elements.py, mas mantém aqui a lógica
    específica de quais campos existem no dataset mode.
    """

    def __init__(self, mode):
        self.mode = mode
        self.frame = None

        font = mode.host.font

        self.dd_source = TextBoxDropdown(
            pg.Rect(0, 0, 440, 30),
            font,
            options=["Simulado", "Real (encoder + UWB)", "Real (odometria 2026)", "Real (PCNT + UWB 2026)"],
        )

        self.dd_sim_kind = TextBoxDropdown(
            pg.Rect(0, 0, 440, 30),
            font,
            options=["Front", "Rear", "Mid", "BC"],
        )
        # Só BC de duas tags: a geometria é uma propriedade do
        # experimento, jamais uma escolha escondida no solver.
        self.dd_tag_preset = TextBoxDropdown(
            pg.Rect(0,0,440,30),font,
            options=["Auto (JSON / legado)", "Frente / trás",
                     "Esquerda / direita", "Personalizado (JSON)"],
        )

        self.dd_dataset = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
        self.dd_encoder = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
        self.dd_uwb = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
        self.dd_odom = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[], max_len=255)
        self.dd_odom_session = TextBoxDropdown(
            pg.Rect(0, 0, 440, 30), font, options=["1", "2"], max_len=8
        )
        # Sidecar JSON é a referência ao UWB tratado; o CSV correspondente
        # é inferido somente após validação do par e do PCNT de origem.
        self.dd_preprocessed_uwb = TextBoxDropdown(
            pg.Rect(0, 0, 440, 30), font, options=["(Nenhum)"], max_len=255
        )
        self.dd_anchors = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
        self.dd_route = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
        self.dd_map = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])

        self.buttons = ModalButtonBar(font, ok_text="Carregar", cancel_text="Cancelar")
        
        self.rows_sim = [
            FormDropdownRow("Fonte:", self.dd_source),
            FormDropdownRow("Tipo:", self.dd_sim_kind),
            FormDropdownRow("Dataset:", self.dd_dataset),
            FormDropdownRow("Âncoras:", self.dd_anchors),
            FormDropdownRow("Rota ref.:", self.dd_route),
            FormDropdownRow("Mapa:", self.dd_map),
        ]

        self.rows_sim_bc = [
            *self.rows_sim[:2],
            FormDropdownRow("Tags:", self.dd_tag_preset),
            *self.rows_sim[2:],
        ]

        self.rows_real = [
            FormDropdownRow("Fonte:", self.dd_source),
            FormDropdownRow("Encoder real:", self.dd_encoder),
            FormDropdownRow("UWB real:", self.dd_uwb),
            FormDropdownRow("Âncoras:", self.dd_anchors),
            FormDropdownRow("Rota ref.:", self.dd_route),
            FormDropdownRow("Mapa:", self.dd_map),
        ]

        self.rows_real_bc = [
            *self.rows_real[:3],
            FormDropdownRow("Tags:", self.dd_tag_preset),
            *self.rows_real[3:],
        ]

        self.rows_odom_uwb = [
            FormDropdownRow("Fonte:", self.dd_source),
            FormDropdownRow("Arquivo PCNT:", self.dd_odom),
            FormDropdownRow("Sessão (bruto):", self.dd_odom_session),
            FormDropdownRow("UWB tratado:", self.dd_preprocessed_uwb),
            FormDropdownRow("Âncoras:", self.dd_anchors),
            FormDropdownRow("Rota ref.:", self.dd_route),
            FormDropdownRow("Mapa:", self.dd_map),
        ]

        self.rows_odometry = [
            FormDropdownRow("Fonte:", self.dd_source),
            FormDropdownRow("Arquivo PCNT:", self.dd_odom),
            FormDropdownRow("Sessão (bruto):", self.dd_odom_session),
            FormDropdownRow("Rota ref.:", self.dd_route),
            FormDropdownRow("Mapa:", self.dd_map),
        ]

        self.open_dropdown = None
        self.max_dropdown_items = 7
        self.dropdown_scroll = 0

        
        self.refresh_options()

    def open(self):
        self.refresh_options()

        # Sincroniza estado atual do DatasetMode para os dropdowns.
        source = getattr(self.mode, "dataset_source_type", "simulated")
        if source == "real_encoder_uwb":
            self._set_dropdown_value(self.dd_source, "Real (encoder + UWB)")
        elif source == "real_odometry_2026":
            self._set_dropdown_value(self.dd_source, "Real (odometria 2026)")
        elif source == "real_odom_uwb_2026":
            self._set_dropdown_value(self.dd_source, "Real (PCNT + UWB 2026)")
        else:
            self._set_dropdown_value(self.dd_source, "Simulado")

        if not self._get_dropdown_value(self.dd_odom_session):
            self._set_dropdown_value(self.dd_odom_session, "2")
        current_overlay=getattr(self.mode,"_real_preproc_sidecar_path",None)
        if current_overlay:
            self._set_dropdown_value(self.dd_preprocessed_uwb,str(current_overlay))
        elif not self._get_dropdown_value(self.dd_preprocessed_uwb):
            self._set_dropdown_value(self.dd_preprocessed_uwb,"(Nenhum)")

        kind = getattr(self.mode, "simulated_dataset_kind", "Front")

        kind_map = {
            "front": "Front",
            "top": "Front",
            "rear": "Rear",
            "bot": "Rear",
            "bottom": "Rear",
            "mid": "Mid",
            "bc": "BC",
        }

        self._set_dropdown_value(
            self.dd_sim_kind,
            kind_map.get(str(kind).strip().lower(), "Front"),
        )

        preset=getattr(self.mode,"_modal_tag_preset","auto")
        labels={
            "auto":"Auto (JSON / legado)",
            "front_rear":"Frente / trás",
            "left_right":"Esquerda / direita",
            "custom":"Personalizado (JSON)",
        }
        self._set_dropdown_value(
            self.dd_tag_preset,labels.get(preset,"Auto (JSON / legado)")
        )
        self.mode.dataset_modal_open = True

    def close(self):
        self.mode.dataset_modal_open = False

    def refresh_options(self):
        """
        Atualiza listas de arquivos usando os diretórios do DatasetMode.
        Esta versão não depende de DatasetMode._list_files(), para evitar
        incompatibilidade de assinatura.
        """
        m = self.mode

        def safe_list(dirname, exts):
            return list_files_by_extension(dirname, exts)

        dataset_dir = (
            getattr(m, "datasets_dir", None)
            or getattr(m, "dataset_dir", None)
            or "datasets"
        )

        real_data_dir = (
            getattr(m, "real_data_dir", None)
            or getattr(m, "real_dir", None)
            or dataset_dir
        )

        anchors_dir = (
            getattr(m, "anchors_dir", None)
            or getattr(m, "anchor_dir", None)
            or "anchors"
        )

        routes_dir = (
            getattr(m, "routes_dir", None)
            or getattr(m, "route_dir", None)
            or "routes"
        )

        maps_dir = (
            getattr(m, "maps_dir", None)
            or getattr(m, "map_dir", None)
            or "maps"
        )

        self._set_dropdown_options(
            self.dd_source,
            ["Simulado", "Real (encoder + UWB)", "Real (odometria 2026)", "Real (PCNT + UWB 2026)"],
        )

        self._set_dropdown_options(
            self.dd_sim_kind,
            ["Front", "Rear", "Mid", "BC"],
        )
        self._set_dropdown_options(
            self.dd_tag_preset,
            ["Auto (JSON / legado)", "Frente / trás",
             "Esquerda / direita", "Personalizado (JSON)"],
        )

        dataset_files = safe_list(dataset_dir, (".txt", ".csv", ".jsonl"))

        # Remove arquivos de trajetória exportada, caso estejam na mesma pasta.
        dataset_files = [
            f for f in dataset_files
            if "_traj" not in f.lower()
        ]

        self._set_dropdown_options(
            self.dd_dataset,
            dataset_files,
        )

        encoder_files = safe_list(real_data_dir, (".csv", ".txt"))
        uwb_files = safe_list(real_data_dir, (".csv", ".txt"))

        # Preferência por nomes contendo encoder/uwb, mas sem zerar lista se não encontrar.
        encoder_filtered = [f for f in encoder_files if "encoder" in f.lower()]
        uwb_filtered = [f for f in uwb_files if "uwb" in f.lower()]

        self._set_dropdown_options(
            self.dd_encoder,
            encoder_filtered if encoder_filtered else encoder_files,
        )

        self._set_dropdown_options(
            self.dd_uwb,
            uwb_filtered if uwb_filtered else uwb_files,
        )

        # O novo pipeline aceita arquivo bruto ou CSV por sessão do conversor.
        self._set_dropdown_options(
            self.dd_odom,
            list_robot2026_sources(
                getattr(m, "robot2026_raw_dir", "dados_reais/robot2026"),
                getattr(m, "robot2026_processed_dir", "resultados/robot2026"),
            ),
        )
        self._set_dropdown_options(self.dd_odom_session, ["1", "2", "3", "4", "5"])
        # Arquivos exportados são apresentados como relatórios sidecar
        # para não confundir CSV UWB-only com um PCNT bruto.
        from pathlib import Path
        pre_dir=Path(dataset_dir)
        sidecars=sorted(
            (str(p) for p in pre_dir.glob("*.preproc.json") if p.is_file()),
            key=str.lower
        ) if pre_dir.is_dir() else []
        last=getattr(m,"_last_uwb_preproc_sidecar",None)
        if last and Path(last).is_file() and str(last) not in sidecars:
            sidecars.insert(0,str(last))
        self._set_dropdown_options(
            self.dd_preprocessed_uwb,
            ["(Nenhum)", "(Escolher relatório...)"]+sidecars
        )

        self._set_dropdown_options(
            self.dd_anchors,
            safe_list(anchors_dir, (".json",)),
        )

        self._set_dropdown_options(
            self.dd_route,
            ["(Nenhuma)"] + safe_list(routes_dir, (".json",)),
        )

        self._set_dropdown_options(
            self.dd_map,
            ["(Nenhum)"] + safe_list(maps_dir, (".json",)),
        )

    def _set_dropdown_options(self, dd, options):
        options = list(options or [])

        dd.options = options
        dd.options_all = options
        dd.options_filtered = options

        current_value = getattr(dd, "selected_value", None)
        current_text = getattr(dd, "text", "")

        # Se o valor selecionado ainda existe, preserva.
        if current_value and current_value in options:
            return

        # Se o texto atual ainda existe exatamente, usa como selected_value.
        if current_text and current_text in options:
            dd.selected_value = current_text
            return

        # Se não existe mais, limpa.
        if current_text:
            dd.selected_value = ""
            dd.set_text("")

    def _draw_open_dropdown_list(self, screen, dd):
        options = self._dropdown_options(dd)

        font = self.mode.host.font
        rect = dd.rect

        item_h = 26

        if not options:
            list_rect = pg.Rect(rect.x, rect.bottom + 2, rect.w, item_h)
            pg.draw.rect(screen, (255, 255, 255), list_rect)
            pg.draw.rect(screen, (80, 80, 90), list_rect, 1)

            txt = font.render("Nenhum arquivo encontrado", True, (120, 120, 120))
            screen.blit(txt, (list_rect.x + 6, list_rect.y + 5))
            return

        max_items = min(self.max_dropdown_items, len(options))

        max_scroll = max(0, len(options) - max_items)
        self.dropdown_scroll = max(0, min(self.dropdown_scroll, max_scroll))

        start = self.dropdown_scroll
        end = start + max_items
        visible_options = options[start:end]

        list_rect = pg.Rect(
            rect.x,
            rect.bottom + 2,
            rect.w,
            item_h * max_items,
        )

        pg.draw.rect(screen, (255, 255, 255), list_rect)
        pg.draw.rect(screen, (80, 80, 90), list_rect, 1)

        mouse_pos = pg.mouse.get_pos()

        for i, opt in enumerate(visible_options):
            item_rect = pg.Rect(
                rect.x,
                rect.bottom + 2 + i * item_h,
                rect.w,
                item_h,
            )

            if item_rect.collidepoint(mouse_pos):
                pg.draw.rect(screen, (225, 235, 255), item_rect)

            txt = font.render(str(opt), True, (25, 25, 25))
            screen.blit(txt, (item_rect.x + 6, item_rect.y + 5))

        # Indicador simples de scroll
        if len(options) > max_items:
            indicator = f"{start + 1}-{min(end, len(options))}/{len(options)}"
            txt = font.render(indicator, True, (90, 90, 90))
            screen.blit(txt, (list_rect.right - txt.get_width() - 6, list_rect.bottom - item_h + 5))

    def is_real(self):
        return self._get_dropdown_value(self.dd_source) == "Real (encoder + UWB)"

    def is_odometry(self):
        return self._get_dropdown_value(self.dd_source) == "Real (odometria 2026)"

    def is_odom_uwb(self):
        return self._get_dropdown_value(self.dd_source) == "Real (PCNT + UWB 2026)"

    def active_rows(self):
        if self.is_odom_uwb():
            return self.rows_odom_uwb
        if self.is_odometry():
            return self.rows_odometry
        if self.is_real():
            return self.rows_real_bc
        return (
            self.rows_sim_bc
            if self._get_dropdown_value(self.dd_sim_kind)=="BC"
            else self.rows_sim
        )

    def draw(self):
        screen = self.mode.host.screen
        font = self.mode.host.font
        bigfont = self.mode.host.bigfont

        rows = self.active_rows()
        h = 150 + len(rows) * 60

        self.frame = ModalFrame.centered(
            screen,
            width=780,
            height=h,
            title="Configurar Dataset",
            font=font,
            bigfont=bigfont,
            y_offset=-20,
        )

        self.frame.draw(screen)

        x, y = self.frame.content_origin(left=28, top=78)

        # Desenha primeiro os dropdowns fechados.
        opened_rows = []

        for i, row in enumerate(rows):
            row.set_position(x, y + i * 60)
            row.draw(screen, font)

            if row.opened:
                opened_rows.append(row)

        # Botões no rodapé.
        by = self.frame.rect.bottom - 48
        self.buttons.set_position(self.frame.rect.right - 24, by)
        self.buttons.draw(screen)

        # Dropdown aberto deve ser desenhado por último para ficar por cima.
        if self.open_dropdown is not None:
            self._draw_open_dropdown_list(screen, self.open_dropdown)

    def handle_event(self, event):
        if not self.mode.dataset_modal_open:
            return False

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                if self.open_dropdown is not None:
                    self._close_open_dropdown()
                else:
                    self.close()
                return True

            if event.key == pg.K_RETURN:
                self.apply()
                return True

            return True

        if event.type == pg.MOUSEWHEEL:
            if self.open_dropdown is not None:
                options = self._dropdown_options(self.open_dropdown)
                max_scroll = max(0, len(options) - self.max_dropdown_items)

                self.dropdown_scroll -= int(event.y)
                self.dropdown_scroll = max(0, min(self.dropdown_scroll, max_scroll))

                return True
            
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos

            # 1) Se havia lista aberta, clique nela tem prioridade.
            if self.open_dropdown is not None:
                if self._handle_open_dropdown_click(pos):
                    return True

                # Clique fora da lista fecha a lista, mas ainda pode abrir outro campo.
                self._close_open_dropdown()

            # 2) Botões.
            if self.buttons.hit_ok(pos):
                self.apply()
                return True

            if self.buttons.hit_cancel(pos):
                self.close()
                return True

            # 3) Clique em algum dropdown.
            clicked_dropdown = None

            for row in self.active_rows():
                dd = row.dropdown

                if dd.rect.collidepoint(pos):
                    clicked_dropdown = dd
                    break

            # Fecha todos primeiro.
            for row in self.active_rows():
                dd = row.dropdown
                dd.active = False
                dd.dropdown_open = False

            # Abre o clicado.
            if clicked_dropdown is not None:
                clicked_dropdown.active = True
                clicked_dropdown.dropdown_open = True
                self.open_dropdown = clicked_dropdown
                self.dropdown_scroll = 0
                return True
            
            # 4) Clique fora fecha o modal.
            if self.frame and not self.frame.contains(pos):
                self.close()
                return True

            return True

        return False

    def _dropdown_options(self, dd):
        return list(getattr(dd, "options_all", None) or getattr(dd, "options", []) or [])


    def _dropdown_list_rect(self, dd):
        options = self._dropdown_options(dd)

        if not options:
            return None

        item_h = 26
        max_items = min(self.max_dropdown_items, len(options))

        return pg.Rect(
            dd.rect.x,
            dd.rect.bottom + 2,
            dd.rect.w,
            item_h * max_items,
        )


    def _handle_open_dropdown_click(self, pos):
        dd = self.open_dropdown

        if dd is None:
            return False

        options = self._dropdown_options(dd)
        list_rect = self._dropdown_list_rect(dd)

        if list_rect is None:
            self._close_open_dropdown()
            return False

        if list_rect.collidepoint(pos):
            item_h = 26
            local_idx = int((pos[1] - list_rect.y) // item_h)

            max_items = min(self.max_dropdown_items, len(options))
            real_idx = self.dropdown_scroll + local_idx

            if 0 <= local_idx < max_items and 0 <= real_idx < len(options):
                full_value = str(options[real_idx])

                self._set_dropdown_value(dd, full_value)

                dd.active = False
                dd.dropdown_open = False
                self.open_dropdown = None
                self.dropdown_scroll = 0

                if dd is self.dd_preprocessed_uwb and full_value=="(Escolher relatório...)":
                    # Seleção de sidecar externo: não tentar abrir um CSV
                    # como PCNT. O motor validará o par CSV+JSON depois.
                    from src.ui.algo_modes.preprocessing_modal import _select_native_file
                    from pathlib import Path
                    file=_select_native_file(sidecar=True)
                    if file and Path(file).name.endswith(".preproc.json"):
                        self._set_dropdown_value(self.dd_preprocessed_uwb,file)
                    else:
                        self._set_dropdown_value(self.dd_preprocessed_uwb,"(Nenhum)")
                    return True

                if dd is self.dd_source:
                    self.refresh_options()
                    # Evita reutilizar a rota dos experimentos simulados
                    # como se fosse referência física do robô 2026.
                    if self.is_odometry() or self.is_odom_uwb():
                        if self.is_odom_uwb():
                            self._set_dropdown_value(self.dd_odom_session, "1")
                        self._set_dropdown_value(self.dd_route, "(Nenhuma)")
                        self._set_dropdown_value(self.dd_map, "(Nenhum)")

                return True

        return False

    def _close_open_dropdown(self):
        """
        Fecha com segurança o dropdown atualmente aberto.
        """
        dd = self.open_dropdown

        if dd is not None:
            dd.dropdown_open = False
            dd.active = False

        self.open_dropdown = None
        self.dropdown_scroll = 0

    def apply(self):
        """
        Copia a seleção do modal para o DatasetMode e chama a aplicação real.
        Usa selected_value para evitar truncamento do nome dos arquivos.
        """
        m = self.mode

        is_real = self.is_real()
        is_odometry = self.is_odometry()
        is_odom_uwb = self.is_odom_uwb()
        m.dataset_source_type = (
            "real_odom_uwb_2026" if is_odom_uwb else
            "real_odometry_2026" if is_odometry else
            "real_encoder_uwb" if is_real else "simulated"
        )

        if is_odometry or is_odom_uwb:
            m._modal_odom_file = self._get_dropdown_value(self.dd_odom)
            m._modal_odom_session = self._get_dropdown_value(self.dd_odom_session) or "1"
            m._modal_preproc_sidecar = (
                self._get_dropdown_value(self.dd_preprocessed_uwb)
                if is_odom_uwb else "(Nenhum)"
            )
            m._modal_dataset_file = ""
            m._modal_real_encoder_file = ""
            m._modal_real_uwb_file = ""
        elif is_real:
            m._modal_real_encoder_file = self._get_dropdown_value(self.dd_encoder)
            m._modal_real_uwb_file = self._get_dropdown_value(self.dd_uwb)
            m._modal_dataset_file = ""
        else:
            m._modal_dataset_file = self._get_dropdown_value(self.dd_dataset)
            m._modal_real_encoder_file = ""
            m._modal_real_uwb_file = ""

            selected_kind = self._get_dropdown_value(self.dd_sim_kind) or "Front"
            m._modal_simulated_kind = selected_kind
            m.simulated_dataset_kind = selected_kind

        tags_selected=self._get_dropdown_value(self.dd_tag_preset)
        m._modal_tag_preset={
            "Auto (JSON / legado)":"auto",
            "Frente / trás":"front_rear",
            "Esquerda / direita":"left_right",
            "Personalizado (JSON)":"custom",
        }.get(tags_selected,"auto")
        m._modal_anchor_file = self._get_dropdown_value(self.dd_anchors)
        route_value = self._get_dropdown_value(self.dd_route)
        map_value = self._get_dropdown_value(self.dd_map)
        m._modal_route_file = "" if route_value == "(Nenhuma)" else route_value
        m._modal_map_file = "" if map_value == "(Nenhum)" else map_value

        ok = m._apply_dataset_config_from_modal_values()

        if ok is not False:
            self.close()

    def _set_dropdown_value(self, dd, value: str):
        """
        Define o valor real selecionado no dropdown.

        O TextBoxDropdown pode cortar o texto exibido para caber no campo.
        Por isso guardamos o valor completo em selected_value.
        """
        value = str(value or "")

        dd.selected_value = value
        dd.set_text(value)


    def _get_dropdown_value(self, dd) -> str:
        """
        Recupera o valor real selecionado.

        Prioridade:
        1. selected_value completo;
        2. texto atual, caso o usuário tenha digitado manualmente.
        """
        selected = getattr(dd, "selected_value", None)

        if selected:
            return str(selected).strip()

        return str(getattr(dd, "text", "") or "").strip()
    