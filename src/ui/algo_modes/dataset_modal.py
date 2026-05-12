from __future__ import annotations

import pygame as pg
import os

from src.ui.ui_elements import (
    TextBoxDropdown,
    ModalFrame,
    FormDropdownRow,
    ModalButtonBar,
)


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
            options=["Simulado", "Real (encoder + UWB)"],
        )

        self.dd_sim_kind = TextBoxDropdown(
            pg.Rect(0, 0, 440, 30),
            font,
            options=["Front", "Rear", "Mid", "BC"],
        )

        self.dd_dataset = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
        self.dd_encoder = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
        self.dd_uwb = TextBoxDropdown(pg.Rect(0, 0, 440, 30), font, options=[])
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

        self.rows_real = [
            FormDropdownRow("Fonte:", self.dd_source),
            FormDropdownRow("Encoder real:", self.dd_encoder),
            FormDropdownRow("UWB real:", self.dd_uwb),
            FormDropdownRow("Âncoras:", self.dd_anchors),
            FormDropdownRow("Rota ref.:", self.dd_route),
            FormDropdownRow("Mapa:", self.dd_map),
        ]

        self.open_dropdown = None
        self.max_dropdown_items = 7
        
        self.refresh_options()

    def open(self):
        self.refresh_options()

        # Sincroniza estado atual do DatasetMode para os dropdowns.
        source = getattr(self.mode, "dataset_source_type", "simulated")
        if source == "real_encoder_uwb":
            self.dd_source.set_text("Real (encoder + UWB)")
        else:
            self.dd_source.set_text("Simulado")

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

        self.dd_sim_kind.set_text(kind_map.get(str(kind).strip().lower(), "Front"))

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
            if not dirname:
                return []

            try:
                if not os.path.isdir(dirname):
                    return []

                files = []

                for name in os.listdir(dirname):
                    full = os.path.join(dirname, name)

                    if not os.path.isfile(full):
                        continue

                    if name.lower().endswith(tuple(e.lower() for e in exts)):
                        files.append(name)

                return sorted(files, key=str.lower)

            except Exception:
                return []

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
            ["Simulado", "Real (encoder + UWB)"],
        )

        self._set_dropdown_options(
            self.dd_sim_kind,
            ["Front", "Rear", "Mid", "BC"],
        )

        self._set_dropdown_options(
            self.dd_dataset,
            safe_list(dataset_dir, (".txt", ".csv", ".jsonl")),
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

        self._set_dropdown_options(
            self.dd_anchors,
            safe_list(anchors_dir, (".json",)),
        )

        self._set_dropdown_options(
            self.dd_route,
            safe_list(routes_dir, (".json",)),
        )

        self._set_dropdown_options(
            self.dd_map,
            safe_list(maps_dir, (".json",)),
        )

    def _set_dropdown_options(self, dd, options):
        options = list(options or [])

        dd.options = options
        dd.options_all = options
        dd.options_filtered = options

        # Se o texto atual não existe mais nas opções, limpa.
        if getattr(dd, "text", "") and dd.text not in options:
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
        list_rect = pg.Rect(
            rect.x,
            rect.bottom + 2,
            rect.w,
            item_h * max_items,
        )

        pg.draw.rect(screen, (255, 255, 255), list_rect)
        pg.draw.rect(screen, (80, 80, 90), list_rect, 1)

        mouse_pos = pg.mouse.get_pos()

        for i, opt in enumerate(options[:max_items]):
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

    def is_real(self):
        return (self.dd_source.text or "").strip().lower().startswith("real")

    def active_rows(self):
        return self.rows_real if self.is_real() else self.rows_sim

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
            idx = int((pos[1] - list_rect.y) // item_h)

            if 0 <= idx < min(self.max_dropdown_items, len(options)):
                dd.set_text(str(options[idx]))

                # Fecha usando a referência local, não self.open_dropdown depois de zerar.
                dd.active = False
                dd.dropdown_open = False
                self.open_dropdown = None

                if dd is self.dd_source:
                    self.refresh_options()

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

    def apply(self):
        """
        Copia a seleção do modal para o DatasetMode e chama a aplicação real.
        """
        m = self.mode

        is_real = self.is_real()
        m.dataset_source_type = "real_encoder_uwb" if is_real else "simulated"

        if is_real:
            m._modal_real_encoder_file = self.dd_encoder.text.strip()
            m._modal_real_uwb_file = self.dd_uwb.text.strip()
            m._modal_dataset_file = ""
        else:
            m._modal_dataset_file = self.dd_dataset.text.strip()
            m._modal_real_encoder_file = ""
            m._modal_real_uwb_file = ""

            selected_kind = self.dd_sim_kind.text.strip() or "Front"
            m._modal_simulated_kind = selected_kind
            m.simulated_dataset_kind = selected_kind

        m._modal_anchor_file = self.dd_anchors.text.strip()
        m._modal_route_file = self.dd_route.text.strip()
        m._modal_map_file = self.dd_map.text.strip()

        ok = m._apply_dataset_config_from_modal_values()

        if ok is not False:
            self.close()

    