"""Janela PyGame para análise e exportação UWB independente do batch.

A escolha de arquivos usa diálogo nativo do SO (tkinter carregado apenas
no clique). Analisar e Salvar são ações DISTINTAS: nenhum resultado
modifica silenciosamente o Dataset, o perfil físico ou a fonte bruta.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pygame as pg

from src.ui.botton import Button
from src.uwb.preprocessing import UwbPreprocessConfig, load_uwb_preprocess_config
from src.uwb.preprocessing_workflow import (
    execute_preprocessing, load_preprocessing_input, save_preprocessed_dataset,
)


def _select_native_file(*, save=False, initialdir=None, initialfile=None):
    """Diálogo do sistema, sem importar Tk nem abrir janela antes do clique."""
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk()
    root.withdraw()
    root.attributes("-topmost",True)
    try:
        options={
            "parent":root,
            "initialdir":str(initialdir or Path.cwd()),
        }
        if save:
            options.update({
                "title":"Salvar cópia pré-processada (CSV)",
                "defaultextension":".csv",
                "filetypes":[("CSV de ranges e desvios","*.csv")],
                "initialfile":str(initialfile or "dataset_preprocessado.csv"),
            })
            return filedialog.asksaveasfilename(**options) or None
        options.update({
            "title":"Escolher dados UWB (PCNT ou ranges)",
            "filetypes":[
                ("Dados UWB","*.csv *.txt *.jsonl"),
                ("CSV","*.csv"),
                ("TXT","*.txt"),
                ("JSONL","*.jsonl"),
                ("Todos os arquivos","*.*"),
            ],
        })
        return filedialog.askopenfilename(**options) or None
    finally:
        root.destroy()


class UwbPreprocessingModal:
    """Wizard simples de fonte -> política -> relatório -> salvar opcional."""

    def __init__(self, mode):
        self.mode=mode
        self.opened=False
        self.source_path=""
        self.anchor_path=""
        self.source_type="pcnt"
        self.session_id=1
        self.policy=UwbPreprocessConfig(mode="flag")
        self.last_result=None
        self.last_report=None
        self.message=""
        self.scroll=0
        font=mode.host.font
        self.buttons={
            "select":Button((0,0,190,32),"Escolher arquivo",font,bg=(236,242,255)),
            "current":Button((0,0,190,32),"Usar arquivo atual",font),
            "anchors":Button((0,0,190,32),"Escolher âncoras",font,bg=(238,242,255)),
            "kind":Button((0,0,195,32),"Formato: PCNT",font),
            "session":Button((0,0,120,32),"Sessão: 1",font),
            "policy":Button((0,0,170,32),"Política: flag",font),
            "analyze":Button((0,0,160,34),"Analisar",font,bg=(230,250,235),fg=(0,105,50)),
            "save":Button((0,0,180,34),"Salvar cópia CSV",font,bg=(232,245,255),fg=(35,67,135)),
            "close":Button((0,0,100,34),"Fechar",font),
        }
        self.rect=None

    def _default_current(self):
        mode=self.mode
        if getattr(mode,"dataset_source_type","")=="real_odom_uwb_2026":
            data=getattr(mode,"_robot2026_data",None)
            if data is not None:
                return str(data.path),"pcnt",int(data.session_id)
        path=getattr(mode,"_dataset_path",None)
        if path and Path(path).is_file():
            return str(path),"ranges",1
        return "", "ranges", 1

    def _reset_result(self):
        self.last_report=None
        self.last_result=None
        self.scroll=0

    def open(self):
        self.opened=True
        self.source_path,self.source_type,self.session_id=self._default_current()
        self.policy=UwbPreprocessConfig(mode="flag")
        anchors=getattr(self.mode,"_anchors_path",None)
        self.anchor_path=str(anchors or "")
        if anchors and str(anchors).lower().endswith(".json") and Path(anchors).is_file():
            try:
                saved=load_uwb_preprocess_config(anchors)
                if saved is not None:
                    self.policy=replace(saved,mode="flag" if saved.mode=="off" else saved.mode)
            except (OSError,ValueError) as error:
                self.message=f"Configuração do pré-processamento inválida: {error}"
            else:
                self.message=""
        else:
            self.message="Selecione o JSON de âncoras em Configurar Dataset"
        self._reset_result()

    def close(self):
        self.opened=False

    def _source(self):
        if not self.source_path:
            raise ValueError("Escolha um arquivo de dados UWB antes de analisar")
        anchors=self.anchor_path
        if not anchors or not Path(anchors).is_file():
            raise ValueError("Selecione o JSON de âncoras antes de analisar")
        return anchors

    def _choose_file(self):
        default=Path(self.source_path).parent if self.source_path else Path.cwd()
        chosen=_select_native_file(initialdir=default)
        if chosen:
            self.source_path=chosen
            # Selecionar um arquivo não pressupõe o tipo; escolha
            # explicitamente PCNT ou dist/sigma quando necessário.
            if Path(chosen).suffix.lower()==".jsonl":
                self.source_type="ranges"
            self._reset_result()
            self.message="Arquivo selecionado. Confirme formato e política antes de analisar."

    def _choose_anchors(self):
        # Seleção explícita do perfil/geometry SEM trocar o cenário do
        # Dataset Mode: importante ao analisar arquivos de outros ensaios.
        chosen=_select_native_file(
            initialdir=Path(self.anchor_path).parent if self.anchor_path else
                       Path(getattr(self.mode,"anchors_dir","anchor_sets"))
        )
        if not chosen:
            return
        if Path(chosen).suffix.lower()!=".json":
            raise ValueError("Âncoras devem estar em arquivo .json")
        # Só trocar estado local depois de validação; nunca muta Dataset.
        cfg=load_uwb_preprocess_config(chosen)
        self.anchor_path=chosen
        if cfg is not None:
            self.policy=replace(cfg,mode="flag" if cfg.mode=="off" else cfg.mode)
        self._reset_result()
        self.message="Geometria de âncoras selecionada para esta análise."

    def _analyze(self):
        anchor_path=self._source()
        item,anchors=load_preprocessing_input(
            self.source_path,anchor_path,
            source_type=self.source_type,session_id=self.session_id,
        )
        result,report=execute_preprocessing(item,anchors,anchor_path,self.policy)
        self.last_result=result
        self.last_report=report
        self.scroll=0
        self.message=(
            f"Concluído: {report['flagged_cells']} células sinalizadas; "
            f"{report['newly_masked_finite_positive_cells']} mascaradas. "
            "Dados originais intactos."
        )

    def _save(self):
        if self.last_result is None or self.last_report is None:
            raise ValueError("Execute Analisar antes de salvar")
        directory=Path(getattr(self.mode,"dataset_dir",Path("resultados")/"datasets"))
        directory.mkdir(parents=True,exist_ok=True)
        stem=Path(self.source_path).stem
        proposed=f"{stem}_preproc_{self.policy.mode}.csv"
        target=_select_native_file(
            save=True,initialdir=directory,initialfile=proposed
        )
        if not target:
            return
        csv_path,meta=save_preprocessed_dataset(
            target,self.last_result,self.last_report
        )
        self.message=(
            f"Salvos: {csv_path.name} e {meta.name}. "
            "Para reutilizar: Configurar Dataset > Simulado."
        )

    def _place(self):
        screen=self.mode.host.screen
        width=min(830,screen.get_width()-40)
        height=min(620,screen.get_height()-40)
        self.rect=pg.Rect(
            (screen.get_width()-width)//2,
            (screen.get_height()-height)//2,
            width,height,
        )
        r=self.rect
        x,y=r.x+24,r.y+100
        self.buttons["select"].rect.topleft=(x,y)
        self.buttons["current"].rect.topleft=(x+200,y)
        self.buttons["anchors"].rect.topleft=(x+400,y)
        self.buttons["kind"].rect.topleft=(x,y+46)
        self.buttons["session"].rect.topleft=(x+205,y+46)
        self.buttons["policy"].rect.topleft=(x+340,y+46)
        footer=r.bottom-55
        self.buttons["analyze"].rect.topleft=(x,footer)
        self.buttons["save"].rect.topleft=(x+175,footer)
        self.buttons["close"].rect.topleft=(r.right-124,footer)
        self.buttons["kind"].text=(
            "Formato: PCNT bruto" if self.source_type=="pcnt"
            else "Formato: dist/sigma"
        )
        self.buttons["session"].text=f"Sessão: {self.session_id}"
        self.buttons["policy"].text=f"Política: {self.policy.mode}"

    @staticmethod
    def _fit(text,font,max_width):
        """Corta o final do texto sem redimensionar (fonte nítida)."""
        string=str(text)
        if font.size(string)[0]<=max_width:
            return string
        suffix="..."
        lo,hi=0,len(string)
        while lo<hi:
            mid=(lo+hi+1)//2
            if font.size(string[:mid]+suffix)[0]<=max_width:
                lo=mid
            else:
                hi=mid-1
        return string[:lo]+suffix

    def _draw_line(self,screen,font,text,x,y,color=(37,43,56),max_width=None):
        width=max_width or (self.rect.right-x-24)
        txt=font.render(self._fit(text,font,width),True,color)
        screen.blit(txt,(x,y))

    def _report_lines(self):
        r=self.last_report
        if r is None:
            return [
                "Selecione um CSV PCNT bruto ou dataset dist/sigma.",
                "Selecione também a geometria de âncoras correspondente.",
                "Formato PCNT: mantém apenas snapshots UWB novos suficientes.",
                "Formato dist/sigma: arquivo com pares de colunas (m).",
                "Clique em Analisar para ver contagens; Salvar é opcional.",
                "",
                "O arquivo original e o batch atual nunca são modificados.",
            ]
        lines=[
            f"Arquivo: {r['source_filename']} | SHA-256: {r['source_sha256'][:12]}...",
            f"Amostras: {r['rows']} | Tags: {r['tags']} | Âncoras: {r['anchors']}",
            f"Política: {r['config']['mode']} | Células: {r['total_cells']}",
            f"Sinalizadas: {r['flagged_cells']} | Válidas mascaradas: {r['newly_masked_finite_positive_cells']}",
            f"Linhas com alerta: {r['rows_with_any_flag']}",
            "Motivos (um range pode ter mais de uma sinalização):",
        ]
        for name,count in r["flag_counts"].items():
            lines.append(f"  {name}: {count}")
        lines.append("Sinalizações por âncora (índice 0..N-1):")
        by_anchor=r["per_anchor_flagged_cells"]
        for start in range(0,len(by_anchor),5):
            lines.append("  "+",  ".join(
                f"A{j}: {by_anchor[j]}" for j in range(start,min(start+5,len(by_anchor)))
            ))
        if r["first_30_flagged_cells"]:
            lines.append("Primeiros eventos: PCNT/index / tag / âncora / motivo")
            for e in r["first_30_flagged_cells"][:12]:
                lines.append(
                    f"  {e['source_sample_index']} / T{e['tag_index']} / "
                    f"A{e['anchor_index']} / {','.join(e['flags'])}"
                )
        else:
            lines.append("Nenhuma medição sinalizada com os critérios atuais.")
        lines.extend(r["source_notes"])
        lines.append("O CSV salvo usa só ranges/sigmas; não contém odometria.")
        return lines

    def draw(self):
        if not self.opened:
            return
        self._place()
        screen=self.mode.host.screen
        font=self.mode.host.font
        bold=self.mode.host.bigfont
        overlay=pg.Surface(screen.get_size(),pg.SRCALPHA)
        overlay.fill((10,20,40,145))
        screen.blit(overlay,(0,0))
        pg.draw.rect(screen,(252,253,255),self.rect,border_radius=8)
        pg.draw.rect(screen,(65,84,117),self.rect,2,border_radius=8)
        self._draw_line(screen,bold,"Pré-processamento UWB",self.rect.x+24,self.rect.y+16)
        self._draw_line(
            screen,font,
            "Escolha um arquivo e uma política. Prévia antes de salvar; sem sobrescrever dados.",
            self.rect.x+24,self.rect.y+55,
        )
        self._draw_line(
            screen,font,
            "Arquivo: "+(self.source_path or "(nenhum)") +
            " | Âncoras: "+(Path(self.anchor_path).name if self.anchor_path else "(nenhuma)"),
            self.rect.x+24,self.rect.y+78,
        )
        for button in self.buttons.values():
            button.draw(screen)
        status_y=self.rect.y+185
        if self.message:
            self._draw_line(screen,font,self.message,self.rect.x+24,status_y,(26,102,71))
        # Caixa de relatório rolável, sem esticar ou comprimir caracteres.
        report=pg.Rect(self.rect.x+20,self.rect.y+213,self.rect.w-40,self.rect.h-280)
        pg.draw.rect(screen,(247,249,253),report,border_radius=4)
        pg.draw.rect(screen,(182,191,207),report,1,border_radius=4)
        lines=self._report_lines()
        available=max(1,report.h//23-1)
        self.scroll=max(0,min(self.scroll,max(0,len(lines)-available)))
        old_clip=screen.get_clip()
        try:
            screen.set_clip(old_clip.clip(report.inflate(-8,-8)))
            for i,line in enumerate(lines[self.scroll:self.scroll+available]):
                self._draw_line(
                    screen,font,line,report.x+12,report.y+8+i*23,
                    max_width=report.w-24,
                )
        finally:
            screen.set_clip(old_clip)
        if len(lines)>available:
            self._draw_line(
                screen,font,f"{self.scroll+1}-{min(len(lines),self.scroll+available)} / {len(lines)}",
                report.right-100,report.bottom-23,(100,100,110),
                max_width=90,
            )

    def handle_event(self,event):
        """Consome TODOS os eventos enquanto o modal está aberto."""
        if not self.opened:
            return False
        if event.type==pg.KEYDOWN and event.key==pg.K_ESCAPE:
            self.close()
            return True
        if event.type==pg.MOUSEWHEEL:
            self.scroll=max(0,self.scroll-int(event.y)*2)
            return True
        if event.type!=pg.MOUSEBUTTONDOWN or event.button!=1:
            return True
        self._place()
        pos=event.pos
        try:
            if self.buttons["close"].hit(pos):
                self.close()
            elif self.buttons["select"].hit(pos):
                self._choose_file()
            elif self.buttons["current"].hit(pos):
                self.source_path,self.source_type,self.session_id=self._default_current()
                self._reset_result()
                self.message="Arquivo atual selecionado."
            elif self.buttons["anchors"].hit(pos):
                self._choose_anchors()
            elif self.buttons["kind"].hit(pos):
                self.source_type="ranges" if self.source_type=="pcnt" else "pcnt"
                self._reset_result()
                self.message="Formato alterado. Clique Analisar."
            elif self.buttons["session"].hit(pos):
                self.session_id=1 if self.session_id>=10 else self.session_id+1
                self._reset_result()
            elif self.buttons["policy"].hit(pos):
                order=("off","flag","mask")
                self.policy=replace(
                    self.policy,mode=order[(order.index(self.policy.mode)+1)%len(order)]
                )
                self._reset_result()
                self.message=f"Política alterada para {self.policy.mode}."
            elif self.buttons["analyze"].hit(pos):
                self._analyze()
            elif self.buttons["save"].hit(pos):
                self._save()
        except (ValueError,OSError,TypeError,KeyError,RuntimeError,ImportError) as exc:
            self.message=f"Pré-processamento: {exc}"
            self.mode.host._set_msg(self.message)
        return True
