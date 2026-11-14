"""Execução independente de pré-processamento para a interface Dataset.

Fluxo separado de run_batch: escolher arquivo -> analisar -> salvar uma
CÓPIA explícita. O programa nunca sobrescreve CSVs brutos/JSONs do
usuário nem substitui silenciosamente as matrizes já carregadas.

Compatibilidade:
- PCNT bruto 2026: usa mapeamento SPI do JSON e preserva índices PCNT.
- CSV/TXT intercalado [range,sigma] por âncora/tag, em metros.
- JSONL legado z_k: seleciona explicitamente a tag frontal (sem sigma).
- CSV exportado pode voltar a ser carregado pelo dataset simulado com
  o mesmo layout. Um dataset extraído de PCNT não contém odometria:
  para fusão, continuar utilizando o PCNT bruto e sua proveniência.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from src.uwb.preprocessing import UwbPreprocessConfig, preprocess_uwb_batch


@dataclass(frozen=True)
class PreprocessingInput:
    """Medições normalizadas, fonte original e alinhamento."""
    path: Path
    source_type: str
    distances: np.ndarray
    deviations: np.ndarray | None
    sample_indices: np.ndarray
    timestamps_s: np.ndarray | None
    notes: tuple[str, ...] = ()


def _read_interleaved(path, anchor_count):
    """CSV/TXT da plataforma: dist,sigma,dist,sigma... por âncora.

    O formato com duas tags usa 2*N pares na ordem:
    A0_front,A0_rear,A1_front,A1_rear,...; não assume que a
    primeira linha é cabeçalho e não descarta silenciosamente dados.
    """
    lines = []
    width = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        delimiter = ";" if ";" in line else "," if "," in line else None
        columns = line.split(delimiter) if delimiter else line.split()
        if not columns:
            continue
        try:
            values = [float(token.strip()) for token in columns]
        except ValueError as exc:
            raise ValueError(
                f"Linha {line_number} não numérica; selecione arquivo dist/sigma sem cabeçalho"
            ) from exc
        if width is None:
            width = len(values)
            if width not in (2*anchor_count,4*anchor_count):
                raise ValueError(
                    f"Esperadas {2*anchor_count} (1 tag) ou {4*anchor_count} "
                    f"(2 tags) colunas dist/sigma; encontrado {width}. "
                    "Para CSV PCNT escolha o formato PCNT bruto."
                )
        if len(values) != width:
            raise ValueError(f"Linha {line_number}: {len(values)} colunas; esperadas {width}")
        lines.append(values)
    if not lines:
        raise ValueError("Arquivo de medições sem registros numéricos")
    data = np.asarray(lines, dtype=float)
    distances, sigmas = data[:, 0::2], data[:, 1::2]
    if distances.shape[1] == anchor_count:
        return distances, sigmas
    count = len(data)
    return (
        distances.reshape(count,anchor_count,2).transpose(0,2,1),
        sigmas.reshape(count,anchor_count,2).transpose(0,2,1),
    )


def _read_jsonl_front(path, anchor_count):
    """Formato do simulador legado z_k; documenta perda da tag traseira."""
    frames = []
    has_rear = False
    for line_number,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL inválido na linha {line_number}") from exc
        raw = data.get("z_k")
        if not isinstance(raw, list):
            raise ValueError(f"JSONL linha {line_number}: esperado z_k em lista")
        if len(raw) == 2*anchor_count:
            frames.append([raw[2*i] for i in range(anchor_count)])
            has_rear = True
        elif len(raw) == anchor_count:
            frames.append(raw)
        else:
            raise ValueError(
                f"JSONL linha {line_number}: z_k precisa ter N ou 2*N ranges"
            )
    if not frames:
        raise ValueError("JSONL sem medições z_k")
    return np.asarray(frames,dtype=float), (
        ("z_k com duas tags: somente frontal exportada (padrão legado)",)
        if has_rear else ()
    )


def load_preprocessing_input(path, anchors_path, *, source_type, session_id=1):
    """Lê arquivo escolhido sem modificar a fonte, no mesmo layout XYZ."""
    source = Path(path)
    config_path = Path(anchors_path)
    if not source.is_file():
        raise ValueError(f"Arquivo de medições inexistente: {source}")
    if not config_path.is_file():
        raise ValueError("Selecione antes um JSON de âncoras válido no Dataset")
    # Leitura mínima de geometria: sem importar Pygame/UI no motor
    # de análise. Mantém a convenção do loader compartilhado: Nx2
    # pode usar anchor_height_m/anchor_z_m/z_anchor, default 1.0 m.
    settings=json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(settings,dict):
        raise ValueError("JSON de âncoras deve conter objeto")
    anchors=np.asarray(
        settings.get("anchors_xy",settings.get("anchors",[])),dtype=float
    )
    if anchors.ndim!=2 or anchors.shape[1] not in (2,3) or len(anchors)<1:
        raise ValueError("Pré-processamento requer geometria Nx2 ou Nx3")
    if anchors.shape[1]==2:
        height=float(settings.get(
            "anchor_height_m",settings.get(
                "anchor_z_m",settings.get("z_anchor",1.0)
            )
        ))
        anchors=np.column_stack((anchors,np.full(len(anchors),height)))
    if not np.isfinite(anchors).all():
        raise ValueError("Geometria de âncoras contém valores não finitos")
    n = len(anchors)
    if source_type=="pcnt":
        from src.ui.algo_modes.dataset_uwb2026_pipeline import (
            load_robot2026_single_tag_uwb,
            read_pcnt_anchor_slot_mapping,
            read_robot2026_profile_from_anchor_json,
            read_uwb2026_processing_settings,
        )
        mapping = read_pcnt_anchor_slot_mapping(config_path,n)
        data = load_robot2026_single_tag_uwb(
            source,session_id=int(session_id),anchor_count=n,
            slot_to_anchor_index=mapping,
            profile=read_robot2026_profile_from_anchor_json(config_path),
            **read_uwb2026_processing_settings(config_path),
        )
        inp = PreprocessingInput(
            path=source,source_type="pcnt",
            distances=data.distances.copy(), deviations=data.deviations.copy(),
            sample_indices=data.sample_indices.copy(),
            timestamps_s=data.timestamps_s.copy(),
            notes=(
                f"PCNT sessao {session_id}: indices originais preservados",
                "Dataset exportado contém somente UWB, sem odometria PCNT",
            ),
        )
    elif source_type=="ranges":
        if source.suffix.lower()==".jsonl":
            d,supplement=_read_jsonl_front(source,n)
            sigmas=None
            notes=supplement
        elif source.suffix.lower() in (".csv",".txt"):
            d,sigmas=_read_interleaved(source,n)
            notes=()
        else:
            raise ValueError("Formato suportado: CSV/TXT intercalado ou JSONL z_k")
        inp=PreprocessingInput(
            path=source, source_type="ranges",distances=d,
            deviations=sigmas, sample_indices=np.arange(len(d),dtype=np.int64),
            timestamps_s=None,notes=notes,
        )
    else:
        raise ValueError("source_type deve ser pcnt ou ranges")
    return inp,anchors


def execute_preprocessing(input_data, anchors, anchor_config_path, config):
    """Executa política e retorna resultado + relatório detalhado imutável.

    'flag' não altera arrays; 'mask' preserva linha e coluna e substitui
    uma célula por NaN. Um report de auditoria NÃO substitui ground truth.
    """
    cfg=(UwbPreprocessConfig.from_dict(config) if isinstance(config,dict) else config)
    result=preprocess_uwb_batch(
        anchors,input_data.distances,input_data.deviations,
        config=cfg,sample_indices=input_data.sample_indices,
        timestamps_s=input_data.timestamps_s,
    )
    bits={
        1:"range_nao_finito",2:"range_nao_positivo",
        4:"acima_max_range",8:"abaixo_min_vertical",
        16:"sigma_invalido",32:"sigma_excessivo",
        64:"possivelmente_repetida",
    }
    examples=[]
    for position in np.argwhere(result.flags!=0)[:30]:
        coords=tuple(int(x) for x in position)
        row, anchor=coords[0],coords[-1]
        tag=coords[1] if len(coords)==3 else 0
        reason=int(result.flags[coords])
        examples.append({
            "row":row,
            "source_sample_index":int(result.sample_indices[row]),
            "timestamp_s":(
                float(result.timestamps_s[row])
                if result.timestamps_s is not None else None
            ),
            "tag_index":tag,"anchor_index":anchor,
            "distance_original_m":(
                float(input_data.distances[coords])
                if np.isfinite(input_data.distances[coords]) else None
            ),
            "flags":[name for bit,name in bits.items() if reason & bit],
        })
    source=input_data.path
    anchor_path=Path(anchor_config_path)
    report={
        **result.summary,
        "source_type":input_data.source_type,
        "source_filename":source.name,
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "anchor_filename":anchor_path.name,
        "anchor_sha256":hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "source_notes":list(input_data.notes),
        "first_30_flagged_cells":examples,
        "original_sample_indices":result.sample_indices.tolist(),
        "original_timestamps_s":(
            result.timestamps_s.tolist() if result.timestamps_s is not None else None
        ),
        "usage_note":(
            "A saída CSV serve aos algoritmos UWB estáticos. "
            "Não contém encoder/ground truth; para fusão posterior "
            "use o PCNT original e índices/timestamps deste relatório."
        ),
    }
    return result,report


def save_preprocessed_dataset(path,result,report):
    """Salva CSV dist/sigma legível pelo Dataset e sidecar JSON, sem overwrite.

    CSV intercalado, sem cabeçalho, como espera load_sim_txt_dataset().
    Com duas tags as colunas são A0_front,A0_rear,A1_front,A1_rear.
    Não infere sigma: datasets sem sigma são exportados com NaN,
    explicitamente anotado, e podem não servir aos solvers ponderados.
    """
    dest=Path(path)
    if dest.suffix.lower()!=".csv":
        raise ValueError("Arquivo exportado deve terminar em .csv")
    meta=dest.with_suffix(".preproc.json")
    if dest.resolve()==meta.resolve() or dest.exists() or meta.exists():
        raise FileExistsError("Destino CSV ou relatório já existe; escolha outro nome")
    matrix=np.asarray(result.distances,dtype=float)
    if matrix.ndim==3:
        m,t,n=matrix.shape
        distances=matrix.transpose(0,2,1).reshape(m,n*t)
        sigma=(result.deviations.transpose(0,2,1).reshape(m,n*t)
               if result.deviations is not None else None)
    else:
        distances=matrix
        sigma=result.deviations
    if sigma is None:
        sigma=np.full_like(distances,np.nan)
    paired=np.empty((len(distances),2*distances.shape[1]),dtype=float)
    paired[:,0::2]=distances
    paired[:,1::2]=sigma
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    created=[]
    try:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("x",encoding="utf-8",newline="") as handle:
            created.append(dest)
            writer=csv.writer(handle,delimiter=",",lineterminator="\n")
            writer.writerows(paired)
        with meta.open("x",encoding="utf-8") as handle:
            created.append(meta)
            handle.write(payload)
    except Exception:
        for file in reversed(created):
            file.unlink(missing_ok=True)
        raise
    return dest,meta
