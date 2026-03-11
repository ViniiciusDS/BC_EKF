# src/uwb/node_params_serialization.py
"""
Funções de serialização/desserialização de NodeParams para JSON.

Permite salvar e carregar configurações completas de clock drift,
delays, bias, etc. junto com as posições das âncoras.
"""
from __future__ import annotations
from typing import Dict, Any
from src.uwb.node_params import NodeParams


def node_params_to_dict(params: NodeParams) -> Dict[str, Any]:
    """
    Converte NodeParams para dicionário JSON-serializável.
    
    Exemplo:
        >>> params = NodeParams(clock=ClockModel(drift_ppm=20.0), ...)
        >>> d = node_params_to_dict(params)
        >>> print(d)
        {'clock': {'drift_ppm': 20.0}, 'ant': {'tx_ns': 0.0, 'rx_ns': 0.0}, ...}
    """
    return {
        "clock": {
            "drift_ppm": float(params.clock.drift_ppm),
        },
        "ant": {
            "tx_ns": float(params.ant.tx_ns),
            "rx_ns": float(params.ant.rx_ns),
        },
        "range_bias_m": float(params.range_bias_m),
    }


def dict_to_node_params(d: Dict[str, Any]) -> NodeParams:
    """
    Converte dicionário JSON para NodeParams.
    
    Usa valores padrão se algum campo estiver ausente.
    
    Exemplo:
        >>> d = {'clock': {'drift_ppm': 20.0}, 'ant': {'tx_ns': 0.0, 'rx_ns': 0.0}}
        >>> params = dict_to_node_params(d)
        >>> print(params.clock.drift_ppm)
        20.0
    """
    from src.uwb.node_params import ClockModel, AntennaDelays
    
    clock_data = d.get("clock", {})
    ant_data = d.get("ant", {})
    
    return NodeParams(
        clock=ClockModel(
            drift_ppm=float(clock_data.get("drift_ppm", 0.0)),
        ),
        ant=AntennaDelays(
            tx_ns=float(ant_data.get("tx_ns", 0.0)),
            rx_ns=float(ant_data.get("rx_ns", 0.0)),
        ),
        range_bias_m=float(d.get("range_bias_m", 0.0)),
    )


def shared_state_to_dict(shared_uwb) -> Dict[str, Any]:
    """
    Exporta SharedUwbState completo para dicionário.
    
    Inclui:
    - Posições das âncoras
    - Parâmetros de cada âncora
    - Parâmetros da tag
    - Seed
    - Metadados
    
    Exemplo:
        >>> data = shared_state_to_dict(shared_uwb)
        >>> with open("config.json", "w") as f:
        >>>     json.dump(data, f, indent=2)
    """
    from datetime import datetime
    
    return {
        "anchors_xy": list(shared_uwb.anchors_xy),
        
        "tag_params": node_params_to_dict(shared_uwb.tag_params),
        
        "anchor_params": {
            str(i): node_params_to_dict(p)
            for i, p in shared_uwb.anchor_params.items()
        },
        
        "seed": int(shared_uwb.seed),
        
        "meta": {
            "count": len(shared_uwb.anchors_xy),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }
    }


def dict_to_shared_state(data: Dict[str, Any], shared_uwb):
    """
    Carrega configuração de dicionário para SharedUwbState existente.
    
    Atualiza o shared_uwb in-place (não cria novo).
    
    Uso:
        >>> with open("config.json") as f:
        >>>     data = json.load(f)
        >>> dict_to_shared_state(data, shared_uwb)
    """
    # Atualiza posições
    anchors = data.get("anchors_xy", [])
    shared_uwb.anchors_xy[:] = [(float(x), float(y)) for x, y in anchors]
    
    # Atualiza parâmetros da tag
    if "tag_params" in data:
        shared_uwb.tag_params = dict_to_node_params(data["tag_params"])
    
    # Atualiza parâmetros das âncoras
    if "anchor_params" in data:
        shared_uwb.anchor_params = {
            int(k): dict_to_node_params(v)
            for k, v in data["anchor_params"].items()
        }
    
    # Seed (opcional - pode não querer sobrescrever)
    if "seed" in data:
        shared_uwb.seed = int(data["seed"])
        if shared_uwb.pipeline:
            shared_uwb.pipeline.seed = shared_uwb.seed
    
    # Sincroniza
    shared_uwb.reindex_anchor_params()
    shared_uwb.sync_pipeline_from_state()


##################################################
# BACKWARD COMPATIBILITY (formato antigo → novo)
##################################################

def upgrade_anchors_file_format(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converte formato antigo (só posições) para novo (posições + params).
    
    Formato antigo:
        {
          "anchors_xy": [[x1,y1], [x2,y2], ...],
          "meta": {"count": N}
        }
    
    Formato novo:
        {
          "anchors_xy": [[x1,y1], [x2,y2], ...],
          "anchor_params": {"0": {...}, "1": {...}},
          "tag_params": {...},
          "meta": {...}
        }
    """
    if "anchor_params" in old_data:
        # Já está no formato novo
        return old_data
    
    # Cria formato novo com parâmetros default
    n_anchors = len(old_data.get("anchors_xy", []))
    
    new_data = {
        "anchors_xy": old_data.get("anchors_xy", []),
        
        "tag_params": node_params_to_dict(NodeParams()),  # default
        
        "anchor_params": {
            str(i): node_params_to_dict(NodeParams())     # default para cada âncora
            for i in range(n_anchors)
        },
        
        "meta": old_data.get("meta", {})
    }
    
    # Preserva metadados antigos
    if "meta" in old_data:
        new_data["meta"].update(old_data["meta"])
    
    return new_data


###############################
# HELPER para validação
###############################

def validate_anchors_data(data: Dict[str, Any]) -> bool:
    """
    Valida se um dicionário é um arquivo de âncoras válido.
    
    Retorna True se válido, False caso contrário.
    """
    try:
        # Campos obrigatórios
        if "anchors_xy" not in data:
            return False
        
        if not isinstance(data["anchors_xy"], list):
            return False
        
        # Cada âncora deve ser [x, y]
        for xy in data["anchors_xy"]:
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                return False
        
        # Se tem anchor_params, valida estrutura
        if "anchor_params" in data:
            if not isinstance(data["anchor_params"], dict):
                return False
            
            for k, v in data["anchor_params"].items():
                if not isinstance(v, dict):
                    return False
                
                # Valida campos de NodeParams (nova estrutura com nested clock/ant)
                if "clock" in v and not isinstance(v["clock"], dict):
                    return False
                if "ant" in v and not isinstance(v["ant"], dict):
                    return False
                # range_bias_m é obrigatório
                if "range_bias_m" not in v:
                    return False
        
        return True
    
    except Exception:
        return False


###############################
# EXEMPLO DE USO
###############################

if __name__ == "__main__":
    import json
    from src.uwb.shared_state import SharedUwbState
    
    # 1. Criar configuração de teste
    shared_uwb = SharedUwbState.make_default(seed=42)
    shared_uwb.anchors_xy = [(0, 0), (10, 0), (10, 10), (0, 10)]
    shared_uwb.anchor_params[0].clock.drift_ppm = 20.0
    shared_uwb.anchor_params[1].clock.drift_ppm = -10.0
    shared_uwb.anchor_params[2].range_bias_m = 0.15
    shared_uwb.reindex_anchor_params()
    
    # 2. Salvar
    data = shared_state_to_dict(shared_uwb)
    with open("test_anchors.json", "w") as f:
        json.dump(data, f, indent=2)
    print(" Salvo em test_anchors.json")
    
    # 3. Carregar
    with open("test_anchors.json") as f:
        loaded_data = json.load(f)
    
    new_shared = SharedUwbState.make_default(seed=123)
    dict_to_shared_state(loaded_data, new_shared)
    
    print(f" Carregado: {len(new_shared.anchors_xy)} âncoras")
    print(f"  Anchor 0 drift: {new_shared.anchor_params[0].clock.drift_ppm} ppm")
    print(f"  Anchor 1 drift: {new_shared.anchor_params[1].clock.drift_ppm} ppm")
    print(f"  Anchor 2 bias: {new_shared.anchor_params[2].range_bias_m} m")
    
    # 4. Validar
    assert validate_anchors_data(loaded_data), "Validação falhou!"
    print(" Validação passou")