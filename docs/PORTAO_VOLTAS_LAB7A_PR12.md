# PR #12 — Portão de voltas físicas + perfil experimental LAB7A

## Objetivo

No LAB7A, a trajetória física parte em **+X**, curva para **-Y** e
continua no sentido horário. O PR #11 diagnosticou que uma hipótese
com canais WR/WL e convenções de sinal remapeados reproduz essa direção.

Neste PR, há duas alterações independentes:

1. Gerar **um segundo JSON** experimental que preserve o layout,
   coordenadas das âncoras, associação dos slots SPI, características UWB,
   raio da roda, base e CPR; mas use a hipótese experimental:
   `swap_encoder_columns=false`, `invert_left=false`,
   `invert_right=true`, `initial_theta_deg=0`,
   `left_scale=right_scale=1.0`.
2. Substituir as janelas fixas por **voltas marcadas por cruzamento
   do portão inicial**, usando odometria ou a trajetória UWB do
   **LS atual** depois de `Rodar batch`.

A rotina não modifica CSV, firmware, rota nominal, algoritmo UWB,
calculador de métricas nem JSON original. **A referência de rota
continua não sendo ground truth temporal**.

## Preparação do JSON experimental

No PowerShell, na raiz do worktree do PR12:

```powershell
$orig = "..\BC_EKF"
python -m tools.create_robot2026_field_variant `
  --source "$orig\anchor_sets\laboratorio_7anc_ancoras.json" `
  --out ".\anchor_sets\laboratorio_7anc_pr12_experimental.json" `
  --start-x 0.15 --start-y 1.38 `
  --direction-deg 0 `
  --gate-source uwb_lms
```

**O JSON gerado deve ser revisado**: o portão [0.15, 1.38] m e
a direção +X vêm do croqui do laboratório; a posição exata da tag
na passagem pode diferir. O JSON original não é sobrescrito.

`--gate-source uwb_lms` (recomendado): usa a trajetória estimada
pelo **LS atual** depois do batch. Até o batch, a interface
expõe apenas **janelas temporais**, rotuladas como tais;
quando o batch termina, passa a mostrar as voltas detectadas.
Se o LS estiver desligado ou outra variante estiver selecionada,
o status avisa para usar **LS atual**.

`--gate-source odometry`: detecta cruzamentos pela odometria corrigida
assim que o CSV é carregado, sem executar algoritmos. É útil para
diagnóstico, mas a deriva acumulada pode fazer o robô estimado não
cruzar o portão depois de algumas voltas. **Zero cruzamentos não
significa zero voltas físicas!**

## Executar no Pygame

O Dataset Mode atualmente procura JSONs em `anchor_sets/` e logs
em `dados_reais/robot2026/`. Se o CSV está na pasta principal,
copie-o para a pasta de logs deste worktree ou execute a aplicação
no diretório principal **após o merge**. Não mova/edite o original.

```powershell
python -m apps.main_interactive
```

Em `Comparar Algoritmos → Configurar Dataset`:

- Fonte: `Real (PCNT + UWB 2026)`
- Arquivo: `Bruto: pcntLog_LAB7A.csv`
- Sessão: `1`
- Âncoras: `laboratorio_7anc_pr12_experimental.json`
- Rota: `laboratorio_7anc_rota.json`, apenas referência geométrica
- Mapa: `laboratorio_7anc_mapa.json`, se disponível

Selecione **LS atual** na coluna de algoritmos e clique
`Rodar batch`. O painel lateral exibe a fonte do portão
(`Portao UWB/LMS: N voltas`), e o mapa desenha uma pequena linha
verde orientada transversalmente a +X.

O botão **Modo: Uma volta / Intervalo / Todas** continua funcionando,
com **Anterior / Próxima**. As linhas UWB estimadas e odométricas
são exibidas por fatias do mesmo registro temporal. Não houve
quebra/cópia dos dados brutos, apenas fatias para visualização.

## Critérios do detector

O novo campo `view_2026` aceita:

```json
{
  "segmentation": "start_gate",
  "window_duration_s": 80.0,
  "gate_source": "uwb_lms",
  "gate_xy_m": [0.15, 1.38],
  "gate_direction_xy": [1.0, 0.0],
  "gate_half_width_m": 0.4,
  "gate_hysteresis_m": 0.12,
  "gate_min_lap_s": 30.0,
  "gate_min_lap_distance_m": 4.5,
  "gate_departure_m": 0.8,
  "gate_max_step_m": 0.8
}
```

A detecção só aceita cruzamento do lado negativo ao positivo do
portão, dentro de sua largura lateral, após o robô se afastar
da origem, e somente se passou tempo suficiente e foi percorrida
distância incremental mínima. Saltos de posição grandes são
ignorados e desarmam o detector. O primeiro trecho é considerado
parcial quando a captura começa longe do portão; o último trecho
também é sempre parcial até novo cruzamento válido.

Esses limiares são **experimentais**, não medidos diretamente com
ground truth. Ajuste `gate_xy_m` e a largura com base no croqui e
nos dados UWB antes de interpretar quantidades de voltas.
O `gate_source=uwb_lms` utiliza somente snapshots UWB válidos
com índices associados ao PCNT, portanto as fatias da interface
são coerentes com os timestamps, sem interpolar UWB.

**Atenção:** a fonte UWB/LMS usa posições estimadas com ruído;
sem filtro temporal, cruzamentos perdidos/falsos ainda são possíveis.
A proteção por histerese, deslocamento e tempo é conservadora,
mas não substitui uma marcação física ou câmera de referência.
Um contador de voltas físico sincronizado continua desejável.

## Compatibilidade

Os perfis existentes com `view_2026.segmentation: "time_windows"`
continuam mostrando **janelas** de 80 segundos. `lap_counter`
mantém o modo legado. Os perfis sem opção não passam a usar
o portão automaticamente. Nenhuma escala de reta PR #10 foi
aplicada ao JSON novo.

## Testes

```powershell
python -m unittest discover -s tests -p "test_dataset_start_gate_2026.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

Validar manualmente no Windows:
- Carregamento do JSON original: comportamento anterior inalterado;
- JSON novo: portão desenhado e status da fonte claramente visível;
- `uwb_lms`: antes do batch, apenas janelas; depois do LS atual,
  voltas com portão; sem LS, aviso explícito;
- Troca para outro CSV: cache de voltas e estado visual devem ser
  reiniciados;
- Uma volta, faixa e todas: sem linhas artificiais ligando fatias;
- F8, F9 e Analyzer não sofrem regressão.

O PR permanece Draft até validação local com o log LAB7A e conferência
visual do número de cruzamentos em relação aos registros do laboratório.
