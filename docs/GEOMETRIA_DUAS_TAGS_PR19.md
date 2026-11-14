# PR #19 — Montagem plug-and-play de DUAS tags UWB

## Objetivo

Configurar a posição rígida de cada tag em coordenadas **locais do
robô**, sem alterar manualmente o código dos algoritmos:

- Presets **frente/trás** e **esquerda/direita**;
- Coordenadas XYZ personalizadas para cada tag e ordem explícita
  dos dois canais de ranging;
- Compatibilidade com arquivos antigos: sem o bloco
  `tag_geometry` e com seleção `Auto`, o modelo histórico
  **[+l,0,z], [-l,0,z]** continua válido;
- Mesma geometria utilizada no gerador sintético UWB,
  nas equações de observação e jacobianas do BC-EKF,
  nos modos Dataset BC simulado e Real (encoder + UWB);
- Sem alteração na predição de movimento dos encoders.

### Referencial cartesiano local do robô

- **+X:** para a frente do veículo;
- **+Y:** para a esquerda do veículo;
- **+Z:** para cima do veículo;
- Origens: ponto de referência da pose `(x, y, theta)`, tipicamente
  o centro cinemático do robô (não necessariamente centro da carcaça);
- Todas as coordenadas em **metros**.

Os offsets giram com `theta`, não com o mapa: para cada
`(a,b,c)` de uma tag,

```text
x_tag = x_robot + cos(theta)*a - sin(theta)*b
y_tag = y_robot + sin(theta)*a + cos(theta)*b
z_tag = c
```

Distância a uma âncora `i`: `sqrt((x_tag-Axi)^2 + (y_tag-Ayi)^2 + (z_tag-Azi)^2)`.
O Jacobiano correspondente é calculado com o mesmo transformador
em `src/bc_ekf.py`. Não há hipótese fixa de tag longitudinal.

## Presets pela interface do Dataset

**Configurar Dataset** mostra uma linha **Tags:** quando:
- Fonte = `Simulado` e Tipo = `BC` (duas tags);
- Fonte = `Real (encoder + UWB)` (formato legado de dois canais).

Opções:
1. **Auto (JSON / legado)** — usa o bloco `tag_geometry` do JSON de
   âncoras; se ausente, frente/trás histórico;
2. **Frente / trás** — força preset longitudinal usando
   `baseline_m` e `height_m` do JSON (se disponíveis) ou
   padrões de `config.TAG_BASELINE` e `config.TAG_HEIGHT`;
3. **Esquerda / direita** — força preset transversal com o mesmo
   baseline e altura;
4. **Personalizado (JSON)** — exige `tag_geometry.tags` no arquivo
   de âncoras, com duas tags e XYZ completos.

O preset é mostrado na identificação do dataset e registrado
em `bc_ekf_data["tag_geometry_info"]`. A interface **não grava**
alterações no arquivo original ao trocar presets.

## JSON de montagem lateral (dois canais)

Acrescente ao JSON de âncoras já usado pelo ensaio:

```json
{
  "tag_geometry": {
    "preset": "left_right",
    "baseline_m": 0.30,
    "height_m": 0.215,
    "channel_order": ["left", "right"]
  }
}
```

**Os valores 0.30 / 0.215 m são apenas EXEMPLOS.** Ajuste aos
valores reais medidos. `baseline_m` é distância total entre os
centros das antenas, não meia distância. Com ordem acima, canal
UWB 0 é a tag esquerda e canal 1 é a direita.

Se o hardware/CSV registrar **direita antes de esquerda**, use:

```json
"channel_order": ["right", "left"]
```

Essa opção ajusta as **posições previstas por canal**, e não
troca silenciosamente colunas do arquivo bruto.

## JSON de montagem totalmente personalizada

```json
{
  "tag_geometry": {
    "preset": "custom",
    "tags": [
      {"name": "tag_A", "x_m": 0.04, "y_m": 0.16, "z_m": 0.23},
      {"name": "tag_B", "x_m": -0.02, "y_m": -0.17, "z_m": 0.25}
    ]
  }
}
```

Os itens devem estar **na ordem efetiva dos canais 0 e 1** do
arquivo UWB. O primeiro é o primeiro range de cada âncora e o
segundo é o segundo range. Em logs legados, normalmente as
colunas chamam-se `range_front` e `range_rear`, mas **esses nomes
são históricos**: seus valores devem ser associados às antenas
físicas pela ordem escolhida. O algoritmo não adivinha ID físico
de cada tag. O ideal futuro é incluir identificadores explícitos
e persistentes no log do firmware.

Para deixar inequívoco:

- `A0_front`/primeiro `range_front` é SEMPRE **canal 0**;
- `A0_rear`/primeiro `range_rear` é SEMPRE **canal 1**;
- Em `left_right + ["left","right"]`: canal 0 vai para
  posição esquerda e canal 1 para direita;
- Em `left_right + ["right","left"]`: canal 0 vai para
  direita e canal 1 para esquerda.

**Não confundir trocar a posição geométrica da tag com inverter
os sinais dos encoders!** A orientação do referencial do robô
permanece a mesma. O BC-EKF ainda depende da consistência dos
sinais físicos, dos timestamps e do mapeamento das âncoras.

## Compatibilidade e escopo real

### BC simulado e Real (encoder + UWB)

Estão ligados à configuração. Ambos utilizam dois ranges por
âncora (N âncoras => 2N canais por snapshot). No dataset simulado
a opção BC espera colunas intercaladas
`A0_ch0,A0_ch1,A1_ch0,A1_ch1...`, cada uma com distância+sigma.
No real legado, o parser aceita
`range_front, range_rear` por âncora e timestamp.

Quando mudar o preset da geometria, o dataset sintético também
deve ter sido gerado com essa geometria, ou haverá descompasso:
o software NÃO altera medições já adquiridas para simular outra
montagem física.

### PCNT + UWB 2026 (uma tag) e EKF 1 tag

**Não se tornam automaticamente de duas tags neste PR.** O
adaptador atual `src/ui/algo_modes/dataset_uwb2026_pipeline.py`
interpreta `5+3*N_slots` colunas como **uma tag**. A função
de duas tags precisa de associação temporal e de canais que
deve refletir o **formato efetivo do firmware**. NÃO é seguro
inferir uma segunda tag a partir de slots de âncoras, duplicar
ranges ou presumir que uma segunda coluna pertence à outra tag.

O EKF 1 tag do PR #18 permanece com seu próprio deslocamento
`single_tag_ekf.tag_offset_x_m/y_m` e altura
`single_tag_ekf.tag_height_m`. Para avaliar só uma tag lateral,
especifique seu offset físico separadamente. O BC-EKF
de duas tags e o EKF de uma tag **não devem ser confundidos**.

## Testar no Windows

```powershell
cd E:\engenhariaeletrica\Mestrado_estimacaoparam\BC_EKF
git fetch origin
git worktree add --detach ..\BC_EKF_PR19 origin/feature/configurable-two-tag-geometry-presets-2026
cd ..\BC_EKF_PR19

python -m unittest discover -s tests -p "test_two_tag_geometry_2026.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

Validação recomendada:
1. Repita um dataset BC **antigo**, com preset Auto. A rota e
   resultados devem coincidir com a branch main (baseline).
2. Alterne o preset **Frente/trás** no mesmo dataset e confirme
   a mesma trajetória (sem mudança física do arquivo).
3. Para o **novo ensaio lateral**, use o arquivo **duas tags**
   compatível, o JSON com baseline/alturas corretos e selecione
   `Auto` (ou `Esquerda/direita`). Confirme a associação
   canal 0/1 antes de avaliar erros.
4. Para uma montagem assimétrica, use `Personalizado (JSON)` e
   confira o log da geometria carregada e os offsets no diagnóstico.
5. Execute os testes completos. Não julgue precisão usando
   rota nominal como se fosse ground truth sincronizado.

## Testes de regressão

`tests/test_two_tag_geometry_2026.py` cobre:
- equivalência do modelo legado e deslocamentos explícitos;
- diferenças frente/trás e esquerda/direita;
- canais trocados voluntariamente;
- altura e offsets personalizados;
- Jacobiano vs. diferenças finitas;
- modelo sintético vs. modelo de observação;
- BC-EKF `run_bc_ekf_step` e batch;
- integração real legado/simulado BC;
- validação rigorosa de parâmetros e ligação UI.

**PR Draft; não executar merge antes da validação real com o log
de duas tags, especialmente a ordem dos canais.**

## Perfis de tags separados das âncoras (recomendado para laboratório)

Para não modificar o JSON de âncoras, utilize arquivos independentes
na pasta `tag_profiles/`. A interface agora mostra dois campos
no Dataset de duas tags:

- **Tags:** escolha Auto, Frente/trás, Esquerda/direita,
  Personalizado (JSON);
- **Perfil tags:** selecione `(Do JSON das âncoras)` para usar a
  configuração embutida/legado ou selecione outro
  `tag_profiles/*.json` do diretório.

O perfil de tags tem prioridade sobre o bloco `tag_geometry`
embutido no JSON de âncoras; o preset do menu pode sobrescrever
apenas a orientação do preset, reutilizando baseline e altura
do perfil. Para offsets independentes e assimétricos,
escolha `Personalizado (JSON)` ou Auto com perfil custom.

**Passo a passo no laboratório:**
1. Copie `tag_profiles/EXEMPLO_esquerda_direita.json` como
   `tag_profiles/tags_lab_laterais.json`.
2. Meça os centros das antenas em relação ao centro cinemático
   e atualize `baseline_m` e `height_m`.
   Se não forem simétricas, copie `EXEMPLO_personalizado.json`
   e preencha XYZ individualmente.
3. Confirme a ordem de canais gravada pelo firmware.
4. Defina `meta.example_only=false` na cópia validada.
5. Na interface, selecione `Auto (JSON / legado)` +
   `tags_lab_laterais.json` em **Perfil tags**.

Os dois exemplos são bloqueados por `meta.example_only=true`
para evitar que medidas ilustrativas sejam usadas em testes reais.
Nenhum arquivo de âncoras é reescrito; assim os hashes dos
relatórios gerados anteriormente no PR17 não mudam apenas
porque a posição das tags foi alterada.

O perfil separado armazena exclusivamente a geometria das tags;
não altera os canais PCNT UWB, o modelo de encoder, o número de
âncoras nem a configuração da rota.
