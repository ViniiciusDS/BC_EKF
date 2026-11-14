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

## Ensaio 14/11/2026 — novo CSV 39 colunas com PCNT e duas tags

**Arquivo analisado:** `pcntLog_14_11_26_1o.csv`, fornecido
pelo laboratório. Não é necessário convertê-lo manualmente para
`range_front/range_rear`, nem transformá-lo em dataset simulado.

O firmware agora produz um **cabeçalho explícito**:

```text
time_ms,left,right,t1_a1_avg,t1_a1_std,...,t1_a9_avg,t1_a9_std,
t2_a1_avg,t2_a1_std,...,t2_a9_avg,t2_a9_std
```

Identificação física confirmada pelo usuário:
- Tag 1 (T1) = **direita** do robô (Y local negativo);
- Tag 2 (T2) = **esquerda** (Y local positivo);
- Separação entre centros = **0,36 m**;
- Altura das duas tags = **0,215 m**, igual ao ensaio anterior;
- Posições das âncoras A3–A8 mantidas;
- A9 ocupa o lugar **anteriormente nomeado A2**:
  `(0, 2.68, 1.76)` metros.

### Configurações reais adicionadas ao PR

- `anchor_sets/lab_duas_tags_A3_A9_14_11_2026.json`:
  geometria em metros, ordem **A3,A4,A5,A6,A7,A8,A9**,
  `uwb_dual_2026.anchor_ids_uwb=[3,4,5,6,7,8,9]`, unidades
  de ranges e desvios do novo firmware **cm**, encoder com
  raio 0.03425m, base 0.229m, 4560 CPR, invert_right=true,
  e pose inicial experimental herdada do LAB7A.
- `tag_profiles/lab_14_11_2026_t1_direita_t2_esquerda_36cm.json`:
  preset left_right, baseline 0.36m, altura 0.215m e
  **channel_order=[right,left]**.

**Hipótese que ainda precisa de verificação física:** os offsets
X=0 e Y=±0.18m pressupõem centro das duas tags simétrico
em relação à origem cinemática do robô. A distância de 36 cm
e o lado físico foram confirmados, mas a simetria em relação ao
centro cinemático não foi medida neste diálogo. Caso não sejam
simétricas, mudar o perfil para custom com XYZ individuais.

### Diagnóstico do arquivo enviado

Leitura direta do CSV de laboratório:
- **4.209 linhas PCNT**, 39 colunas, aproximadamente 1.052s
  de ensaio (17,5 minutos);
- A1/A2 zeradas; **A3–A9** ativas nas duas tags;
- Após sete linhas iniciais sem UWB, as 4.202 restantes
  possuem sete ranges positivos por tag;
- **510 snapshots UWB conjuntos novos**, renovados
  aproximadamente a cada 2,06 segundos. Linhas PCNT
  intermediárias repetem as leituras;
- Houve **339 pares de ranges tag1/tag2** (de 3.570 pares)
  com diferença superior à distância física das tags
  mais margem de 5 cm. São sinalizações diagnósticas,
  **não mascaramento automático**. Causas possíveis incluem
  viés/NLOS, não simultaneidade física do ranging,
  associação incorreta dos canais ou geometria real
  diferente da hipótese. Não é possível atribuir causa
  a partir desses números isolados.

O novo `dataset_two_tag2026_pipeline.py` lê por cabeçalho,
mantém todo PCNT para odometria, converte cm→m, ignora
as leituras zero dos slots não ativos, detecta atualizações
novas **de ambas as tags** e conserva somente elas como
observações UWB em instantes PCNT existentes. O BC-EKF
faz predição nos 4.209 instantes, usando `dt` real;
corrige apenas quando aparece uma das 510 observações
novas. A curva fusionada completa possui 4.209 poses,
mas as métricas/analyzer utilizam as 510 poses correspondentes
aos instantes UWB dos métodos estáticos.

O parser anterior de **PCNT+UWB de uma tag** permanece
inalterado. Assim, o novo log não pode ser acidentalmente
interpretado como 5+3*N colunas.

### Como carregar no Windows

1. Salvar/copiar o arquivo do laboratório em
   `dados_reais/robot2026/pcntLog_14_11_26_1o.csv`
   no **worktree PR19**;
2. `python -m apps.main_interactive`;
3. `Comparar Algoritmos (Dataset / Step)` → **Configurar Dataset**;
4. **Fonte:** `Real (PCNT + 2 tags 2026)`;
5. **Arquivo PCNT:** `Bruto: pcntLog_14_11_26_1o.csv`;
6. **Sessão:** `1`;
7. **Tags:** `Auto (JSON / legado)`;
8. **Perfil tags:** `lab_14_11_2026_t1_direita_t2_esquerda_36cm.json`;
9. **Âncoras:** `lab_duas_tags_A3_A9_14_11_2026.json`;
10. **Rota ref.:** arquivo da rota nominal LAB7A, caso esteja
    presente na pasta `routes/`. Se não houver, use nenhuma.
    Não há ground truth sincronizado;
11. **Carregar** → marcar **BC-EKF** e, opcionalmente, outros
    métodos UWB → **Rodar batch**.

A interface mostra a contagem PCNT/UWB, canal escolhido e
número de inconsistências geométricas sinalizadas. Por
segurança, esta configuração usa inicialmente **janelas por
tempo**, não afirma que são voltas físicas. É possível
aplicar depois o detector de portão a partir da trajetória
UWB real, verificando posição do início e sentido físico.

**Atenção ao Analyzer:** métodos estáticos recebem
os ranges da **T1 direita** (posição da tag); o BC-EKF
estima o **centro do robô**. Distâncias à pista nominal
não são ground truth e não devem ser usadas como uma
comparação justa de erro entre tag deslocada e centro
sem primeiro normalizar o referencial.

### Testes novos

`tests/test_dual_tag_pcnt2026.py` gera logs com o mesmo
esquema de 39 colunas, incluindo A1/A2 inativas,
verifica a conversão de unidades, reuso de ranges,
mapeamento de A9, sinal e integração PCNT, séries
completas e amostras comparáveis do BC-EKF, sessões,
formatos inválidos e a nova opção de interface.

```powershell
python -m unittest discover -s tests -p "test_dual_tag_pcnt2026.py" -v
python -m unittest discover -s tests -v
```

**Não executar merge sem teste real de interface no Windows.**
