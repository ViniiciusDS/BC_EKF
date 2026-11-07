# PR #6 — Pré-teste de campo do robô 2026, 4 âncoras

**Fonte:** observações do laboratório em 06/11/2026 e `pcntLog_passeio_01.csv` recebido no chat. O arquivo bruto completo **não foi adicionado ao GitHub**.

## Dados físicos do ensaio

| Âncora | x (m) | y (m) | z (m) | Colunas do CSV |
| --- | ---: | ---: | ---: | --- |
| A2 | 0.77 | 1.77 | 1.85 | 9,10,11 |
| A3 | 0 | 0 | 1.63 | 12,13,14 |
| A4 | -2.66 | 0.13 | 1.28 | 15,16,17 |
| A5 | -3.04 | 2.10 | 0.86 | 18,19,20 |

Formato bruto:
- **1:** timestamp ESP32 (ms), **2:** WR, **3:** WL com sinal invertido, conforme mensagem do laboratório;
- **4:** contador de eventos da pista (não assumir voltas físicas), **5:** estado;
- **6–8:** primeiro slot UWB vazio, média=0 e desvio=-255;
- **9–20:** quatro âncoras A2–A5, triplas de média (cm), variância (cm²) e desvio padrão (cm).

O JSON usa `pcnt_slot_to_anchor_index: [null, 0, 1, 2, 3]`. O programa converte cm para metros e seleciona somente snapshots com **quatro ranges novos e válidos** para que todos os quatro canais estejam presentes.

**Arquivo:** `anchor_sets/passeio_01_4_ancoras.json`

## Rota e condição de ground truth
O laboratório forneceu linhas MATLAB com polígono fechado. Foi convertido para:
`routes/passeio_01_rota_nominal.json`, de aproximadamente **6,05 m de perímetro**.

Essa é uma **geometria de referência da pista**, NÃO uma sequência de posições verdadeiras sincronizadas ponto a ponto. É válido comparar a distância perpendicular de posições estimadas a essa polilinha, mas não chamar o resultado de RMSE de posição verdadeiro.

## Diagnóstico exploratório do CSV real (fora da aplicação Pygame)

- 3091 linhas, 20 colunas, 772,454 segundos de captura; timestamps sem reset, mediana ~250 ms.
- 4 colunas de ranging quase sempre positivas: 3085 linhas com quatro ranges positivos.
- 1309 linhas com as quatro triplas novas (pela comparação de média/variância/desvio com registro anterior); 1311 linhas com >=3 triplas novas.
- Intervalo mediano entre novas triplas observado no **registro**: ~505 ms por slot. Não é timestamp da aquisição física nem duração de TWR.
- A contagem da coluna 4 tem 192 incrementos; a odometria estima ~58 m de deslocamento, compatível com ~9,6 perímetros de pista. Assim, o PR usa **janelas de tempo de 80 s** para visualizar esse arquivo; elas NÃO são voltas.
- Uma verificação 3D de LS/GN realizada fora da plataforma acompanha a forma da pista. Ao avaliar 3085 posições resultantes, a distância RMS à polilinha de referência foi aproximadamente **0,196 m**, mediana ~0,100 m, 95º percentil ~0,430 m. Estes valores são **indicadores geométricos preliminares**, NÃO RMSE de posição.

## Cuidado com os encoders
A mensagem identifica os canais **WR (coluna 2) e WL (coluna 3, negativo)**.
O perfil de teste usa:
```json
"swap_encoder_columns": true,
"invert_left": true,
"invert_right": false
```
Este é o tratamento literal da mensagem, mas **a orientação e o referencial do robô ainda requerem confirmação**. Comparações exploratórias mostram que a inversão de orientação altera significativamente a coincidência entre o traçado odométrico e os pontos UWB.

A origem inicial configurada no JSON (`x=-1.15 m, y=0.86 m, theta=180°`) é **estimada a partir das primeiras medidas UWB e da direção aparente**; não foi medida independentemente. Não utilizar para afirmar erro odométrico absoluto.

Caso o laboratório confirme que o sinal/etiqueta física dos encoders é o oposto do informado, altere **apenas o JSON**, por exemplo:
```json
"swap_encoder_columns": false,
"invert_left": false,
"invert_right": true
```
Faça um movimento reto e uma curva conhecida antes da decisão definitiva.

## Como testar já no Windows

1. Faça checkout da branch `feature/field-pretest-2026-4anchors`.
2. Copie o arquivo original do laboratório, sem modificá-lo, para:
   `dados_reais/robot2026/pcntLog_passeio_01.csv`.
3. Execute o diagnóstico:
```powershell
python -m tools.check_robot2026_ensaio --log ".\dados_reais\robot2026\pcntLog_passeio_01.csv" --anchors ".\anchor_sets\passeio_01_4_ancoras.json" --session 1 --out ".\resultados\robot2026\check_passeio_01.json"
```
4. Execute `python -m apps.main_interactive`, vá ao **Dataset (batch)** e selecione:
   - Fonte: **Real (PCNT + UWB 2026)**
   - Arquivo: **Bruto: pcntLog_passeio_01.csv**
   - Sessão: **1**
   - Âncoras: **passeio_01_4_ancoras.json**
   - Rota ref.: **passeio_01_rota_nominal.json**
   - Mapa: **(Nenhum)**
5. Carregue, verifique se as âncoras correspondem ao polígono; selecione **Trilateração 3D atual** ou **GN atual** para o primeiro teste rápido. Depois os demais, um por vez, observando os tempos de execução.
6. Use as **janelas de tempo** nos controles inferiores, não o contador físico não validado.

## Para amanhã, com 6 ou 7 âncoras

O parser agora aceita `5+3*N` colunas. Por exemplo, 6 slots => 23 colunas; 7 slots => 26; 8 slots => 29. Se o firmware mantiver um slot inicial inativo, para 7 âncoras serão necessários 8 slots => **29 colunas**.

O JSON deve conter as novas coordenadas em metros e exatamente um índice por slot, com `null` para slots sem âncora. Defina `uwb_2026.min_valid_anchors` coerentemente com o método escolhido.

**Não é garantido** que o firmware novo emitirá os campos nessa ordem; teste o cabeçalho/primeiras linhas antes de iniciar o batch. Algoritmos 3D específicos de quatro âncoras poderão exigir seleção de subconjunto; LS/GN gerais são preferíveis na primeira rodada com seis ou sete âncoras.

## Pendências científicas
- Confirmar valores reais da pose inicial e a altura efetiva da tag.
- Confirmar sentido de rotação e identificação física WR/WL em uma curva conhecida.
- Registrar instantes reais das medições UWB para fusão temporal (o log atual só fornece o timestamp ESP32 de gravação).
- Validar o EKF de uma tag em etapa posterior; o BC-EKF original precisa de duas tags.
