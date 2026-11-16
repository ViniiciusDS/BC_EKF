# PR #20 — Auditoria de heading PCNT versus UWB de duas tags

## Objetivo

Investigar a **discordância angular** observada no ensaio real
`pcntLog_14_11_26_1o.csv` (PCNT 4.209 linhas, 510 snapshots
UWB atualizados), com foco nos **80–120 s**, sem alterar o BC-EKF,
a geometria, as distâncias brutas ou a calibração do robô.

Esta ferramenta não usa o caminho nominal como ground truth.
Estima a orientação por DOIS processos independentes:

1. **Odometria:** `theta_rad` contínuo do próprio adaptador
   `convert_session`, incluindo incrementos direito/esquerdo;
2. **UWB sem odometria:** ajusta **individualmente** `(x,y)`
   de T1 e T2 aos ranges 3D para âncoras conhecidas (mínimos
   quadrados ponderados com Huber); estima heading pela direção
   mundial do segmento T1→T2 menos a direção local
   `(offset_T2−offset_T1)` do perfil de tags.

Não usa LS de pose anterior, heading da odometria,
estimador BC-EKF ou resultado de trajetória nominal como prior
para resolver posições UWB. A semente anterior de cada tag,
quando utilizada, é apenas sua **própria estimativa UWB**.

A orientação medida por duas tags com baseline físico de apenas
0,36m pode ser muito incerta. A variância de heading fornecida
pela WLS é NOMINAL; desvios de firmware não são necessariamente
incerteza metrológica calibrada. A aquisição PCNT registra o
horário no SD, não timestamps físicos separados das duas tags;
não se deve afirmar simultaneidade perfeita.

## Regra de qualidade (opt-in por diagnóstico, sem mascarar o firmware)

Para cada snapshot com ambas as tags renovadas:
- Exigir pelo menos 4 distâncias válidas por tag;
- Resolver XY individualmente via WLS/Huber multistart;
- Rejeitar como referência angular se RMS residual WLS
  exceder 0,20m em qualquer tag (configurável);
- Rejeitar se a distância 3D inferida entre tags diferir
  mais que 0,12m do baseline configurado;
- Rejeitar se a incerteza angular WLS nominal superar 40°
  (configurável);
- Preservar o snapshot rejeitado no relatório, com motivo.

A configuração **não altera o dataset**, e nenhum algoritmo
de localização descarta pontos com base nesses critérios.

## Quais ângulos são reportados?

- `theta_odom_deg`: ângulo PCNT contínuo (sem wrap);
- `theta_uwb_wrapped_deg`: heading pela geometria das duas
  tags, em ±180° e apenas para épocas aprovadas;
- `difference_wrapped_deg`: diferença circular UWB−PCNT
  no intervalo ±180°, sem falso salto na fronteira de ±π;
- `difference_unwrapped_deg`: diferença entre estimadores
  desenrolada nas épocas boas. Pode ter ambiguidade quando
  o salto verdadeiro entre épocas aprovadas excede 180°;
- `relative_drift_deg`: discordância desenrolada menos o
  primeiro valor válido. Isso remove **somente um offset inicial
  de apresentação**, não realinha a trajetória por algoritmo
  ou cria pseudo ground truth.

Também são reportados por PCNT `delta_left_ticks`,
`delta_right_ticks`, `delta_theta_deg` e
`omega_odom_deg_s`, com os 4.209 instantes preservados.

## Executar na pasta PR20 (Windows)

```powershell
cd E:\engenhariaeletrica\Mestrado_estimacaoparam\BC_EKF
git fetch origin
git worktree add --detach ..\BC_EKF_PR20 origin/feature/dual-tag-heading-audit-2026
cd ..\BC_EKF_PR20

python -m unittest discover -s tests -p "test_dual_tag_heading_audit_2026.py" -v
python -m unittest discover -s tests -v
```

Copie o CSV do laboratório em
`dados_reais/robot2026/pcntLog_14_11_26_1o.csv`
no worktree PR20. A seguir:

```powershell
python tools/audit_dual_tag_heading_2026.py `
  --pcnt "dados_reais/robot2026/pcntLog_14_11_26_1o.csv" `
  --anchors "anchor_sets/lab_duas_tags_A3_A9_14_11_2026.json" `
  --tags "tag_profiles/lab_14_11_2026_t1_direita_t2_esquerda_xyz_custom.json" `
  --session 1 `
  --window-s 10 `
  --focus-start 80 `
  --focus-end 120 `
  --output-dir "resultados/robot2026/auditoria_angular"
```

Em PowerShell, a crase `\` no fim de cada linha é necessária
para continuação. Também é possível escrever tudo em uma linha.

`matplotlib` é opcional para os PNGs; caso não esteja
instalado, use `python -m pip install matplotlib`. Sem
Matplotlib os CSVs/JSON são gerados normalmente, ou use
`--no-plots` para análise leve com NumPy.

### Saídas geradas

Dentro de `resultados/robot2026/auditoria_angular`:

| Arquivo | Conteúdo |
| --- | --- |
| `dual_tag_heading_audit.json` | Resumo completo, parâmetros, geometria, SHA256 das 3 fontes, qualidade, detalhes e limites |
| `dual_tag_heading_audit_snapshots.csv` | 510 snapshots UWB e heading UWB−PCNT, com motivos de rejeição e σ nominal |
| `dual_tag_heading_audit_pcnt.csv` | Todos os incrementos individuais PCNT e heading odométrico |
| `dual_tag_heading_audit_windows.csv` | Resumo em janelas de 10s, quantidade de UWB aceito, diferença angular e giro PCNT |
| `dual_tag_heading_audit_angulos.png` | Heading PCNT contínuo e heading independente UWB desenrolado |
| `dual_tag_heading_audit_divergencia.png` | Discordância angular UWB−PCNT com foco destacado |
| `dual_tag_heading_audit_encoders.png` | Incrementos de contadores das duas rodas |
| `dual_tag_heading_audit_qualidade.png` | Separação inferida das duas tags vs 36cm físico |

Todos os caminhos e hashes têm **proveniência verificável**;
nenhum arquivo de origem é modificado ou substituído.

### Análise esperada (hipóteses, não conclusões pré-fixadas)

Investigar especialmente as janelas 80–90, 90–100, 100–110
e 110–120s:

1. O PCNT mostra incremento abrupto de rotação ou
   diferença crescente de ticks ao longo de várias amostras?
2. Quantos headings UWB independentes e **aprovados** existem
   em cada janela? Quais foram rejeitados e por quê?
3. A divergência UWB−PCNT cresce suavemente, em uma curva,
   ou só muda na reconstrução por janelas?
4. A geometria de baseline e os resíduos de cada tag são
   consistentes durante essa manobra?
5. Há buracos longos sem heading UWB confiável que
   tornem o desenrolamento ±180° ambíguo?

O relatório não otimiza automaticamente `wheel_base_m`,
`left_scale` ou `right_scale`; tais hipóteses devem ser
testadas em uma fase posterior com critérios independentes.

### Testes de regressão

`tests/test_dual_tag_heading_audit_2026.py`: GT
sintético (não confundir com dados de laboratório), deriva
introduzida deliberadamente, ângulos atravessando ±180°,
rejeição de dados ruins, compatibilidade de montagem
não lateral, guardas de dimensões/índices, JSON/CSV
serializável e vinculação ao parser real sem Pygame.

**Status PR Draft até CI aprovado + execução com CSV real
no Windows. Nenhum merge automático.**

## Análise real recebida em 16/11/2026

Foram examinados os relatórios `dual_tag_heading_audit.json`,
`_snapshots.csv`, `_pcnt.csv` e `_windows.csv` e os quatro
gráficos enviados pelo usuário. Os arquivos analisados reportam
SHA-256 do PCNT `5691989936293019482c6fd60337993ff363f9f2ab251af9166cf9015d33de5e`.

**Amostragem e qualidade:**
- 4.209 instantes PCNT, 510 épocas UWB novas, 3.698 repetições;
- apenas **295/510 (57,8%)** headings foram aceitos;
- 185 reprovados por RMS de ajuste UWB e 30 por incompatibilidade
  da separação entre tags;
- incerteza angular WLS **nominal** mediana em épocas boas ≈15,0°
  (não confundir com precisão validada ou ground truth);
- máxima lacuna entre headings UWB aceitos ≈31,0s;
- separação estimada entre tags variou de ≈0,203m a ≈0,666m
  mesmo com baseline físico configurado em 0,360m.

### Primeiro desvio grande entre 80–120s

| Intervalo | PCNT: Δheading acumulado | Alteração da discordância entre 1ª/última UWB válida | UWB aceitos |
| --- | ---: | ---: | ---: |
| 80–90s | −29,87° | −7,16° | 5 |
| **90–100s** | **−6,33°** | **−53,83°** | 4 |
| 100–110s | −2,52° | −8,26° | 5 |
| 110–120s | −83,68° | −31,25° | 4 |

**Pareamento mais claro:** entre **90,53s e 109,10s**:
- `theta_PCN`: −203,37° → −214,41° (**Δ=−11,04°**);
- `theta_UWB`: +166,77° → +95,45° (**Δ=−71,32°**);
- **discordância de mudanças: −60,28°**.
- centros XY UWB: aproximadamente `(0.15,0.00)` →
  `(-0.01,1.06)`m, indicando transição para deslocamento
  no sentido +Y. Também se observa orientação de curso do
  centro UWB ≈172° durante 74–88s (ida −X) e ≈83° durante
  100–109s (ida +Y).

Esses dois indícios são derivados do **mesmo UWB**, não são
duas referências de sensores independentes. Contudo,
juntos apontam mudança de direção na curva que o PCNT
não reflete proporcionalmente. Isto é evidência de
**discordância localizada de giro**, NÃO demonstra por si
só falha do encoder, deslizamento ou calibração de rodas.

O relatório global mostra tendência negativa da
discordância desenrolada de aprox. **−0,197°/s**
(ajuste linear exploratório, R²~0,894). Ao reanalisar somente
épocas com **baseline a ±0,05m** e RMS de ajuste de
cada tag ≤0,12m (95 épocas boas restantes), a inclinação
permanece em ≈**−0,191°/s** (R²~0,929). Isso sugere
componente persistente, mas não converte a orientação UWB
em ground truth.

**Contraprova importante:** há diversas janelas com PCNT
quase retilíneo e discordância UWB oscilando >20°.
Por exemplo, entre 1030–1040s, PCNT varia −3,84° e a
discordância estimada varia −61,87°. Como a maioria das
janelas contém apenas 2–5 headings UWB aprovados,
**não considerar `angular_difference_change_deg` uma
medida confiável de erro físico instantâneo**. Precisamos
comparar incrementos de orientação com incerteza temporal,
aprovação por geometria e, se possível, uma estimação
conjunta de pose rígida por duas tags.

### Correção de visualização após envio dos gráficos

Detectou-se que `angulos.png` desenrolava
`theta_uwb_wrapped` de maneira **independente** enquanto
`divergencia.png` desenrolava a **diferença circular
UWB−PCNT**. Com lacunas longas de UWB, ambos os métodos
podiam escolher ramos diferentes separados por 360°,
produzindo gráficos incompatíveis **a partir de ~631s**,
apesar de os registros PCNT/UWB não terem mudado.

O PR20 passou a:
1. Calcular `theta_uwb_multivoltas_plot =
   theta_odom+delta_circular_desenrolado` **apenas para
   apresentação**. A observação UWB por época continua
   independente dos encoders; o ramo multivoltas pode
   continuar ambíguo após lacunas longas.
2. Exportar, além dos 4 gráficos globais, 4 PNGs `*_foco.png`
   com eixo X e Y ampliados no intervalo de interesse
   (com margem de 10s), para examinar a curva sem compressão.
3. Adicionar teste de regressão para garantir
   `theta_UWB_plot−theta_PCNT = delta_circular_desenrolado`
   mesmo após lacunas e números de voltas elevados.

Para obter esses novos PNGs, atualizar a branch e executar
a mesma CLI **novamente**. O CSV original não será alterado.

### Próximas verificações ANTES de calibrar automaticamente

- Checar um evento de curva por volta, usando o deslocamento
  do centro estimado UWB como **checagem adicional** (com
  hipótese de movimento para frente e baixa derrapagem);
- Repetir estimativa de heading com **ajuste conjunto rígido**
  (XY + theta sob baseline de 36cm), mantendo um estimador
  independente dos encoders como controle;
- Identificar atrasos e diferenças de timestamp T1/T2;
- Repetir análise com faixas de qualidade mais restritivas;
- Testar parametrizações de roda/bases sem gravar valores
  ajustados no perfil original e sem otimizar pela pista nominal.

**O PR20 permanece em Draft. Não fazer merge sem a
validação dos gráficos de foco corrigidos.**
