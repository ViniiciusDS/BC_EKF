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
