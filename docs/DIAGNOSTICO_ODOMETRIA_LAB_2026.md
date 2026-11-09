# PR #11 — Diagnóstico de hipóteses de odometria no campo (2026)

## Motivação e escopo

Após o ensaio de 7 âncoras, os algoritmos UWB acompanharam visualmente
o percurso nominal, enquanto a odometria PCNT ficou deslocada/rotacionada
e apresentou deriva. A causa **ainda não foi comprovada**: pose inicial,
ordem/sinais WR/WL, escala, patinagem ou combinação destes fatores.

Este PR fornece um **comparador de hipóteses não destrutivo**, separado
do Dataset Mode. Não modifica perfis, firmware, CSVs, calibrações, rotas,
UI, estimadores nem o procedimento de segmentação por janelas de 80 s.

## O que é comparado

- Perfil original (do JSON efetivamente usado no ensaio).
- Rotação hipotética de heading inicial +90° e -90°.
- Inversão isolada esquerda/direita ou de ambos os sinais.
- Troca da interpretação das duas colunas de encoder.
- Pose inicial medida independentemente (opcional, fornecida pelo usuário).
- Escalas candidatas provenientes dos relatórios PR #10 (opcional).

Em cada caso, o software integra o **mesmo registro PCNT**, a partir das
mesmas amostras, pela função `convert_session` existente. O resumo
contém contagens, deslocamentos por roda, distância odométrica integrada,
pose final, heading acumulado e ponto final no referencial inicial do
robô. Este último permite separar efeito da pose inicial (que muda
apenas posição/orientação global) de efeito de configuração dos encoders
(que altera a forma local da trajetória).

Se um arquivo de rota JSON for fornecido, o script também calcula a
distância RMS de cada trajetória à polilinha. **Isso não mede acurácia
absoluta**, não é erro de posição sincronizado e não é usado para
escolher um cenário vencedor: o campo `best_case` permanece `null`.

## Exemplo: dados reais de laboratório

Abra PowerShell na raiz do seu worktree do PR #11. Os arquivos do campo
podem permanecer na pasta original BC_EKF. Altere os caminhos caso
tenham sido movidos.

```powershell
$orig = "..\BC_EKF"

python -m tools.compare_field_odometry_2026 `
  --log "$orig\dados_reais\robot2026\pcntLog_LAB7A.csv" `
  --anchors "$orig\anchor_sets\laboratorio_7anc_ancoras.json" `
  --route "$orig\routes\laboratorio_7anc_rota.json" `
  --session 1 `
  --out ".\resultados\robot2026\lab7a_hipoteses.json" `
  --traces-csv ".\resultados\robot2026\lab7a_trajetorias_comparadas.csv"
```

Se houver arquivos JSON de calibração reta produzidos pelo PR #10,
é possível usá-los **apenas como cenários candidatos**; é necessário que
tenham sido gerados com o mesmo perfil de robô (CPR, raio, base, troca,
inversões e escalas base):

```powershell
python -m tools.compare_field_odometry_2026 `
  --log "$orig\dados_reais\robot2026\pcntLog_LAB7A.csv" `
  --anchors "$orig\anchor_sets\laboratorio_7anc_ancoras.json" `
  --route "$orig\routes\laboratorio_7anc_rota.json" `
  --calibration-report "..\BC_EKF_PR10\resultados\robot2026\calibracao_reta_tentativa1.json" `
  --calibration-report "..\BC_EKF_PR10\resultados\robot2026\calibracao_reta_tentativa2.json" `
  --out ".\resultados\robot2026\lab7a_com_escalas.json" `
  --traces-csv ".\resultados\robot2026\lab7a_com_escalas.csv"
```

Se a pose inicial foi fisicamente medida, adicione:
`--pose-x 0.15 --pose-y 1.38 --pose-theta-deg 0`.
Esses valores são **apenas um exemplo do JSON de campo**; não os
declare como medidos sem verificação independente. Fornecer uma pose
diferente muda o alinhamento, não a cinemática local.

## Como interpretar

1. Compare `original` com `pose_medida`, `heading_mais_90` e
   `heading_menos_90`. Distâncias por roda, heading relativo e
   deslocamento final no referencial do robô devem permanecer iguais.
2. Compare `original` com `escalas_reta` para visualizar o efeito
   de uma possível subestimação de distância encontrada nos ensaios
   de 3 m. Os fatores do PR #10 não são calibração definitiva.
3. As variantes de sinal/troca servem para inspecionar efeitos de
   convenção, não para decidir só pela coincidência com a pista.
4. Use o CSV de trajetórias para sobrepor as diferentes curvas numa
   ferramenta de gráficos sem alterar o Dataset Mode.
5. Não separe voltas pelo resultado deste módulo: janelas de 80 s
   continuam sendo apenas janelas temporais. A detecção de voltas
   físicas ficará para o PR #12.

### Saídas

- JSON: resumo de cada hipótese, SHA-256 do log, sessão, dados do
  perfil e proveniência das médias de escala.
- CSV: colunas `hypothesis,timestamp_s,x_m,y_m,theta_deg` para
  reprodução independente dos gráficos.
- Nenhum arquivo de entrada é sobrescrito; saídas existentes também
  são recusadas.

## Validação

```powershell
python -m unittest discover -s tests -p "test_field_odometry_hypotheses.py" -v
python -m unittest discover -s tests -v
python -m tools.compare_field_odometry_2026 --help
```

Depois, executar sobre o arquivo real LAB7A, conferir os valores no
JSON, observar o CSV num gráfico e verificar que a odometria do
simulador original continua exatamente igual. Alterações de parâmetros
físicos requerem um ensaio independente de reta/giro conhecido.
