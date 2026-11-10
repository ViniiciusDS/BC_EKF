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
- Troca isolada da interpretação das duas colunas de encoder.
- Troca **conjunta** da interpretação esquerda/direita e das convenções
  de sinal (com troca das escalas individuais): preserva o avanço
  físico para frente e inverte a convenção do giro odométrico.
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
   Atenção especial a `trocar_colunas_e_sinais`: ela mantém deslocamento
   para frente e inverte a curva, diferentemente da troca isolada de
   colunas, que pode passar a indicar deslocamento para trás.
   Confirme fisicamente o sentido **da primeira curva** antes de usá-la.
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

## Revisão após retorno de campo LAB7A

O operador confirmou que o robô iniciou no sentido **+X**. A configuração
`initial_theta_deg: 0` corresponde a esse dado experimental, assumindo
que o +X do croqui é o +X do referencial do simulador. Portanto, um
heading inicial de -90° **não é uma correção fisicamente admissível**
apenas porque sua distância à polilinha seja menor.

No CSV de comparação de 3351 amostras, com a rota aproximada do laboratório,
observou-se que a hipótese original inicia em +X mas acumula giro positivo,
entrando em +Y na primeira curva. A rota nominal salva prevê +X e depois
-Y (sentido horário). Uma hipótese adicional agora disponível,
`trocar_colunas_e_sinais`, testa se os canais físicos foram trocados,
**sem inverter o sentido de avanço**. Ela troca também as escalas
esquerda/direita, mantendo os valores ligados aos canais originais.

Esta hipótese é um **diagnóstico**, não prova de ordem física WR/WL. A
orientação do percurso e a ligação real dos encoders ainda devem ser
confirmadas. O arquivo de rota foi desenhado com vértices aproximados,
logo a distância RMS é somente referência geométrica.

Os relatórios de reta do PR #10 usam o mapeamento original. Se o
mapeamento físico estiver invertido, a associação de cada escala à roda
esquerda/direita deve ser revista antes de aplicar esses fatores.

Para repetir a análise após atualizar a branch, use novos nomes de saída,
pois o comando protege os relatórios já existentes. Espera-se ver um
novo caso denominado `trocar_colunas_e_sinais` no JSON e no CSV.

Com os relatórios de reta fornecidos, o script passa a produzir
também `trocar_colunas_e_sinais_com_escalas_reta`, que atribui a
escala originalmente vinculada ao canal direito para a nova roda
esquerda e vice-versa. A combinação é apenas uma hipótese experimental,
não uma aplicação automática. Isso evita a comparação inconsistente de
canais remapeados com escalas ligadas à roda física equivocada.

Na geometria nominal anteriormente gerada, a trajetória do LAB7A
`trocar_colunas_e_sinais` corresponde à reflexão da original sobre
a linha de heading inicial (início em +X), e deu RMS geométrica
exploratória próxima de 0,409 m, contra 1,260 m do original. Nos
primeiros 120 s, aproximadamente 0,105 m contra 1,166 m. A trajetória
com escalas sozinhas deu 1,186 m. **Esses números dependem da rota
nominal aproximada** e não validam fisicamente os sinais: confrontar
com a direção da primeira curva, e não escolher apenas por RMS.

## Revisão de campo — direção física confirmada e PR #11

O operador confirmou a trajetória real:
**+X → -Y → -X → +Y → +X**, repetida no sentido horário.
O heading inicial correto na geometria adotada é 0°, não -90°.
No CSV comparativo, a primeira curva principal aparece por volta de
30–45 s; esse intervalo foi identificado exploratoriamente na odometria
e deve ser revisado se o registro físico indicar outros tempos.

### Checagem explícita da primeira curva

Além do RMS nominal, é possível informar um fato observado fisicamente:
o **sentido horário da primeira curva**. Para cada hipótese o JSON
passa a conter `observed_turn_check` com a variação angular calculada
nesse intervalo e um indicador de compatibilidade de sentido.
O cálculo não cria trajetória de referência nem valida acurácia angular.
Deslocamento angular absoluto inferior ou igual a 5° é classificado
como `indeterminate` (não confirma uma curva).

Adicione ao comando anterior:

```powershell
  --turn-start-s 30 `
  --turn-end-s 45 `
  --turn-direction cw `
```

Use esse trecho com **nomes de saída novos** e com os dois relatórios
de calibração informados como nos exemplos anteriores. Não execute
no CSV completo sem especificar o intervalo. As unidades são segundos
relativos ao início da sessão.

Conferência independente com o CSV LAB7A-v2 no intervalo aproximado
30,221–44,971 s:

| Hipótese | Giro odométrico | Direção esperada |
| --- | ---: | --- |
| original | +88,57° | incompatível (CCW) |
| heading_menos_90 | +88,57° | incompatível (CCW) |
| trocar_colunas_e_sinais | -88,57° | compatível (CW) |
| escalas_reta | +103,43° | incompatível (CCW) |
| trocar_colunas_e_sinais_com_escalas_reta | -103,43° | compatível (CW) |

Embora a hipótese combinada com escalas reproduza o sentido,
ela aumenta o deslocamento acumulado total e parece ultrapassar
a posição aproximada da primeira curva da pista. As escalas PR #10
**não** devem ser adotadas como calibração definitiva somente com
aquelas duas tentativas; os tempos do trecho reto devem ser revisados
contra a posição real dos marcadores e o início/fim do movimento.

O operador é quem fornece o sentido físico (CW/CCW). A ferramenta
não valida a identidade elétrica real dos cabos ou do firmware e
não altera o arquivo de configuração. Não fazer merge de uma
correção de perfil neste PR: trata-se apenas de diagnóstico.
