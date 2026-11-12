# PR #14 — Auditoria da deriva da odometria por voltas físicas (LAB7A)

## Motivação

Os relatórios F7 do PR #13, exportados com **A2 correta**
(0,00; 2,68; 1,76 m), identificam 7 retornos UWB/LS, 6 voltas
completas e 2 trechos parciais (3.351 PCNT / 808 snapshots UWB).
O RMS da discordância LS × odometria cresce aproximadamente de
0,413 m até 1,883 m ao longo dessas 6 voltas.

Essa tendência pode indicar deriva acumulada de odometria,
mas NÃO identifica sozinha qual parâmetro está errado nem indica
erro verdadeiro de localização: UWB não é ground truth.

O PR #14 adiciona auditoria reprodutível e **não destrutiva**:
- Recarrega o CSV PCNT original, perfil usado no F7 e limites de
  início/fim de cada volta física.
- Calcula deslocamento odométrico entre retornos consecutivos,
  afastamento acumulado em relação ao primeiro portão, comprimento
  percorrido por volta e heading odométrico total (sem wrap).
- Opcionalmente compara cada giro odométrico ao valor topológico
  de **-360° por circuito horário completo**. Isso é uma HIPÓTESE
  sobre a orientação após cada volta, não medida de ângulo por
  referência independente, e nunca produz calibração de entre-eixos.
- Reproduz a tendência da discordância LS × odometria registrada no F7
  como inclinação descritiva (OLS). Não é incerteza estatística nem
  teste causal de deriva.
- Opcionalmente calcula uma hipótese com duas escalas digitadas
  explicitamente para observar sensibilidade. Nenhuma hipótese
  é selecionada, escrita no JSON ou otimizada pela pista nominal.

## Executar no Windows

Use o worktree `BC_EKF_PR14`, mantendo os arquivos de campo originais
na pasta `BC_EKF` e o relatório F7 na pasta `BC_EKF_PR13`.
O perfil deve ser **a cópia exata em bytes** daquele usado ao gerar
o relatório F7; não renomeie nem reformate seu conteúdo.

```powershell
git fetch origin
git worktree add --detach ..\BC_EKF_PR14 origin/feature/odometry-per-lap-calibration-audit-2026
cd ..\BC_EKF_PR14

python -m unittest discover -s tests -p "test_field_calibration_audit_2026.py" -v
python -m unittest discover -s tests -v
python -m tools.audit_field_odometry_2026 --help
```

**Auditoria do perfil original, sem adotar novos fatores:**

```powershell
python -m tools.audit_field_odometry_2026 `
  --log "..\BC_EKF\dados_reais\robot2026\pcntLog_LAB7A.csv" `
  --anchors "..\BC_EKF_PR13\anchor_sets\lab7a_A2_alternativa_v3.json" `
  --laps "..\BC_EKF_PR13\resultados\robot2026\pcntLog_LAB7A_sessao_01_voltas_001.json" `
  --algorithm lms `
  --expected-lap-turn-deg -360 `
  --out "resultados\robot2026\lab7a_auditoria_pr14.json" `
  --csv "resultados\robot2026\lab7a_auditoria_pr14.csv"
```

Se os arquivos estiverem em outro diretório, ajuste somente os
caminhos. O comando se recusará a analisar quando o SHA-256 do
CSV ou do JSON de âncoras não coincidir com o relatório F7,
ou se as fatias PCNT forem inconsistentes.

**Comparação de sensibilidade opcional** (NÃO é calibração validada):

```powershell
python -m tools.audit_field_odometry_2026 `
  --log "..\BC_EKF\dados_reais\robot2026\pcntLog_LAB7A.csv" `
  --anchors "..\BC_EKF_PR13\anchor_sets\lab7a_A2_alternativa_v3.json" `
  --laps "..\BC_EKF_PR13\resultados\robot2026\pcntLog_LAB7A_sessao_01_voltas_001.json" `
  --algorithm lms `
  --expected-lap-turn-deg -360 `
  --candidate-scales 1.02 0.98 `
  --out "resultados\robot2026\lab7a_sensibilidade_pr14.json" `
  --csv "resultados\robot2026\lab7a_sensibilidade_pr14.csv"
```

Os valores **1,02 / 0,98** acima são apenas uma perturbação ilustrativa,
não devem ser adotados no robô. O CLI não calcula um vencedor.

### Limite do ensaio de reta PR #10

Os dois relatórios anteriores de reta de 3 m foram calculados com
a associação antiga de canais/sinais. Não aplicar diretamente
`left_scale=1,2205` e `right_scale=1,2058` ao perfil corrigido:
a interpretação de esquerda/direita mudou. Para estimar escalas
fisicamente válidas, repetir a análise dos **dados brutos de reta**
com o **perfil corrigido**, rever os tempos do movimento e medir
novamente um trecho isolado no campo.

Para calibrar `wheel_base_m`, precisamos de uma **rotação isolada
de ângulo efetivamente medido**, por exemplo 360° no lugar, com
contagem das rodas opostas e teste de sinal. O giro de -360° por
volta de pista é hipótese topológica e NÃO substitui esse ensaio.

## Como interpretar as saídas

O JSON registra os SHA-256 dos arquivos, sessão e um resumo por
hipótese. O CSV contém uma linha por **volta completa e hipótese**.
Campos relevantes:
- `odom_path_m`: percurso integrado pela odometria entre dois
  índices de retorno UWB;
- `gate_to_gate_closure_m`: distância entre as posições odométricas
  nos dois retornos UWB consecutivos;
- `cumulative_offset_from_first_gate_m`: distância da posição
  odométrica no retorno atual àquela no primeiro retorno registrado;
- `odom_net_heading_deg_unwrapped`: giro total estimado pelos encoders;
- `heading_residual_deg_assumed`: diferença relativamente a -360°
  somente quando fornecida a hipótese;
- `uwb_odom_disagreement_from_f7`: RMS por volta e inclinação
  descritiva de sua tendência.

O retorno UWB pode ocorrer **dentro de um raio de 0,50 m** em vez de
no exato centro geométrico do gate. Assim, o fechamento não representa
erro absoluto medido externamente; não confundir com o gap F7, que usa
o ponto anterior ao índice final da fatia. A diferença de um índice
PCNT é intencional para comparar **retorno a retorno**.

## Proteções e regressões

- Checagem de SHA-256 do log bruto e configuração, além dos nomes e
  da sessão, antes de processar.
- Fatias F7 com índices PCNT contíguos, não sobrepostos e com retorno
  registrado; trechos parciais não são comparados como voltas completas.
- Nenhuma modificação de UWB, estimador, firmware, dados, JSON ou PR13.
- Arquivos de saída exclusivos, sem sobrescrita; regressões para
  rollback de exportação e entradas incompatíveis.
- Testes: fonte inválida, escala inválida, curvas em referência
  topológica, tendência, sintético com reset, integridade de CSV/JSON.

**Aceitação:** CI verde e execução com o LAB7A corrigido, seguida de
revisão dos seis fechamentos e giros no CSV. PR mantido Draft até isso.
