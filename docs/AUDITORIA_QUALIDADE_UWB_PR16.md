# PR #16 — Auditoria não destrutiva da qualidade UWB por âncora

## Objetivo e limites

O PR #15 corrigiu a interface, e a calibração física dos encoders
permanece pausada até a próxima ida ao laboratório. O PR #16 investiga
qualidade dos **enlaces UWB por slot SPI**, usando o CSV bruto
`pcntLog_LAB7A.csv` e a configuração corrigida de 7 âncoras.

**Não modifica** PCNT, firmware, mapeamento de âncoras, localização,
odometria, rotas, F7/F8/F9, pesos WLS, filtro do Dataset Mode ou
o comportamento dos solvers. O diagnóstico é executado por CLI e
gera relatório JSON + CSV por slot.

### Evidências fornecidas pelo outro pesquisador

O resumo externo relata 1.865 amostras de `dist.mat` (453 vetores
distintos), 24 medições que violam a diferença vertical entre tag e
âncora (A4: 16; A8: 8) e resíduos elevados de LOAO para esses enlaces.
**Não reproduzimos esses resultados neste PR**: o arquivo `dist.mat`
original ainda precisará ser fornecido, e não sabemos se sua cadência
e seus índices correspondem aos 3.351 PCNT/808 snapshots UWB do LAB7A.

O resumo também indica A2=(0,2.68,1.76) m, coerente com o croqui
corrigido do LAB7A, e tag a **21,5 cm de altura**. São referências
relatadas; para o CSV de campo, confirme a posição efetiva da tag e
que Z tem a mesma origem das âncoras.

## Verificações implementadas

- **Por slot SPI:** nome do slot e índice da âncora (base zero),
  coordenadas XYZ, quantidade de triplas novas numericamente válidas,
  triplas repetidas e triplas inválidas/sentinelas.
- **Limite físico de distância:** exige `range >= |Z_ancora-Z_tag|`,
  admitindo `height_tolerance_m` opcional para incerteza da medição.
  Conta somente leituras NOVAS; repetições são informadas à parte.
  Não remove medidas nem muda processamento.
- **Cobertura temporal:** linhas do log contendo ao menos 3 e 4
  ranges mapeados NOVOS, mais linhas com solução LOAO válida.
- **LOAO (leave-one-anchor-out):** usa apenas leituras NOVAS
  provenientes da MESMA linha PCNT, com ao menos **4 âncoras**
  fisicamente plausíveis. Exclui uma âncora, estima XY com as demais
  via mínimos quadrados geométricos 2D e altura Z da tag fixada,
  e mede o resíduo assinado do enlace excluído. Computa
  RMS/P95/máximo de resíduos absolutos por âncora e contagens
  acima de um limiar diagnóstico (0,30 m por padrão).
- **Proveniência:** SHA-256 do CSV bruto e JSON de configuração,
  nome dos arquivos, sessão, unidades, altura adotada e avisos.
- **Sessões:** reset de timestamp separa sessões, sem concatenar
  leituras de aquisições diferentes.

A qualidade do ajuste depende de geometria não colinear e do
mapeamento SPI correto. O LOAO linear por diferenças de quadrados
é um indicador exploratório: quando uma das OUTRAS âncoras está
com viés, o resíduo atribuído à âncora excluída também é afetado.
O LOAO não representa RMSE de posição nem comprova NLOS.

### Unidades importantes

O firmware registra as triplas `[média, variância, desvio]` em
`[cm, cm², cm]`. As distâncias e desvios do relatório são
convertidos para metros. `new_valid` significa **válido em formato,
não necessariamente fisicamente plausível**. O relatório separa
`physically_impossible_new` para que o problema seja observável.

O modelo não assume que timestamp de gravação PCNT representa o
instante físico da comunicação UWB.

## Execução no Windows / PowerShell

A partir da pasta principal `BC_EKF` (com worktrees como nos PRs
anteriores):

```powershell
git fetch origin
git worktree add --detach ..\BC_EKF_PR16 origin/feature/uwb-link-quality-audit-2026
cd ..\BC_EKF_PR16

python -m unittest discover -s tests -p "test_uwb_link_quality_2026.py" -v
python -m unittest discover -s tests -v
python -m tools.audit_uwb_links_2026 --help
```

Com CSV do laboratório na pasta principal e perfil corrigido
`lab7a_A2_alternativa_v3.json` no worktree do PR #13:

```powershell
python -m tools.audit_uwb_links_2026 `
  --log "..\BC_EKF\dados_reais\robot2026\pcntLog_LAB7A.csv" `
  --anchors "..\BC_EKF_PR13\anchor_sets\lab7a_A2_alternativa_v3.json" `
  --session 1 `
  --tag-height-m 0.215 `
  --height-tolerance-m 0.0 `
  --loa-outlier-threshold-m 0.30 `
  --out "resultados\robot2026\lab7a_uwb_links_pr16.json" `
  --csv "resultados\robot2026\lab7a_uwb_links_pr16.csv"
```

**Atenção:** `0.215` m é a altura relatada do dispositivo do
experimento. Confira antes de usar que esse valor é a altura do
centro de fase/posição efetiva da tag no **mesmo referencial Z**
das âncoras; uma margem de `--height-tolerance-m 0.02` (2 cm)
pode ser usada em uma SEGUNDA execução, com arquivos de saída de
outros nomes, como estudo de sensibilidade.

Os arquivos de saída **nunca são sobrescritos**. Se um nome já
existir, o programa pede um novo destino. O CSV é a tabela por slot;
o JSON contém também as primeiras 20 violações com linha PCNT e
timestamp de gravação relativos à sessão.

## Revisão sugerida

1. Verificar que slot 0 fica inativo, e confirmar com o firmware
   se slots 1..7 realmente correspondem a A2..A8. Caso contrário,
   o rótulo de âncora estaria incorreto.
2. Comparar `new_valid`, `repeated_valid` e `invalid` por slot.
   Não somar snapshots com triplas repetidas como medições
   independentes.
3. Conferir a regra vertical com o croqui corrigido e comparar
   os resultados com e sem tolerância de 2 cm.
4. Avaliar `loa_abs_residual_summary_m` e a quantidade de casos
   sem solução (geometria insuficiente/condição ruim). Resíduos
   maiores indicam inconsistência, mas não causa identificada.
5. Conferir que a execução pré/pós PR #16 no Dataset Mode mantém
   o mesmo número de amostras UWB e os mesmos resultados de
   localização. O PR #16 não é chamado automaticamente pelo batch.
6. Enviar JSON e CSV para comparação independente.

## Política de validação

- Não comparar diretamente os números deste CSV com `dist.mat`
  até haver proveniência e mapeamento temporal.
- Não substituir `min_sigma_m`, não inserir filtros por `LOAO` e
  não alterar âncoras com base apenas nesses indicadores.
- Não fazer merge até CI aprovado e revisão de saídas reais
  do LAB7A. O PR permanece **Draft**.
