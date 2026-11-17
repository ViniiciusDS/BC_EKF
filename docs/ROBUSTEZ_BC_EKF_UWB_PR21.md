# PR #21 — BC-EKF com dt real, UWB parcial e diagnósticos NIS

## Objetivo

Tornar o filtro BC-EKF de **duas tags rigidamente ligadas ao robô**
mais robusto a dados reais e plug-and-play, sem introduzir uma
segunda implementação do modelo de geometria rígida, sem forçar
ajustes de calibração e sem alterar CSV bruto ou âncoras.

A medição de duas tags continua sendo a distância 3D calculada
por `src/bc_ekf.py::_measurement_model` sobre a **pose única do
centro do robô** `[x,y,theta]` e offsets de cada tag XYZ conhecidos.
O passo de predição usa as velocidades PCNT.

## Mudanças

1. **Relógio PCNT real**: `run_bc_ekf_from_data(timestamps_s=...)`
   propaga o estado com `dt = t[k]-t[k-1]` a cada amostra.
   Q continua a covariância de processo definida para um passo
   nominal `T`; por isso usa `Q_k = Q * dt/T`. Essa é uma
   convenção discreta simples e documentada, não uma identificação
   experimental de Q. `timestamps_s` deve ser finito/estritamente
   crescente; relógios inválidos geram erro explícito.
2. **Dados UWB esparsos**: com timestamps reais, qualquer instante
   contendo range novo e válido pode atualizar o filtro; não o
   limitar ao índice módulo `update_ratio`. Sem timestamps, mantém
   exatamente a agenda por índice e o `T` fixo da API histórica.
3. **Correção parcial**: um range ausente `NaN`, `Inf`, zero
   ou negativo é excluído apenas do *vetor e submatrizes de
   medição daquele passo*. Os outros ranges podem contribuir
   com a correção. O estado a priori continua o mesmo quando
   não há observações suficientes.
4. **Estabilidade numérica**: `np.linalg.solve` substitui a
   inversa explícita de S, e a covariância posterior usa a
   **forma de Joseph**, simetrizada para reduzir a deriva
   numérica. Isso pode produzir diferenças numéricas mínimas
   em resultados do código anterior, sem mudar as equações.
5. **Qualidade e diagnósticos**: `NIS = y^T S^{-1} y`,
   `NIS / n_observações`, RMS e máximo da inovação,
   contagem e índices válidos, correções parciais e
   razões de rejeição. NIS é uma medida de **consistência
   estatística do modelo e covariâncias**, não ground
   truth nem precisão geométrica absoluta.
6. **Gates opcionais**: `nis_gate_per_dof` rejeita o
   lote se NIS/dof exceder o limiar; `range_gate_sigma`
   remove medidas cuja inovação padronizada supera o
   limiar, recalculando S com a submatriz correspondente.
   **Ambos os gates ficam DESLIGADOS por padrão**;
   não é recomendável definir limiares sem estudos de Q/R
   e sincronização temporal.
7. **Compatibilidade step**: `run_bc_ekf_step` utiliza a
   mesma rotina e, quando `debug=True`, devolve os campos
   originais `innov,S,K,h,H,x_pred,P_pred,dt` mais os
   índices e indicadores de qualidade.
8. **Compatibilidade de simulação**: `run_bc_ekf` sintético
   tinha referência à variável `time_axis` inexistente.
   Retornou ao modelo de passo constante T.
   Os wrappers `run_bc_ekf` e `run_bc_ekf_custom_commands`
   também usam a correção UWB compartilhada e protegem
   `update_ratio` para `T > 0,2s`.

## Como usar em código

```python
states = run_bc_ekf_from_data(
    T=T_nominal,
    anchors=anchors_3xA,
    odometry_noisy=velocidades_2xN,
    z_hist=distancias_2AxN,
    l=baseline/2,
    z_c=altura_media,
    sigma_uwb=.15,
    x0=pose_inicial,
    timestamps_s=pcnt_timestamps_N,
    tag_offsets_xyz=offsets_2x3,
)
```

A API antiga continua retornando **matriz 3×N**. Se for
necessário obter diagnósticos:

```python
states, report = run_bc_ekf_from_data(
    # mesmos argumentos acima
    # ...
    return_diagnostics=True,
    min_valid_ranges=1,
    nis_gate_per_dof=None,
    range_gate_sigma=None,
)
print(report["summary"])
```

`report["steps"]` possui um registro JSON-serializável para
cada transição PCNT (do índice 1 ao N−1), incluindo os índices
de observações UWB utilizadas e sua contagem. `run_batch`
habilita o retorno internamente e disponibiliza os dados em
`resultados["bc_ekf"]["update_diagnostics"]`, mantendo
`["posicoes"]`, `["rmse_xy"]` e `["full_pcnt_poses"]`
inalterados na estrutura.

A UI do Dataset apresenta um resumo de correções aceitas,
parciais e rejeitadas; os diagnósticos completos são
exportáveis pelo CLI abaixo.

## Auditoria dos dados reais (comandos PowerShell)

Na pasta principal `BC_EKF`:

```powershell
git fetch origin
git worktree add --detach ..\BC_EKF_PR21 origin/feature/bc-ekf-robust-uwb-updates-2026
cd ..\BC_EKF_PR21
python -m unittest discover -s tests -p "test_bc_ekf_robust_updates_2026.py" -v
python -m unittest discover -s tests -v
```

Copie o CSV bruto para `dados_reais/robot2026/` no worktree.
Para analisar sem gatilhos de rejeição:

```powershell
python tools/audit_bc_ekf_updates_2026.py `
  --pcnt "dados_reais/robot2026/pcntLog_14_11_26_1o.csv" `
  --anchors "anchor_sets/lab_duas_tags_A3_A9_14_11_2026.json" `
  --tags "tag_profiles/lab_14_11_2026_t1_direita_t2_esquerda_xyz_custom.json" `
  --session 1 `
  --output-dir "resultados/robot2026/auditoria_bc_ekf"
```

O CLI usa somente o dataset de duas tags com cabeçalho,
o mesmo mapeamento A3–A9 e o perfil T1 à direita/T2 à
esquerda do PR #19, mas sua implementação não depende de
números de âncoras fixos. `sigma_uwb` usa, por padrão,
o mesmo piso `max(.15, median(desvios))` do Dataset Mode.

Saídas de cada execução (sufixos incrementais, sem
sobrescrever as anteriores):

- `bc_ekf_updates_001.json`: resumo, NIS elevado, critérios,
  geometrias, SHA-256 das fontes e ressalvas;
- `bc_ekf_updates_001_steps.csv`: cada transição PCNT,
  `dt_s`, ranges válidos/utilizados, NIS e status;
- `bc_ekf_updates_001_poses.csv`: estimativa completa
  do centro do robô em todos os instantes PCNT.

Para **simulação de sensibilidade**, executar em outra
rodada com, por exemplo, `--nis-gate-per-dof 9` e
`--range-gate-sigma 4`; esses valores são **apenas
ilustrativos, não parâmetros recomendados para o campo**.
As saídas terão sufixos diferentes e poderão ser comparadas
sem alterar nenhum CSV/JSON de entrada.

A associação de tags pode ser personalizada via
`tag_offsets_xyz`, e não se adivinha qual tag é o canal
T1 ou T2 em dados de firmware desconhecido.

## Limitações

- NIS exige Q, R e modelo temporal bem calibrados para
  interpretação probabilística; dividir por graus de
  liberdade facilita comparação de tamanhos diferentes de
  observações, mas **não é automaticamente um teste de
  significância calibrado**.
- O timestamp PCNT é o momento de registro; a simultaneidade
  física dos ranging T1/T2 pode não ser perfeita.
- Uma medição isolada pode atualizar uma projeção parcial
  do estado, mas não garante observabilidade completa de
  `x,y,theta`. Ajuste `min_valid_ranges` se necessário.
- Outliers muito grandes podem causar desvios sem gates;
  portanto, compare relatórios com/sem gating antes de
  considerar seu acionamento em produção.
- O BC-EKF estima o **centro robótico**, diferentemente dos
  solvers UWB estáticos que estimam a T1. Distância à pista
  nominal **não é ground truth**.
- A correção temporal e a fusão melhoram a metodologia,
  mas **não substituem calibração de raios/entre-eixos**,
  nem eliminam erros causados por multipercurso/NLOS.

## Testes e release

`tests/test_bc_ekf_robust_updates_2026.py` cobre a
integração variável, legado por índice, correção parcial,
observações não finitas, minimização de ranges, gates opt-in,
NIS, covariância Joseph, modo step, batch e simulador.
O workflow continua executando os testes anteriores para
garantir que regressões da geometria não sejam introduzidas.

**PR #21 permanece Draft até CI e validação do CSV real
no Windows. Não fazer merge sem aprovação do usuário.**
