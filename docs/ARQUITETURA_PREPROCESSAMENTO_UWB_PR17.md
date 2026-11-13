# PR #17 — Pré-processamento UWB genérico, configurável e não destrutivo

## Decisão de arquitetura

**O BC_EKF é um simulador generalista, não uma ferramenta LAB7A.**
O LAB7A serve de regressão e referência experimental, mas a arquitetura
deve funcionar com qualquer geometria e número compatível de âncoras,
dados reais ou sintéticos, uma ou mais tags e diferentes solvers.

**Três responsabilidades separadas:**
1. **Adaptadores de entrada** convertem unidades, IDs de âncoras e
   timestamps e identificam o que é medição válida/novo snapshot.
2. **Pré-processador** recebe arrays UWB já normalizados em metros e
   opcionalmente *sinaliza* ou *mascara* medições conforme política
   declarativa. Não conhece PCNT, CSV, LAB7A, LMS ou BC-EKF.
3. **Estimador** produz posição/pose, podendo implementar robustez
   interna dependente da previsão e inovação (fase futura).
   Comparações e métricas continuam separadas do pré-processamento.

## API de domínio

`src/uwb/preprocessing.py` disponibiliza:
- `UwbPreprocessConfig`: política imutável e validada;
- `preprocess_uwb_batch(anchors_Nx3, distances, deviations=None, ...)`:
  distâncias **M×N** (uma tag) ou **M×T×N** (T tags), N âncoras;
- `UwbPreprocessResult`: cópias dos ranges/sigmas, matriz de flags,
  máscara de aceitação, `sample_indices`, `timestamps_s` e sumário JSON;
- `load_uwb_preprocess_config(anchor_json)`: leitor opcional da política
  em arquivos de geometria existentes, válido para fonte real/sintética.

Todos os comprimentos em metros; a geometria de distância vertical
requer **coordenadas Nx3**, e cada tag exige altura medida e referencial
Z compatível. Nenhuma cota fixa do laboratório é assumida.

### Modos de operação

| `mode` | Sinaliza? | Modifica input dos solvers? | Objetivo |
|---|---|---|---|
| (campo ausente) | Não | **Não** | Legado idêntico |
| `off` | Não | Não | Baseline explícito |
| `flag` | Sim | **Não** | Diagnóstico sem alterar trajetórias |
| `mask` | Sim | Substitui apenas células marcadas por NaN | Estudo experimental do efeito do tratamento |

Nenhuma modalidade modifica arquivos de origem, reordena linhas,
interpola UWB ou produz amostras faltantes. `mask` pode deixar um
snapshot com ranges insuficientes; essa falta NÃO é automaticamente
corrigida nem substituída por uma pose sintética.

### Verificações opcionais

- Ranges não finitos e não positivos;
- `max_range_m` (limite explícito, nenhuma distância máxima universal);
- `tag_heights_m` + `vertical_tolerance_m` (diferença vertical);
- desvios não finitos ou negativos e `max_reported_sigma_m`;
- `detect_consecutive_repeats`: igualdade exata de range e desvio
  (se disponível) em linhas consecutivas. **Apenas suspeita de reutilização**,
  não prova de ausência de medição UWB nova; desligado por padrão.

Os motivos são **bits independentes** para permitir diagnóstico
sem apagar informação. O relatório retorna contagens globais e por
âncora, além do número de células válidas que foram mascaradas.

## Integração ao simulador

O `run_batch(..., preprocess_config=...)` recebe política opcional.
A mesma preparação é aplicada **uma única vez antes dos métodos UWB
estáticos**, e cada algoritmo recebe a mesma matriz de entrada.
O resultado de cada método inclui a chave `preprocessing` com sumário
somente quando uma política foi explicitamente informada.

O Dataset Mode lê a opção do **JSON de âncoras escolhido**, seja dataset
simulado ou PCNT+UWB real, e passa a política para `run_batch`.
Sem o campo, não há alteração de resultados nem novas configurações.

**Limite atual:** o BC-EKF frontal/traseiro de duas tags usa
`z_hist` próprio, não a matriz UWB M×N dos métodos estáticos.
O pré-processador de domínio suporta M×T×N, porém o adaptador
desse caminho BC ainda não está implementado. Se houver tentativa
explícita de aplicar política `flag`/`mask` à fonte simulada BC,
a interface BLOQUEIA com explicação em vez de fingir que tratou o
`z_hist`. O BC-EKF legado permanece inalterado. A integração por
tag/estado observável será tratada em PR separado.

### Como ativar

No JSON de âncoras utilizado pelo dataset, adicionar apenas se quiser
ligar o recurso:

```json
{
  "anchors_xy": [[0, 0, 1.5], [3, 0, 1.5], [0, 3, 1.5], [3, 3, 1.5]],
  "uwb_preprocessing": {
    "mode": "flag",
    "tag_heights_m": 0.5,
    "vertical_tolerance_m": 0.02,
    "max_range_m": 20.0,
    "max_reported_sigma_m": 1.0,
    "detect_consecutive_repeats": false
  }
}
```

**Valores acima são ilustrativos.** Use coordenadas medidas e altura
da tag referentes ao cenário real. Para comparação justa, clone o
arquivo de configuração em variantes separadas, mantenha as mesmas
âncoras e experimente `off`, `flag` e depois `mask`, sem sobrescrever
a geometria original. `max_reported_sigma_m` precisa de sigmas
disponíveis; senão a validação bloqueará a política.

No modo `flag`, trajetórias de todos os métodos devem permanecer
**idênticas à referência sem pré-processamento**, e o rodapé informa
quantas células foram sinalizadas (0 efetivamente mascaradas).
No modo `mask`, o rodapé informa quantas medidas válidas foram
mascaradas; comparar também a cobertura de estimativas válidas.

Para uso programático (independente da interface):

```python
from src.uwb.preprocessing import UwbPreprocessConfig, preprocess_uwb_batch
from src.uwb.algoritmos_estaticos import run_batch

policy = UwbPreprocessConfig(mode="flag", tag_heights_m=0.5)
audit = preprocess_uwb_batch(anchors, ranges, sigmas, config=policy)
print(audit.summary)  # métricas JSON; sem modificar arrays recebidos

results = run_batch(
    anchors_Nx3=anchors, distances=ranges, deviations=sigmas,
    algoritmos=["lms", "gauss_newton", "lmsp"],
    preprocess_config=policy,  # política comum antes de cada solver
)
```

## Validação / Windows

Na pasta `BC_EKF` que agora está na `main`:

```powershell
git fetch origin
git worktree add --detach ..\BC_EKF_PR17 origin/feature/generic-uwb-preprocessing-policies-2026
cd ..\BC_EKF_PR17
python -m unittest discover -s tests -p "test_uwb_preprocessing_2026.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

Checklist:
1. Dataset com JSON antigo (sem `uwb_preprocessing`): mesmos
   resultados, contagem de samples, posições, algoritmos e Analyzer.
2. Dataset com `mode=flag`: mesmo resultado dos algoritmos, novas
   contagens de flags visíveis na mensagem de execução.
3. Dataset com `mode=mask`: ranges incompatíveis viram NaN apenas
   na cópia entregue ao batch; dimensões M×N mantidas e origem intacta.
4. Repetir a comparação com diferentes geometrias e datasets
   simulados. Nenhum valor do LAB7A deve ser especial.
5. Verificar que o BC simulado de duas tags NÃO sofre política
   silenciosa e que a versão antiga continua executando normalmente.
6. Inspecionar os testes de regressão de F7/F8 e análises existentes:
   sem impacto nas execuções sem política.

## Próximos módulos (fora do escopo)

- **PR #18**: filtro de fusão UWB+odometria com uma tag e interface
  comum aos estimadores; predição por encoder, correção direta por
  ranges nos instantes disponíveis, observabilidade e covariâncias.
- **PR #19**: robustez interna de solvers/filtros (Huber, inovação,
  seleção de subgrupos) com políticas reproduzíveis.
- **PR #20**: comparação justa e Monte Carlo em vários cenários,
  incluindo LAB7A, sem confundir rota nominal e ground truth.

**PR #17 permanece Draft até CI e validação local.**
