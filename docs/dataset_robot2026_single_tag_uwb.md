# Ensaio 2026: PCNT + UWB (uma tag)

## Pré-requisitos
O novo modo **Real (PCNT + UWB 2026)** requer um CSV bruto ESP32 com
**20 colunas, sem cabeçalho**:

- 0: timestamp no ESP32 (ms)
- 1–2: contagem PCNT esquerda/direita
- 3–4: contador de voltas e status de pista
- 5–19: para cada um dos **cinco slots SPI**, média de distância (cm),
  variância (cm²) e desvio padrão (cm).

O arquivo de odometria anterior, com **5 colunas**, continua disponível no
modo **Real (odometria 2026)**. Não contém medições UWB.

## Âncoras: mapeamento obrigatório
Crie um JSON com as **coordenadas reais medidas** e
uma lista de cinco slots SPI. Exemplo de *estrutura* (substitua os valores
das posições pelos do seu próprio levantamento):

```json
{
  "anchors_xy": [
    [1.1, 1.2, 1.0],
    [3.2, 1.3, 1.0],
    [1.2, 4.0, 1.0],
    [3.1, 4.1, 1.0],
    [2.1, 2.5, 1.0]
  ],
  "pcnt_slot_to_anchor_index": [0, 1, 2, 3, 4]
}
```

**Atenção: as coordenadas acima são ilustrativas, não medidas da sua pista.
Não copie o mapeamento 0..4 sem conferir os identificadores reais no STM32.**
Cada posição da lista indica a linha da matriz de âncoras correspondente ao
slot SPI 0..4. Para slots sem âncora, use `null`; ao menos três devem
estar mapeados. O campo `anchor_ids_uwb` antigo não é prova da ordem
dos slots SPI.

## Procedimento no Dataset Mode
1. Escolha **Real (PCNT + UWB 2026)**.
2. Selecione **Bruto: [nome do log de 20 colunas]**.
3. Escolha a sessão após eventuais reinícios de timestamp.
4. Carregue o JSON de âncoras com mapeamento SPI conferido.
5. Deixe rota e mapa em **(Nenhuma)** e **(Nenhum)** até haver referência do
   ensaio físico.
6. Selecione Trilateração, LS, GN ou WLS; clique **Rodar batch**.
7. Alterne as voltas para comparar odometria (preta), pista aproximada
   (azul tracejada, não ground truth) e estimativas UWB.

## Limitações científicas
- A odometria é preservada integralmente. O batch UWB usa apenas quadros
  com pelo menos três distâncias **novas** e válidas.
- Não são interpolados ranges ausentes e registros UWB repetidos são
  considerados potencialmente antigos. Se não houver quadros simultâneos
  suficientes, o carregamento informa explicitamente a limitação.
- O timestamp é o momento de **gravação** no ESP32, não o instante físico
  da medição TWR. As 150 leituras acumuladas podem acrescentar latência.
- Média e desvio são convertidos para metros. Foi adotado piso de sigma
  de **0,02 m** para evitar pesos infinitos em WLS; deve ser calibrado.
- **BC-EKF** requer duas tags e baseline, portanto não roda neste modo.
- Sem ground truth independente, o Analyzer não fornece RMSE de posição
  verdadeiro. A pista aproximada não entra na avaliação.
- Se o hardware UWB atualizar cada âncora em um momento diferente,
  precisamos de timestamps por âncora para uma fusão temporal rigorosa.

## Testar
```powershell
python -m unittest discover -s tests -p "test_dataset_uwb2026_pipeline.py" -v
python -m unittest discover -s tests -p "test_dataset_odometry_pipeline.py" -v
python -m unittest discover -s tests -p "test_dataset_lap_view.py" -v
python -m unittest discover -s tests -p "test_dataset_track_reference.py" -v
python -m unittest discover -s tests -p "test_pcntlog_adapter.py" -v
python -m apps.main_interactive
```
