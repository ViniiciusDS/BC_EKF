# PR #7 — Analyzer no modo PCNT + UWB e diagnóstico da odometria

## O que estava errado

O Analyzer não aparecia porque `_compute_dataset_stats` devolvia
`None` para `real_odom_uwb_2026` mesmo com resultados de batch válidos.
Isso era uma proteção contra apresentar a pista geométrica nominal como
**ground truth sincronizado**, mas acabou impedindo qualquer comparação.

O conserto mantém a proteção científica e permite:
- **Rota nominal:** distância ortogonal das posições UWB à polilinha MATLAB
  (RMS, média e máximo, em metros). **NÃO é RMSE de posição**.
- **UWB x odometria:** diferença ponto a ponto entre estimativas e encoder
  nos `sample_indices` **originais do PCNT**, não nos índices reduzidos
  pela janela da UI. Uma amostra UWB só é comparada ao encoder do
  mesmo registro bruto; NaNs não alteram a correspondência.
- **Dispersão:** distância ao centróide do conjunto (não representa
  precisão de localização nem tracking error).
- **Rota sincronizada** é indisponível nesse modo porque os waypoints
  MATLAB não contêm timestamps.

A interface identifica explicitamente `(nao GT)`, mostra RMS de
DISTÂNCIA (em vez de RMSE) e posiciona o resumo PCNT à direita do
Analyzer quando há espaço. O botão Mostrar/Ocultar volta a produzir
efeito visual após o batch. O atalho F10 também é liberado no modo
PCNT+UWB (continua bloqueado para odometria sem UWB).

## Por que a odometria preta não segue a rota nominal

O JSON do PR #6 adotava, como hipótese, pose inicial:
`x=-1.15 m, y=0.86 m, theta=180°`. O ângulo não foi medido.

Reintegrei o `pcntLog_passeio_01.csv` do laboratório (3091 linhas,
CPR 4560, raio=0.03425 m, base=0.229 m e convenções de sinais do
PR #6) e calculei a distância das posições à rota MATLAB:

| Pose inicial | RMS nos primeiros 80s | RMS nos 772s |
|---|---:|---:|
| (-1.15, 0.86) m, 180° | 0.850 m | 0.930 m |
| (-1.41, 0.83) m, 7° | 0.152 m | 0.292 m |

A posição inicial aproximada de (-1.41,0.83)m resulta das primeiras
medições UWB, e o heading de ~7° foi determinado minimizando a distância
geométrica à pista nos primeiros 80 segundos. **Isso não é validação
com ground truth independente**, é uma hipótese para inspeção visual.
**Não** use como estimativa de erro verdadeiro da odometria.

O novo JSON é
`anchor_sets/passeio_01_4_ancoras_pose_estimada.json`.
O original `passeio_01_4_ancoras.json` permanece intacto.
NÃO alteramos raio, base, CPR, calibração relativa das rodas nem sinais.
A calibração de `left_scale`, `right_scale`, sentido WR/WL e posição
física de partida deve ser confirmada no laboratório.

## Como comparar no simulador

1. Copie o arquivo bruto `pcntLog_passeio_01.csv` para
   `dados_reais/robot2026/`, sem alterar suas amostras.
2. Fonte **Real (PCNT + UWB 2026)**, sessão 1.
3. Carregue o JSON **passeio_01_4_ancoras_pose_estimada.json**
   e a rota **passeio_01_rota_nominal.json**.
4. Selecione inicialmente **GN atual** ou **trilateração 3D atual** e
   clique em **Rodar batch**.
5. Confirme o Analyzer, clique em **Mostrar/Ocultar**, mude a métrica
   entre *Rota nominal*, *UWB x odometria* e *Dispersão*.
6. Navegue pelas 10 janelas de 80 segundos: os resultados do
   Analyzer continuam sobre as **amostras completas** do batch.
7. Compare o aspecto da trajetória preta com o JSON antigo (180°).
   A configuração estimada não substitui uma pose inicial medida.

## Para a bateria de sábado
- Meça a **pose inicial** no mesmo referencial das âncoras e registre a
  orientação da tag/carrinho com precisão suficiente.
- Marque uma reta e uma curva de sentido conhecido para confirmar
  sinal, ordem e fator de escala dos dois encoders.
- Se viável, registre referência XY temporalmente sincronizada
  (mocap, vídeo calibrado, marcador físico com timestamp etc.).
- Se o robô seguir sempre a mesma polilinha, distância à pista nominal
  é um **indicador de aderência geométrica**, não de precisão absoluta.
- Não reduza o tamanho do dataset para esconder outliers; preserve
  todas as amostras, identificando valores inválidos.

## Testes de regressão

```powershell
python -m unittest discover -s tests -p "test_dataset_field_analyzer.py" -v
python -m unittest discover -s tests -p "test_field_passeio_2026.py" -v
python -m unittest discover -s tests -p "test_dataset_uwb2026_pipeline.py" -v
python -m apps.main_interactive
```
