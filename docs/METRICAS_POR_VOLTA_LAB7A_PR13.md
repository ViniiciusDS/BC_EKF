# PR #13 — Indicadores por volta física (PCNT + UWB, 2026)

## Motivação

No LAB7A, a configuração de âncoras corrigida (A2=(0, 2.68, 1.76) m)
gerou 808 snapshots UWB / 805 posições LS válidas, e o PR #12
identificou **7 eventos de retorno**, incluindo **6 voltas completas**
e dois trechos parciais. Os algoritmos foram avaliados até então sobre
o arquivo inteiro; isso oculta mudanças entre voltas.

O PR #13 permite exportar **métricas de cada segmento** a partir do
mesmo batch já calculado e das fatias do detector físico. Ele não altera
mapeamento de âncoras, odometria, gate, estimadores nem referências.

## Como usar

A partir da raiz `BC_EKF_PR13`, execute os testes e abra o Pygame:

```powershell
python -m unittest discover -s tests -p "test_dataset_field_lap_metrics.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

O worktree precisa ter o CSV LAB7A e seus JSONs disponíveis nos
diretórios padrão. No Dataset Mode:
1. Carregue `Real (PCNT + UWB 2026)`, sessão 1, usando
   **`lab7a_A2_alternativa_v3.json`**, o arquivo de 7 âncoras com
   A2 corrigida — não a variante literal A2 errada.
2. Carregue `laboratorio_7anc_rota.json` como referência espacial.
3. Ative `LS atual` (outros estimadores são opcionais).
4. Clique em **Rodar batch** e espere o portão reconhecer as voltas.
5. Pressione **F7** quando o Dataset não estiver com o modal aberto.
   A barra lateral mostrará `F7: exportar indicadores por volta`.

O F7 gera em `resultados/robot2026/` dois arquivos:
`pcntLog_LAB7A_sessao_01_voltas_001.json` e
`pcntLog_LAB7A_sessao_01_voltas_001.csv`.
Se executar novamente, gera `..._002`, `..._003` etc.,
**sem substituir resultados antigos**. O relatório não contém
Ground Truth inventado e preserva índices originais do PCNT.

## Métricas por segmento e algoritmo

**Segmento:** identificador, se está completo ou parcial, timestamps
inicial/final, duração, amostras PCNT, snapshots UWB, distância
acumulada na trajetória estimada pelos encoders e **gap de
fechamento odométrico** (distância entre primeiro e último ponto
do trecho).

**Algoritmo:** implementação exata, quantidade de posições válidas,
fração válida, distância RMS/média/p95/máxima **à polilinha nominal**
se fornecida, RMS/média/p95/máxima da diferença **UWB × odometria
no mesmo timestamp** e distância entre primeira e última posição
UWB estimada no trecho.

As métricas são descritivas e **não devem ser citadas como erro real**:

- Distância RMS à rota nominal: distância espacial à linha, sem
  correspondência temporal. A geometria do croqui possui vértices
  aproximados. **Não é RMSE verdadeiro de localização.**
- Discordância UWB×odometria: diferença entre duas estimativas, sem
  determinar qual está correta.
- Gap de fechamento: diferença de posição estimada nos extremos da
  fatia, sujeita à largura da região de retorno e aos timestamps.
  **Não é erro real de fechamento medido externamente.**
- Trechos incompletos aparecem no relatório como `complete:false`,
  não entram na contagem de voltas completas.
- O campo `implementation` diferencia variações de LS, GN, WLS etc.

O arquivo JSON inclui também o nome e SHA-256 do log PCNT, JSON
de âncoras, sessão, instantes de retorno e notas sobre as limitações
científicas para rastreabilidade.

## Proteções contra falsos resultados

- Os índices de snapshots UWB são associados aos índices reais
  do PCNT. Não se recorta pela posição na matriz de estimadores.
- Os `LapSlice` usam uma amostra de sobreposição exclusivamente para
  continuidade do desenho. No cálculo cada amostra PCNT e UWB pertence
  a **um único segmento**, sem contar o ponto compartilhado duas vezes.
- Um estimador com NaN/Inf conserva a posição temporal da amostra;
  são ignoradas apenas posições não finitas, e a cobertura permanece
  explícita. Não truncar séries de comprimentos incompatíveis.
- Se não existem voltas físicas completas ou o portão voltou ao
  fallback de janelas temporais, o F7 bloqueia a exportação:
  **janelas não são voltas**.
- Se o usuário executar um novo batch sem `LS atual`, a segmentação
  antiga não é reutilizada; volta para as janelas e deixa de autorizar F7.
- Os parâmetros do PR #10 não são aplicados e os dados de aquisição
  continuam inalterados.

## Testes

```powershell
python -m unittest discover -s tests -p "test_dataset_field_lap_metrics.py" -v
python -m unittest discover -s tests -v
```

Também validar manualmente F7 após o batch do LAB7A e abrir o CSV
em Excel ou Python. Esperado: oito segmentos (parcial + seis
completos + parcial); a soma de `pcnt_samples` deve ser **3.351**,
e a soma de `uwb_snapshots` deve ser **808** (cada snapshot contado
uma única vez).

O PR permanece **Draft** até validação dos arquivos exportados.
Depois disso poderemos estudar tendência por volta e planejar
calibração odométrica fisicamente medida.
