# Pré-diagnóstico de campo — PCNT + UWB 2026 (PR #8)

## Como usar no PyGame

1. Execute `python -m apps.main_interactive`.
2. No **Dataset (batch)**, carregue o arquivo **Bruto: ...** PCNT+UWB, sessão e JSON de âncoras corretos.
3. Pressione **F9** para gerar um diagnóstico da sessão carregada.
4. Consulte `resultados/robot2026/<nome_do_log>_sessao_XX_diagnostico.json`.

O diagnóstico roda sobre o **CSV bruto completo da sessão**, independentemente
da janela de 80 s selecionada na interface, sem editar nenhuma amostra.
Também funciona para o modo PCNT sem UWB: neste caso o status é
`odometry_only`.

## Conteúdo do relatório

- Quantidade de sessões, linhas e duração;
- Quantidade de slots, mapeamento, configuração de âncoras e CPR;
- Quantidade de snapshots com o mínimo configurado de ranges **novos**;
- Por slot: triplas novas válidas, triplas válidas repetidas e inválidas;
- Avisos informativos, inclusive baixa proporção de snapshots completos;
- Nome do CSV e JSON de configuração utilizados.

O status `ready_for_static_batch` indica somente **pré-condições de
importação**, não qualidade métrica ou calibração dos sensores.

## Limitações importantes

- A trinca UWB repetida é detectada por igualdade exata da média,
  variância e desvio; coincidências físicas ocasionais podem ocorrer.
- Não são recriados snapshots a partir de medidas antigas ou interpoladas.
- Timestamps do CSV são os de registro do ESP32, não de ranging TWR.
- Uma sessão começa após reset ou timestamp não crescente; não corresponde
  necessariamente a uma volta nem a um teste completo.
- Uma pista nominal não é ground truth sincronizado.
- A distribuição dos ranges não comprova associação física dos slots
  nem sentido de rotação dos encoders.
- O JSON de diagnóstico pode sobrescrever relatório anterior com mesmo
  nome de arquivo e sessão; guarde versões para comparações entre ensaios.
- O PR não altera leitores, filtragem, algoritmos nem poses.

## Testes

```powershell
python -m unittest discover -s tests -p "test_field_diagnostic.py" -v
python -m unittest discover -s tests -p "test_dataset_field_analyzer.py" -v
python -m unittest discover -s tests -p "test_field_passeio_2026.py" -v
python -m unittest discover -s tests -p "test_dataset_uwb2026_pipeline.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

Confirme visualmente no Windows que F9 gera o JSON tanto em
PCNT puro quanto em PCNT+UWB; navegue nas janelas e repita o atalho.

## Checklist do ensaio físico

- Salvar um CSV por ensaio e anotar o nome, hora e montagem de âncoras.
- Medir posição inicial da tag e orientação do robô (mesmo referencial).
- Fazer reta curta e giro de sentido conhecido para validar WR/WL.
- Conferir slot UWB 0 inativo e associação dos slots 1–4.
- Guardar os brutos originais e os JSONs usados em cada ensaio.
- Não chamar aderência geométrica à pista de RMSE de posição.
