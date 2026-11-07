# PR #9 — Painel de qualidade UWB e verificação descritiva de encoders

## Interface PyGame
1. Carregue uma sessão PCNT+UWB no Dataset Mode.
2. Pressione **F8** para mostrar/ocultar o painel de qualidade.
3. Pressione **F9** para salvar o relatório JSON atualizado em `resultados/robot2026`.
4. Pressione **Esc** com o painel aberto para fechá-lo, sem sair do modo.

F8 calcula o relatório **uma vez ao abrir** e guarda o resultado enquanto
o painel está visível; não reprocessa o CSV a cada frame.
Ao carregar outro arquivo, o painel e o cache são descartados.
Os resultados referem-se à sessão inteira, não à janela de 80 s em exibição.

## Informações
- Proporção de linhas PCNT com snapshot UWB novo e completo;
- Por slot SPI: ranges novos, repetidos e inválidos;
- Distância total incremental estimada pela odometria;
- Velocidade linear máxima estimada (módulo);
- Resumo JSON adicional: deslocamento líquido estimado, velocidade angular
  máxima, fração de incrementos negativos de cada encoder, convenções e CPR.

## Limitações científicas
- Diagnóstico de consistência, **não calibração automática**.
- Sinais positivos/negativos não comprovam correspondência física WR/WL:
  conferir com reta e giro de sentido conhecido.
- Taxa de snapshots é relativa às linhas do log, não frequência física
  de ranging nem garantia de precisão de posicionamento.
- Ground truth verdadeiro não existe nos logs reais atuais.
- Um pico de velocidade exige inspeção do movimento e do timestamp;
  não indica por si só defeito de sensor.
- F8 é um painel compacto de consulta, não substitui o JSON F9.

## Testes antes do merge
```powershell
python -m unittest discover -s tests -p "test_field_diagnostic.py" -v
python -m unittest discover -s tests -p "test_dataset_field_analyzer.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```
Verificar F8/F9 com CSV PCNT puro, PCNT+UWB, duas sessões e troca de
arquivo. Testar o modo Dataset com 4 e 7 âncoras e confirmar que
os algoritmos continuam funcionando com/sem painel.
