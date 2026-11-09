# PR #10 — Calibração assistida dos encoders (somente diagnóstico)

Este módulo **não calibra automaticamente** o robô. Ele compara
incrementos dos encoders de um movimento isolado com distância ou ângulo
**medido independentemente**, e retorna candidatos que precisam ser
confirmados em ensaios repetidos.

## Preparação do ensaio

1. Salve o CSV bruto PCNT sem cabeçalho (5 ou 20/29 colunas).
2. Registre no mesmo referencial a distância **para frente** de uma reta
   ou o giro assinado em graus (**positivo anti-horário**, visto de cima).
3. Faça **um movimento por arquivo**, se possível. Se o arquivo possuir
   várias manobras, anote início/fim e use `--start-s` e `--end-s`.
4. Use o JSON que corresponde ao hardware testado e mantenha o original.

## Executar no PowerShell, na raiz BC_EKF

**Reta para frente de 1 metro:**

```powershell
python -m tools.check_encoder_calibration_2026 --log "dados_reais/robot2026/ensaio_reta.csv" --anchors "anchor_sets/passeio_01_4_ancoras_pose_estimada.json" --straight-m 1.0 --out "resultados/robot2026/calibracao_reta.json"
```

**Rotação anti-horária de 90 graus:**

```powershell
python -m tools.check_encoder_calibration_2026 --log "dados_reais/robot2026/ensaio_giro.csv" --anchors "anchor_sets/passeio_01_4_ancoras_pose_estimada.json" --turn-deg 90 --out "resultados/robot2026/calibracao_giro.json"
```

Para giro horário, use `--turn-deg -90`. Para um recorte,
adicione `--session 1 --start-s 2.0 --end-s 7.0`.

Não omita o JSON se quiser reproduzir o mapeamento/sinais atuais
(`swap_encoder_columns`, `invert_left`, `invert_right`, CPR,
raio, escalas e base). Sem ele, utiliza-se o perfil padrão, que pode
**não** corresponder ao robô físico.

## Como interpretar

- `observed.left_wheel_m` e `right_wheel_m` são deslocamentos
  com sinal, pela convenção do perfil.
- Na **reta**, candidatos `left_scale_candidate` e
  `right_scale_candidate` só aparecem se ambas as rodas avançam
  **e seus deslocamentos são coerentes**. Divergência acima de
  `max(0,02 m; 10% do deslocamento médio odométrico)` gera aviso e
  bloqueia as sugestões. Esses limites são **heurísticos de segurança**,
  não evidências de calibração correta.
- No **giro no próprio eixo**, `wheel_base_m_candidate` só aparece
  se as rodas giram em sentidos opostos, o sentido do giro concorda
  com a referência e a soma dos deslocamentos das rodas permanece
  próxima de zero. Curva com translação relevante gera apenas avisos.
  A tolerância utilizada também é heurística:
  `abs(dl + dr) <= max(0,02 m; 10%*(abs(dl)+abs(dr)))`.
- Para calibrar a base, primeiro confirme os sinais e as escalas
  dos encoders; erros de escala se propagam para a base sugerida.
- Um único movimento não permite estimar independentemente raio das
  rodas, CPR, base e sinais: medições independentes são obrigatórias.
- A distância estimada pelo integrador e o ângulo são **odometria**,
  não ground truth. O número informado em CLI é a referência externa.
- Se houver patinagem, manobras misturadas ou movimento fora dos
  limites do recorte, a recomendação fica comprometida.
- Para testar `WR/WL`, registre uma reta curta e um giro isolado com
  sentido conhecido, compare o relatório e só então ajuste o JSON
  manualmente após confirmar fisicamente as convenções.
- CSV original e arquivo de âncoras não são reescritos. O `--out`
  não pode apontar para nenhum desses dois arquivos e **não substitui**
  relatórios anteriores: se já existir, escolha outro nome.

## Testes de regressão

```powershell
python -m unittest discover -s tests -p "test_assisted_encoder_calibration.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

Este PR entrega um **comando CLI e saída JSON**; a incorporação de
controles gráficos no Pygame fica para uma etapa posterior e não
bloqueia os ensaios de campo.

**Validação física pendente:** executar pelo menos três repetições
de reta e giro, registrar comprimento e ângulo medidos, inspecionar
avisos e confirmar o perfil manualmente antes de aplicar parâmetros.
