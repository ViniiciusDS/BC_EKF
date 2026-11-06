# PR #5 — Preparação da bateria de testes de campo (7/11/2026)

## Objetivo: sem mudanças no Python durante o ensaio
O firmware grava um arquivo CSV/TXT sem cabeçalho. **Ele pode ter
qualquer nome**, desde que esteja em `dados_reais/robot2026/`. Ao abrir
**Configurar Dataset**, a lista é atualizada automaticamente.

O único requisito de configuração que **não pode ser inferido do log**
é a geometria física da montagem: as coordenadas das âncoras, a
correspondência com os slots SPI e a pose inicial (quando quiser
comparar odometria e ranges no mesmo referencial).

## Preparar a montagem (uma vez para cada disposição de âncoras)
1. Copie `docs/examples/robot2026_ensaio_EXEMPLO.json` para
   `anchor_sets/ensaio_S01.json`.
2. Meça a posição (x, y, z) em **metros** de cada âncora, usando
   origem e orientação únicas. Edite `anchors_xy`.
3. Após editar as coordenadas, defina `meta.example_only` como `false`.
   A configuração ilustrativa é bloqueada enquanto esse campo for `true`.
4. Confirme com o laboratório qual slot SPI do log (0..4) pertence a
   cada posição, e edite `pcnt_slot_to_anchor_index`. **Não** presuma
   que a ordem 0..4 é igual à ordem das posições.
5. Em `robot_2026`, mantenha CPR=4560, raio=0,03425 m e base=0,229 m,
   caso a calibração e o hardware não tenham mudado. Ajuste
   `initial_x_m`, `initial_y_m` e `initial_theta_deg` para a pose
   inicial **no mesmo referencial das âncoras**. Se desconhecida,
   a visualização fica relativa e não serve para medir erro absoluto.
6. Mudou a montagem? Crie **outro JSON**, por exemplo
   `anchor_sets/ensaio_S02.json`, preservando o anterior.

**Não copie as coordenadas do arquivo EXEMPLO como se fossem medidas.**
As opções de rota e mapa continuam independentes: selecione
`(Nenhuma)`/`(Nenhum)` se não houver arquivo correspondente.
Uma rota em `routes/*.json` usa `{"waypoints":[[x,y],...]}`; se
for de outro ensaio, **não** utilize como ground truth.

## A cada bateria de gravação
1. Copie do SD para `dados_reais/robot2026/`, sem editar as linhas.
   Use nomes distintos: `20261107_S01_T01.csv`, `20261107_S01_T02.csv` etc.
2. Antes de abrir o simulador, faça a inspeção (não muda dados):
   ```powershell
   python -m tools.check_robot2026_ensaio --log ".\dados_reais\robot2026\20261107_S01_T01.csv" --anchors ".\anchor_sets\ensaio_S01.json" --session 1 --out ".\resultados\robot2026\check_S01_T01.json"
   ```
3. Resultado `ready_for_static_batch`: tem pelo menos um snapshot
   UWB válido para iniciar a comparação dos métodos estáticos.
   `odometry_only`: arquivo antigo de 5 colunas, permite reconstruir
   odometria; não há ranges. `blocked`: examine os avisos (slots,
   geometria, medidas novas, formato ou sessão).
4. Execute `python -m apps.main_interactive`. Vá a
   **Comparar Algoritmos → Configurar Dataset** e selecione:
   - Fonte: **Real (PCNT + UWB 2026)** se for arquivo de 20 colunas.
     Para 5 colunas, use **Real (odometria 2026)**.
   - Arquivo PCNT: **Bruto: 20261107_S01_T01.csv**.
   - Sessão: **1** ou outra sessão após reinício do timestamp.
   - Âncoras: **ensaio_S01.json**.
   - Rota/mapa: configuração correspondente ou nenhuma.
5. Clique **Carregar**, confirme âncoras e PCNT, selecione
   inicialmente os métodos mais rápidos e clique **Rodar batch**.
   Teste as voltas no painel lateral. Para novos layouts, troque
   apenas o JSON e/ou arquivo no modal, sem editar código.

## Observações fundamentais
- O formato combinado precisa de **20 colunas** (5 PCNT + 5 triplas
  UWB). Só usamos ranges válidos e novos; o leitor exige três novas
  distâncias por snapshot. Se o firmware atualiza âncoras em instantes
  diferentes, o software **não cria sincronismo fictício**: o
  pré-verificador aponta a ausência de snapshots suficientes.
- A geometria não pode ser adivinhada: medir x/y/z e confirmar o slot
  físico é indispensável. Para métodos 3D, pelo menos quatro âncoras
  em geometria 3D adequada podem ser necessárias; âncoras coplanares
  tornam a altitude ambígua sem restrição adicional.
- O timestamp PCNT representa gravação no ESP32 e não necessariamente
  o instante da medição UWB. Atrasos devem ser caracterizados.
- A pista aproximada da primeira volta **não é ground truth**.
- O BC-EKF original de duas tags não funciona como estimador de uma
  tag: nesta etapa execute trilateração/LS/GN/WLS. O EKF de uma tag
  com odometria é uma implementação futura.
- Não sobrescreva os logs do SD nem os layouts históricos. Guarde
  o CSV bruto, JSON de âncoras, eventuais rotas e o relatório de
  pré-verificação com o mesmo identificador de ensaio.
- Se a montagem mudar **durante** uma gravação, não use um único
  JSON para o arquivo todo: separe a gravação em trechos, cada um
  com sua geometria conhecida, ou refaça o ensaio.

## Sobre os dados que ainda faltam para sábado
Precisamos confirmar no hardware a **ordem física de SPI 0..4**,
as **coordenadas reais** e a disponibilidade de um **CSV com 20
colunas**. O leitor foi preparado, mas isso não constitui validação
com o firmware atual até o primeiro log real ser processado.
