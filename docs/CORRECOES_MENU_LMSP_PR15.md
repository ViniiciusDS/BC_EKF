# PR #15 — Correções de interface: botão de logs e LMS-P padrão

## Escopo

Duas regressões de usabilidade identificadas no Windows:

1. O rótulo **"Abrir pasta de logs"** transbordava o botão de
   180 px do menu principal com a fonte Arial 22 em negrito.
   Foi alinhado com os botões da linha inferior (220 px); além disso,
   `Button.draw()` agora mantém o texto na resolução original do
   `font.render()` e recorta qualquer excesso sem interpolação.
   A área de clique é preservada.
2. O `AlgoTestScreen` criava `selected["lmsp"] = False`
   enquanto o estado inicial de variante dizia `"lmsp"` (LMS-P atual).
   Isso podia deixar o botão visualmente desligado mesmo quando a
   seleção concreta de variante já era LMS-P. O estado inicial
   agora tem quatro algoritmos ativos: Trilateração 3D, LS atual,
   GN atual e LMS-P atual. BC-EKF continua desligado como antes.
   O botão continua alternando off/variantes normalmente.
   Textos e cores são atualizados ao alternar Dataset/Step.

Não há mudanças em matemática UWB, datasets, PCNT, calibração física
dos encoders, posição das âncoras ou arquivos de campo.

## Testes de regressão

```powershell
python -m unittest discover -s tests -p "test_ui_menu_default_algos_2026.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

O teste automatizado **não depende de Pygame no CI**: verifica a
geometria, o ciclo de variantes, a renderização nativa do botão com
objetos simulados e a ligação do código real via AST. A validação visual em Windows é necessária
para conferir DPI, fonte e clique.

## Checklist manual antes do merge

- Abrir o menu: texto inteiro dentro do retângulo de **Abrir pasta
  de logs**, **sem aspecto borrado**. Clicar e verificar que abre o diretório de logs.
- Conferir também as fontes dos botões menores do Dataset: não devem
  ficar reduzidas ou embaçadas em comparação à versão anterior.
- Entrar pela primeira vez em **Comparar Algoritmos (Dataset/Step)**.
  Verificar que Trilateração, LS atual, GN atual, **LMS-P atual**
  aparecem ativos. BC-EKF permanece off.
- Executar um dataset conhecido **uma única vez**, sem clicar no
  LMS-P, e conferir que o batch inclui o algoritmo `lmsp`.
- Clicar em LMS-P para alternar para WLS sigma, WLS-RKF, RWLS Wang17,
  off, e novamente LMS-P atual, sem afetar os demais métodos.
- Alternar Dataset ↔ Step ↔ Dataset: não exibir rótulo de variante
  antiga no modo errado nem perder o estado escolhido pelo usuário.

O botão de logs não foi renomeado nem teve o comportamento alterado,
apenas suas dimensões e proteção de texto.

## Continuação da análise UWB (fora do escopo desta correção)

O resumo independente `resumo_experimento_uwb_gdop_multilateracao(1).md`
recebido do laboratório relata **outra representação do experimento**:
`dist.mat` com 1.865 linhas x sete âncoras, 453 vetores UWB únicos
e cerca de 75,8% de repetições; esses números não podem ser fundidos
sem mais análise ao LAB7A de 3.351 PCNT/808 snapshots filtrados.
Os resultados do resumo são **relatados, não recalculados**:
- A2=(0,268,176) cm, coerente com o croqui corrigido do LAB7A;
- altura da tag 21,5 cm;
- cota de um chanfro corrigida de 124 para 24 cm;
- 24 medições fisicamente incompatíveis (16 A4, 8 A8),
  equivalentes a seis vetores distintos após deduplicação;
- HDOP geométrico médio relatado ~0,949, que não indica acurácia
  real sob NLOS/multipercurso;
- resíduos de exclusão de âncora particularmente elevados em
  A4 e A8; nenhum diagnóstico de outlier é prova de falha isolada;
- WLS robusto com piso ilustrativo de 3 cm para os desvios,
  não equivalentes automaticamente à covariância real do EKF.

A análise dos enlaces, LOAO, verificação de limites físicos e WLS
robusto devem integrar um PR futuro de **diagnóstico somente**,
com o `dist.mat` original e/ou os ranges brutos sincronizados
disponíveis; não alterar solvers ou pesos por simples resumo.

## Ajuste posterior à validação visual (antes do merge)

A primeira implementação limitava todos os textos com
`pygame.transform.smoothscale`. Em botões pequenos de 28 px de altura,
a folga vertical forçava uma redução de tamanho mesmo quando o texto
original cabia, deixando os caracteres **embaçados**. O feedback do
usuário confirmou a regressão visual. O código foi corrigido para
**desenhar diretamente a superfície produzida por `font.render()`**,
com um `clip` temporário ao retângulo do botão. O clipping anterior
é sempre restaurado, inclusive em caso de exceção. Assim, as letras
continuam nítidas e qualquer excesso não vaza sobre controles vizinhos.
O menu continua com botões de 220 px de largura; a solução não altera
fontes globalmente nem o algoritmo de renderização do Pygame.

Foram acrescentados testes específicos de regressão contra
`smoothscale`, verificação do uso da mesma superfície renderizada e
restauração do recorte. O CI mais recente para essa correção passou.
