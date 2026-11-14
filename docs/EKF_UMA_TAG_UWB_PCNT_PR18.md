# PR #18 — EKF com UMA tag UWB + odometria (ranges diretos)

## Por que esta mudança?

O BC_EKF é uma plataforma generalista para comparação de algoritmos
de localização, geometrias e fontes simuladas/reais. Até o PR #17, o
Dataset PCNT+UWB 2026 com uma tag possuía métodos UWB puros e
odometria, mas **não fazia fusão**. O BC-EKF legado exige duas tags
(front/rear), e não deve receber matrizes de uma tag por engano.

Este PR introduz **um estimador adicional**, sem modificar o algoritmo
BC-EKF existente de duas tags:

- **Estado:** `[x,y,theta]` (pose planar de um robô diferencial).
- **Predição:** velocidades linear/angular do PCNT, intervalo a
  intervalo, com integração midpoint da mesma convenção da odometria.
- **Correção:** ranges UWB **brutos**, diretamente às âncoras 3D
  e uma tag com altura e alavanca XY configuráveis.
- **Incerteza:** filtro EKF e Jacobianas analíticas, covariância
  por propagação e atualização **Joseph**, sigmas de range com piso
  conservador e coeficientes de movimento configuráveis.
- **Alinhamento:** N instantes PCNT + M snapshots UWB com índices
  originais; correção só em `sample_indices`, sem interpolação.
- **Resultados:** N poses fusionadas e M poses para Analyzer/F7,
  covariâncias Nx3x3, fração de correções, ranges usados, inovação
  RMS por snapshot e contadores quando faltam ranges.

Não há análise de verdade terrestre no LAB7A; o uso da polilinha
nominal permanece uma comparação espacial **sem ground truth**.
No teste sintético do PR a trajetória verdadeira é de fato conhecida
e permite calcular RMSE verdadeiro.

## API genérica

`src/fusion/single_tag_range_ekf.py` não importa Pygame, PCNT,
CSV, firmware ou a UI. Recebe apenas arrays NumPy em unidades SI:

```python
from src.fusion.single_tag_range_ekf import (
    SingleTagEkfConfig, run_single_tag_range_ekf,
)
result = run_single_tag_range_ekf(
    timestamps_s=t_pcnt,       # N timestamps PCNT em segundos
    v_m_s=v_pcnt,              # N velocidades sobre k-1 → k
    omega_rad_s=w_pcnt,        # N velocidades angulares k-1 → k
    anchors_xyz=anchors,       # A × 3 metros
    ranges_m=ranges,           # M × A metros; NaN = indisponível
    deviations_m=sigmas,       # opcional, M × A, em metros
    sample_indices=idx_pcnt,   # M índices inteiros crescentes
    sample_timestamps_s=t_pcnt[idx_pcnt], # verificação opcional
    initial_pose=[x0, y0, theta0_rad],
    config=SingleTagEkfConfig(tag_height_m=z_tag),
)
print(result.summary)
# result.poses_xytheta.shape == (N,3)
# result.sampled_xytheta.shape == (M,3)
```

Um snapshot com menos de `min_ranges_for_update` (3 por padrão)
segue com predição somente, sem inserir falsa leitura. Configurar 1
ou 2 ranges é permitido apenas explicitamente; a observabilidade
instantânea é insuficiente para pose completa. Tag central em robô
parado **não observa theta** diretamente. O heading pode ser
parcialmente corrigido ao longo do movimento, pelas covariâncias
cruzadas do filtro; qualidade depende da trajetória e das âncoras.

Não há fusão com dois tags neste módulo. Dados `BC` simulados e
a fonte legada front/rear continuam usando `run_bc_ekf_from_data`.

## Configuração do Dataset Real

Em **Configurar Dataset → Real (PCNT + UWB 2026)**, selecione:
- PCNT original e sessão correta;
- JSON de geometria corrigida, com `robot_2026` e mapeamento
  `pcnt_slot_to_anchor_index`;
- opcional: arquivo `UWB tratado` do PR #17, que mantém o PCNT
  bruto para odometria e só substitui os ranges por índices;
- rota/mapa opcionais, usados **somente como referência**.

O quinto botão do Dataset (antigo BC-EKF) passa a exibir **EKF 1 tag**
nesta fonte real, com cor laranja. Nos demais modos, mantém
**BC-EKF** e seu código de duas tags sem alteração. Habilite o botão
e execute **Rodar batch**. A trajetória de fusão aparece junto
de trilateração, LS, GN, LMS-P e odometria, quando selecionados.
A mensagem do batch mostra quantas correções UWB foram aplicadas.

O EKF lê o bloco OPCIONAL `single_tag_ekf` do mesmo JSON de geometria
que já transporta o perfil PCNT. Para máxima reprodutibilidade,
recomendamos um novo arquivo de configuração (cópia), preservando a
geometria corrigida. Exemplo de **bloco adicional**, sem reescrever
os campos existentes:

```json
"single_tag_ekf": {
  "tag_height_m": 0.215,
  "tag_offset_x_m": 0.0,
  "tag_offset_y_m": 0.0,
  "initial_position_sigma_m": 0.6,
  "initial_heading_sigma_deg": 35.0,
  "range_sigma_floor_m": 0.15,
  "use_reported_sigma": true,
  "motion_sigma_m_sqrt_m": 0.08,
  "turn_sigma_rad_sqrt_rad": 0.12,
  "stationary_sigma_m_sqrt_s": 0.005,
  "heading_sigma_rad_sqrt_s": 0.02,
  "min_ranges_for_update": 3
}
```

**0.215 m é apenas o valor relatado do LAB7A**, não um parâmetro
intrínseco do programa. Verifique a altura física e o referencial
das âncoras; para outros ensaios, altere o valor. Offset XY 0
assume tag central: configure `tag_offset_x_m/y_m` caso a tag
não esteja exatamente no ponto de referência cinemática do robô.

Se o JSON não contiver `single_tag_ekf.tag_height_m`, o loader
pode herdar **exclusivamente** o valor numérico explícito de
`uwb_preprocessing.tag_heights_m` (do PR #17). Se nenhum estiver
declarado, exibe erro amigável, sem inventar altura.

**Os coeficientes de ruído são hipóteses iniciais ajustáveis**,
não foram calibrados para um hardware específico. O sigma do
firmware recebe piso de 15 cm por padrão porque medidas de firmware
podem representar estatísticas internas, e não ruído calibrado;
isso não substitui caracterização metrológica.

## Cuidado com a comparação off/flag/mask

- `flag`: não altera os ranges; EKF e solvers puros usam o mesmo
  conjunto original de M snapshots.
- `mask`: valores marcados recebem NaN e são ignorados na correção
  do EKF. Se houver menos de 3 ranges em um snapshot, o EKF
  executa apenas predição. Nenhum índice PCNT é removido.
- Overlay processado real do PR #17: matrizes já validadas por
  hashes, índices, timestamps e geometria. **Não reaplicar** a
  mesma política por acidente.
- Ao alterar o arquivo JSON de geometria para inserir
  `single_tag_ekf`, um relatório `UWB tratado` exportado antes
  pode deixar de validar o **hash do JSON**. Use uma cópia consistente
  ou regenere o sidecar a partir da mesma configuração. Não
  desabilite a verificação de hash.

## Execução no Windows

Na pasta original BC_EKF, que deve permanecer na main:

```powershell
git fetch origin
git worktree add --detach ..\BC_EKF_PR18 origin/feature/single-tag-range-ekf-pcnt-2026
cd ..\BC_EKF_PR18

python -m unittest discover -s tests -p "test_single_tag_range_ekf_2026.py" -v
python -m unittest discover -s tests -v
python -m apps.main_interactive
```

No PCNT LAB7A de uma tag, com geometria e odometria corrigidas:

1. Carregar o mesmo PCNT e o perfil de 7 âncoras; conferir
   3.351 registros e 808 snapshots, e a segmentação por portão
   UWB/LMS quando o LS também estiver selecionado.
2. Ativar **EKF 1 tag**. Executar batch com trilateração,
   LS, GN e LMS-P como baseline; repetir com só EKF selecionado.
3. Conferir na legenda/analyzer a linha laranja e a quantidade
   de correções UWB. Os resultados UWB estáticos devem se
   manter idênticos ao PR #17 quando a configuração é igual.
4. Comparar odometria, métodos UWB puros e fusão. Distâncias
   à rota nominal e odometria NÃO são ground truth de posição.
5. Testar com `UWB tratado: (Nenhum)`, flag e mask; observar
   correções aplicadas e disponibilidade sem inventar amostras.
6. Pressionar F7 quando a segmentação de voltas reais estiver
   disponível. Relatório deve incluir `bc_ekf` com
   `implementation = single_tag_range_ekf`, ao lado dos
   métodos puros, com as **mesmas M amostras** indexadas.

## Limitações deliberadas do PR #18

- A correção usa o timestamp PCNT da linha, que pode diferir
  do instante físico do ranging; não inventa sincronização
  subamostral ou corrige latência sem calibração.
- UWB de UMA tag pode não observar theta instantaneamente.
  Nenhuma afirmação de convergência angular universal.
- Sem rejeição robusta por inovação, perda Huber ou WLS robusto;
  serão comparados futuramente. Uma leitura anômala pode
  perturbar o filtro e não deve ser omitida da avaliação.
- Não calibra raios/entre-eixos dos encoders e não compensa
  deslizamento sistemático; o EKF só estima a pose sob modelo.
- Suporte direto ao formato real PCNT+UWB 2026 de uma tag;
  o motor é genérico e reutilizável com outros adaptadores.
- Configurações de robô, âncoras e filtro permanecem no JSON
  existente por compatibilidade. Separação desses perfis em
  arquivos independentes fica para futura mudança versionada.

**Política:** Draft até CI aprovado, testes no Windows e comparação
dos dados reais. Nenhum merge automático antes de aprovação.
