# Perfis de montagem de tags UWB

Este diretório contém **arquivos de geometria das tags**, separados das
configurações de âncoras para não alterar a configuração física do
ambiente nem os hashes dos relatórios já gerados.

O Dataset Mode oferece este diretório em **Configurar Dataset →
Perfil tags** quando for selecionado **Simulado / BC** ou **Real
(encoder + UWB)**, com duas tags.

1. Copie um `EXEMPLO_*.json` para outro nome
   (por exemplo `minhas_tags_laterais.json`).
2. Meça as coordenadas X, Y e Z das antenas, em metros, no
   referencial do robô: +X frente, +Y esquerda, +Z para cima.
3. Confirme rigorosamente a associação entre canal 0 e canal 1
   do arquivo UWB e a tag física esquerda/direita.
4. Atualize `meta.example_only` para `false`.
5. Em Configurar Dataset, selecione **Tags = Auto (JSON / legado)**
   e **Perfil tags = minhas_tags_laterais.json**.

Os exemplos **não podem ser usados diretamente**: isso evitará que
coordenadas ilustrativas sejam confundidas com medições físicas.

Para não usar perfil separado, escolha
`Perfil tags = (Do JSON das âncoras)`. O software então utilizará
`tag_geometry` embutido no arquivo de âncoras se existir; caso
contrário, manterá a montagem antiga frente/trás.

**Atenção:** o adaptador de logs PCNT+UWB 2026 continua sendo para
**uma tag**. Os perfis deste diretório são usados no BC-EKF de
duas tags com dados intercalados dos dois canais, não adaptam
automaticamente novos layouts de log do firmware.
