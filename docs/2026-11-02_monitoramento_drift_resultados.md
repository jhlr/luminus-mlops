# Monitoramento: resultados de drift medidos

Data: 2026-11-02 | Como reproduzir: README, seção "Gerar carga" | Matemática: `src/luminus_mlops/monitoring/drift.py`

Referência: 500 imagens do treino do Imagenette 320 px (divisão determinística, seed 0). Cada carga abaixo foi 200 chamadas
a `/predict` (`load_gen.py --n 200`, laço fechado, 5% de chamadas inválidas), contra o container Docker de inferência; o D e o
p vêm de `GET /data/drift?since=...&until=...` do `monitor` sobre exatamente o intervalo da carga. As janelas têm de 186 a 191
imagens (as chamadas inválidas não têm imagem). KS por canal sobre histogramas 0-255 agregados com peso igual por imagem; níveis
D >= 0,05 amarelo, D >= 0,10 vermelho.

| Carga | Imagens | D L | D R | D G | D B | Nível (L, R, G, B) | p (L, R, G, B) |
|---|---|---|---|---|---|---|---|
| `holdout`, sem drift (controle) | 191 | 0,015 | 0,013 | 0,011 | 0,020 | verde nos 4 | 1, 1, 1, 1 |
| `val`, sem drift (outra divisão, mesma distribuição) | 186 | 0,027 | 0,021 | 0,031 | 0,034 | verde nos 4 | 1, 1, 1, 1 |
| `woof` (Imagewoof, só cães: drift natural) | 186 | 0,037 | 0,050 | 0,045 | 0,075 | verde, amarelo, verde, amarelo | 0,99, 0,88, 0,95, 0,44 |
| `holdout`, `mild` | 191 | 0,161 | 0,138 | 0,184 | 0,176 | vermelho nos 4 | 0,0015, 0,011, 0,00017, 0,00038 |
| `holdout`, `strong` | 191 | 0,540 | 0,671 | 0,506 | 0,325 | vermelho nos 4 | 2e-35, 2e-54, 4e-31, 4e-13 |

PSI de algumas variáveis (10 bins por quantis da referência; categórica: uma categoria por classe):

| Carga | luminance_mean | sharpness | file_bytes | confidence | entropy | predicted_class |
|---|---|---|---|---|---|---|
| `holdout`, sem drift | 0,104 | 0,091 | 0,076 | 0,031 | 0,054 | 3,10 |
| `val`, sem drift | 0,048 | 0,064 | 0,047 | 0,047 | 0,057 | 2,81 |
| `woof` | 0,095 | 0,196 | 0,142 | 0,552 | 0,598 | 21,2 |
| `holdout`, `mild` | 0,788 | 0,440 | 0,636 | 0,040 | 0,068 | 3,03 |
| `holdout`, `strong` | 11,3 | 7,45 | 12,4 | 1,88 | 1,98 | 5,44 |

## Leitura

1. **O controle fica verde** nos dois conjuntos sem drift (D de 0,011 a 0,034), com a referência de 500 imagens e janelas de ~190.
2. `strong` e `mild` ficam vermelhos nos 4 canais; o D de `strong` (0,33 a 0,67) é dez vezes o limiar vermelho.
3. **`woof` é o drift natural:** os pixels quase não mexem (D de 0,037 a 0,075, dois canais amarelos, nenhum vermelho, p > 0,4),
   mas a **saída** muda muito: confiança e entropia (PSI 0,55 e 0,60) e a classe prevista. Um monitor que olha só o pixel não   veria isso; é o motivo de monitorar entrada e saída juntas.
4. **Resultado inconveniente, reportado como medido:** o PSI de `predicted_class` dá 3,1 e 2,8 nos controles sem drift. Não é
   drift: são ~190 predições espalhadas por até 1000 classes do ImageNet, e categorias vistas só de um lado entram com o epsilon
   (1e-6) do PSI. Para variável categórica de cardinalidade alta, o PSI sobre classes não serve com janelas pequenas; leia o
   KS dos canais e o PSI de `confidence` e `entropy`. Não ajustamos limiar para mascarar isso.
5. `mild` quase não mexe na confiança (PSI 0,04) mas mexe forte na luminância (0,79): o modelo é robusto àquela mudança, e só a
   entrada denuncia o deslocamento.
6. Nenhum limiar foi ajustado para fazer o controle passar.
