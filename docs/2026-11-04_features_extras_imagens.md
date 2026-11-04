# Três características extras de imagem (atividade de MLOps)

Atividade: além das médias de Luminância, R, G e B, propor 3 características da imagem (numéricas ou categóricas), extraí-las com código gerado por uma ferramenta de IA (Claude) e mostrar definição, código, três imagens e o resultado.

O projeto deste repositório já extrai largura, altura, tamanho do arquivo, as médias L/R/G/B, o desvio-padrão da luminância (contraste) e a nitidez (variância do Laplaciano) em `src/luminus_mlops/monitoring/features.py`. As três abaixo são diferentes dessas.

## Definições

| # | Característica | Tipo | Definição | Por que importa no monitoramento |
|---|---|---|---|---|
| 1 | `colorfulness` | numérica, 0 a ~100+ | Métrica de Hasler e Süsstrunk (2003): com `rg = R - G` e `yb = 0.5(R+G) - B`, vale `sqrt(std(rg)² + std(yb)²) + 0.3 * sqrt(mean(rg)² + mean(yb)²)`. 0 é cinza puro. | Média de cor igual pode esconder fotos desbotadas ou saturadas demais; muda com filtro, câmera e iluminação. |
| 2 | `edge_density` | numérica, 0 a 1 | Fração de pixels cujo gradiente de Sobel (sobre a luminância, normalizado em níveis de cinza por pixel) passa de 25. | Mede complexidade do conteúdo (cena cheia de detalhe vs. fundo liso), independente de brilho e cor. |
| 3 | `dominant_color` | categórica (11 nomes) | Nome da cor mais frequente, pixel a pixel no espaço HSV: preto (V < 0,2), branco (S < 0,2 e V > 0,8), cinza (resto dos pouco saturados) ou um de red, orange, yellow, green, cyan, blue, purple, pink pelo matiz. | Uma categórica de entrada permite acompanhar proporção de classes no tempo (chi-quadrado, PSI categórico). |

## Código

Arquivo: [scripts/extra_features.py](../scripts/extra_features.py) (só `numpy` e `Pillow`).

```python
def colorfulness(rgb: np.ndarray) -> float:
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def edge_density(rgb: np.ndarray, threshold: float = 25.0) -> float:
    luma = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float64)
    gx = (luma[:-2, 2:] + 2 * luma[1:-1, 2:] + luma[2:, 2:] - luma[:-2, :-2] - 2 * luma[1:-1, :-2] - luma[2:, :-2]) / 8
    gy = (luma[2:, :-2] + 2 * luma[2:, 1:-1] + luma[2:, 2:] - luma[:-2, :-2] - 2 * luma[:-2, 1:-1] - luma[:-2, 2:]) / 8
    return float((np.hypot(gx, gy) > threshold).mean())


def dominant_color(rgb: np.ndarray) -> str:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float64)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    # classifica cada pixel (preto/branco/cinza ou faixa de matiz) e devolve a moda
    ...
```

O arquivo completo tem a tabela de faixas de matiz e o `main` que recebe os caminhos das imagens. Execução:

```bash
python scripts/extra_features.py docs/2026-11-04_features_extras_imagens/img*.jpg
```

## Imagens

Três fotos reais do conjunto Imagenette (`data/imagenette2-320/val`), uma de cada classe, escolhidas para dar resultados diferentes.

| Imagem | Arquivo | Classe |
|---|---|---|
| 1 | ![img1](2026-11-04_features_extras_imagens/img1_motosserra.jpg) | motosserra (homem podando árvore, céu claro) |
| 2 | ![img2](2026-11-04_features_extras_imagens/img2_trompa.jpg) | trompa (instrumento dourado) |
| 3 | ![img3](2026-11-04_features_extras_imagens/img3_bomba.jpg) | bomba de combustível |

## Resultado

Saída do script nas três imagens:

| Imagem | `colorfulness` | `edge_density` | `dominant_color` |
|---|---|---|---|
| img1_motosserra.jpg | 24.03 | 0.3659 | white |
| img2_trompa.jpg | 39.07 | 0.1534 | orange |
| img3_bomba.jpg | 46.70 | 0.0887 | gray |

Leitura:

1. `edge_density` separa bem as cenas: a árvore sem folhas contra o céu tem 37% de pixels de borda, a bomba (superfícies lisas) tem 9%.
2. `dominant_color` é um nome só, então perde informação: a bomba é vermelha e cinza, mas o cinza ocupa mais pixels e vence.
3. `colorfulness` é mais alta na bomba (46,7, com o corpo vermelho) do que na trompa (39,1), embora a categórica da trompa seja "orange" e a da bomba "gray": as duas medem coisas diferentes (intensidade da cor vs. cor mais frequente).

## Limites

1. Foram só 3 imagens, o que basta para mostrar o código funcionando, não para validar os limiares (25 para borda, 0,2 e 0,8 para S e V).
2. Os limiares são escolhas minhas, não calibradas em dados.
3. Estas variáveis não foram adicionadas ao `features.py` do projeto; o serviço continua com as 9 escalares e os 4 histogramas.
