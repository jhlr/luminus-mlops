# Como ler o `/metrics` do classificador (exemplo real)

Data: 2026-11-02 | Planilha com fórmulas vivas: `docs/2026-11-02_monitoramento_metrics_analysis.xlsx`
| Snapshot bruto usado aqui: `docs/2026-11-02_monitoramento_metrics_snapshot.txt` | Gerado por
`scripts/make_metrics_xlsx.py` a partir do snapshot.

O exemplo vem de uma execução real: serviço no ar (`just up-inference`, container Docker, 1 worker, CPU do Mac do autor) e
`scripts/load_gen.py --n 200 --traffic holdout` (5% de chamadas inválidas), mais 1 chamada manual com
`samples/sample_01.png`. Total de 201 chamadas a `/predict`: 192 com status 200, 4 com 400 e 5 com 500.
As imagens são do holdout do Imagenette (treino fora da referência). Os números abaixo são medidos, não estimados.

## 1. O que cada série significa

Nomes e rótulos reais do snapshot (BentoML 1.4):

| Série | Tipo | Rótulos | O que é |
|---|---|---|---|
| `bentoml_service_request_total` | counter | `endpoint`, `http_response_code`, `runner_name`, `service_name`, `service_version` | Quantas requisições terminaram, por endpoint e código HTTP. Só cresce (até o processo reiniciar). |
| `bentoml_service_request_duration_seconds_bucket` | histograma | os mesmos + `le` | Quantas requisições levaram **até** `le` segundos. É **cumulativo**: o bucket `le="0.05"` inclui os de `le="0.045"`. |
| `bentoml_service_request_duration_seconds_sum` / `_count` | counter | os mesmos | Soma das durações (s) e número de observações. |
| `bentoml_service_request_in_progress` | gauge | `endpoint`, ... | Requisições em andamento agora (sobe e desce). |
| `bentoml_service_last_request_timestamp_seconds` | gauge | `pid`, ... | Instante da última requisição. |
| `luminus_input_luminance_mean_bucket/_sum/_count` | histograma | `le` | Luminância média (0-255) da imagem de entrada, buckets de 16 em 16. |
| `luminus_prediction_confidence_bucket/_sum/_count` | histograma | `le` | Confiança top-1 (softmax), buckets de 0,1 em 0,1. |
| `luminus_prediction_entropy_bucket/_sum/_count` | histograma | `le` | Entropia (bits) do softmax de 1000 classes. |
| `luminus_prediction_confidence_band_total` | counter | `band` (3 valores) | Predições por faixa de confiança: `low` (< 0,5), `medium` (0,5 a 0,8), `high` (>= 0,8). |

**Contador cumulativo:** o valor é o total desde que o processo subiu. Para saber "quantas por
segundo" usa-se a diferença entre duas leituras (`rate()`/`increase()` no Prometheus).
**Buckets `le`** ("menor ou igual"): `le="+Inf"` é igual a `_count`. Como são cumulativos, a
contagem de um bucket isolado é a diferença para o anterior.

Exemplo, `/predict` com status 200 (snapshot): `_count = 192`, `_sum = 9,0083 s`, logo a
média é `_sum / _count = 46,92 ms`.

Série customizada no mesmo snapshot: `luminus_prediction_confidence_band_total` = 57 low, 34 medium, 101 high
(soma 192, igual ao número de respostas 200). `luminus_input_luminance_mean_count = 192`, `_sum = 22028,3`, média 114,7.

## 2. Histograma de latência: de-acumular por diferença

Buckets finais (seção 6): 0,01 / 0,025 / 0,035 / 0,0375 / 0,04 / 0,0415 / 0,043 / 0,045 / 0,05 / 0,06 / 0,07 / 0,0725 / 0,075 / 0,0775 / 0,08 / 0,09 / 0,1 / 0,15 / 0,25 / 0,5 / 1 / 2,5 / 5 s.

| `le` (s) | acumulado | contagem no bucket (diferença) | fração |
|---|---|---|---|
| até 0,04 | 0 | 0 | 0% |
| 0,0415 | 3 | 3 | 1,6% |
| 0,043 | 11 | 8 | 4,2% |
| 0,045 | 84 | 73 | 38,0% |
| 0,05 | 179 | 95 | 49,5% |
| 0,06 | 187 | 8 | 4,2% |
| 0,07 | 188 | 1 | 0,5% |
| 0,0775 | 189 | 1 | 0,5% |
| 0,08 | 191 | 2 | 1,0% |
| 0,25 | 192 | 1 | 0,5% |
| demais até +Inf | 192 | 0 | 0% |

Na planilha, aba `histogram`, cada contagem é `=raw!B{i}-raw!B{i-1}`. Para o gráfico de barras
usa-se essa coluna (nunca o acumulado). Maior fração por bucket: 49,5% (0,045 a 0,05 s), por pouco abaixo da meta de 50%.

## 3. "Boxplot" de latência a partir dos buckets (só uma aproximação)

O Prometheus não guarda observações, só contagens por bucket. Os quartis saem por interpolação
linear dentro do bucket, a mesma lógica de `histogram_quantile`: posição = `q x total`; acha-se o
primeiro bucket cujo acumulado alcança a posição; valor = `inferior + (superior - inferior) x
(posição - acumulado_antes) / (acumulado_no_bucket - acumulado_antes)`. A interpolação supõe
observações uniformes dentro do bucket.

| | Q1 | mediana | Q3 | whisker mín. | whisker máx. |
|---|---|---|---|---|---|
| Dos buckets (aba `quantiles`) | 44,01 ms | 45,63 ms | 48,16 ms | 40 ms | 250 ms |
| Exato, de `latency_ms` do log | 43,68 ms | 44,54 ms | 45,89 ms | 40,26 ms | 177,60 ms |

Whiskers dos buckets = limite inferior do primeiro bucket não vazio e limite superior do último
não vazio. A diferença para o exato mostra o erro da aproximação (aqui de 1 a 2 ms, porque os buckets são
estreitos perto da mediana); para quartis exatos use o log. Q1, mediana e Q3 da planilha foram recalculados em Python de forma
independente (`histogram_quantile` à mão sobre o snapshot) e batem (0,044014 / 0,045632 / 0,048158 s); a planilha foi
recalculada também no LibreOffice (headless) e deu os mesmos valores.

## 4. Taxa de erro

Com os nomes reais:

```promql
# acumulada desde o início do processo (5xx e 4xx separados)
sum(bentoml_service_request_total{endpoint="/predict",http_response_code=~"5.."}) / sum(bentoml_service_request_total{endpoint="/predict"})
sum(bentoml_service_request_total{endpoint="/predict",http_response_code=~"4.."}) / sum(bentoml_service_request_total{endpoint="/predict"})
# em janela de 5 minutos
sum(increase(bentoml_service_request_total{endpoint="/predict",http_response_code=~"5.."}[5m])) / sum(increase(bentoml_service_request_total{endpoint="/predict"}[5m]))
sum(rate(bentoml_service_request_total{endpoint="/predict",http_response_code=~"5.."}[5m])) / sum(rate(bentoml_service_request_total{endpoint="/predict"}[5m]))
```

No snapshot (`/predict`): 200 = 192, 400 = 4, 500 = 5, total 201, taxa 5xx = 5/201 = 2,5% e taxa 4xx = 4/201 = 2,0%
(aba `errors`). Atenção: `/readyz` e `/livez` também geram séries em `bentoml_service_request_total`; por isso os filtros
`endpoint="/predict"` acima.

### Códigos HTTP medidos para entrada inválida

| Entrada inválida | Código | Linha `response` do log (`error`) |
|---|---|---|
| Sem o campo `image` (multipart sem `image`) | 400 | `ValidationError/invalid_request` |
| Corpo vazio | 400 | `ValidationError/invalid_request` |
| Bytes que não são imagem (`image` com texto) | **500** | `HTTPError/http_500` |
| JPEG truncado | **500** | `OSError/invalid_image` |

O 500 para "bytes que não são imagem" vem do BentoML, não do nosso código: o decodificador de
`PIL.Image` levanta `UnidentifiedImageError` dentro da validação da entrada e o BentoML a trata como
erro interno em vez de 4xx. Como a função da API nunca roda nesse caso, o log só sabe o status
(por isso `HTTPError/http_500`, sem a classe da exceção). Não alteramos esse comportamento. Para o JPEG
truncado a função roda e registra `OSError/invalid_image`.

## 5. Retenção, precisamente

- **Métricas** vivem na memória do processo de inferência e **zeram quando ele reinicia**; a
  aplicação não persiste nada.
- **Prometheus** guarda no TSDB por 15 dias (`--storage.tsdb.retention.time=15d`, definido explicitamente
  em `docker/monitoring/compose.yml`).
- **Log em arquivo** (`serving-logs/inference.log`) dura até a rotação descartar o 5o backup:
  50 MB por arquivo, 5 backups, no máximo 300 MB. Com cerca de 5,8 KB por par requisição+resposta medido,
  um arquivo comporta cerca de 9.000 pares e os 6 arquivos cerca de 54.000 pares.

## 6. Medições registradas

1. **Código HTTP de entrada inválida:** tabela da seção 4.
2. **Séries reais do `/metrics`:** tabela da seção 1 (todos os nomes vêm do snapshot).
3. **Distribuição de latência** (log, `latency_ms`, 192 chamadas 200 de imagens Imagenette 320 px, sequencial, 1 worker, container Docker):
   a 1a chamada da janela (a manual, com `samples/sample_01.png`) levou **177,6 ms**; o modelo carrega no `__init__`, então o que
   sobra é aquecimento do PyTorch e do cache. Sem ela (n = 191): mínimo 40,3 ms, **p50 = 44,5 ms, p95 = 50,7 ms**, máximo 78,5 ms.
   Com ela: p50 = 44,5 ms, p95 = 50,9 ms. Latência do cliente (`load_gen`, inclui rede local e upload): média 46,7 ms,
   p50 47,5 ms, p95 53,5 ms. No host (`just serve`, sem Docker) a mesma carga deu p50 = 76,6 ms e p95 = 88,6 ms. A diferença foi medida e não foi investigada.
   **Buckets:** a grade calibrada só com o host (menor bucket em 0,05 s) colocava 88% das chamadas do Docker (836 de 945) no bucket 0,025 a 0,05 s.
   A grade final acima, com buckets de 1,5 a 2,5 ms nas regiões do host (0,07 a 0,08 s) e do
   Docker (0,04 a 0,05 s), deixa no máximo 49,5% por bucket no Docker. É a grade deste hardware; em outra máquina, meça e ajuste
   (`DURATION_BUCKETS` em `src/luminus_mlops/monitoring/metrics.py`).
4. **Bytes por par requisição+resposta no log real:** cerca de 5.800 bytes em média (a linha `request` carrega os 4
   histogramas de 256 inteiros e é quase tudo; a `response` tem uns 370 B).
5. **Contagem de arquivos dos conjuntos de imagens:** Imagenette 320 px: treino 9.469 (referência 500, holdout 8.969), val 3.925;
   Imagewoof 320 px: val 3.929. Verificado com `split_reference_images` e `traffic_images`.
6. **BentoML 1.4:** o hook de middleware existe (`Service.add_asgi_middleware`) e funciona;
   `ctx: bentoml.Context` na API é aceito e não altera o esquema de entrada; `prometheus_client` direto aparece no
   mesmo `/metrics` com 1 worker, desde que os objetos sejam criados dentro do worker (o BentoML define
   `PROMETHEUS_MULTIPROC_DIR` depois do import do módulo); buckets de duração configuráveis por
   `@bentoml.service(metrics={"duration": {"buckets": [...]}})`. **A duração do BentoML e a `latency_ms` do log concordam:**
   soma de duração do BentoML 9,0083 s contra soma de `latency_ms` 8,8675 s em 192 chamadas (diferença de 0,73 ms por
   chamada). `@bentoml.api` só serve POST; as rotas GET `/data/*` entram por `@bentoml.asgi_app`.
7. **`RotatingFileHandler` criado de forma preguiçosa no worker rotaciona em processo servido:**
   sim, verificado (`tests/monitoring/test_rotation.py`, `SERVING_LOG_MAX_BYTES=20000`: 8 chamadas geram
   `inference.log.1` e as 16 linhas aparecem somadas sem perda).
