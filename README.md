# Luminus MLOps: monitoramento de um classificador de imagens

Trabalho de MLOps (Cesar School), grupo Luminus. Um classificador pronto da internet (torchvision **MobileNetV3-Small**, pesos
do ImageNet, CPU) atrás de um serviço BentoML, com **log de requisição/resposta em arquivo**, `GET /metrics` (Prometheus),
drift (KS e PSI) contra um conjunto de referência, Loki, Prometheus e Grafana. Nada é treinado aqui.

Monitorar aqui significa, antes de tudo, **gravar em arquivo um rastro de cada requisição e de cada resposta**. Seguindo o padrão
da disciplina, o log guarda variáveis extraídas da imagem (médias e histogramas 0-255 de L, R, G, B, contraste, nitidez, tamanho)
e da saída do modelo (classe prevista, confiança, margem, entropia). Nunca os pixels.

## Ver o log crescer (caminho mais curto, sem Docker e sem baixar dataset)

Precisa de [uv](https://docs.astral.sh/uv/) e [just](https://github.com/casey/just).

```bash
just setup                                           # 1. instala as dependências
just samples                                         # 2. gera 5 imagens sintéticas em samples/
just serve                                           # 3. terminal A: serviço em http://localhost:3000
tail -f serving-logs/inference.log                   # 4. terminal B: o log (uma linha JSON por evento)
just load --images-dir samples --n 50                # 5. terminal C: gera carga (laço fechado, 1 chamada por vez)
curl -s localhost:3000/metrics | grep luminus_       # 6. métricas Prometheus
curl -s "localhost:3000/data/variables?format=csv" | head -3   # 7. variáveis extraídas do log
```

Cada chamada a `POST /predict` grava **duas linhas** ligadas por `request_id`: `request` (variáveis `in_*` da imagem) e `response`
(status, `latency_ms`, variáveis `out_*`, `error`). Chamadas inválidas (corpo vazio, campo ausente) geram só a linha `response`.
`/metrics`, `/livez`, `/readyz` e `/data/*` não escrevem no log. Resposta do serviço:

```bash
curl -s -F image=@samples/sample_01.png localhost:3000/predict   # ou: just predict
# {"label": "wall clock", "confidence": 0.0562, "top5": [{"label": "wall clock", "confidence": 0.0562}, ...]}
```

## Caminho Docker (os três ambientes + Grafana)

```bash
just data                  # 1. baixa Imagenette 320 px (~340 MB) e Imagewoof 320 px (~330 MB) em data/ (só para gerar carga real)
just up-inference          # 2. inferência em http://localhost:3000 (log em ./serving-logs)
just up-monitoring         # 3. Prometheus :9090, Loki :3100, Alloy :12345, Grafana :3001, monitor :3002
just load --n 200          # 4. carga sem drift (controle); veja as variações abaixo
tail -f serving-logs/inference.log
open http://localhost:3001 # 5. Grafana, dashboard "Luminus: monitoramento" (sem login)
just down-monitoring && just down-inference
```

`artifacts/reference.csv` e `reference_hist.csv` já estão no repositório; para refazê-los: `just build-reference-docker`
(ambiente de referência, imagem própria) ou `just build-reference` (local).

### Gerar carga

`just load` roda `scripts/load_gen.py` (cliente `httpx`, fora dos ambientes). Chamadas em **laço fechado**: a próxima só sai quando
a anterior termina (`--concurrency`, padrão 1), porque o serviço tem um único worker. Argumentos úteis:

| Argumento | Valores |
|---|---|
| `--traffic` | `holdout` (imagens de treino fora da referência: controle sem drift), `val` (outra divisão do Imagenette), `woof` (Imagewoof, só cães: drift natural) |
| `--drift` | `none`, `mild` (brilho x0,75 e +15 de vermelho), `strong` (brilho x0,4, tom azulado e ruído) |
| `--n`, `--invalid-pct`, `--images-dir` | número de chamadas (padrão 200), % de inválidas (padrão 5), pasta de imagens qualquer |

Exemplo: `just load --n 200 --traffic holdout --drift strong` e acompanhe o painel "Drift" ficar vermelho.

## Arquitetura: três ambientes separados

```
 REFERÊNCIA ("treino")        INFERÊNCIA                         MONITORAMENTO
 docker/reference             docker/inference                   docker/monitoring
 extra `reference`            extra `inference`                  Prometheus, Loki, Alloy, Grafana,
 build_reference.py           bentoml serve :3000                `monitor` (BentoML + pandas + scipy)
        |                     escreve o log JSONL                         ^        ^
        | artifacts/*.csv     expõe /metrics  ---- scrape (luminus-net) --+        |
        +--- somente leitura --------------------------------------------+        |
                              ./serving-logs (bind mount) --somente leitura-------+ (Alloy e monitor)
```

| Ambiente | Contém | Não contém |
|---|---|---|
| Referência | torch, torchvision, scipy; gera `artifacts/reference*.csv` com o modelo real | BentoML, código de serviço |
| Inferência | BentoML, torch, torchvision, escrita do log, `/data/variables` | pandas, scipy, mlflow, análise de drift |
| Monitoramento | Prometheus, Loki, Alloy, Grafana, `monitor` | torch, pesos do modelo |

Cada um sobe e desce sozinho (`just up-inference`, `just up-monitoring`, `just build-reference-docker`); a inferência continua
gravando o log com o monitoramento parado. O contêiner de inferência roda com o sistema de arquivos somente leitura (só `/tmp` e
`serving-logs/` são graváveis). **Um único worker** BentoML escreve o log, porque o `RotatingFileHandler` (50 MB, 5 backups) não é
seguro entre processos.

Tamanhos de imagem, tempos de build e latência no Docker foram medidos neste Mac: `docs/2026-11-04_medicoes_docker_e_loki.md`.

## Modelo e dados

- **Modelo:** `torchvision.models.mobilenet_v3_small`, pesos `MobileNet_V3_Small_Weights.IMAGENET1K_V1` (~10 MB, baixados do
  model zoo do torchvision, `download.pytorch.org`, no build da imagem). O código do torchvision é BSD-3; os pesos foram
  treinados no ImageNet, cujos termos de uso são para pesquisa não comercial, e o torchvision não publica uma licença à parte para
  os pesos: use só para estudo.
- **Dados:** [Imagenette 320 px](https://github.com/fastai/imagenette) (fast.ai, 10 classes do ImageNet, ~340 MB) e
  [Imagewoof 320 px](https://github.com/fastai/imagenette) (10 raças de cães, ~330 MB), baixados por `just data` para `data/`
  (fora do git). **Referência** = 500 imagens do treino do Imagenette (caminhos ordenados, embaralhados com seed 0, as primeiras 500);
  `holdout` = o resto do treino. Com 500 imagens a referência é pequena: leia cada KS e PSI com essa ressalva.

## Variáveis do log

| Entrada (`in_*`) | Saída (`out_*`) |
|---|---|
| `width`, `height`, `file_bytes` | `predicted_class` (categórica) |
| `luminance_mean`, `r_mean`, `g_mean`, `b_mean` | `confidence` (softmax top-1) |
| `luminance_std` (contraste), `sharpness` (variância do Laplaciano) | `margin` (top-1 menos top-2) |
| `hist_l`, `hist_r`, `hist_g`, `hist_b` (256 bins) | `entropy` (bits do softmax), `top5_mass` |

Três variáveis extras de imagem (colorfulness, edge density, cor dominante) foram propostas e testadas fora do serviço: `docs/2026-11-04_features_extras_imagens.md`.

Mais `ts`, `request_id`, `status_code`, `latency_ms` e `error`. Privacidade: sem pixels; `error` é a classe da exceção mais um código
fixo (nunca a mensagem); o IP do cliente só entra como os 12 primeiros hex do SHA-256.

## `/metrics`, Prometheus, Loki, Grafana

- **`GET /metrics`** (formato Prometheus, nativo do BentoML): `bentoml_service_request_total`,
  `bentoml_service_request_duration_seconds_{bucket,sum,count}`, mais `luminus_input_luminance_mean`, `luminus_prediction_confidence`,
  `luminus_prediction_entropy` (histogramas) e `luminus_prediction_confidence_band_total`. **Como ler, com exemplo real:
  `docs/2026-11-02_monitoramento_metrics_analysis.md` e a planilha `.xlsx` (fórmulas vivas)**: histograma e boxplot de latência a
  partir dos buckets, taxa de erro e retenção (métricas na memória do processo, Prometheus 15 dias, log até 300 MB).
- **Prometheus** (`:9090`) raspa `inference:3000/metrics` e `monitor:3000/metrics` a cada 5 s. Consultas:
  ```promql
  histogram_quantile(0.95, sum by (le) (rate(bentoml_service_request_duration_seconds_bucket{endpoint="/predict"}[1m])))
  sum(rate(bentoml_service_request_total{endpoint="/predict",http_response_code=~"5.."}[5m])) / sum(rate(bentoml_service_request_total{endpoint="/predict"}[5m]))
  sum(rate(bentoml_service_request_total{endpoint="/predict"}[1m]))
  ```
- **Loki** (`:3100`): o Loki guarda logs mas não lê arquivos; o **Alloy** lê `serving-logs/inference.log` e envia as linhas.
  Rótulos só `job` e `event`. Exemplo: `{job="luminus", event="response"} | json | status_code >= 500`. O Alloy lê só o arquivo ativo, para o Loki não reler as linhas dos arquivos rotacionados (medição em `docs/2026-11-04_medicoes_docker_e_loki.md`).
- **Grafana** (`:3001`, sem login, tudo provisionado por arquivo): um dashboard com operação (taxa, erros, latência p50/p95/p99 e
  histograma), entrada (histogramas L, R, G, B contra a referência, boxplots), saída (boxplots de confiança e entropia, histogramas),
  drift (KS D por canal e nível) e logs (Loki). O Grafana não tem box plot nativo: usamos o painel Business Charts (ECharts) alimentado
  por `/data/summary` via Infinity. Gerado por `scripts/build_dashboards.py`. Os 25 painéis foram consultados pela API do Grafana e
  todos devolveram dados após uma carga.

## Extrair as variáveis e calcular KS e PSI (E2)

`GET /data/variables?since=&until=&limit=&format=json|csv&include_hist=true|false` (uma linha por `request_id`) existe na inferência
(`:3000`) e no `monitor` (`:3002`). Só o `monitor` tem `/data/reference`, `/data/reference/hist`, `/data/summary` e
`/data/drift?channel=L|R|G|B|all&since=&until=`. O `monitor` também expõe os gauges `luminus_drift_ks_d`, `luminus_drift_ks_pvalue`,
`luminus_drift_level` (-1 inconclusivo, 0 verde, 1 amarelo, 2 vermelho) e `luminus_drift_psi` da janela das últimas 100 imagens.

```python
import httpx
d = httpx.get("http://localhost:3002/data/drift").json()          # KS e PSI da janela deslizante
print(d["channels"]["L"])                                          # {'d': ..., 'p_value': ..., 'level_name': ...}
print(d["psi"]["luminance_mean"])                                  # {'value': ..., 'band': ...}
```

KS por canal: D = máximo da diferença entre as CDFs dos histogramas agregados (peso igual por imagem), p-valor com n em imagens;
amarelo em D >= 0,05, vermelho em D >= 0,10, inconclusivo com menos de 8 imagens. PSI: faixas 0,10 e 0,25 (convenção da indústria, não
regra provada). Resultados medidos por carga, incluindo o controle: `docs/2026-11-02_monitoramento_drift_resultados.md`.

## Onde ler primeiro (para estudar o código)

1. `src/luminus_mlops/monitoring/logging_setup.py`: como e onde o log é escrito (JSON Lines, rotação, timestamp, IP em hash).
2. `src/luminus_mlops/monitoring/tracing.py`: o middleware que grava as linhas `request` e `response` de cada chamada.
3. `src/luminus_mlops/monitoring/features.py`: variáveis extraídas da imagem. Depois `outputs.py`: variáveis da saída do modelo.
4. `src/luminus_mlops/service.py` e `inference.py`: o serviço BentoML e o modelo, juntos em pouco código.
5. `src/luminus_mlops/monitoring/drift.py`: a matemática do KS, do PSI e os limiares (um único dono).
6. `src/luminus_mlops/monitoring/metrics.py`: métricas Prometheus próprias; `log_reader.py` e `data_api.py`: ler o log de volta.
7. `monitor_service.py` e `monitoring/analysis.py`: o serviço do ambiente de monitoramento. `scripts/load_gen.py`: o gerador de carga.
8. `tests/monitoring/`: cada regra tem um teste; `just check` roda `mypy` (strict) e `pytest`.

## Notas e limites

- Entrada que não é imagem devolve **500** (comportamento do BentoML antes de chamar a nossa função); corpo vazio ou campo ausente, 400.
- A `latency_ms` do log e a duração do BentoML concordam (diferença de ~0,7 ms por chamada); a primeira chamada de um processo é mais lenta.
- Os buckets de latência (`DURATION_BUCKETS`) foram calibrados com a latência medida neste Mac; em outra máquina, meça e ajuste.
- PSI de `predicted_class` fica alto mesmo sem drift (1000 classes, janelas pequenas); ver `docs/2026-11-02_monitoramento_drift_resultados.md`.
- Documentos: `docs/2026-11-02_sdd_monitoramento_mlops.md` (projeto, em inglês), análise de `/metrics` e resultados de drift.
