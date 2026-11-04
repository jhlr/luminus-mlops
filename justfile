# Luminus MLOps: setup, servico, carga e ambientes Docker. Veja o README.md para o passo a passo.

DATA_URL := "https://s3.amazonaws.com/fast-ai-imageclas"

# instala as dependencias de todos os ambientes (uv.lock; --all-extras = inference + reference + monitor)
setup:
    uv sync --all-extras

# baixa e extrai Imagenette (~340 MB) e Imagewoof (~330 MB), 320px (fast.ai), em data/ (pula se ja existir)
data:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p data
    for name in imagenette2-320 imagewoof2-320; do
        if [ -d "data/$name/train" ]; then echo "data/$name already there, skipping."; continue; fi
        curl -fL "{{DATA_URL}}/$name.tgz" | tar -xz -C data
    done

# regenera as 5 imagens sinteticas de exemplo em samples/
samples:
    uv run python scripts/gen_sample_images.py

# serve o classificador em http://localhost:3000 (caminho local, sem Docker)
serve:
    uv run --extra inference bentoml serve luminus_mlops.service:ClassifierService --port 3000

# uma chamada de exemplo ao servico em execucao
predict:
    curl -s -F image=@samples/sample_01.png http://localhost:3000/predict | python3 -m json.tool

# gera o conjunto de referencia (artifacts/reference*.csv) com o modelo real; o ambiente de "treino"
build-reference:
    uv run --extra reference python scripts/build_reference.py

# gera carga contra o servico em execucao (ex.: just load --n 100 --drift strong --traffic holdout)
load *ARGS:
    uv run --extra monitor python scripts/load_gen.py {{ARGS}}

# checagem de tipos e testes
check:
    uv run --all-extras mypy
    uv run --all-extras pytest tests

# --- Docker: ambientes separados de referencia, inferencia e monitoramento (docs/2026-11-02_sdd_monitoramento_mlops.md, secao 2) ---

# cria a rede compartilhada se faltar (os dois compose a declaram `external`)
_luminus-net:
    docker network inspect luminus-net >/dev/null 2>&1 || docker network create luminus-net

# ambiente de REFERENCIA: gera artifacts/reference*.csv dentro da propria imagem (data/ somente leitura, artifacts/ gravavel)
build-reference-docker:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p artifacts
    export UID GID="$(id -g)"
    docker compose -f docker/reference/compose.yml run --rm --build build-reference

# ambiente de INFERENCIA em http://localhost:3000 (log em ./serving-logs)
up-inference: _luminus-net
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p serving-logs   # criado pelo usuario do host (senao o Docker cria como root)
    export UID GID="$(id -g)" SERVICE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    docker compose -f docker/inference/compose.yml up -d --build --wait --wait-timeout 300

down-inference:
    docker compose -f docker/inference/compose.yml down

# ambiente de MONITORAMENTO: Prometheus :9090, Loki :3100, Alloy :12345, Grafana :3001, monitor :3002 (independente da inferencia)
up-monitoring: _luminus-net
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p serving-logs
    export UID GID="$(id -g)" SERVICE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    docker compose -f docker/monitoring/compose.yml up -d --build --wait --wait-timeout 300

down-monitoring:
    docker compose -f docker/monitoring/compose.yml down
