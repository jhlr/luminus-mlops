# Luminus MLOps: setup, serve, load and Docker environments. See README.md for the step-by-step.

DATA_URL := "https://s3.amazonaws.com/fast-ai-imageclas"

# install the dependencies of all environments (uv.lock; --all-extras = inference + reference + monitor)
setup:
    uv sync --all-extras

# download and extract Imagenette (~340 MB) and Imagewoof (~330 MB), 320px (fast.ai), into data/ (skipped if present)
data:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p data
    for name in imagenette2-320 imagewoof2-320; do
        if [ -d "data/$name/train" ]; then echo "data/$name already there, skipping."; continue; fi
        curl -fL "{{DATA_URL}}/$name.tgz" -o "data/$name.tgz"
        tar -xzf "data/$name.tgz" -C data
        mkdir -p ~/.Trash && mv "data/$name.tgz" ~/.Trash/
    done

# regenerate the 5 synthetic sample images in samples/
samples:
    uv run python scripts/gen_sample_images.py

# serve the classifier at http://localhost:3000 (local path, no Docker)
serve:
    uv run --extra inference bentoml serve luminus_mlops.service:ClassifierService --port 3000

# one example call against the running service
predict:
    curl -s -F image=@samples/sample_01.png http://localhost:3000/predict | python3 -m json.tool

# build the reference set (artifacts/reference*.csv) with the real model; the "training" environment
build-reference:
    uv run --extra reference python scripts/build_reference.py

# generate load against the running service (e.g.: just load --n 100 --drift strong --traffic holdout)
load *ARGS:
    uv run --extra monitor python scripts/load_gen.py {{ARGS}}

# drift analysis over the log (KS per channel, PSI, boxplots) into reports/<label>/ (e.g.: just analyze --last 20 --label strong)
analyze *ARGS:
    uv run --extra monitor python scripts/analyze_log.py {{ARGS}}

# type check and tests
check:
    uv run mypy
    uv run pytest tests

# --- Docker: separate reference, inference and monitoring environments (docs/2026-11-02_sdd_monitoramento_mlops.md, section 2) ---

# create the shared network if missing (both compose files declare it `external`)
_luminus-net:
    docker network inspect luminus-net >/dev/null 2>&1 || docker network create luminus-net

# REFERENCE environment: build artifacts/reference*.csv inside its own image (data/ read-only, artifacts/ writable)
build-reference-docker:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p artifacts
    export UID GID="$(id -g)"
    docker compose -f docker/reference/compose.yml run --rm --build build-reference

# INFERENCE environment at http://localhost:3000 (log in ./serving-logs)
up-inference: _luminus-net
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p serving-logs   # created by the host user (otherwise Docker creates it as root)
    export UID GID="$(id -g)" SERVICE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    docker compose -f docker/inference/compose.yml up -d --build --wait --wait-timeout 300

down-inference:
    docker compose -f docker/inference/compose.yml down

# MONITORING environment: Prometheus :9090, Loki :3100, Alloy :12345, Grafana :3001, monitor :3002 (independent of inference)
up-monitoring: _luminus-net
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p serving-logs
    export UID GID="$(id -g)" SERVICE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    docker compose -f docker/monitoring/compose.yml up -d --build --wait --wait-timeout 300

down-monitoring:
    docker compose -f docker/monitoring/compose.yml down
