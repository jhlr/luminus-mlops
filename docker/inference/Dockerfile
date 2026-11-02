# INFERENCE environment (SDD section 2): luminus_mlops + BentoML + torch/torchvision (MobileNetV3-Small).
# No pandas, scipy, mlflow, datasets or data. The ImageNet weights (~10 MB) are downloaded at BUILD time into
# /opt/torch, so the container needs no network and no mounts to serve.
# Build context = repository root:  docker build -f docker/inference/Dockerfile .
FROM python:3.13.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_NO_CACHE=1 PATH=/opt/venv/bin:$PATH

# 1) Exact dependencies from uv.lock (extra `inference`). The lock resolves torch with CUDA on Linux; here torch and
#    torchvision (same versions as the lock) come from the CPU-only PyTorch index and the nvidia/triton packages are left out
#    (except `nvidia-ml-py`, the `pynvml` that BentoML imports when it measures the machine resources).
COPY pyproject.toml uv.lock ./
RUN uv venv --seed /opt/venv \
 && uv export --frozen --no-dev --extra inference --no-emit-project --no-hashes --no-annotate -o /tmp/requirements.txt \
 && grep -E '^(torch|torchvision)==' /tmp/requirements.txt | sed 's/ ;.*//' > /tmp/torch.txt \
 && grep -v -E '^(torch|torchvision|triton|nvidia-|cuda-)' /tmp/requirements.txt > /tmp/rest.txt \
 && grep -E '^nvidia-ml-py==' /tmp/requirements.txt >> /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && rm -f /tmp/requirements.txt /tmp/rest.txt /tmp/torch.txt

# 2) The library, editable in /app.
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python --no-deps -e .

# 3) The model weights, fetched once at build time (world-readable: the container runs as an arbitrary UID).
ENV TORCH_HOME=/opt/torch
RUN python -c "from luminus_mlops.inference import ImageClassifier; ImageClassifier()" && chmod -R a+rX /opt/torch

# Arbitrary user (`user: UID:GID` in compose): HOME and caches must be writable.
ENV HOME=/tmp BENTOML_HOME=/tmp/bentoml XDG_CACHE_HOME=/tmp/cache SERVING_LOG_DIR=/serving-logs PYTHONUNBUFFERED=1
ARG SERVICE_VERSION=unknown
ENV SERVICE_VERSION=${SERVICE_VERSION}

EXPOSE 3000
CMD ["bentoml", "serve", "luminus_mlops.service:ClassifierService", "--host", "0.0.0.0", "--port", "3000"]
