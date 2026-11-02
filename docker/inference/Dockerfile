# Ambiente de INFERENCIA (SDD secao 2): luminus_mlops + BentoML + torch/torchvision (MobileNetV3-Small).
# Sem pandas, scipy, mlflow, datasets nem dados. Os pesos do ImageNet (~10 MB) sao baixados no BUILD em
# /opt/torch, entao o container nao precisa de rede nem de montagens para servir.
# Contexto de build = raiz do repositorio:  docker build -f docker/inference/Dockerfile .
FROM python:3.13.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_NO_CACHE=1 PATH=/opt/venv/bin:$PATH

# 1) Dependencias exatas do uv.lock (extra `inference`). O lock resolve o torch com CUDA no Linux; aqui o torch e o
#    torchvision (mesmas versoes do lock) vem do indice PyTorch so de CPU e os pacotes nvidia/triton ficam de fora
#    (exceto `nvidia-ml-py`, o `pynvml` que o BentoML importa ao medir os recursos da maquina).
COPY pyproject.toml uv.lock ./
RUN uv venv --seed /opt/venv \
 && uv export --frozen --no-dev --extra inference --no-emit-project --no-hashes --no-annotate -o /tmp/requirements.txt \
 && grep -E '^(torch|torchvision)==' /tmp/requirements.txt | sed 's/ ;.*//' > /tmp/torch.txt \
 && grep -v -E '^(torch|torchvision|triton|nvidia-|cuda-)' /tmp/requirements.txt > /tmp/rest.txt \
 && grep -E '^nvidia-ml-py==' /tmp/requirements.txt >> /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && rm -f /tmp/requirements.txt /tmp/rest.txt /tmp/torch.txt

# 2) A biblioteca, editavel em /app.
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python --no-deps -e .

# 3) Os pesos do modelo, baixados uma vez no build (legiveis por todos: o container roda com um UID arbitrario).
ENV TORCH_HOME=/opt/torch
RUN python -c "from luminus_mlops.inference import ImageClassifier; ImageClassifier()" && chmod -R a+rX /opt/torch

# Usuario arbitrario (`user: UID:GID` no compose): HOME e caches precisam ser gravaveis.
ENV HOME=/tmp BENTOML_HOME=/tmp/bentoml XDG_CACHE_HOME=/tmp/cache SERVING_LOG_DIR=/serving-logs PYTHONUNBUFFERED=1
ARG SERVICE_VERSION=unknown
ENV SERVICE_VERSION=${SERVICE_VERSION}

EXPOSE 3000
CMD ["bentoml", "serve", "luminus_mlops.service:ClassifierService", "--host", "0.0.0.0", "--port", "3000"]
