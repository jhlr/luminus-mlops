# REFERENCE ("training") environment (SDD section 2): torch/torchvision + scipy + Pillow + numpy. Nothing is trained here:
# the job runs the real model over the reference images and writes artifacts/reference*.csv.
# No BentoML, Prometheus client, pandas or serving code path. Build context = repository root.
FROM python:3.13.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_NO_CACHE=1 PATH=/opt/venv/bin:$PATH

COPY pyproject.toml uv.lock ./
RUN uv venv --seed /opt/venv \
 && uv export --frozen --no-dev --extra reference --no-emit-project --no-hashes --no-annotate -o /tmp/requirements.txt \
 && grep -E '^(torch|torchvision)==' /tmp/requirements.txt | sed 's/ ;.*//' > /tmp/torch.txt \
 && grep -v -E '^(torch|torchvision|triton|nvidia-|cuda-)' /tmp/requirements.txt > /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/rest.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && rm -f /tmp/requirements.txt /tmp/rest.txt /tmp/torch.txt

COPY src ./src
COPY scripts/build_reference.py ./scripts/build_reference.py
RUN uv pip install --python /opt/venv/bin/python --no-deps -e .

ENV TORCH_HOME=/opt/torch
RUN python -c "from luminus_mlops.inference import ImageClassifier; ImageClassifier()" && chmod -R a+rX /opt/torch
ENV HOME=/tmp XDG_CACHE_HOME=/tmp/cache PYTHONUNBUFFERED=1

CMD ["python", "scripts/build_reference.py", "--out-dir", "/app/artifacts"]
