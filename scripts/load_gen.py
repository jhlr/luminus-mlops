"""Gerador de carga: cliente HTTP que chama `POST /predict` (SDD secao 8).

  just load                       # 200 chamadas, uma por vez, trafego holdout, sem drift
  uv run python scripts/load_gen.py --n 100 --traffic holdout --drift strong

`--traffic`: holdout (imagens de treino do Imagenette fora da referencia, controle sem drift), val (outra
divisao do Imagenette, mesma distribuicao) e woof (Imagewoof: so caes, drift natural de classe).
`--images-dir DIR` usa as imagens de uma pasta qualquer. `--drift`: transformacao sintetica (Pillow) aplicada antes do envio: mild (brilho x0.75 e +15 de
vermelho) ou strong (brilho x0.4, azulado e ruido gaussiano). `--invalid-pct` mistura requisicoes
invalidas (bytes que nao sao imagem, corpo vazio, campo `image` ausente, JPEG truncado).
Carga em laco fechado: `--concurrency` (padrao 1) e o numero de chamadas em andamento ao mesmo tempo; a
proxima so sai quando uma anterior termina (um unico worker). Despejar chamadas a uma taxa fixa
(`--rps`, opcional) acima da capacidade forma fila no servidor. `--rps` so limita o ritmo maximo de envio.
"""

from __future__ import annotations

import argparse
import io
import random
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from luminus_mlops.monitoring.split import DEFAULT_SEED, TRAFFIC_SETS, list_images, traffic_images

INVALID_KINDS = ("not_an_image", "empty_body", "missing_field", "truncated_jpeg")
NOISE_SIGMA = 12.0


@dataclass(frozen=True)
class Job:
    kind: str  # "valid" ou um item de INVALID_KINDS
    path: Path


def apply_drift(image: Image.Image, drift: str, rng: np.random.Generator) -> Image.Image:
    """Transformacao sintetica de iluminacao/cor sobre a imagem RGB."""
    if drift == "none":
        return image
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    if drift == "mild":
        pixels = pixels * 0.75
        pixels[..., 0] += 15
    elif drift == "strong":
        pixels = pixels * 0.4
        pixels[..., 0] *= 0.6
        pixels[..., 2] *= 1.3
        pixels = pixels + rng.normal(0.0, NOISE_SIGMA, size=pixels.shape)
    else:
        raise ValueError(f"drift desconhecido: {drift!r}")
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def encode(path: Path, drift: str, rng: np.random.Generator) -> tuple[bytes, str]:
    """Bytes e content-type a enviar; sem drift vai o arquivo original."""
    if drift == "none":
        return path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with Image.open(path) as image:
        drifted = apply_drift(image, drift, rng)
    buffer = io.BytesIO()
    if path.suffix.lower() == ".png":
        drifted.save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"
    drifted.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue(), "image/jpeg"


def build_jobs(images: list[Path], n: int, invalid_pct: float, seed: int) -> list[Job]:
    rng = random.Random(seed)
    pool = images * (n // len(images)) + rng.sample(images, n % len(images))
    jobs: list[Job] = []
    invalid_seen = 0
    for path in pool:
        if rng.random() * 100 < invalid_pct:
            jobs.append(Job(INVALID_KINDS[invalid_seen % len(INVALID_KINDS)], path))
            invalid_seen += 1
        else:
            jobs.append(Job("valid", path))
    return jobs


def send(client: httpx.Client, url: str, job: Job, drift: str, seed: int) -> tuple[int, float]:
    """Uma chamada; devolve (status, latencia em segundos)."""
    rng = np.random.default_rng([seed, zlib.crc32(job.path.name.encode())])
    endpoint = f"{url.rstrip('/')}/predict"
    started = time.perf_counter()
    if job.kind == "valid":
        payload, content_type = encode(job.path, drift, rng)
        response = client.post(endpoint, files={"image": (job.path.name, payload, content_type)})
    elif job.kind == "not_an_image":
        response = client.post(endpoint, files={"image": ("x.jpg", b"this is not an image", "image/jpeg")})
    elif job.kind == "empty_body":
        response = client.post(endpoint)
    elif job.kind == "missing_field":
        response = client.post(endpoint, data={"outro_campo": "x"})
    else:  # truncated_jpeg
        data = job.path.read_bytes()
        response = client.post(endpoint, files={"image": ("t.jpg", data[: len(data) // 3], "image/jpeg")})
    return response.status_code, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1, help="chamadas em andamento ao mesmo tempo (laco fechado)")
    parser.add_argument("--rps", type=float, default=None, help="limite opcional de chamadas enviadas por segundo")
    parser.add_argument("--traffic", choices=TRAFFIC_SETS, default="holdout")
    parser.add_argument("--images-dir", type=Path, default=None, help="usa as imagens desta pasta (ex.: samples) no lugar de --traffic")
    parser.add_argument("--drift", choices=("none", "mild", "strong"), default="none")
    parser.add_argument("--invalid-pct", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency deve ser >= 1")
    if args.rps is not None and args.rps <= 0:
        raise SystemExit("--rps deve ser > 0")

    images = list_images(args.images_dir) if args.images_dir else traffic_images(args.traffic, args.seed)
    if not images:
        raise SystemExit("nenhuma imagem encontrada; rode `just data` (baixa o Imagenette e o Imagewoof) ou use --images-dir")
    jobs = build_jobs(images, args.n, args.invalid_pct, args.seed)
    print(f"{len(jobs)} chamadas a {args.url}/predict, trafego={args.images_dir or args.traffic} ({len(images)} imagens), drift={args.drift}, concorrencia={args.concurrency}, rps={'sem limite' if args.rps is None else args.rps}")

    statuses: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    latencies: list[float] = []
    started = time.perf_counter()
    with httpx.Client(timeout=300) as client, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for index, job in enumerate(jobs):
            if args.rps is not None:
                delay = started + index / args.rps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            futures.append((job, pool.submit(send, client, args.url, job, args.drift, args.seed)))
        for job, future in futures:
            status, latency = future.result()
            statuses[str(status)] += 1
            kinds[f"{job.kind}->{status}"] += 1
            latencies.append(latency)

    print("status:", dict(sorted(statuses.items())))
    print("por tipo:", dict(sorted(kinds.items())))
    ms = np.array(latencies) * 1000
    print(f"latencia do cliente: media={ms.mean():.1f} ms  p50={np.percentile(ms, 50):.1f} ms  p95={np.percentile(ms, 95):.1f} ms  (n={len(ms)})")


if __name__ == "__main__":
    main()
