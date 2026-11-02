"""Fixtures dos testes de monitoramento: sobem o serviço BentoML de verdade (subprocesso)."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
SAMPLE_NAMES = ["sample_01.png", "sample_02.png", "sample_03.png", "sample_04.png", "sample_dark.png"]
SERVICE_TARGET = "luminus_mlops.service:ClassifierService"
LOG_NAME = "inference.log"
STARTUP_TIMEOUT_S = 240.0


@dataclass
class LiveService:
    url: str
    log_dir: Path
    process: subprocess.Popen[bytes]

    @property
    def log_file(self) -> Path:
        return self.log_dir / LOG_NAME

    def predict(self, name: str = "sample_01.png") -> httpx.Response:
        with (SAMPLES / name).open("rb") as fh:
            return httpx.post(f"{self.url}/predict", files={"image": (name, fh, "image/png")}, timeout=300)

    def log_lines(self) -> list[str]:
        if not self.log_file.exists():
            return []
        return self.log_file.read_text(encoding="utf-8").splitlines()

    def log_events(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.log_lines()]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_service(log_dir: Path, extra_env: dict[str, str] | None = None, target: str = SERVICE_TARGET) -> LiveService:
    port = _free_port()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SERVING_LOG_DIR": str(log_dir), **(extra_env or {})}
    process = subprocess.Popen(
        [sys.executable, "-m", "bentoml", "serve", target, "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    service = LiveService(url=f"http://127.0.0.1:{port}", log_dir=log_dir, process=process)
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"bentoml serve saiu com codigo {process.returncode}")
        try:
            if httpx.get(f"{service.url}/readyz", timeout=2).status_code == 200:
                return service
        except httpx.TransportError:
            time.sleep(1)
    stop_service(service)
    raise TimeoutError("servico nao ficou pronto a tempo")


def stop_service(service: LiveService) -> None:
    if service.process.poll() is None:
        os.killpg(os.getpgid(service.process.pid), signal.SIGTERM)
        try:
            service.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(service.process.pid), signal.SIGKILL)
            service.process.wait()


@pytest.fixture(scope="session", autouse=True)
def samples() -> Path:
    """Garante as 5 amostras sinteticas de `just samples` (seed fixa)."""
    if not all((SAMPLES / name).exists() for name in SAMPLE_NAMES):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_sample_images.py")], check=True, cwd=ROOT)
    return SAMPLES


@pytest.fixture(scope="session")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveService]:
    live = start_service(tmp_path_factory.mktemp("serving-logs"))
    yield live
    stop_service(live)


@pytest.fixture()
def fresh_service(tmp_path: Path) -> Iterator[LiveService]:
    """Servico novo com log vazio (contagem exata de linhas)."""
    live = start_service(tmp_path / "serving-logs")
    yield live
    stop_service(live)
