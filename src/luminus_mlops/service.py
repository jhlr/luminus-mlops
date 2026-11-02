"""Servico BentoML: classificacao top-5 de uma imagem enviada (MobileNetV3-Small, ImageNet).

Monitoramento (`luminus_mlops.monitoring`): um middleware ASGI grava uma linha `request` e uma
`response` no log JSON Lines por chamada de `/predict` (variaveis derivadas da imagem e da
saida, nunca pixels). Um unico worker escreve o arquivo. `GET /data/variables` devolve as linhas
do log do proprio servico (as rotas GET passam por `@bentoml.asgi_app`).

Rodar:
  bentoml serve luminus_mlops.service:ClassifierService --port 3000   # or: just serve
"""

from __future__ import annotations

import os
import time

import bentoml
from PIL import Image as pil

from .inference import ImageClassifier
from .monitoring.data_api import build_data_app
from .monitoring.features import extract_features, uploaded_size
from .monitoring.logging_setup import get_serving_logger
from .monitoring.metrics import DURATION_BUCKETS, get_metrics
from .monitoring.outputs import derive_outputs, top_k
from .monitoring.tracing import RequestTrace, TraceMiddleware

__all__ = ["ClassifierService"]

DEBUG_SLEEP_HEADER = "X-Debug-Sleep-Ms"


@bentoml.asgi_app(build_data_app(), path="/data")  # GET /data/variables (@bentoml.api so serve POST)
class _Classifier:
    def __init__(self) -> None:
        self.classifier = ImageClassifier()
        get_serving_logger()  # cria o handler do log AQUI, dentro do worker (nunca no import)
        self.metrics = get_metrics()  # idem: prometheus_client so depois de PROMETHEUS_MULTIPROC_DIR definido

    @bentoml.api
    def predict(self, image: pil.Image, ctx: bentoml.Context) -> dict[str, object]:
        """Classifica a imagem enviada: rotulo top-1, sua confianca e o top-5."""
        trace: RequestTrace = ctx.request.state.trace
        with trace.stage("invalid_image"):
            features = extract_features(image, uploaded_size(image))  # decodifica a imagem
        trace.log_request(features)

        if os.environ.get("ALLOW_DEBUG_SLEEP") == "1":  # so para dar cauda ao histograma de latencia
            sleep_ms = ctx.request.headers.get(DEBUG_SLEEP_HEADER, "")
            if sleep_ms.isdigit():
                time.sleep(int(sleep_ms) / 1000)

        with trace.stage("inference_failed"):
            pred = self.classifier.predict_image(image)
        outputs = derive_outputs(pred)
        trace.set_outputs(outputs)
        self.metrics.observe(features, outputs)

        best = top_k(pred)
        return {
            "label": outputs.predicted_class,
            "confidence": outputs.confidence,
            "top5": [{"label": name, "confidence": round(prob, 4)} for name, prob in best],
        }


ClassifierService = bentoml.service(name="classifier", workers=1, metrics={"duration": {"buckets": list(DURATION_BUCKETS)}})(_Classifier)
ClassifierService.add_asgi_middleware(TraceMiddleware, endpoints=("/predict",))
