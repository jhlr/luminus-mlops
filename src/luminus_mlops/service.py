"""BentoML service: top-5 classification of an uploaded image (MobileNetV3-Small, ImageNet).

Monitoring (`luminus_mlops.monitoring`): an ASGI middleware writes one `request` and one `response`
line to the JSON Lines log per `/predict` call (variables derived from the image and from the
output, never pixels). A single worker writes the file. `GET /data/variables` returns the rows of
the service's own log (GET routes through `@bentoml.asgi_app`).

Run:
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


@bentoml.asgi_app(build_data_app(), path="/data")  # GET /data/variables (@bentoml.api only serves POST)
class _Classifier:
    def __init__(self) -> None:
        self.classifier = ImageClassifier()
        get_serving_logger()  # creates the log handler HERE, inside the worker (never at import)
        self.metrics = get_metrics()  # same: prometheus_client only after PROMETHEUS_MULTIPROC_DIR is set

    @bentoml.api
    def predict(self, image: pil.Image, ctx: bentoml.Context) -> dict[str, object]:
        """Classify the uploaded image: top-1 label, its confidence and the top-5."""
        trace: RequestTrace = ctx.request.state.trace
        with trace.stage("invalid_image"):
            features = extract_features(image, uploaded_size(image))  # decodes the image
        trace.log_request(features)

        if os.environ.get("ALLOW_DEBUG_SLEEP") == "1":  # only to give the latency histogram a tail
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
