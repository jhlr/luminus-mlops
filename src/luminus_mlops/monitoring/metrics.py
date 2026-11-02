"""Metricas Prometheus proprias do servico (SDD secao 6), expostas no `/metrics` do BentoML.

Usam `prometheus_client` direto (o BentoML 1.4 considera `bentoml.metrics` obsoleto). Os objetos
sao criados de forma preguicosa, dentro do worker: o BentoML define `PROMETHEUS_MULTIPROC_DIR`
DEPOIS de importar o modulo do servico, e um `prometheus_client` importado antes ficaria em modo
processo unico, invisivel para o coletor multiprocesso do `/metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .outputs import CONFIDENCE_BANDS, confidence_band

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

    from .analysis import DriftReport
    from .features import ImageFeatures
    from .outputs import OutputVariables

LUMINANCE_BUCKETS: tuple[float, ...] = tuple(float(v) for v in range(16, 256, 16))  # 16..240 (+Inf implicito)
CONFIDENCE_BUCKETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ENTROPY_BUCKETS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)  # bits (max log2(1000) = 9.97)
# Buckets do histograma de duracao do BentoML. Ajustados com a latencia real medida (docs/*_metrics_analysis.md).
DURATION_BUCKETS: tuple[float, ...] = (0.01, 0.025, 0.035, 0.0375, 0.04, 0.0415, 0.043, 0.045, 0.05, 0.06, 0.07, 0.0725, 0.075, 0.0775, 0.08, 0.09, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0)


@dataclass
class ServingMetrics:
    input_luminance: Histogram
    confidence: Histogram
    entropy: Histogram
    confidence_band: Counter

    def observe(self, features: ImageFeatures, outputs: OutputVariables) -> None:
        self.input_luminance.observe(features.luminance_mean)
        self.confidence.observe(outputs.confidence)
        self.entropy.observe(outputs.entropy)
        self.confidence_band.labels(band=confidence_band(outputs.confidence)).inc()


_metrics: ServingMetrics | None = None


def get_metrics() -> ServingMetrics:
    """Cria (uma vez por processo) os histogramas e o contador do servico."""
    global _metrics
    if _metrics is None:
        from prometheus_client import Counter, Histogram

        band = Counter("luminus_prediction_confidence_band", "Predicoes por faixa de confianca (low, medium, high)", ["band"])
        for value in CONFIDENCE_BANDS:  # as 3 series existem desde o inicio
            band.labels(band=value)
        _metrics = ServingMetrics(
            input_luminance=Histogram(
                "luminus_input_luminance_mean",
                "Luminancia media (0-255) da imagem de entrada",
                buckets=LUMINANCE_BUCKETS,
            ),
            confidence=Histogram(
                "luminus_prediction_confidence", "Confianca (probabilidade top-1) das predicoes", buckets=CONFIDENCE_BUCKETS
            ),
            entropy=Histogram("luminus_prediction_entropy", "Entropia (bits) do softmax das predicoes", buckets=ENTROPY_BUCKETS),
            confidence_band=band,
        )
    return _metrics


@dataclass
class DriftGauges:
    """Gauges de drift do `monitor` (SDD secao 7): valores vindos de `analysis.DriftReport`.

    Gauge no modo multiprocesso precisa de `multiprocess_mode`; `livemostrecent` mantem o ultimo valor
    escrito pelo processo vivo. Valor indefinido (ex.: PSI de referencia constante) vai como NaN.
    """

    ks_d: Gauge
    ks_pvalue: Gauge
    level: Gauge
    psi: Gauge
    window_images: Gauge

    def update(self, report: DriftReport) -> None:
        nan = float("nan")
        for label, channel in report.channels.items():
            self.ks_d.labels(channel=label).set(nan if channel.d is None else channel.d)
            self.ks_pvalue.labels(channel=label).set(nan if channel.p is None else channel.p)
            self.level.labels(channel=label).set(int(channel.level))
        for variable, result in report.psi.items():
            self.psi.labels(variable=variable).set(nan if result.value is None else result.value)
        self.window_images.set(report.n_window)


_drift_gauges: DriftGauges | None = None


def get_drift_gauges() -> DriftGauges:
    """Cria (uma vez por processo) os gauges de drift; lazy pelo mesmo motivo de `get_metrics`."""
    global _drift_gauges
    if _drift_gauges is None:
        from prometheus_client import Gauge

        mode: Literal["livemostrecent"] = "livemostrecent"
        _drift_gauges = DriftGauges(
            ks_d=Gauge("luminus_drift_ks_d", "KS D (max |F_ref - F_janela|) por canal", ["channel"], multiprocess_mode=mode),
            ks_pvalue=Gauge("luminus_drift_ks_pvalue", "p-valor do KS por canal (n em imagens)", ["channel"], multiprocess_mode=mode),
            level=Gauge("luminus_drift_level", "Nivel de alerta por canal: -1 inconclusivo, 0 verde, 1 amarelo, 2 vermelho", ["channel"], multiprocess_mode=mode),
            psi=Gauge("luminus_drift_psi", "PSI por variavel (janela x referencia)", ["variable"], multiprocess_mode=mode),
            window_images=Gauge("luminus_drift_window_images", "Imagens na janela deslizante de drift", multiprocess_mode=mode),
        )
    return _drift_gauges
