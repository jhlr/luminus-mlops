"""Gera o dashboard unico do Grafana (essencial: operacao, entrada, saida, drift e logs).

  uv run python scripts/build_dashboards.py

Escreve `monitoring/grafana/provisioning/dashboards/luminus.json`; o Grafana os carrega sozinho (provider em
`dashboards.yml`). O JSON gerado e versionado: este script e so o dono dos padroes de painel (um box plot,
um histograma de pixels, um stat de nivel de drift... definidos uma vez e repetidos por variavel), para
nao manter 70 paineis a mao. Os baldes (buckets) vem de `luminus_mlops.monitoring.metrics`, os nomes das
variaveis de `features.py`/`outputs.py`: nada e copiado.

Fontes: Prometheus (series de `inference:3000` e `monitor:3000`), Loki (log via Alloy) e Infinity
(rotas `/data/*` do `monitor`: valores por imagem, histogramas 0-255 e quartis). Box plots: o Grafana nao
tem box plot nativo; usam o painel Business Charts (ECharts, `volkovlabs-echarts-panel`) alimentado por
`/data/summary` (min, Q1, mediana, Q3, max).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from luminus_mlops.monitoring.metrics import CONFIDENCE_BUCKETS, DURATION_BUCKETS, LUMINANCE_BUCKETS
from luminus_mlops.paths import ROOT

Json = dict[str, Any]

OUT_DIR = ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards"
PROM: Json = {"type": "prometheus", "uid": "prometheus"}
LOKI: Json = {"type": "loki", "uid": "loki"}
INF: Json = {"type": "yesoreyeram-infinity-datasource", "uid": "infinity"}
MONITOR = "http://monitor:3000"
ECHARTS = "volkovlabs-echarts-panel"
PRED = 'endpoint="/predict"'
REQ = "bentoml_service_request_total"
DUR = "bentoml_service_request_duration_seconds_bucket"
TIME_PARAMS: list[Json] = [
    {"key": "since", "value": "${__from:date:iso}"},
    {"key": "until", "value": "${__to:date:iso}"},
]
CHANNELS = {"L": "luminance_mean", "R": "r_mean", "G": "g_mean", "B": "b_mean"}
CHANNEL_COLORS = {"L": "#9aa0a6", "R": "#e24d42", "G": "#3ea65a", "B": "#3d85de"}
LEVEL_MAPPINGS: list[Json] = [
    {
        "type": "value",
        "options": {
            "-1": {"text": "inconclusivo", "color": "gray", "index": 0},
            "0": {"text": "verde", "color": "green", "index": 1},
            "1": {"text": "amarelo", "color": "yellow", "index": 2},
            "2": {"text": "vermelho", "color": "red", "index": 3},
        },
    }
]


# --- consultas -----------------------------------------------------------------------------------
def prom(expr: str, ref: str = "A", legend: str = "", instant: bool = False, fmt: str = "time_series") -> Json:
    target: Json = {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legend or "__auto", "format": fmt}
    if instant:
        target.update({"instant": True, "range": False})
    return target


def loki(expr: str, ref: str = "A", legend: str = "", query_type: str = "range") -> Json:
    target: Json = {"refId": ref, "datasource": LOKI, "expr": expr, "queryType": query_type}
    if legend:
        target["legendFormat"] = legend
    return target


def col(selector: str, text: str, kind: str = "number") -> Json:
    return {"selector": selector, "text": text, "type": kind}


def infinity(
    path: str,
    root: str,
    columns: Sequence[Json],
    ref: str = "A",
    params: Sequence[Json] = (),
) -> Json:
    """Consulta Infinity (JSON, parser backend: o `root_selector` e uma expressao JSONata)."""
    return {
        "refId": ref,
        "datasource": INF,
        "type": "json",
        "source": "url",
        "format": "table",
        "parser": "backend",
        "url": f"{MONITOR}{path}",
        "url_options": {"method": "GET", "params": list(params)},
        "root_selector": root,
        "columns": list(columns),
    }


def variables(columns: Sequence[Json], root: str = "$[status_code=200]", ref: str = "A", limit: int = 10000) -> Json:
    """Uma linha por requisicao do intervalo do dashboard (`/data/variables`)."""
    params = [*TIME_PARAMS, {"key": "limit", "value": str(limit)}]
    return infinity("/data/variables", root, columns, ref, params)


def summary(root: str, columns: Sequence[Json], ref: str = "A") -> Json:
    return infinity("/data/summary", root, columns, ref, TIME_PARAMS)


def map_rows(source: str, fields: dict[str, str]) -> str:
    """JSONata: `$map(<source>, function($r){ {<campo>: <expressao>} })`."""
    body = ", ".join(f'"{name}": {expr}' for name, expr in fields.items())
    return f"$map({source}, function($r){{ {{{body}}} }})"


# --- paineis -------------------------------------------------------------------------------------
class Board:
    """Um dashboard: acumula paineis com posicao automatica (24 colunas) e escreve o JSON."""

    def __init__(self, uid: str, title: str, description: str, refresh: str = "30s", time_from: str = "now-3h") -> None:
        self.uid, self.title, self.description = uid, title, description
        self.refresh, self.time_from = refresh, time_from
        self.panels: list[Json] = []
        self._next_id = 1
        self._x = 0
        self._y = 0
        self._row_h = 0

    def row(self, title: str) -> None:
        self._newline()
        self.panels.append(
            {"id": self._take_id(), "type": "row", "title": title, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": self._y}, "panels": []}
        )
        self._y += 1

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _newline(self) -> None:
        if self._x:
            self._y += self._row_h
            self._x = 0
            self._row_h = 0

    def add(self, panel: Json, w: int = 12, h: int = 8) -> None:
        if self._x + w > 24:
            self._newline()
        panel["id"] = self._take_id()
        panel["gridPos"] = {"h": h, "w": w, "x": self._x, "y": self._y}
        self.panels.append(panel)
        self._x += w
        self._row_h = max(self._row_h, h)

    def dump(self) -> Json:
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": ["luminus"],
            "timezone": "browser",
            "editable": False,
            "graphTooltip": 1,
            "schemaVersion": 41,
            "version": 1,
            "refresh": self.refresh,
            "time": {"from": self.time_from, "to": "now"},
            "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]},
            "templating": {"list": []},
            "annotations": {"list": []},
            "links": [{"type": "dashboards", "title": "Luminus", "tags": ["luminus"], "asDropdown": True, "includeVars": False, "keepTime": True}],
            "panels": self.panels,
        }

    def write(self) -> Path:
        path = OUT_DIR / f"{self.uid}.json"
        path.write_text(json.dumps(self.dump(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def panel(kind: str, title: str, targets: Sequence[Json], description: str = "", **extra: Any) -> Json:
    base: Json = {"type": kind, "title": title, "targets": list(targets), "description": description}
    base["datasource"] = targets[0]["datasource"] if targets else PROM
    base.update(extra)
    return base


def timeseries(
    title: str,
    targets: Sequence[Json],
    description: str = "",
    unit: str = "",
    style: str = "line",
    stack: str = "none",
    minimum: float | None = None,
    maximum: float | None = None,
    thresholds: Sequence[Json] | None = None,
    transformations: Sequence[Json] = (),
    overrides: Sequence[Json] = (),
) -> Json:
    custom: Json = {
        "drawStyle": style,
        "lineWidth": 1,
        "fillOpacity": 60 if style == "bars" else 10,
        "showPoints": "auto",
        "spanNulls": True,
        "stacking": {"mode": stack, "group": "A"},
    }
    if thresholds:
        custom["thresholdsStyle"] = {"mode": "line"}
    defaults: Json = {"unit": unit, "custom": custom}
    if minimum is not None:
        defaults["min"] = minimum
    if maximum is not None:
        defaults["max"] = maximum
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": list(thresholds)}
    return panel(
        "timeseries",
        title,
        targets,
        description,
        fieldConfig={"defaults": defaults, "overrides": list(overrides)},
        options={"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}, "tooltip": {"mode": "multi", "sort": "none"}},
        transformations=list(transformations),
    )


def stat(
    title: str,
    targets: Sequence[Json],
    description: str = "",
    unit: str = "",
    decimals: int | None = None,
    thresholds: Sequence[Json] | None = None,
    mappings: Sequence[Json] = (),
    color_mode: str = "value",
) -> Json:
    defaults: Json = {
        "unit": unit,
        "mappings": list(mappings),
        "thresholds": {"mode": "absolute", "steps": list(thresholds or [{"color": "blue", "value": None}])},
        "color": {"mode": "thresholds"},
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return panel(
        "stat",
        title,
        targets,
        description,
        fieldConfig={"defaults": defaults, "overrides": []},
        options={"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "colorMode": color_mode, "graphMode": "none", "textMode": "value", "justifyMode": "center"},
    )


def table(title: str, targets: Sequence[Json], description: str = "", transformations: Sequence[Json] = (), overrides: Sequence[Json] = ()) -> Json:
    return panel(
        "table",
        title,
        targets,
        description,
        fieldConfig={"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}}, "overrides": list(overrides)},
        options={"showHeader": True, "cellHeight": "sm"},
        transformations=list(transformations),
    )


def ordered(names: Sequence[str]) -> Json:
    """Ordem das colunas (o parser backend do Infinity as devolve em ordem alfabetica)."""
    return {"id": "organize", "options": {"indexByName": {name: index for index, name in enumerate(names)}}}


def histogram(title: str, targets: Sequence[Json], description: str = "", unit: str = "", bucket_size: float | None = None, bucket_count: int = 20) -> Json:
    options: Json = {"legend": {"showLegend": False, "displayMode": "list", "placement": "bottom"}, "combine": False, "fillOpacity": 80}
    if bucket_size is not None:
        options["bucketSize"] = bucket_size
    else:
        options["bucketCount"] = bucket_count
    return panel("histogram", title, targets, description, fieldConfig={"defaults": {"unit": unit, "custom": {"fillOpacity": 80, "lineWidth": 1}}, "overrides": []}, options=options)


def barchart(title: str, targets: Sequence[Json], x_field: str, description: str = "", unit: str = "", transformations: Sequence[Json] = (), orientation: str = "auto") -> Json:
    return panel(
        "barchart",
        title,
        targets,
        description,
        fieldConfig={"defaults": {"unit": unit, "custom": {"fillOpacity": 80, "lineWidth": 1, "axisPlacement": "auto"}}, "overrides": []},
        options={"xField": x_field, "orientation": orientation, "showValue": "auto", "stacking": "none", "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}, "xTickLabelRotation": -45},
        transformations=list(transformations),
    )


def pixel_histogram(channel: str) -> Json:
    """Histograma 0-255 da janela (barras) com o da referencia sobreposto (linha); ambos em fracao do total (peso igual por imagem)."""
    key = channel
    root = (
        f"( $w := histograms.{key}.window; $r := histograms.{key}.reference; $sw := $sum($w); $sr := $sum($r); "
        '$map([0..255], function($i){ {"bin": $i, "janela": $sw > 0 ? $w[$i]/$sw : 0, "referencia": $r[$i]/$sr} }) )'
    )
    target = summary(root, [col("bin", "Valor (0-255)"), col("janela", "Janela"), col("referencia", "Referência")])
    overrides: list[Json] = [
        {"matcher": {"id": "byName", "options": "Valor (0-255)"}, "properties": [{"id": "unit", "value": "none"}]},
        {"matcher": {"id": "byName", "options": "Janela"}, "properties": [
            {"id": "custom.drawStyle", "value": "bars"}, {"id": "custom.fillOpacity", "value": 70},
            {"id": "color", "value": {"mode": "fixed", "fixedColor": CHANNEL_COLORS[channel]}}]},
        {"matcher": {"id": "byName", "options": "Referência"}, "properties": [
            {"id": "custom.drawStyle", "value": "line"}, {"id": "custom.lineWidth", "value": 2},
            {"id": "color", "value": {"mode": "fixed", "fixedColor": "#5a5a5a"}}]},
    ]
    return panel(
        "trend",
        f"Histograma {channel}: janela x referência",
        [target],
        f"Histograma 0-255 do canal {channel} somado sobre as imagens do intervalo (barras) contra o da referência (linha), "
        "como fração do total, com peso igual por imagem (cada histograma é normalizado antes da soma, então fotos grandes não dominam). Fonte: /data/summary via Infinity.",
        fieldConfig={"defaults": {"unit": "percentunit", "custom": {"lineWidth": 1, "fillOpacity": 30, "showPoints": "never"}}, "overrides": overrides},
        options={"xField": "Valor (0-255)", "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}, "tooltip": {"mode": "multi"}},
    )


_JS_THEME = "const ink = (context.grafana && context.grafana.theme && context.grafana.theme.colors.text.primary) || '#aaa';"
BOXPLOT_JS = (
    "const s = context.panel.data.series[0];"
    "if (!s) { return {}; }"
    "const col = (n) => { const f = s.fields.find((x) => x.name === n); return f ? Array.from(f.values) : []; };"
    f"{_JS_THEME}"
    "const names = col('variavel');"
    "const box = (p) => names.map((_, i) => [col(p + '_min')[i], col(p + '_q1')[i], col(p + '_median')[i], col(p + '_q3')[i], col(p + '_max')[i]]);"
    "return {"
    " textStyle: { color: ink },"
    " tooltip: { trigger: 'item', confine: true },"
    " legend: { data: ['Janela', 'Referência'], textStyle: { color: ink } },"
    " grid: { left: 90, right: 20, top: 40, bottom: 30 },"
    " xAxis: { type: 'category', data: names, axisLabel: { color: ink } },"
    " yAxis: { type: 'value', scale: true, axisLabel: { color: ink }, splitLine: { lineStyle: { opacity: 0.2 } } },"
    " series: ["
    "  { name: 'Janela', type: 'boxplot', data: box('w'), itemStyle: { color: 'rgba(61,133,222,0.35)', borderColor: '#3d85de' } },"
    "  { name: 'Referência', type: 'boxplot', data: box('r'), itemStyle: { color: 'rgba(150,150,150,0.35)', borderColor: '#8a8a8a' } }"
    " ] };"
)
def echarts(title: str, targets: Sequence[Json], code: str, description: str) -> Json:
    return panel(
        ECHARTS,
        title,
        targets,
        description,
        options={"renderer": "canvas", "editorMode": "code", "getOption": code, "editor": {"language": "javascript", "format": "auto"}, "visualEditor": {"dataset": [], "series": [], "code": ""}, "map": "none", "themeEditor": {"name": "default", "config": "{}"}},
        fieldConfig={"defaults": {}, "overrides": []},
    )


def boxplot(title: str, names: Sequence[str], description: str = "") -> Json:
    """Box plot (janela x referência) das variáveis `names` a partir dos quartis de /data/summary."""
    fields = {"variavel": "$k"}
    for prefix, side in (("w", "window"), ("r", "reference")):
        for stat_name in ("min", "q1", "median", "q3", "max"):
            fields[f"{prefix}_{stat_name}"] = f"$n.{side}.{stat_name}"
    body = ", ".join(f'"{k}": {v}' for k, v in fields.items())
    root = f'$map({json.dumps(list(names))}, function($k){{ ( $n := $lookup($$.numeric, $k); {{{body}}} ) }})'
    columns = [col("variavel", "variavel", "string")] + [col(f"{p}_{s}", f"{p}_{s}") for p in ("w", "r") for s in ("min", "q1", "median", "q3", "max")]
    return echarts(
        title,
        [summary(root, columns)],
        BOXPLOT_JS,
        (description + " " if description else "")
        + "Box plot do intervalo do dashboard (azul) contra a referência (cinza): caixa Q1 a Q3, traço na mediana, bigodes no mínimo e no máximo. "
        "Painel Business Charts (ECharts, volkovlabs-echarts-panel 7.2.5) alimentado por /data/summary via Infinity.",
    )


def _le(bound: float) -> str:
    """Rotulo `le` como o prometheus_client o escreve (0.25, 1.0, 16.0, +Inf)."""
    return "+Inf" if bound == float("inf") else repr(float(bound))


def deaccumulated(metric: str, buckets: Sequence[float], title: str, description: str, matchers: str = "") -> Json:
    """Histograma de um histograma Prometheus: cada barra e a diferenca entre buckets `le` consecutivos (janela `$__range`)."""

    def cumulative(bound: float) -> str:
        selector = ",".join(filter(None, [matchers, f'le="{_le(bound)}"']))
        return f"round(sum(increase({metric}_bucket{{{selector}}}[$__range])))"

    targets: list[Json] = []
    bounds = [*buckets, float("inf")]
    for index, bound in enumerate(bounds):
        if index == 0:
            expr, label = cumulative(bound), f"≤ {bound:g}"
        else:
            expr = f"{cumulative(bound)} - {cumulative(bounds[index - 1])}"
            label = f"> {bounds[index - 1]:g}" if bound == float("inf") else f"{bounds[index - 1]:g} a {bound:g}"
        targets.append(prom(expr, ref=f"Q{index}", legend=label, instant=True))
    reduce: Json = {"id": "reduce", "options": {"mode": "seriesToRows", "reducers": ["lastNotNull"]}}
    organize: Json = {"id": "organize", "options": {"renameByName": {"Field": "faixa", "Last *": "observações"}}}
    return barchart(title, targets, "faixa", description, transformations=[reduce, organize])



# --- dashboard -----------------------------------------------------------------------------------
def build_dashboard() -> Board:
    b = Board("luminus", "Luminus: monitoramento", "Operacao, imagem de entrada, saida do modelo, drift (KS) e logs do classificador de imagens.")

    b.row("Operação")
    b.add(stat("Total de requisições (/predict)", [prom(f"sum({REQ}{{{PRED}}})", instant=True)], "Contador acumulado desde que o processo de inferência subiu (zera ao reiniciar).", decimals=0), 6, 5)
    b.add(stat("Requisições em andamento", [prom(f"sum(bentoml_service_request_in_progress{{{PRED}}})", instant=True)], "Com um único worker, mais de 1 significa fila.", decimals=0), 6, 5)
    b.add(
        stat(
            "Taxa de erro 5xx (5 min)",
            [prom(f'(sum(rate({REQ}{{{PRED},http_response_code=~"5.."}}[5m])) or vector(0)) / sum(rate({REQ}{{{PRED}}}[5m]))', instant=True)],
            "Falhas do servidor: fração das requisições dos últimos 5 minutos.",
            unit="percentunit",
            decimals=1,
            thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 0.01}],
        ),
        12,
        5,
    )
    b.add(
        timeseries(
            "Requisições por segundo por código de status",
            [prom(f"sum by (http_response_code) (rate({REQ}{{{PRED}}}[$__rate_interval]))", legend="{{http_response_code}}")],
            "rate() do contador bentoml_service_request_total em /predict, por código HTTP.",
            unit="reqps",
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Taxa de erro 4xx e 5xx",
            [
                prom(f'(sum(rate({REQ}{{{PRED},http_response_code=~"4.."}}[1m])) or vector(0)) / sum(rate({REQ}{{{PRED}}}[1m]))', ref="A", legend="4xx (entrada inválida)"),
                prom(f'(sum(rate({REQ}{{{PRED},http_response_code=~"5.."}}[1m])) or vector(0)) / sum(rate({REQ}{{{PRED}}}[1m]))', ref="B", legend="5xx (servidor)"),
            ],
            "Fração das requisições com erro, janela de 1 minuto.",
            unit="percentunit",
            minimum=0,
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Latência p50, p95 e p99",
            [prom(f"histogram_quantile({q}, sum by (le) (rate({DUR}{{{PRED},http_response_code=\"200\"}}[1m])))", ref=ref, legend=f"p{int(q * 100)}") for ref, q in (("A", 0.5), ("B", 0.95), ("C", 0.99))],
            "histogram_quantile sobre os buckets de duração do BentoML (aproximação por interpolação dentro do bucket).",
            unit="s",
        ),
        12,
        8,
    )
    b.add(
        deaccumulated(
            "bentoml_service_request_duration_seconds",
            DURATION_BUCKETS,
            "Histograma de latência (intervalo do dashboard)",
            "Contagem por faixa de latência: diferença entre buckets `le` consecutivos.",
            matchers=f'{PRED},http_response_code="200"',
        ),
        12,
        8,
    )

    b.row("Entrada (imagem)")
    for channel in CHANNELS:
        b.add(pixel_histogram(channel), 12, 8)
    b.add(boxplot("Boxplot das médias de L, R, G, B", list(CHANNELS.values())), 12, 8)
    b.add(boxplot("Boxplot de luminance_std (contraste)", ["luminance_std"]), 6, 8)
    b.add(boxplot("Boxplot de sharpness (nitidez)", ["sharpness"]), 6, 8)

    b.row("Saída (modelo)")
    b.add(boxplot("Boxplot de confidence, margin e top5_mass", ["confidence", "margin", "top5_mass"]), 12, 8)
    b.add(boxplot("Boxplot de entropy (bits)", ["entropy"]), 12, 8)
    b.add(
        deaccumulated(
            "luminus_prediction_confidence",
            CONFIDENCE_BUCKETS,
            "Histograma de confiança top-1",
            "Contagem por faixa de confiança, da métrica customizada luminus_prediction_confidence.",
        ),
        12,
        8,
    )
    b.add(
        deaccumulated(
            "luminus_input_luminance_mean",
            LUMINANCE_BUCKETS,
            "Histograma de luminância média da entrada",
            "Contagem por faixa de luminância média, da métrica customizada luminus_input_luminance_mean.",
        ),
        12,
        8,
    )

    b.row("Drift (KS contra a referência)")
    for channel in CHANNELS:
        b.add(
            stat(
                f"Drift canal {channel}",
                [prom(f'luminus_drift_level{{channel="{channel}"}}', instant=True)],
                "Nível do KS (D): verde (< 0,05), amarelo (>= 0,05), vermelho (>= 0,10), cinza (menos de 8 imagens).",
                mappings=LEVEL_MAPPINGS,
                thresholds=[{"color": "gray", "value": None}, {"color": "green", "value": 0}, {"color": "yellow", "value": 1}, {"color": "red", "value": 2}],
                color_mode="background",
            ),
            6,
            4,
        )
    b.add(
        timeseries(
            "KS D por canal (L, R, G, B)",
            [prom("luminus_drift_ks_d", legend="{{channel}}")],
            "D = máximo da diferença entre as CDFs do histograma agregado da janela (últimas 100 imagens) e o da referência, peso igual por imagem. Linhas: 0,05 e 0,10.",
            minimum=0,
            maximum=1,
            thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 0.05}, {"color": "red", "value": 0.10}],
        ),
        24,
        8,
    )

    b.row("Logs (Loki)")
    log_options: Json = {"showTime": True, "wrapLogMessage": False, "prettifyLogMessage": False, "enableLogDetails": True, "sortOrder": "Descending", "dedupStrategy": "none"}
    b.add(panel("logs", "Log ao vivo", [loki('{job="luminus"}')], "Linhas request e response, enviadas ao Loki pelo Alloy.", options=log_options), 24, 10)
    b.add(panel("logs", "Erros (status_code >= 500)", [loki('{job="luminus", event="response"} | json | status_code >= 500')], "O status_code é extraído do JSON na consulta (nunca é rótulo).", options=log_options), 24, 8)
    return b


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    board = build_dashboard()
    path = board.write()
    print(f"{path.relative_to(ROOT)}: {sum(1 for p in board.panels if p['type'] != 'row')} paineis")


if __name__ == "__main__":
    main()
