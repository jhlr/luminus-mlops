"""Planilha didatica do `/metrics` (SDD secao 6.1), com FORMULAS VIVAS (openpyxl).

  uv run python scripts/make_metrics_xlsx.py --snapshot docs/2026-11-01_monitoramento_metrics_snapshot.txt \
      --out docs/2026-11-01_monitoramento_metrics_analysis.xlsx

Abas: `raw` (le/acumulado colados do snapshot), `histogram` (contagem por bucket por diferenca),
`quantiles` (Q1, mediana, Q3 por interpolacao linear no estilo `histogram_quantile`, e whiskers
pelos limites dos buckets nao vazios) e `errors` (taxa de erro por codigo HTTP). Troque os numeros
da aba `raw`/`errors` por outro snapshot e tudo se recalcula.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font
from prometheus_client.parser import text_string_to_metric_families

ENDPOINT = "/predict"
DURATION = "bentoml_service_request_duration_seconds"
REQUESTS = "bentoml_service_request_total"
BOLD = Font(bold=True)


def read_snapshot(text: str, status: str = "200") -> tuple[list[tuple[str, float]], float, dict[str, float]]:
    """(buckets cumulativos de /predict para `status`, soma da duracao, requisicoes por codigo HTTP)."""
    buckets: dict[float, float] = {}
    total_sum = 0.0
    by_code: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            labels = sample.labels
            if labels.get("endpoint") != ENDPOINT:
                continue
            if sample.name == f"{DURATION}_bucket" and labels.get("http_response_code") == status:
                buckets[float(labels["le"])] = sample.value
            elif sample.name == f"{DURATION}_sum" and labels.get("http_response_code") == status:
                total_sum = sample.value
            elif sample.name == REQUESTS:
                by_code[labels["http_response_code"]] = sample.value
    ordered = sorted(buckets.items())
    return [("+Inf" if math.isinf(le) else str(le), count) for le, count in ordered], total_sum, by_code


def build(snapshot: Path, out: Path) -> None:
    buckets, total_sum, by_code = read_snapshot(snapshot.read_text(encoding="utf-8"))
    n = len(buckets)
    first, last = 2, 1 + n  # linhas de dados em raw/histogram
    wb = Workbook()
    raw = wb.create_sheet("raw")
    wb.remove(wb["Sheet"])  # aba padrao vazia
    raw.append(["le (s)", "acumulado (bucket)"])
    for le, count in buckets:
        raw.append([float(le) if le != "+Inf" else "+Inf", count])
    raw["D1"], raw["E1"] = "soma da duracao (s)", total_sum
    raw["D2"], raw["E2"] = "contagem (= bucket +Inf)", f"=B{last}"
    raw["D3"], raw["E3"] = "media (s) = soma / contagem", "=E1/E2"
    raw["D5"] = f"Fonte: {snapshot.name}, serie {DURATION}_bucket, endpoint={ENDPOINT}, http_response_code=200."
    for cell in ("A1", "B1", "D1", "D2", "D3"):
        raw[cell].font = BOLD

    hist = wb.create_sheet("histogram")
    hist.append(["de (s)", "ate (s)", "contagem no bucket", "fracao"])
    for row in range(first, last + 1):
        hist.cell(row, 1, "=0" if row == first else f"=raw!A{row - 1}")
        hist.cell(row, 2, f"=raw!A{row}")
        hist.cell(row, 3, f"=raw!B{row}" if row == first else f"=raw!B{row}-raw!B{row - 1}")
        hist.cell(row, 4, f"=C{row}/raw!$B${last}")
    for cell in ("A1", "B1", "C1", "D1"):
        hist[cell].font = BOLD

    q = wb.create_sheet("quantiles")
    q.append(["estatistica", "q", "posicao (q x total)", "indice do bucket", "limite inferior", "limite superior", "acumulado antes", "acumulado no bucket", "valor (s)"])
    cum = f"raw!$B${first}:$B${last}"
    le = f"raw!$A${first}:$A${last}"
    total = f"raw!$B${last}"
    for i, (name, quantile) in enumerate([("Q1", 0.25), ("mediana", 0.5), ("Q3", 0.75)], start=2):
        q.cell(i, 1, name)
        q.cell(i, 2, quantile)
        q.cell(i, 3, f"=B{i}*{total}")
        q.cell(i, 4, f'=COUNTIF({cum},"<"&C{i})+1')
        q.cell(i, 5, f"=IF(D{i}=1,0,INDEX({le},D{i}-1))")
        q.cell(i, 6, f"=INDEX({le},D{i})")
        q.cell(i, 7, f"=IF(D{i}=1,0,INDEX({cum},D{i}-1))")
        q.cell(i, 8, f"=INDEX({cum},D{i})")
        # bucket +Inf: como o histogram_quantile do Prometheus, devolve o maior limite finito
        q.cell(i, 9, f"=IF(ISNUMBER(F{i}),E{i}+(F{i}-E{i})*(C{i}-G{i})/(H{i}-G{i}),E{i})")
    q["A6"], q["I6"] = "whisker minimo (limite inferior do 1o bucket nao vazio)", f"=IF(COUNTIF({cum},0)=0,0,INDEX({le},COUNTIF({cum},0)))"
    q["A7"], q["I7"] = "whisker maximo (limite superior do ultimo bucket nao vazio)", f'=IF(ISNUMBER(INDEX({le},COUNTIF({cum},"<"&{total})+1)),INDEX({le},COUNTIF({cum},"<"&{total})+1),INDEX({le},COUNTIF({cum},"<"&{total})))'
    q["A9"] = "Aproximacao: quartis exatos exigem o latency_ms do log (a interpolacao supoe observacoes uniformes dentro do bucket)."
    for col in "ABCDEFGHI":
        q[f"{col}1"].font = BOLD

    err = wb.create_sheet("errors")
    err.append(["http_response_code", "requisicoes (acumulado)"])
    for code, count in sorted(by_code.items()):
        err.append([code, count])
    end = 1 + len(by_code)
    err["D1"], err["E1"] = "total de requisicoes", f"=SUM(B2:B{end})"
    err["D2"], err["E2"] = "5xx (erro do servidor)", f'=SUMIF(A2:A{end},"5*",B2:B{end})'
    err["D3"], err["E3"] = "4xx (erro do cliente)", f'=SUMIF(A2:A{end},"4*",B2:B{end})'
    err["D4"], err["E4"] = "taxa 5xx", "=E2/E1"
    err["D5"], err["E5"] = "taxa 4xx", "=E3/E1"
    err["D7"] = f"Fonte: serie {REQUESTS}, endpoint={ENDPOINT}. PromQL: sum({REQUESTS}{{http_response_code=~\"5..\"}}) / sum({REQUESTS})"
    for cell in ("A1", "B1", "D1", "D2", "D3", "D4", "D5"):
        err[cell].font = BOLD

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.snapshot, args.out)
    print(f"gravado: {args.out}")


if __name__ == "__main__":
    main()
