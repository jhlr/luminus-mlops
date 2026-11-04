# Medições de Docker e Loki

Trechos que ficavam no README, movidos para cá para o README ficar curto. Os números são os medidos na época.

## Docker

Medido neste Mac (Docker Desktop): imagem de inferência **1,46 GB**, monitor **772 MB**, referência **1,54 GB**; build da inferência
com cache frio e subida 1 min 53 s, monitoramento (inclui baixar Prometheus, Loki, Alloy, Grafana) 1 min 3 s, build da referência
2 min 35 s (o job em si, 500 imagens, leva 58 s). Latência por chamada no Docker: p50 44,5 ms, p95 50,7 ms
(`docs/2026-11-02_monitoramento_metrics_analysis.md`).

## Loki e rotação do log

Rótulos só `job` e `event`. Exemplo: `{job="luminus", event="response"} | json | status_code >= 500`. O Alloy lê só o arquivo ativo:
ao incluir os rotacionados (`inference.log.1`...) o Loki guardou 2.676 linhas contra 1.945 no disco (relidos do início); só com o
ativo ficou **1.945 = 1.945, também depois de 2 rotações forçadas** (`SERVING_LOG_MAX_BYTES=2000000`).
