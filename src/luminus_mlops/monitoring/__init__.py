"""Observabilidade do servico Luminus: log de requisicao/resposta, metricas e drift.

Cada responsabilidade tem um unico dono (docs/2026-11-01_sdd_monitoramento_mlops.md, Decisao 3):
`features` (variaveis da imagem), `outputs` (variaveis da saida do modelo), `drift` (KS, PSI e
limiares), `split` (referencia x holdout), `log_reader` (leitura do log). Nenhum modulo grava ou
registra pixels, serial ou valor de leitura.
"""
