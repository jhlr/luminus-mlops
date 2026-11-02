"""Gera o conjunto de referencia do monitoramento (SDD Decisao 7): `artifacts/reference.csv` e
`artifacts/reference_hist.csv`.

Roda `features.py`, `outputs.py` e o classificador real (MobileNetV3-Small) sobre as fotos de
referencia (divisao deterministica de `luminus_mlops.monitoring.split`). E o "treino" deste projeto:
um job em lote do ambiente de REFERENCIA (nada e treinado):

  just build-reference
  # ou: uv run python scripts/build_reference.py [--images-dir DIR] [--seed N]

reference.csv: uma linha por imagem, mesmas colunas escalares do log sem prefixo, sem nome de
arquivo. reference_hist.csv: histogramas 0-255 por canal, cada imagem normalizada para somar 1 e depois
somadas (`drift.aggregate_histograms`, peso igual por imagem; o total de cada coluna e o numero
de imagens). `--hist-only` reescreve so esse arquivo (sem rodar o modelo).
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from luminus_mlops.inference import ImageClassifier
from luminus_mlops.monitoring.drift import aggregate_histograms
from luminus_mlops.monitoring.features import CATEGORICAL_FEATURES, CHANNELS, HIST_BINS, SCALAR_FEATURES, extract_features, uploaded_size
from luminus_mlops.monitoring.outputs import SCALAR_OUTPUT_COLUMNS, derive_outputs
from luminus_mlops.monitoring.split import DEFAULT_SEED, IMAGENETTE_DIR, split_reference_images
from luminus_mlops.paths import ARTIFACTS_DIR

REFERENCE_COLUMNS = (*SCALAR_FEATURES, *CATEGORICAL_FEATURES, *SCALAR_OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=IMAGENETTE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--hist-only", action="store_true", help="reescreve so reference_hist.csv (sem rodar o modelo)")
    args = parser.parse_args()

    reference, holdout = split_reference_images(args.dataset_dir, args.seed)
    print(f"referencia: {len(reference)} imagens (holdout reservado: {len(holdout)}), seed={args.seed}")
    classifier = None if args.hist_only else ImageClassifier()
    rows: list[dict[str, object]] = []
    hists: dict[str, list[Sequence[int]]] = {channel: [] for channel in CHANNELS}
    started = time.perf_counter()
    for index, path in enumerate(reference, start=1):
        with Image.open(path) as image:
            size = uploaded_size(image)
            features = extract_features(image, size)
            if classifier is not None:
                rows.append({**features.scalars(), **derive_outputs(classifier.predict_image(image)).scalars()})
        for channel in CHANNELS:
            hists[channel].append(getattr(features, f"hist_{channel}"))
        if index % 20 == 0 or index == len(reference):
            print(f"  {index}/{len(reference)}  ({time.perf_counter() - started:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.hist_only:
        with (args.out_dir / "reference.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REFERENCE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    sums = {channel: aggregate_histograms(hists[channel]) for channel in CHANNELS}
    with (args.out_dir / "reference_hist.csv").open("w", newline="", encoding="utf-8") as handle:
        hist_writer = csv.writer(handle)
        hist_writer.writerow(["bin", *CHANNELS])
        for bin_index in range(HIST_BINS):
            hist_writer.writerow([bin_index, *(sums[channel][bin_index] for channel in CHANNELS)])
    print(f"gravado: {args.out_dir / 'reference_hist.csv'} ({len(reference)} imagens)" + ("" if args.hist_only else f" e reference.csv ({len(rows)} linhas)"))


if __name__ == "__main__":
    main()
