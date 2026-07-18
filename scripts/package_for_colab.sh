#!/usr/bin/env bash
# Empaqueta jacobian-lens + jlens-harness + sleep-harness en un zip para Colab.
# Los notebooks lo esperan en Drive:  MyDrive/sleep_lab/sleep_lab_bundle.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"    # …/internal-agent
OUT="${1:-$ROOT/sleep_lab_bundle.zip}"

cd "$ROOT"
rm -f "$OUT"
zip -qr "$OUT" jacobian-lens jlens-harness sleep-harness \
    -x "*/.venv/*" "*/.git/*" "*__pycache__*" "*.egg-info*" \
       "*/resultados/*" "*/wandb/*" "*.pt" "*/notebooks/*"

echo "bundle: $OUT ($(du -h "$OUT" | cut -f1))"
echo "subilo a Google Drive en: MyDrive/sleep_lab/sleep_lab_bundle.zip"
