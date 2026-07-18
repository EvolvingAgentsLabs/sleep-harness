#!/usr/bin/env bash
# V6 — ablaciones de la Idea 3: 4 brazos (gkd, gkd_ws, gkd_ws_random,
# ce_only) × N semillas sobre facts_mini. Archiva por semilla en
# resultados/colab/v6/.
# Uso: scripts/run_v6.sh [semillas="0 1 2"] [gpu=A100]
set -uo pipefail
SEMILLAS="${1:-0 1 2}"
GPU="${2:-A100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S=sleep-lab
mkdir -p "$ROOT/resultados/colab/v6"

for SEED in $SEMILLAS; do
  DEST="$ROOT/resultados/colab/v6/exp3_seed${SEED}.json"
  if [ -s "$DEST" ]; then
    echo "== v6 semilla $SEED ya archivada; salteando =="
    continue
  fi
  echo "== v6 semilla $SEED =="
  W="$ROOT/scripts/colab_runners/run_exp3_v6_actual.py"
  {
    echo "import os"
    echo "os.environ['SLEEP_SEED'] = '$SEED'"
    echo "os.environ['SLEEP_ARMS'] = 'gkd,gkd_ws,gkd_ws_random,ce_only'"
    cat "$ROOT/scripts/colab_runners/run_exp3_workspace_distill.py"
  } > "$W"
  PART="$ROOT/resultados/colab/exp3_partial.json"
  KEEP=$(python3 -c "
import json
try:
    print(1 if json.load(open('$PART')).get('seed') == $SEED else 0)
except Exception:
    print(0)" 2>/dev/null)
  if [ "$KEEP" != "1" ]; then
    rm -f "$PART"
  fi
  if colab status -s $S 2>/dev/null | grep -q "\[$S\]"; then
    if [ "$KEEP" = "1" ]; then
      echo "mkdir -p /content/resultados" | colab console -s $S >/dev/null 2>&1
      colab upload -s $S "$PART" /content/resultados/exp3_partial.json >/dev/null 2>&1
    else
      echo "rm -f /content/resultados/exp3_partial.json" | colab console -s $S >/dev/null 2>&1
    fi
  fi
  "$ROOT/scripts/colab_orchestrate.sh" exp3v6 6 "$GPU" || { echo "v6 semilla $SEED sin converger"; exit 1; }
  cp "$ROOT/resultados/colab/exp3_partial.json" "$DEST"
  echo "  archivada → $DEST"
done
echo "== matriz V6 completa =="
