#!/usr/bin/env bash
# Plan V2 — matriz multi-semilla de exp1 sobre facts_v2, en A100.
# 1) genera el bundle de dreams en GPU (una vez); 2) corre las 4 condiciones
# × N semillas (selección + orden de SFT), archivando cada semilla.
# Uso: scripts/run_v2.sh [semillas="0 1 2 3"] [gpu=A100]
set -uo pipefail
SEMILLAS="${1:-0 1 2 3}"
GPU="${2:-A100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S=sleep-lab
mkdir -p "$ROOT/resultados/colab/v2"

# --- 1) bundle de dreams en GPU (idempotente: lo saltea si ya existe local)
BUNDLE_LOCAL="$ROOT/resultados/colab/v2/rem_facts_v2_seed0.json"
if [ ! -s "$BUNDLE_LOCAL" ]; then
  echo "== generando bundle facts_v2 en GPU =="
  "$ROOT/scripts/colab_orchestrate.sh" bundle_v2 4 "$GPU" || exit 1
  cp "$ROOT/resultados/colab/rem_facts_v2_seed0.json" "$BUNDLE_LOCAL" 2>/dev/null || true
fi
[ -s "$BUNDLE_LOCAL" ] || { echo "sin bundle v2; abortando"; exit 1; }

# --- 2) matriz de semillas
for SEED in $SEMILLAS; do
  DEST="$ROOT/resultados/colab/v2/exp1_seed${SEED}.json"
  if [ -s "$DEST" ]; then
    echo "== semilla $SEED ya archivada; salteando =="
    continue
  fi
  echo "== semilla $SEED =="
  # wrapper por semilla (presupuesto calibrado + bundle v2 + semilla)
  W="$ROOT/scripts/colab_runners/run_exp1_v2_actual.py"
  {
    echo "import os"
    echo "os.environ['SLEEP_SEED'] = '$SEED'"
    echo "os.environ['SLEEP_BUNDLE'] = 'rem_facts_v2_*.json'"
    echo "os.environ['SLEEP_SFT_EPOCHS'] = '5'"
    echo "os.environ['SLEEP_SFT_LR'] = '2e-4'"
    echo "os.environ['SLEEP_SFT_DUP'] = '2'"
    cat "$ROOT/scripts/colab_runners/run_exp1_dream_filter.py"
  } > "$W"
  # conservar el parcial SOLO si pertenece a esta semilla (un relanzamiento
  # del driver no debe borrar el progreso de la semilla en curso)
  PART="$ROOT/resultados/colab/exp1_partial.json"
  KEEP=$(python3 -c "
import json
try:
    print(1 if json.load(open('$PART')).get('seed') == $SEED else 0)
except Exception:
    print(0)" 2>/dev/null)
  if [ "$KEEP" != "1" ]; then
    rm -f "$PART" "$ROOT/resultados/colab/exp1_aux.json"
  else
    echo "  parcial de la semilla $SEED conservado ($(python3 -c "
import json; print(len(json.load(open('$PART'))['condiciones']))") condiciones)"
  fi
  if colab status -s $S 2>/dev/null | grep -q "\[$S\]"; then
    if [ "$KEEP" = "1" ]; then
      echo "mkdir -p /content/resultados" | colab console -s $S >/dev/null 2>&1
      colab upload -s $S "$PART" /content/resultados/exp1_partial.json >/dev/null 2>&1
    else
      echo "rm -f /content/resultados/exp1_partial.json" | colab console -s $S >/dev/null 2>&1
    fi
    colab upload -s $S "$BUNDLE_LOCAL" /content/resultados/rem_facts_v2_seed0.json >/dev/null 2>&1
  fi
  "$ROOT/scripts/colab_orchestrate.sh" exp1v 6 "$GPU" || { echo "semilla $SEED sin converger"; exit 1; }
  cp "$ROOT/resultados/colab/exp1_partial.json" "$DEST"
  echo "  archivada → $DEST"
done
echo "== matriz V2 completa =="
