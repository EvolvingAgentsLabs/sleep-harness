#!/usr/bin/env bash
# Pasada LEXICAL (control de la Idea 1): agrega la 5ª condición a cada
# semilla ya archivada de la matriz V2, reusando su parcial (solo corre lo
# nuevo). Re-archiva la tabla de 5 condiciones.
# Uso: scripts/run_v2_lex.sh [semillas="0 1 2 3"] [gpu=A100]
set -uo pipefail
SEMILLAS="${1:-0 1 2 3}"
GPU="${2:-A100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S=sleep-lab

for SEED in $SEMILLAS; do
  ARCH="$ROOT/resultados/colab/v2/exp1_seed${SEED}.json"
  [ -s "$ARCH" ] || { echo "semilla $SEED sin archivo base; salteando"; continue; }
  if python3 -c "
import json, sys
sys.exit(0 if 'lexical' in json.load(open('$ARCH'))['condiciones'] else 1)" 2>/dev/null; then
    echo "== semilla $SEED ya tiene lexical; salteando =="
    continue
  fi
  echo "== lexical, semilla $SEED =="
  W="$ROOT/scripts/colab_runners/run_exp1_lex_actual.py"
  {
    echo "import os"
    echo "os.environ['SLEEP_SEED'] = '$SEED'"
    echo "os.environ['SLEEP_BUNDLE'] = 'rem_facts_v2_*.json'"
    echo "os.environ['SLEEP_CONDS'] = 'none,grad,jspace,combinado,lexical'"
    echo "os.environ['SLEEP_SFT_EPOCHS'] = '5'"
    echo "os.environ['SLEEP_SFT_LR'] = '2e-4'"
    echo "os.environ['SLEEP_SFT_DUP'] = '2'"
    cat "$ROOT/scripts/colab_runners/run_exp1_dream_filter.py"
  } > "$W"
  # el parcial de la semilla (4 condiciones hechas) es el punto de partida
  cp "$ARCH" "$ROOT/resultados/colab/exp1_partial.json"
  if colab status -s $S 2>/dev/null | grep -q "\[$S\]"; then
    echo "mkdir -p /content/resultados" | colab console -s $S >/dev/null 2>&1
    colab upload -s $S "$ARCH" /content/resultados/exp1_partial.json >/dev/null 2>&1
    # bundle CON scores lexical (pisa cualquier versión vieja cacheada)
    colab upload -s $S "$ROOT/bundles/rem_facts_v2_seed0.json" \
      /content/resultados/rem_facts_v2_seed0.json >/dev/null 2>&1
  fi
  "$ROOT/scripts/colab_orchestrate.sh" exp1lex 5 "$GPU" || { echo "lexical semilla $SEED sin converger"; exit 1; }
  cp "$ROOT/resultados/colab/exp1_partial.json" "$ARCH"
  echo "  re-archivada con lexical → $ARCH"
done
echo "== pasada lexical completa =="
