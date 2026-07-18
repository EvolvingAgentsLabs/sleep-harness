#!/usr/bin/env bash
# Orquestador resiliente para runners en Colab: si la VM muere (keep-alive
# roto -> reclaims), recrea la sesión, redeploya el lab, re-sube el parcial
# y relanza. El runner saltea condiciones ya completadas.
#
# Uso: scripts/colab_orchestrate.sh <exp1|exp3> [max_intentos] [gpu]
set -uo pipefail

EXP="${1:?uso: colab_orchestrate.sh <exp1|exp3> [max_intentos] [gpu]}"
MAX="${2:-5}"
GPU="${3:-T4}"
S=sleep-lab
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$ROOT/../sleep_lab_bundle.zip"
case "$EXP" in
  exp1)  RUNNER="$ROOT/scripts/colab_runners/run_exp1_dream_filter.py"; NCOND=4; PNAME=exp1 ;;
  exp1f) RUNNER="$ROOT/scripts/colab_runners/run_exp1_fuerte.py"; NCOND=4; PNAME=exp1 ;;
  exp1v) RUNNER="$ROOT/scripts/colab_runners/run_exp1_v2_actual.py"; NCOND=4; PNAME=exp1 ;;
  exp1lex) RUNNER="$ROOT/scripts/colab_runners/run_exp1_lex_actual.py"; NCOND=5; PNAME=exp1 ;;
  exp3)  RUNNER="$ROOT/scripts/colab_runners/run_exp3_workspace_distill.py"; NCOND=2; PNAME=exp3 ;;
  exp3v6) RUNNER="$ROOT/scripts/colab_runners/run_exp3_v6_actual.py"; NCOND=4; PNAME=exp3 ;;
  bundle_v2) RUNNER="$ROOT/scripts/colab_runners/run_v2_bundle.py"; NCOND=1; PNAME=bundle_v2 ;;
  *) echo "exp desconocido: $EXP"; exit 1 ;;
esac
PARTIAL_LOCAL="$ROOT/resultados/colab/${PNAME}_partial.json"
PARTIAL_REMOTO="/content/resultados/${PNAME}_partial.json"
AUX_LOCAL="$ROOT/resultados/colab/${PNAME}_aux.json"
AUX_REMOTO="/content/resultados/${PNAME}_aux.json"
mkdir -p "$ROOT/resultados/colab"

completas() {  # cuántas condiciones tiene el parcial local
  if [ "$EXP" = "bundle_v2" ]; then
    # el "parcial" del bundle es el bundle mismo: 1 si existe con dreams
    python3 -c "
import json
try:
    b = json.load(open('$ROOT/resultados/colab/rem_facts_v2_seed0.json'))
    print(1 if b.get('dreams') else 0)
except Exception:
    print(0)"
    return
  fi
  python3 -c "
import json, sys
try:
    print(len(json.load(open('$PARTIAL_LOCAL')).get('condiciones', {})))
except Exception:
    print(0)"
}

for intento in $(seq 1 "$MAX"); do
  echo "== intento $intento/$MAX ($(date +%H:%M)) | condiciones completas: $(completas)/$NCOND =="

  if ! colab status -s $S 2>/dev/null | grep -q "\[$S\]"; then
    echo "  sesión muerta: recreando…"
    colab new -s $S --gpu "$GPU" 2>&1 | tail -1
    "$ROOT/scripts/package_for_colab.sh" "$ZIP" >/dev/null
    colab upload -s $S "$ZIP" /content/sleep_lab_bundle.zip >/dev/null 2>&1
    echo "cd /content && rm -rf lab && mkdir -p lab resultados && unzip -q sleep_lab_bundle.zip -d lab && rm -f lab/sleep-harness/bundles/rem_exp2_c2.json lab/sleep-harness/bundles/rem_facts_mini_seed1.json" \
      | colab console -s $S >/dev/null 2>&1
    colab install -s $S "transformers>=5.5" "torchao>=0.16" peft accelerate datasets rapidfuzz sentence-transformers 2>&1 | tail -1
  fi

  if [ -f "$PARTIAL_LOCAL" ] && [ "$(completas)" -gt 0 ]; then
    # solo parciales válidos y con contenido (un download de un remoto
    # ausente deja un archivo vacío que rompería el resume)
    echo "mkdir -p /content/resultados" | colab console -s $S >/dev/null 2>&1
    colab upload -s $S "$PARTIAL_LOCAL" "$PARTIAL_REMOTO" >/dev/null 2>&1 \
      && echo "  parcial re-subido ($(completas) condiciones)"
  fi

  # aux (scores/baselines cacheados): re-subir si es JSON válido
  if [ -s "$AUX_LOCAL" ] && python3 -c "import json; json.load(open('$AUX_LOCAL'))" 2>/dev/null; then
    echo "mkdir -p /content/resultados" | colab console -s $S >/dev/null 2>&1
    colab upload -s $S "$AUX_LOCAL" "$AUX_REMOTO" >/dev/null 2>&1
  fi

  # kernel limpio SIEMPRE: un intento previo en esta sesión deja la GPU llena
  colab restart-kernel -s $S >/dev/null 2>&1
  sleep 10

  # exec con watchdog: si la VM es reclamada a mitad del exec, el proceso del
  # CLI queda colgado para siempre — vigilar la sesión y cortarlo
  TMPOUT="$ROOT/resultados/colab/${EXP}_exec.log"
  colab exec -s $S -f "$RUNNER" --timeout 7200 > "$TMPOUT" 2>&1 &
  EXEC_PID=$!
  while kill -0 "$EXEC_PID" 2>/dev/null; do
    sleep 60
    if ! colab status -s $S 2>/dev/null | grep -q "\[$S\]"; then
      echo "  la sesión murió durante el exec: cortando y reintentando"
      kill "$EXEC_PID" 2>/dev/null
      break
    fi
  done
  wait "$EXEC_PID" 2>/dev/null
  tail -20 "$TMPOUT"

  # rescatar el parcial pase lo que pase (sin pisar uno bueno con uno vacío)
  if [ "$EXP" = "bundle_v2" ]; then
    colab download -s $S /content/resultados/rem_facts_v2_seed0.json \
      "$ROOT/resultados/colab/rem_facts_v2_seed0.json" >/dev/null 2>&1
  fi
  TMP="$PARTIAL_LOCAL.tmp"
  colab download -s $S "$PARTIAL_REMOTO" "$TMP" >/dev/null 2>&1
  if [ -s "$TMP" ] && python3 -c "import json; json.load(open('$TMP'))" 2>/dev/null; then
    mv "$TMP" "$PARTIAL_LOCAL"
  else
    rm -f "$TMP"
  fi
  TMPA="$AUX_LOCAL.tmp"
  colab download -s $S "$AUX_REMOTO" "$TMPA" >/dev/null 2>&1
  if [ -s "$TMPA" ] && python3 -c "import json; json.load(open('$TMPA'))" 2>/dev/null; then
    mv "$TMPA" "$AUX_LOCAL"
  else
    rm -f "$TMPA"
  fi
  N=$(completas)
  echo "  tras el intento: $N/$NCOND condiciones"
  if [ "$N" -ge "$NCOND" ]; then
    echo "== COMPLETO en el intento $intento =="
    exit 0
  fi
done
echo "== agotados $MAX intentos con $(completas)/$NCOND condiciones =="
exit 1
