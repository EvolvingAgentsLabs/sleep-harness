#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; S=sleep-lab
LOG=resultados/colab/multiciclo_exec.log
done_arms() { python3 -c "
import json
try: b=json.load(open('resultados/colab/multiciclo_partial.json')).get('brazos',{})
except Exception: b={}
print(len(b))" 2>/dev/null; }
for intento in $(seq 1 12); do
  n=$(done_arms); echo "[loop $(date +%H:%M)] intento $intento: $n/2 brazos"
  [ "$n" -ge 2 ] && break
  if ! colab status -s $S 2>/dev/null | grep -q "\[$S\]"; then
    echo "  recreando A100…"; colab new -s $S --gpu ${GPU:-L4} 2>&1 | tail -1
    scripts/package_for_colab.sh >/dev/null
    colab upload -s $S "$ROOT/../sleep_lab_bundle.zip" /content/sleep_lab_bundle.zip >/dev/null 2>&1
    echo "cd /content && rm -rf lab && mkdir -p lab resultados && unzip -q sleep_lab_bundle.zip -d lab" | colab console -s $S >/dev/null 2>&1
    colab install -s $S "transformers>=5.5" "torchao>=0.16" peft accelerate datasets rapidfuzz 2>&1 | tail -1
  fi
  [ -s resultados/colab/multiciclo_partial.json ] && { echo "mkdir -p /content/resultados"|colab console -s $S >/dev/null 2>&1; colab upload -s $S resultados/colab/multiciclo_partial.json /content/resultados/multiciclo_partial.json >/dev/null 2>&1; }
  : > "$LOG"
  colab exec -s $S -f scripts/colab_runners/run_multiciclo.py --timeout 5400 > "$LOG" 2>&1 &
  EPID=$!
  STALL=0
  while kill -0 "$EPID" 2>/dev/null; do
    sleep 60
    # rescatar partial
    colab download -s $S /content/resultados/multiciclo_partial.json "$LOG.p" >/dev/null 2>&1 && python3 -c "import json;json.load(open('$LOG.p'))" 2>/dev/null && mv "$LOG.p" resultados/colab/multiciclo_partial.json 2>/dev/null || rm -f "$LOG.p"
    # stall detection: si el log no crece en 4 min, o la sesión murió, matar exec
    AGE=$(( $(date +%s) - $(stat -f %m "$LOG" 2>/dev/null || echo 0) ))
    if [ "$AGE" -gt 600 ] || ! colab status -s $S 2>/dev/null | grep -q "\[$S\]"; then
      echo "  exec estancado/sesión caída (log age ${AGE}s) → mato y reintento"
      kill "$EPID" 2>/dev/null; sleep 2; colab stop -s $S >/dev/null 2>&1; break
    fi
  done
  wait "$EPID" 2>/dev/null
  colab download -s $S /content/resultados/multiciclo_partial.json resultados/colab/multiciclo_partial.json >/dev/null 2>&1 || true
done
colab stop -s $S 2>&1 | tail -1
echo "[loop] FIN con $(done_arms)/2 brazos"
