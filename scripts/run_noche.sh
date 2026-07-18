#!/usr/bin/env bash
# Piloto unificado: matriz V2 -> pasada lexical -> V6 -> apagar sesión.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for ronda in $(seq 1 100); do
  n=$(ls resultados/colab/v2/exp1_seed*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "[piloto $(date +%H:%M)] ronda $ronda: $n/4 semillas V2"
  [ "$n" -ge 4 ] && break
  pgrep -f "run_v2.sh" >/dev/null || { echo "[piloto] relanzando driver V2"; scripts/run_v2.sh "0 1 2 3" A100 >> resultados/colab/v2/driver.log 2>&1 & }
  sleep 300
done
pkill -f "run_v2.sh" 2>/dev/null; pkill -f "colab_orchestrate" 2>/dev/null; pkill -f "colab exec -s sleep-lab" 2>/dev/null; sleep 5
n=$(ls resultados/colab/v2/exp1_seed*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -ge 4 ]; then
  echo "[piloto] V2 completa; pasada lexical"
  scripts/run_v2_lex.sh "0 1 2 3" A100 2>&1 | tee resultados/colab/v2/lex.log | tail -3
  echo "[piloto] V6"
  scripts/run_v6.sh "0 1 2" A100 2>&1 | tee resultados/colab/v6/driver.log | tail -3
else
  echo "[piloto] V2 incompleta ($n/4) tras 40 rondas; freno sin controles"
fi
colab stop -s sleep-lab 2>&1 | tail -1
echo "[piloto] FIN ($(date +%H:%M)) con V2=$n/4"
