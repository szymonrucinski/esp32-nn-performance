#!/usr/bin/env bash
# Deploy + latency-benchmark INT8 models on the ESP32-S3 (Xtensa LX7) via ESP-DL.
#
# For each model it: embeds the .espdl, writes model_config.h, builds, flashes,
# reads the firmware's `result=[OK ... ms/inf=.. MAC/cycle=.. min=.. max=..]`
# line, and collects it into results/bench_results.csv.
#
# Energy (mJ/inf) is measured separately with the PPK2 (measure_power_ppk2.py);
# if a results/power_<model>_int8.csv exists, its mean energy is folded into the
# table here, otherwise the mJ column is left blank.
#
#   ./bench.sh                       # all models
#   ./bench.sh SqueezeNet            # just one (or several) by name
#   ESPMON_SECONDS=45 ./bench.sh     # longer capture window per model
#
# Build/flash run inside espressif/idf:release-v5.3 (docker compose service
# "esp-idf"), as root because the build dir is root-owned from prior runs.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/espdl_bench"
ESPDL_DIR="$ROOT_DIR/model_ckpts/espdl"
RESULTS_DIR="$ROOT_DIR/results"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/bench_results.csv"

# name | .espdl file | MMAC (from the IEEE TIM table). Single source of truth.
MODELS=(
  "MobileNetV3  mobilenetv3_int8.espdl  6.12"
  "MCUNetV1     mcunetv1_s8.espdl       20.41"
  "EfficientNet efficientnet_s8.espdl   32.17"
  "SqueezeNet   squeezenet_s8.espdl     51.61"
)

# Optional CLI filter: `./bench.sh EfficientNet MCUNetV1` runs just those.
if [[ $# -gt 0 ]]; then
  WANTED=" $* "
  FILTERED=()
  for entry in "${MODELS[@]}"; do
    read -r n _ _ <<< "$entry"
    [[ "$WANTED" == *" $n "* ]] && FILTERED+=("$entry")
  done
  [[ ${#FILTERED[@]} -gt 0 ]] || { echo "ERROR: no model matched '$*'"; exit 1; }
  MODELS=("${FILTERED[@]}")
fi

PORT="$(readlink -f "${ESP32_PORT:-/dev/ttyACM0}")"
[[ -e "$PORT" ]] || { echo "ERROR: $PORT not found"; exit 1; }
ESPMON_SECONDS="${ESPMON_SECONDS:-35}"

# Mean of the energy_mJ column in a PPK2 CSV, or "" if the file is absent.
mean_energy_mj() {
  local f="$RESULTS_DIR/power_$(echo "$1" | tr '[:upper:]' '[:lower:]')_int8.csv"
  [[ -f "$f" ]] || { echo ""; return; }
  awk -F, 'NR>1 && $5!="" {s+=$5; n++} END {if (n) printf "%.3f", s/n}' "$f"
}

echo "model,mmac,ms_per_inf,mj_per_inf,mac_per_cycle,min_ms,max_ms,status" > "$CSV"
declare -a SUMMARY

for entry in "${MODELS[@]}"; do
  read -r NAME ESPDL MMAC <<< "$entry"
  echo
  echo "==================================================================="
  echo ">>> $NAME  ($ESPDL, ${MMAC} MMAC)"
  echo "==================================================================="

  if [[ ! -f "$ESPDL_DIR/$ESPDL" ]]; then
    echo "  SKIP: $ESPDL_DIR/$ESPDL missing"
    SUMMARY+=("$NAME|$MMAC|-|-|-|missing espdl")
    echo "$NAME,$MMAC,,,,,,missing_espdl" >> "$CSV"
    continue
  fi

  # Embed model + write the per-model config the firmware compiles in.
  cp "$ESPDL_DIR/$ESPDL" "$BENCH_DIR/main/model.espdl"
  printf '#pragma once\n#define MODEL_NAME "%s"\n#define MODEL_MMAC %sf\n' \
    "$NAME" "$MMAC" > "$BENCH_DIR/main/model_config.h"

  LINE="$(ESP32_PORT="$PORT" docker compose -f "$ROOT_DIR/docker-compose.yml" run --rm -T \
      --user 0:0 -e ESPPORT="$PORT" -e ESPMON_SECONDS="$ESPMON_SECONDS" esp-idf bash -lc '
    set -e
    export HOME=/workspace/.esp-idf-home
    cd espdl_bench
    idf.py build >/dev/null 2>&1
    idf.py -p "$ESPPORT" -b 460800 flash >/dev/null 2>&1
    PY=$(ls /opt/esp/python_env/*/bin/python | head -n1)
    "$PY" - <<PYEOF
import os,time,serial
ser=serial.Serial(os.environ["ESPPORT"],115200,timeout=0.2)
end=time.time()+float(os.environ["ESPMON_SECONDS"])
buf=""
while time.time()<end:
    c=ser.read(4096)
    if c: buf+=c.decode("utf-8","replace")
    if "result=[OK" in buf: break
ser.close()
for l in buf.splitlines():
    if "result=[OK" in l:
        print(l.strip()); break
PYEOF
  ' 2>/dev/null | grep "result=\[OK" | head -1 || true)"

  if [[ -z "$LINE" ]]; then
    echo "  RESULT: no benchmark line captured (crash or timeout)"
    SUMMARY+=("$NAME|$MMAC|-|-|-|FAILED")
    echo "$NAME,$MMAC,,,,,,failed" >> "$CSV"
    continue
  fi

  # Parse: result=[OK <name> ms/inf=.. MAC/cycle=.. min=.. max=..]
  MS=$(sed -n 's/.*ms\/inf=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MAC=$(sed -n 's/.*MAC\/cycle=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MIN=$(sed -n 's/.*min=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MAX=$(sed -n 's/.* max=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MJ=$(mean_energy_mj "$NAME")    # from PPK2 CSV if present, else blank
  echo "  RESULT: ms/inf=$MS  MAC/cycle=$MAC  (min=$MIN max=$MAX)  mJ/inf=${MJ:-n/a}"
  SUMMARY+=("$NAME|$MMAC|$MS|${MJ:-"-"}|$MAC|OK")
  echo "$NAME,$MMAC,$MS,$MJ,$MAC,$MIN,$MAX,ok" >> "$CSV"
done

echo
echo "==================================================================="
echo " ESP32-S3 (Xtensa LX7 @240MHz) — ESP-DL INT8 benchmark"
echo "==================================================================="
printf "| %-12s | %6s | %9s | %8s | %9s | %-6s |\n" \
  "Model" "MMAC" "ms/inf" "mJ/inf" "MAC/cyc" "status"
printf "|%14s|%8s|%11s|%10s|%11s|%8s|\n" \
  "--------------" "--------" "-----------" "----------" "-----------" "--------"
for row in "${SUMMARY[@]}"; do
  IFS='|' read -r n mmac ms mj mac st <<< "$row"
  printf "| %-12s | %6s | %9s | %8s | %9s | %-6s |\n" "$n" "$mmac" "$ms" "$mj" "$mac" "$st"
done
echo
echo "CSV: $CSV"
