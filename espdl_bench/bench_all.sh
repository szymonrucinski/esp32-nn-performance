#!/usr/bin/env bash
# Deploy + benchmark all fitting models on the ESP32-S3 (Xtensa LX7) via ESP-DL.
#
# For each model it: embeds the .espdl, sets model_config.h, builds, flashes,
# reads the serial STATUS line the firmware prints, and collects ms/inf,
# mJ/inf and MAC/cycle into a table + CSV (results/bench_results.csv).
#
#   ./bench_all.sh                 # run all models
#   ESPMON_SECONDS=45 ./bench_all.sh   # longer capture window per model
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

# model name | .espdl file | MMAC (from the IEEE TIM table)
MODELS=(
  "MobileNetV3 mobilenetv3_s8.espdl 6.12"
  "MCUNetV1    mcunetv1_s8.espdl    20.41"
  "SqueezeNet  squeezenet_s8.espdl  51.61"
)

PORT="$(readlink -f "${ESP32_PORT:-/dev/ttyACM0}")"
[[ -e "$PORT" ]] || { echo "ERROR: $PORT not found"; exit 1; }
ESPMON_SECONDS="${ESPMON_SECONDS:-35}"

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
import os,sys,time,serial
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

  # Parse: result=[OK <name> ms/inf=.. mJ/inf=.. MAC/cycle=.. min=.. max=..]
  MS=$(sed -n 's/.*ms\/inf=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MJ=$(sed -n 's/.*mJ\/inf=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MAC=$(sed -n 's/.*MAC\/cycle=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MIN=$(sed -n 's/.*min=\([0-9.]*\).*/\1/p' <<< "$LINE")
  MAX=$(sed -n 's/.* max=\([0-9.]*\).*/\1/p' <<< "$LINE")
  echo "  RESULT: ms/inf=$MS  mJ/inf=$MJ  MAC/cycle=$MAC  (min=$MIN max=$MAX)"
  SUMMARY+=("$NAME|$MMAC|$MS|$MJ|$MAC|OK")
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
