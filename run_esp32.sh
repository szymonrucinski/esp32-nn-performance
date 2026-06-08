#!/usr/bin/env bash
# Build + flash + monitor the MobileNetV2 (ESP-DL imagenet_cls) example on an
# ESP32-S3 using the official espressif/idf:release-v5.3 Docker image.
#
#   ./run_esp32.sh            # build, flash, then interactive monitor
#   ESPMON_SECONDS=20 ./run_esp32.sh   # non-interactive: dump 20s of serial then exit
#
# The ESP32-S3 built-in USB-JTAG enumerates as /dev/ttyACM0.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

detect_port() {
  if [[ -n "${ESPPORT:-}" ]]; then echo "$ESPPORT"; return 0; fi
  local p
  for p in /dev/serial/by-id/*; do
    [[ -e "$p" ]] || continue
    if [[ "$p" =~ (Espressif|JTAG|CP210|CH340|FTDI|UART|wch) ]]; then echo "$p"; return 0; fi
  done
  p="$(ls -1 /dev/ttyACM* 2>/dev/null | head -n1 || true)"; [[ -n "$p" ]] && { echo "$p"; return 0; }
  p="$(ls -1 /dev/ttyUSB* 2>/dev/null | head -n1 || true)"; [[ -n "$p" ]] && { echo "$p"; return 0; }
  echo "ERROR: no ESP32 serial port found (ls /dev/ttyACM* /dev/ttyUSB*)" >&2
  return 1
}

ESPPORT_LINK="$(detect_port)"
ESPPORT_REAL="$(readlink -f "$ESPPORT_LINK")"
[[ -e "$ESPPORT_REAL" ]] || { echo "ERROR: $ESPPORT_LINK does not resolve to a device" >&2; exit 1; }

TTY_GROUP_ID="$(stat -c %g "$ESPPORT_REAL")"
ESPFLASH_BAUD="${ESPFLASH_BAUD:-460800}"
ESPMON_BAUD="${ESPMON_BAUD:-115200}"
ESPMON_SECONDS="${ESPMON_SECONDS:-}"

echo "Using ESPPORT=$ESPPORT_REAL (group gid=$TTY_GROUP_ID)"
echo "Baud: flash=$ESPFLASH_BAUD monitor=$ESPMON_BAUD"

cd "$ROOT_DIR"

DOCKER_TTY_ARGS=()
if [[ -t 1 && -t 0 ]]; then DOCKER_TTY_ARGS=(-it); else DOCKER_TTY_ARGS=(-T); fi

export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"

ESP32_PORT="$ESPPORT_REAL" docker compose run --rm "${DOCKER_TTY_ARGS[@]}" \
  -e "ESPPORT=$ESPPORT_REAL" \
  -e "TTY_GROUP=$TTY_GROUP_ID" \
  -e "ESPFLASH_BAUD=$ESPFLASH_BAUD" \
  -e "ESPMON_BAUD=$ESPMON_BAUD" \
  -e "ESPMON_SECONDS=${ESPMON_SECONDS:-}" \
  esp-idf bash -lc '
    set -euo pipefail
    export HOME=/workspace/.esp-idf-home
    mkdir -p "$HOME"
    git config --global --add safe.directory "*" 2>/dev/null || true
    cd project
    if [[ ! -f sdkconfig ]]; then
      idf.py set-target esp32s3
    fi
    idf.py build
    idf.py -p "$ESPPORT" -b "$ESPFLASH_BAUD" flash
    if [[ -t 0 && -z "${ESPMON_SECONDS:-}" ]]; then
      exec idf.py -p "$ESPPORT" monitor
    else
      : "${ESPMON_SECONDS:=20}"
      PY="$(ls /opt/esp/python_env/*/bin/python | head -n1)"
      exec "$PY" - <<PYEOF
import os, sys, time, serial
port = os.environ["ESPPORT"]; baud = int(os.environ.get("ESPMON_BAUD","115200"))
deadline = time.time() + float(os.environ.get("ESPMON_SECONDS","20"))
ser = serial.Serial(port, baudrate=baud, timeout=0.2)
try:
    while time.time() < deadline:
        c = ser.read(4096)
        if c: sys.stdout.write(c.decode("utf-8","replace")); sys.stdout.flush()
finally:
    ser.close()
PYEOF
    fi
  '
