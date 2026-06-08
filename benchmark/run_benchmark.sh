#!/usr/bin/env bash
# Benchmark all tflite models on ESP32-S3.
# Usage: ./run_benchmark.sh [model_name]
#   No args = run all that fit in flash. Pass a name to run one.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../model_ckpts/tflite"
BENCHMARK_DIR="$SCRIPT_DIR"

declare -A MODEL_NAMES=(
    ["SqueezeNet"]="SqueezeNet"
    ["MobileNetV3"]="MobileNetV3"
    ["MCUNetV1"]="MCUNetV1"
    ["EfficientNet"]="EfficientNet"
)

declare -A MODEL_MMAC=(
    ["SqueezeNet"]="51.61"
    ["MobileNetV3"]="6.12"
    ["MCUNetV1"]="20.41"
    ["EfficientNet"]="32.17"
)

MAX_FLASH_BYTES=$((7 * 1024 * 1024))  # ~7MB usable in 8MB partition

run_model() {
    local key="$1"
    local tflite_file
    tflite_file=$(ls "$MODELS_DIR"/${key}_*.tflite 2>/dev/null | head -1)

    if [[ -z "$tflite_file" ]]; then
        echo "ERROR: No tflite file found for $key"
        return 1
    fi

    local size
    size=$(stat -c%s "$tflite_file")
    echo "=== $key: $(( size / 1024 / 1024 )) MB ==="

    if (( size > MAX_FLASH_BYTES )); then
        echo "SKIP: $key ($size bytes) exceeds flash limit ($MAX_FLASH_BYTES bytes)"
        echo "========== RESULTS =========="
        echo "Model: $key — Does not fit"
        echo "============================="
        return 0
    fi

    # Symlink model
    ln -sf "$tflite_file" "$BENCHMARK_DIR/main/model.tflite"

    echo "Building and flashing $key..."
    cd "$BENCHMARK_DIR"

    # Clean build to pick up new model
    rm -rf build/esp-idf/main/model.tflite.S build/esp-idf/main/CMakeFiles

    # Build inside docker
    docker compose -f "$SCRIPT_DIR/../docker-compose.yml" run --rm -T \
        -e "ESPPORT=${ESPPORT:-/dev/ttyACM0}" \
        -w /workspace/benchmark \
        esp-idf bash -lc "
            export HOME=/workspace/.esp-idf-home
            mkdir -p \$HOME
            git config --global --add safe.directory '*' 2>/dev/null || true
            if [[ ! -f sdkconfig ]]; then
                idf.py set-target esp32s3
            fi
            idf.py -DMODEL_NAME='\"${MODEL_NAMES[$key]}\"' -DMODEL_MMAC=${MODEL_MMAC[$key]} build
            idf.py -p \$ESPPORT flash
        "

    echo "Monitoring output for 60s..."
    # Capture serial output
    timeout 60 python3 -c "
import serial, sys, time
ser = serial.Serial('${ESPPORT:-/dev/ttyACM0}', 115200, timeout=0.5)
deadline = time.time() + 55
try:
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', 'replace').strip()
        if line:
            print(line)
            sys.stdout.flush()
            if 'MAC/cycle' in line:
                break
finally:
    ser.close()
" 2>/dev/null || true

    echo ""
}

if [[ $# -ge 1 ]]; then
    run_model "$1"
else
    for key in SqueezeNet MobileNetV3 MCUNetV1 EfficientNet; do
        run_model "$key"
        echo ""
    done
fi
