# INT8 CNN Benchmark on ESP32-S3 (Xtensa LX7)

Latency, energy and MAC/cycle of INT8 CNNs on the ESP32-S3 (Xtensa LX7, dual core,
240 MHz) with [ESP-DL](https://github.com/espressif/esp-dl). Fills the Xtensa LX7
column of an IEEE TIM cross-platform table.

Energy is measured with a Nordic PPK2 source meter on the 3V3 rail. 10 runs of
continuous inference per model.

## Results (INT8, EuroSAT)

| Model        | MMAC  | ms/inf | mJ/inf         | MAC/cycle | top-1 (float to quant) |
|--------------|-------|--------|----------------|-----------|------------------------|
| MobileNetV3  | 6.12  | 34.81  | 8.846 ± 0.013  | 0.7325    | 98% to ~10% (PTQ broke) |
| MCUNetV1     | 20.41 | 75.93  | 19.883 ± 0.037 | 1.1200    | n/a (1000-class proxy)  |
| EfficientNet | 32.17 | 252.48 | 63.028 ± 0.155 | 0.5309    | 98.4% to 19.2% (PTQ broke) |
| SqueezeNet   | 51.61 | 130.03 | 34.685 ± 0.055 | 1.6537    | 97% to 90.4%           |

Active power is 250 to 267 mW board level for all models. Energy per MAC is
0.67 to 1.96 mJ/MMAC, in line with the reference below.

## Host setup

Benchmark and power scripts run in a [uv](https://docs.astral.sh/uv/) venv:

```bash
uv venv esp32-s3-profiling
source esp32-s3-profiling/bin/activate
uv pip install -r requirements-host.txt
```

## Docker (build, flash, quantize)

| Image | Build |
|-------|-------|
| `espressif/idf:release-v5.3` | `docker compose pull` |
| `model-convert` (onnx2tf) | `docker build -t model-convert -f convert/Dockerfile convert/` |
| `esp-ppq` (quant, numpy<2) | `docker build -t esp-ppq -f convert/Dockerfile.esp-ppq convert/` |

## Run

```bash
./espdl_bench/bench_all.sh             # latency + MAC/cycle, all models
python3 espdl_bench/measure_power_ppk2.py   # PPK2 energy (board on PPK2, USB out)
```

Output goes to `results/`. Quantize with `convert/quantize_espdl.py` (ONNX to INT8

## Hardware

- ESP32-S3-EYE, 8 MB octal PSRAM, 8 MB flash. USB-JTAG shows up as `/dev/ttyACM0`
  (`303a:1001`).
- Needs ESP-DL 3.3.5 (3.3.4 loader crashes on INT16 requant params).
- PPK2 source mode: `VOUT` to `J9 pin 2 (VDD_3V3)`, `GND` to `J9 GND`. Board boots
  from VOUT, no USB needed.

## Reference

Same chip, same meter (PPK2), CNN inference energy:

> B. Karic, N. Herrmann, J. Stenkamp, P. Scharf, F. Gieseke, A. Schwering.
> *Send Less, Save More: Energy-Efficiency Benchmark of Embedded CNN Inference vs.
> Data Transmission in IoT.* arXiv:2510.24829 (2025).
> <https://arxiv.org/abs/2510.24829>
