"""Quantize ONNX models to ESPDL format using esp-ppq for ESP32-S3."""
import os
import sys
import glob
import zipfile

# Patch esp-ppq BEFORE importing: disable RequantizeLinear insertion.
# esp-ppq inserts RequantizeLinear ops when scale_diff >= 1e-5, but fails to
# export their parameters, causing ESP-DL FbsLoader to crash (null pointer).
# NOTE: Must run in same process — Docker containers are ephemeral.
import site, subprocess
for _sp in site.getsitepackages():
    _p = os.path.join(_sp, "esp_ppq/parser/espdl/export_patterns.py")
    if os.path.exists(_p):
        subprocess.run(["sed", "-i",
            "s/scale_diff >= 1e-5/scale_diff >= 1e5/g; s/zeropoint_diff >= 1e-1/zeropoint_diff >= 1e5/g",
            _p], check=True)
        _cache = os.path.join(os.path.dirname(_p), "__pycache__")
        if os.path.exists(_cache):
            for _f in os.listdir(_cache):
                if "export_patterns" in _f:
                    os.remove(os.path.join(_cache, _f))
        # Verify
        with open(_p) as f:
            assert "scale_diff >= 1e5" in f.read(), "Patch failed!"
        print(f"PATCHED: RequantizeLinear disabled")
        break

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

MODELS_DIR = "/workspace/model_ckpts/onnx_fixed"
OUTPUT_DIR = "/workspace/model_ckpts/espdl"
EUROSAT_DIR = "/workspace/model_ckpts/eurosat"
TARGET = "esp32s3"
NUM_BITS = 8
BATCH_SIZE = 32
NUM_CALIB_BATCHES = 16
INPUT_SHAPE = (1, 3, 64, 64)

os.makedirs(OUTPUT_DIR, exist_ok=True)

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from esp_ppq.core import TargetPlatform

# Layers esp-ppq flags with exponent overflow (output_exp - in_exp - w_exp < 0).
# In INT8 this produces a negative requant shift, so the optimized conv SIMD
# kernel reads out of bounds -> LoadProhibited at inference. Promote these
# specific layers to INT16, which has the headroom to keep the exponent valid.
INT16_LAYERS = {
    "MobileNetV3": ["node_Conv_508", "node_conv2d_2", "node_conv2d_13"],
}

def download_eurosat():
    """Download EuroSAT dataset if not present."""
    if os.path.exists(EUROSAT_DIR) and len(os.listdir(EUROSAT_DIR)) > 0:
        print(f"EuroSAT already at {EUROSAT_DIR}")
        return
    os.makedirs(EUROSAT_DIR, exist_ok=True)
    url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
    zip_path = "/tmp/eurosat.zip"
    print(f"Downloading EuroSAT from {url}...")
    import urllib.request
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(EUROSAT_DIR)
    os.remove(zip_path)
    print(f"EuroSAT extracted to {EUROSAT_DIR}")

def make_eurosat_dataloader(batch_size, num_batches):
    """Load real EuroSAT images as calibration data."""
    # NOTE: these models were trained with plain [0,1] inputs (ToTensor only),
    # NOT ImageNet mean/std. Calibrating with ImageNet norm shifts every
    # activation range -> wrong exponents (caused conv2d_13 overflow) and
    # poor quantized accuracy. Verified: original ONNX = 97% top-1 with [0,1],
    # 24% with ImageNet norm.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Find all jpg/tif images
    img_paths = []
    for root, dirs, files in os.walk(EUROSAT_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                img_paths.append(os.path.join(root, f))

    print(f"Found {len(img_paths)} EuroSAT images")
    np.random.shuffle(img_paths)

    total_images = batch_size * num_batches
    img_paths = img_paths[:total_images]

    batches = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                tensors.append(transform(img))
            except Exception:
                tensors.append(torch.randn(3, 64, 64))
        if len(tensors) < batch_size:
            while len(tensors) < batch_size:
                tensors.append(torch.randn(3, 64, 64))
        batches.append(torch.stack(tensors))

    print(f"Created {len(batches)} calibration batches of {batch_size}")
    return batches

MODELS = {
    "EfficientNet": {"mmac": 32.17},
}

download_eurosat()
calib_data = make_eurosat_dataloader(BATCH_SIZE, NUM_CALIB_BATCHES)

for name, info in MODELS.items():
    onnx_files = [f for f in glob.glob(f"{MODELS_DIR}/{name}_nogemm.onnx")]
    if not onnx_files:
        print(f"SKIP {name}: no ONNX file found")
        continue
    onnx_path = onnx_files[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_s8.espdl")

    print(f"\n{'='*60}")
    print(f"Quantizing: {name}")
    print(f"  ONNX: {onnx_path}")
    print(f"  Output: {out_path}")
    print(f"  Target: {TARGET}, bits: {NUM_BITS}")

    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.equalization = True
    quant_setting.equalization_setting.iterations = 4
    quant_setting.equalization_setting.value_threshold = 0.4
    quant_setting.equalization_setting.opt_level = 2

    for op_name in INT16_LAYERS.get(name, []):
        quant_setting.dispatching_table.append(
            operation=op_name, platform=TargetPlatform.ESPDL_S3_INT16)
        print(f"  Mixed-precision: {op_name} -> INT16")

    try:
        dataloader = calib_data

        quantized = espdl_quantize_onnx(
            onnx_import_file=onnx_path,
            espdl_export_file=out_path,
            calib_dataloader=dataloader,
            calib_steps=NUM_CALIB_BATCHES,
            input_shape=list(INPUT_SHAPE),
            target=TARGET,
            num_of_bits=NUM_BITS,
            setting=quant_setting,
            device="cpu",
        )
        size = os.path.getsize(out_path)
        print(f"  OK: {size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("Done! ESPDL models in:", OUTPUT_DIR)
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = os.path.join(OUTPUT_DIR, f)
    print(f"  {f}: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
