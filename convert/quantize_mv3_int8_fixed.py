"""MobileNetV3 INT8 for ESP32-S3 WITH exponent-overflow surgery.

Plain INT8 PTQ leaves 3 convs (Conv_508, conv2d_2, conv2d_13 = SE squeeze)
violating ESP-DL's pow2 constraint output_exp >= in0_exp + in1_exp, which makes
the optimized conv SIMD kernel index a shift-table out of bounds -> LoadProhibited
crash at load. `fix_exponent_overflow` raises each violating layer's output
exponent to the smallest power-of-2 that satisfies the constraint (coarser output,
but loadable). Goal here: make INT8 MV3 actually RUN on-device. Accuracy stays
poor (INT8 is hostile to MV3's SE/hardswish ranges) -- INT16+LSQ is the
deployable path (quantize_mv3_prod.py). This is the "run it in INT8" path.

CPU, no LSQ (GPU-only). Output: mobilenetv3_int8.espdl
"""
import os, glob, site, subprocess

# numpy<2 + RequantizeLinear patch (same as the other quantize scripts).
for _sp in site.getsitepackages():
    _p = os.path.join(_sp, "esp_ppq/parser/espdl/export_patterns.py")
    if os.path.exists(_p):
        subprocess.run(["sed", "-i",
            "s/scale_diff >= 1e-5/scale_diff >= 1e5/g; s/zeropoint_diff >= 1e-1/zeropoint_diff >= 1e5/g",
            _p], check=False)
        break

import torch, numpy as np
from PIL import Image
from torchvision import transforms
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
import esp_ppq.lib as PFL


def _exp(scale):
    return np.log2(np.atleast_1d(scale.detach().cpu().numpy()))

def _master(tqc):
    m = tqc
    while getattr(m, "dominated_by", m) is not m:
        m = m.dominated_by
    return m

def fix_exponent_overflow(graph):
    """Raise output exponent of any Conv/Gemm/MatMul/Mul where
    output_exp < in0_exp + in1_exp to the smallest pow2 that satisfies it."""
    fixed = 0
    for op in graph.operations.values():
        if op.type not in ("Conv", "Gemm", "MatMul", "Mul"):
            continue
        cfg = op.config
        if len(cfg.input_quantization_config) < 2 or not cfg.output_quantization_config:
            continue
        in0, in1 = cfg.input_quantization_config[0], cfg.input_quantization_config[1]
        out = cfg.output_quantization_config[0]
        if in0.scale is None or in1.scale is None or out.scale is None:
            continue
        in0_e, in1_e = int(_exp(in0.scale)[0]), int(_exp(in1.scale)[0])
        out_e = int(_exp(out.scale)[0])
        need = in0_e + in1_e
        if out_e < need:
            m = _master(out)
            m.scale = m.scale * (2.0 ** (need - out_e))
            print(f"  fixed {op.name}: out_exp {out_e} -> {need} (in0={in0_e} in1={in1_e})")
            fixed += 1
    print(f"exponent overflow fixes: {fixed}")

ONNX = "/workspace/model_ckpts/onnx_fixed/MobileNetV3_nogemm.onnx"
OUT = "/workspace/model_ckpts/espdl/mobilenetv3_int8.espdl"
EUROSAT = "/workspace/model_ckpts/eurosat/EuroSAT_RGB"
DEVICE = os.environ.get("DEVICE", "cpu")
NUM_BITS = 8
INPUT_SHAPE = (1, 3, 64, 64)

tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])  # [0,1]
paths = []
for c in sorted(os.listdir(EUROSAT)):
    paths += sorted(glob.glob(os.path.join(EUROSAT, c, "*.jpg")))[:64]
np.random.RandomState(42).shuffle(paths)
paths = paths[:512]
calib = [torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+32]])
         for i in range(0, len(paths), 32)]
print(f"calib: {len(calib)} batches")

qs = QuantizationSettingFactory.espdl_setting()
qs.equalization = True
qs.equalization_setting.iterations = 4
qs.equalization_setting.value_threshold = 0.4
qs.equalization_setting.opt_level = 2
qs.bias_correct = True          # CPU-friendly systematic-error correction

graph = espdl_quantize_onnx(
    onnx_import_file=ONNX,
    espdl_export_file=OUT,
    calib_dataloader=calib,
    calib_steps=len(calib),
    input_shape=list(INPUT_SHAPE),
    target="esp32s3",
    num_of_bits=NUM_BITS,
    setting=qs,
    device=DEVICE,
    error_report=False,
    skip_export=True,           # surgery before export
)

fix_exponent_overflow(graph)

PFL.Exporter(platform=get_target_platform("esp32s3", NUM_BITS)).export(
    file_path=OUT, graph=graph, export_config=True)
print("OK ->", OUT, f"{os.path.getsize(OUT)/1024/1024:.2f} MB")
