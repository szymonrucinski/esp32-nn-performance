"""Production MobileNetV3 quantization for ESP32-S3.

PTQ with power-of-2 INT8 destroys MobileNetV3 (SE + hardswish ranges are hostile
to pow2 scales): INT8 ~10%, INT16 ~55% top-1. The recovery is FULL INT16 plus
esp-ppq's gradient-based LSQ + bias-correction passes (GPU), which restores
~91.5% top-1 (float 98%). Calibration uses plain [0,1] inputs (the training norm).

Run with: docker run --gpus all ... python convert/quantize_mv3_prod.py
"""
import os, glob
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
    """ESP-DL conv/gemm/mul need output_exp >= in0_exp + in1_exp (pow2 fixed-point).
    LSQ can leave a layer (MobileNetV3 SE squeeze conv2d_13) violating this. Raise
    the output scale to the smallest power-of-2 that satisfies it -- coarser output,
    but valid. The SE squeeze feeds a saturating HardSigmoid, so this is harmless."""
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
            m.scale = m.scale * (2.0 ** (need - out_e))   # raise output exponent
            print(f"  fixed {op.name}: out_exp {out_e} -> {need} (in0={in0_e} in1={in1_e})")
            fixed += 1
    print(f"exponent overflow fixes: {fixed}")

ONNX = "/workspace/model_ckpts/onnx_fixed/MobileNetV3_nogemm.onnx"
OUT = "/workspace/model_ckpts/espdl/mobilenetv3_s8.espdl"
EUROSAT = "/workspace/model_ckpts/eurosat/EuroSAT_RGB"
DEVICE = os.environ.get("DEVICE", "cuda")
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
qs.bias_correct = True          # systematic-error correction
qs.lsq_optimization = True      # gradient-based quant-param finetune (needs GPU)

graph = espdl_quantize_onnx(
    onnx_import_file=ONNX,
    espdl_export_file=OUT,
    calib_dataloader=calib,
    calib_steps=len(calib),
    input_shape=list(INPUT_SHAPE),
    target="esp32s3",
    num_of_bits=16,             # full INT16 -- INT8 unrecoverable for this model
    setting=qs,
    device=DEVICE,
    error_report=False,
    skip_export=True,           # surgery before export
)

fix_exponent_overflow(graph)

PFL.Exporter(platform=get_target_platform("esp32s3", 16)).export(
    file_path=OUT, graph=graph, export_config=True)
print("OK ->", OUT, f"{os.path.getsize(OUT)/1024/1024:.2f} MB")
