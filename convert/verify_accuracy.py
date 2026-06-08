"""Verify deployed-quantization fidelity for MobileNetV3 on EuroSAT.

esp-ppq's executor simulates the SAME power-of-2 / exponent arithmetic the
ESP32-S3 runs (including the conv2d_13 exponent-clamp that triggers the export
warning). So comparing the quantized graph against the float graph on held-out
EuroSAT images gives a device-faithful accuracy number.

Reports:
  - float top-1      (sanity: should match the trained model)
  - quantized top-1  (what the ESP32-S3 effectively computes)
  - agreement        (quant argmax == float argmax) -- the pure quant-fidelity metric
"""
import os, glob, site, subprocess

# Same numpy<2 + RequantizeLinear patching the quantize script uses.
for _sp in site.getsitepackages():
    _p = os.path.join(_sp, "esp_ppq/parser/espdl/export_patterns.py")
    if os.path.exists(_p):
        subprocess.run(["sed", "-i",
            "s/scale_diff >= 1e-5/scale_diff >= 1e5/g; s/zeropoint_diff >= 1e-1/zeropoint_diff >= 1e5/g",
            _p], check=False)
        break

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, load_onnx_graph
from esp_ppq.core import TargetPlatform
from esp_ppq.executor import TorchExecutor

MODEL = os.environ.get("MODEL", "MobileNetV3")
ONNX = f"/workspace/model_ckpts/onnx_fixed/{MODEL}_nogemm.onnx"
EUROSAT = "/workspace/model_ckpts/eurosat/EuroSAT_RGB"
CLASSES = sorted(os.listdir(EUROSAT))                 # alphabetical = ImageFolder order
INPUT_SHAPE = (1, 3, 64, 64)
_INT16 = {"MobileNetV3": ["node_Conv_508", "node_conv2d_2", "node_conv2d_13"]}
INT16_LAYERS = _INT16.get(MODEL, [])
print(f"MODEL={MODEL}")
DEVICE = os.environ.get("DEVICE", "cpu")
N_CALIB = int(os.environ.get("N_CALIB", "256"))   # for calibration
N_TEST = int(os.environ.get("N_TEST", "500"))     # held-out test images (disjoint from calib)
SEED = 42

tf = transforms.Compose([    # plain [0,1] -- matches training (see check_onnx_acc)
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def load_split():
    """Return (calib_imgs, test_imgs[(tensor,label)]) with disjoint indices."""
    rng = np.random.RandomState(SEED)
    per_class = {}
    for ci, c in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(EUROSAT, c, "*.jpg")))
        rng.shuffle(files)
        per_class[ci] = files
    calib, test = [], []
    # round-robin so test set is class-balanced
    n_test_pc = N_TEST // len(CLASSES)
    n_calib_pc = N_CALIB // len(CLASSES) + 1
    for ci, files in per_class.items():
        for f in files[:n_test_pc]:
            test.append((f, ci))
        for f in files[n_test_pc:n_test_pc + n_calib_pc]:
            calib.append(f)
    return calib, test

def to_batches(paths, bs=32):
    batches, cur = [], []
    for p in paths:
        try:
            cur.append(tf(Image.open(p).convert("RGB")))
        except Exception:
            continue
        if len(cur) == bs:
            batches.append(torch.stack(cur)); cur = []
    if cur:
        batches.append(torch.stack(cur))
    return batches

calib_paths, test = load_split()
calib_batches = to_batches(calib_paths)
print(f"calib images: {sum(b.shape[0] for b in calib_batches)}, test images: {len(test)}")

# ---- Quantized graph (device-faithful) ----
USE_INT16 = os.environ.get("USE_INT16", "1") == "1"
USE_EQ = os.environ.get("USE_EQ", "1") == "1"
print(f"config: INT16={USE_INT16} EQUALIZE={USE_EQ}")
qs = QuantizationSettingFactory.espdl_setting()
qs.equalization = USE_EQ
qs.equalization_setting.iterations = 4
qs.equalization_setting.value_threshold = 0.4
qs.equalization_setting.opt_level = 2
if USE_INT16:
    for op in INT16_LAYERS:
        qs.dispatching_table.append(operation=op, platform=TargetPlatform.ESPDL_S3_INT16)
# Advanced PTQ recovery passes (gradient-based) for hard-to-quantize models.
if os.environ.get("BIAS_CORRECT") == "1":
    qs.bias_correct = True
if os.environ.get("LSQ") == "1":
    qs.lsq_optimization = True
if os.environ.get("BLOCKWISE") == "1":
    qs.blockwise_reconstruction = True
print(f"passes: bias_correct={qs.bias_correct} lsq={qs.lsq_optimization} "
      f"blockwise={qs.blockwise_reconstruction}")

NUM_BITS = int(os.environ.get("NUM_BITS", "8"))
print(f"config: NUM_BITS={NUM_BITS}")
quantized = espdl_quantize_onnx(
    onnx_import_file=ONNX,
    espdl_export_file="/tmp/mv3_verify.espdl",
    calib_dataloader=calib_batches,
    calib_steps=len(calib_batches),
    input_shape=list(INPUT_SHAPE),
    target="esp32s3", num_of_bits=NUM_BITS, setting=qs, device=DEVICE,
    skip_export=True,   # IMPORTANT: export mutates the graph in place (int/LUT
                        # conversion) -> TorchExecutor would then read garbage.
    error_report=False,
)
quant_exe = TorchExecutor(graph=quantized, device=DEVICE)

# ---- Float graph (reference) ----
float_graph = load_onnx_graph(ONNX)
float_exe = TorchExecutor(graph=float_graph, device=DEVICE)

def logits(exe, x):
    out = exe.forward(x)[0]            # final output tensor
    return out.reshape(out.shape[0], -1)

def argmax_out(exe, x):
    return logits(exe, x).argmax(dim=1).cpu().numpy()

# Debug: dump float vs quant logits for the first 2 test images.
for p, label in test[:2]:
    x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
    fl = logits(float_exe, x)[0].detach().cpu().numpy()
    ql = logits(quant_exe, x)[0].detach().cpu().numpy()
    print(f"  label={label} float_argmax={int(fl.argmax())} quant_argmax={int(ql.argmax())}")
    print(f"    float: {np.round(fl,2)}")
    print(f"    quant: {np.round(ql,2)}")

f_correct = q_correct = agree = total = 0
for p, label in test:
    try:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
    except Exception:
        continue
    fp = int(argmax_out(float_exe, x)[0])
    qp = int(argmax_out(quant_exe, x)[0])
    f_correct += (fp == label)
    q_correct += (qp == label)
    agree += (fp == qp)
    total += 1

print("\n================= MobileNetV3 EuroSAT fidelity =================")
print(f"  test images      : {total}")
print(f"  float top-1      : {100*f_correct/total:.2f}%")
print(f"  quantized top-1  : {100*q_correct/total:.2f}%  (device-faithful)")
print(f"  quant vs float   : {100*agree/total:.2f}% agreement")
print(f"  top-1 drop       : {100*(f_correct-q_correct)/total:+.2f} pts")
print("===============================================================")
