"""Isolate whether low accuracy is preprocessing/mapping vs our ONNX surgery.
Runs ORIGINAL onnx and the nogemm (Gemm->Conv1x1) onnx with onnxruntime on the
same EuroSAT test images, under two normalizations."""
import os, glob, numpy as np
from PIL import Image
import onnxruntime as ort

EUROSAT = "/workspace/model_ckpts/eurosat/EuroSAT_RGB"
ORIG = glob.glob("/workspace/model_ckpts/onnx/MobileNetV3_*ckpt.onnx")[0]
NOGEMM = "/workspace/model_ckpts/onnx_fixed/MobileNetV3_nogemm.onnx"
CLASSES = sorted(os.listdir(EUROSAT))
N_PC = 30  # per class

rng = np.random.RandomState(42)
test = []
for ci, c in enumerate(CLASSES):
    fs = sorted(glob.glob(os.path.join(EUROSAT, c, "*.jpg"))); rng.shuffle(fs)
    for f in fs[:N_PC]:
        test.append((f, ci))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def prep(path, norm):
    im = Image.open(path).convert("RGB").resize((64, 64))
    a = np.asarray(im, np.float32) / 255.0          # HWC [0,1]
    if norm == "imagenet":
        a = (a - MEAN) / STD
    a = a.transpose(2, 0, 1)[None]                   # NCHW
    return a.astype(np.float32)

def run(path_onnx, norm):
    sess = ort.InferenceSession(path_onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    correct = 0
    for f, lab in test:
        out = sess.run(None, {inp: prep(f, norm)})[0].reshape(-1)
        if int(out.argmax()) == lab:
            correct += 1
    return 100.0 * correct / len(test)

for norm in ["imagenet", "plain01"]:
    print(f"norm={norm:9s}  original={run(ORIG, norm):5.2f}%   nogemm={run(NOGEMM, norm):5.2f}%")
print(f"(N={len(test)} images, chance=10%)")
