"""Convert ONNX models to TFLite with TFLM-compatible ops (no dynamic shapes)."""
import os
import sys
import glob
import subprocess
import numpy as np

MODELS_DIR = "/workspace/model_ckpts/onnx"
OUTPUT_DIR = "/workspace/model_ckpts/tflite_static"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for onnx_path in sorted(glob.glob(f"{MODELS_DIR}/*.onnx")):
    name = os.path.basename(onnx_path).split("_")[0]
    out_dir = f"/tmp/onnx2tf_{name}"
    out_tflite = f"{OUTPUT_DIR}/{name}_static_float32.tflite"

    print(f"\n{'='*60}")
    print(f"Converting: {name}")
    print(f"  ONNX: {onnx_path}")
    print(f"  Output: {out_tflite}")

    # First simplify the ONNX model to remove dynamic shapes
    simplified = f"/tmp/{name}_simplified.onnx"
    try:
        import onnxsim
        import onnx
        model = onnx.load(onnx_path)
        model_sim, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_sim, simplified)
            print(f"  Simplified OK")
        else:
            simplified = onnx_path
            print(f"  Simplification failed, using original")
    except Exception as e:
        simplified = onnx_path
        print(f"  Simplification error: {e}, using original")

    # Convert using onnx2tf
    try:
        subprocess.run([
            sys.executable, "-m", "onnx2tf",
            "-i", simplified,
            "-o", out_dir,
            "-osd",  # output saved model directory
            "--non_verbose",
        ], check=True, capture_output=True, text=True)

        # Convert saved model to tflite
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(out_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()

        with open(out_tflite, "wb") as f:
            f.write(tflite_model)
        print(f"  TFLite size: {len(tflite_model) / 1024 / 1024:.1f} MB")

        # Verify ops
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        ops = set()
        for d in interp._get_ops_details():
            ops.add(d['op_name'])
        print(f"  Ops: {sorted(ops)}")
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print(f"  Input: {inp['shape']} {inp['dtype'].__name__}")
        print(f"  Output: {out['shape']} {out['dtype'].__name__}")

        # Check for problematic ops
        bad_ops = {'SHAPE', 'SLICE', 'STRIDED_SLICE', 'PACK', 'CAST'}
        found_bad = ops & bad_ops
        if found_bad:
            print(f"  WARNING: potentially unsupported TFLM ops: {found_bad}")
        else:
            print(f"  OK: No dynamic shape ops")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print("Done! Static TFLite models in:", OUTPUT_DIR)
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = os.path.join(OUTPUT_DIR, f)
    print(f"  {f}: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
