"""Fix ONNX models for ESP-DL compatibility.

esp-ppq/ESP-DL FbsLoader crashes on:
1. Gemm/MatMul ops — parameter serialization bug
2. Clip ops — scalar initializer params not handled

Fixes:
- Gemm → Conv1x1 (keep ReduceMean spatial dims)
- Clip(min, max) → Relu (for Clip(0, 6) aka ReLU6; close enough for INT8)
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import glob
import os

MODELS_DIR = "/workspace/model_ckpts/onnx"
OUTPUT_DIR = "/workspace/model_ckpts/onnx_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def bypass_pre_gemm_flatten(model):
    """Remove Flatten/Reshape nodes that sit between GlobalAvgPool and a Gemm.

    MobileNetV3's classifier is ReduceMean -> Reshape([N,C,1,1]->[N,C]) -> Gemm.
    We keep ReduceMean spatial dims (keepdims=1) and turn the Gemm into a
    Conv1x1, but the intermediate Reshape flattens to 2D, so the Conv1x1 gets a
    2D input and esp-ppq's executor errors. Rewire each Gemm to consume the
    Reshape/Flatten's *input* (the 4D tensor) and drop the flatten node.
    """
    graph = model.graph
    producer = {o: n for n in graph.node for o in n.output}
    gemm_inputs = {n.input[0] for n in graph.node if n.op_type == "Gemm"}

    removed = 0
    for tname in list(gemm_inputs):
        p = producer.get(tname)
        if p is None or p.op_type not in ("Reshape", "Flatten"):
            continue
        src = p.input[0]
        # Rewire any Gemm reading this flatten output to read the 4D source.
        for g in graph.node:
            if g.op_type == "Gemm" and g.input[0] == tname:
                g.input[0] = src
        # Drop the flatten node if nothing else consumes its output.
        still_used = any(tname in n.input for n in graph.node)
        if not still_used:
            graph.node.remove(p)
            removed += 1
    return model, removed


def replace_gemm_with_conv1x1(model):
    """Replace Gemm with Conv1x1 by keeping spatial dims from ReduceMean."""
    graph = model.graph
    replacements = []
    initializers_to_add = []
    reduce_mean_mods = []

    for node in graph.node:
        if node.op_type != "Gemm":
            continue

        alpha = 1.0
        beta = 1.0
        transB = 0
        for attr in node.attribute:
            if attr.name == "alpha": alpha = attr.f
            if attr.name == "beta": beta = attr.f
            if attr.name == "transB": transB = attr.i

        input_name = node.input[0]
        weight_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None
        output_name = node.output[0]

        weight_init = None
        for init in graph.initializer:
            if init.name == weight_name:
                weight_init = init
                break

        if weight_init is None:
            print(f"  Gemm {node.name}: weight not in initializer, skipping")
            continue

        w = numpy_helper.to_array(weight_init)
        if transB:
            w = w.T  # [in, out] -> [out, in] for transB, then we'll reshape
            # Actually: Gemm with transB: Y = A @ B^T + C, B is [out, in]
            # Conv1x1 weight: [out_ch, in_ch, 1, 1]
            w = w.T  # back to [out, in]

        w = w * alpha
        out_features, in_features = w.shape
        print(f"  Gemm {node.name}: {in_features} -> {out_features}")

        # Find chain of ReduceMean nodes feeding into this Gemm
        # Set their keepdims=1 so output stays 4D
        predecessors = find_reduce_mean_chain(graph, input_name)
        if predecessors:
            # Only modify the LAST ReduceMean to keep dims
            # Actually, we need ALL to keep dims for shape propagation
            for rm_node in predecessors:
                reduce_mean_mods.append(rm_node)
                print(f"    Will set keepdims=1 on {rm_node.name}")

        # Conv1x1 weight: [out_channels, in_channels, 1, 1]
        w_conv = w.reshape(out_features, in_features, 1, 1)
        new_weight_name = weight_name + "_conv1x1"
        initializers_to_add.append(
            numpy_helper.from_array(w_conv.astype(np.float32), name=new_weight_name)
        )

        new_nodes = []

        # Conv1x1: input [batch, C, 1, 1] -> [batch, out, 1, 1]
        # Output directly as model output (skip Reshape — benchmark doesn't need 2D)
        conv_out = output_name
        conv_inputs = [input_name, new_weight_name]
        if bias_name:
            if beta != 1.0:
                for init in graph.initializer:
                    if init.name == bias_name:
                        b = numpy_helper.to_array(init) * beta
                        scaled_name = bias_name + "_scaled"
                        initializers_to_add.append(
                            numpy_helper.from_array(b.astype(np.float32), name=scaled_name)
                        )
                        conv_inputs.append(scaled_name)
                        break
            else:
                conv_inputs.append(bias_name)

        new_nodes.append(helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[conv_out],
            name=node.name + "_conv1x1",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        ))

        replacements.append((node, new_nodes))

    # Modify ReduceMean nodes to keepdims=1
    for rm_node in reduce_mean_mods:
        for i, attr in enumerate(rm_node.attribute):
            if attr.name == "keepdims":
                attr.i = 1
                break
        else:
            rm_node.attribute.append(helper.make_attribute("keepdims", 1))

    # Replace Gemm nodes in-place
    for old_node, new_nodes in replacements:
        idx = list(graph.node).index(old_node)
        graph.node.remove(old_node)
        for j, n in enumerate(new_nodes):
            graph.node.insert(idx + j, n)

    for i in initializers_to_add:
        graph.initializer.append(i)

    return model, len(replacements)


def find_reduce_mean_chain(graph, target_output):
    """Find ReduceMean nodes that produce the given output (chain backwards)."""
    chain = []
    current = target_output
    for _ in range(10):
        found = False
        for node in graph.node:
            if node.op_type == "ReduceMean" and current in [o for o in node.output]:
                chain.append(node)
                current = node.input[0]
                found = True
                break
        if not found:
            break
    return chain


def replace_clip_with_relu(model):
    """Replace Clip(0, 6) with Relu. Scalar Clip params crash ESP-DL FbsLoader."""
    graph = model.graph
    init_map = {}
    for init in graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    count = 0
    for i, node in enumerate(graph.node):
        if node.op_type != "Clip":
            continue

        min_val = init_map.get(node.input[1] if len(node.input) > 1 else "", None)
        max_val = init_map.get(node.input[2] if len(node.input) > 2 else "", None)

        if min_val is not None and float(min_val) == 0.0:
            relu_node = helper.make_node(
                "Relu",
                inputs=[node.input[0]],
                outputs=[node.output[0]],
                name=node.name + "_relu"
            )
            graph.node.remove(node)
            graph.node.insert(i, relu_node)
            count += 1

    return model, count


for name in ["MCUNetV1"]:
    files = glob.glob(f"{MODELS_DIR}/{name}_*.onnx")
    files = [f for f in files if "simplified" not in f and "fixed" not in f]
    if not files:
        print(f"SKIP {name}: no ONNX file")
        continue

    onnx_path = files[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}_nogemm.onnx")
    print(f"\n{'='*50}")
    print(f"Fixing: {name}")

    model = onnx.load(onnx_path)
    ops_before = sorted(set(n.op_type for n in model.graph.node))
    print(f"  Ops before: {ops_before}")

    model, num_flat = bypass_pre_gemm_flatten(model)
    print(f"  Removed {num_flat} pre-Gemm Flatten/Reshape")

    model, num_gemm = replace_gemm_with_conv1x1(model)
    print(f"  Replaced {num_gemm} Gemm -> Conv1x1")

    model, num_clip = replace_clip_with_relu(model)
    print(f"  Replaced {num_clip} Clip -> Relu")

    # Must run shape inference after modifying keepdims
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)

    try:
        from onnxsim import simplify
        model, check = simplify(model)
        if check:
            print("  Simplified OK")
    except Exception as e:
        print(f"  Simplification failed (non-fatal): {e}")

    ops_after = sorted(set(n.op_type for n in model.graph.node))
    print(f"  Ops after: {ops_after}")
    print(f"  Gemm gone: {'Gemm' not in ops_after}")
    print(f"  MatMul present: {'MatMul' in ops_after}")

    onnx.checker.check_model(model)
    print("  ONNX validation OK")

    onnx.save(model, out_path)
    print(f"  Saved: {out_path}")

print("\nDone!")
