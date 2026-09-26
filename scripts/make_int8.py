#!/usr/bin/env python3
"""
make_int8.py - compress the WebGPU SD-Turbo ONNX models to int8.

Shrinks the engine download by converting the MatMul/Gemm weights of the
unet and text encoder to int8 (16 bits -> 8 bits per weight), while keeping
convolutions fp16 (no int8 GPU kernels exist for those). Everything stays
inside one ONNX file with fp32 graph IO, so the browser engine code is
unchanged.

"Decompiling and removing parts" of a neural network is not possible
without retraining - the weights ARE the knowledge. int8 quantization is
the safe equivalent, and every model is verified numerically against the
original before shipping.

Usage:
    python3 make_int8.py <src_dir> <out_dir> [--image]

    src_dir contains: unet/model.onnx, text_encoder/model.onnx,
                      vae_decoder/model.onnx   (the fp16 web export)
    out_dir gets:     unet_int8.onnx, text_encoder_int8.onnx,
                      vae_decoder.onnx (copied), report.json
                      (+ sample_fp16.png / sample_int8.png with --image)
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT
QUANT_OPS = {"DynamicQuantizeLinear", "MatMulInteger", "DequantizeLinear",
             "QuantizeLinear", "GemmInteger"}
# deterministic test input: "a peaceful waterfall in a magical forest, cinematic"
TEST_IDS = [49406, 320, 9461, 16403, 530, 320, 7823, 4167, 267, 25602, 49407] + [0] * 66
SIGMA = 14.6146
VAE_SCALE = 0.18215
IR_TARGET = 9      # stay readable by onnxruntime-web 1.19
OPSET_TARGET = 17


def downgrade_versions(m):
    m.ir_version = min(m.ir_version, IR_TARGET)
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = min(op.version, OPSET_TARGET)
    return m


def selective_fp32(m):
    """fp32-ify only the weight initializers that feed MatMul/Gemm - the ops
    the quantizer will convert. The rest of the graph stays fp16."""
    inits = {i.name for i in m.graph.initializer}
    hit = set()
    for n in m.graph.node:
        if n.op_type in ("MatMul", "Gemm"):
            for x in n.input:
                if x in inits:
                    hit.add(x)
    for idx, init in enumerate(m.graph.initializer):
        if init.name in hit and init.data_type == F16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            m.graph.initializer[idx].CopyFrom(
                numpy_helper.from_array(arr, init.name))
    return len(hit)


def repair_types(m):
    """After quantization the graph mixes fp16 (original) and fp32 (quantized
    subgraphs). Walk nodes in order, compute each tensor's dtype and insert
    Cast nodes wherever a node would receive mixed float types."""
    g = m.graph
    ttype = {}
    for i in g.input:
        ttype[i.name] = i.type.tensor_type.elem_type
    for init in g.initializer:
        ttype[init.name] = init.data_type
    new_nodes = []

    for n in g.node:
        if n.op_type == "Cast":
            to = [a.i for a in n.attribute if a.name == "to"][0]
            for o in n.output:
                ttype[o] = to
            new_nodes.append(n)
            continue
        if n.op_type == "Constant":
            new_nodes.append(n)
            continue
        if n.op_type in QUANT_OPS:
            if n.op_type == "DynamicQuantizeLinear":
                x = n.input[0]
                if ttype.get(x) == F16:  # DQL needs fp32 input
                    cname = x + "__qf32"
                    new_nodes.append(helper.make_node(
                        "Cast", [x], [cname], to=F32))
                    ttype[cname] = F32
                    n.input[0] = cname
                ttype[n.output[0]] = TensorProto.UINT8
                ttype[n.output[1]] = F32
                ttype[n.output[2]] = TensorProto.UINT8
            elif n.op_type == "MatMulInteger":
                for o in n.output:
                    ttype[o] = TensorProto.INT32
            elif n.op_type == "DequantizeLinear":
                ttype[n.output[0]] = F32
            new_nodes.append(n)
            continue
        # regular op: make all float inputs agree; prefer fp16
        # (Resize: leave roi/scales alone - only harmonize the data input)
        check = [0] if n.op_type == "Resize" else range(len(n.input))
        ftypes = [ttype.get(n.input[k]) for k in check
                  if n.input[k] and ttype.get(n.input[k]) in (F16, F32)]
        if not ftypes:
            new_nodes.append(n)
            continue
        target = F16 if F16 in ftypes else F32
        for k in check:
            x = n.input[k]
            if not x:
                continue
            t = ttype.get(x)
            if t in (F16, F32) and t != target:
                cname = x + "__c%d" % len(new_nodes)
                new_nodes.append(helper.make_node(
                    "Cast", [x], [cname], to=target))
                ttype[cname] = target
                n.input[k] = cname
        for o in n.output:
            ttype[o] = target
        new_nodes.append(n)

    del g.node[:]
    g.node.extend(new_nodes)

    # graph outputs must keep their declared type (fp32)
    for o in g.output:
        want = o.type.tensor_type.elem_type
        if want == F32 and ttype.get(o.name) == F16:
            newo = o.name + "__out32"
            g.node.append(helper.make_node("Cast", [o.name], [newo], to=F32))
            ttype[newo] = F32
            o.name = newo
    return m


def quantize(src, dst, log):
    t0 = time.time()
    tmp = dst + ".fp32.onnx"
    for p in (tmp, tmp + ".data"):
        if os.path.exists(p):
            os.remove(p)
    model = onnx.load(src)
    n = selective_fp32(model)
    # the fp32-mixed model can exceed the 2 GB protobuf limit -> external data
    onnx.save_model(model, tmp, save_as_external_data=True,
                    location=os.path.basename(tmp) + ".data",
                    size_threshold=0)
    del model
    quantize_dynamic(tmp, dst,
                     op_types_to_quantize=["MatMul", "Gemm"],
                     weight_type=QuantType.QInt8,
                     per_channel=True)
    m = onnx.load(dst)
    m = repair_types(m)
    del m.graph.value_info[:]          # stale annotations confuse ORT
    m = downgrade_versions(m)
    onnx.save(m, dst)
    for p in (tmp, tmp + ".data"):
        if os.path.exists(p):
            os.remove(p)
    log("quantized %-13s (%d weights) -> %s (%.1f min)" %
        (os.path.basename(os.path.dirname(src)), n, os.path.basename(dst),
         (time.time() - t0) / 60))


def make_session(path):
    return InferenceSession(path, providers=["CPUExecutionProvider"])


def run(sess, feeds):
    names = [i.name for i in sess.get_inputs()]
    out = sess.run(None, {n: feeds[n] for n in names})
    return {o.name: v for o, v in zip(sess.get_outputs(), out)}


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def verify(src, int8_path, feeds, log):
    ref = run(make_session(src), feeds)
    got = run(make_session(int8_path), feeds)
    results = {}
    for k in ref:
        c = cosine(ref[k], got[k])
        results[k] = {"cosine": round(c, 6)}
        log("  %s: cosine=%.6f" % (k, c))
    worst = min(v["cosine"] for v in results.values())
    return {"outputs": results, "worst_cosine": worst, "ok": worst > 0.99}


def randn_latents(seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, 4, 64, 64)).astype(np.float32) * SIGMA)


def unet_feeds(embeddings):
    latents = randn_latents(42)
    divi = (SIGMA ** 2 + 1) ** 0.5
    return {"sample": latents / divi, "timestep": np.array([999], np.int64),
            "encoder_hidden_states": embeddings}


def to_png(arr, path):
    from PIL import Image
    x = np.clip(arr[0].astype(np.float32) / 2 + 0.5, 0, 1)
    x = (x.transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8)  # CWR -> RGB
    Image.fromarray(x).save(path)


def generate(pipeline, out_png, log):
    """1-step SD-Turbo, same math as the browser engine."""
    te, unet, vae = pipeline
    emb = list(run(te, {"input_ids": np.array([TEST_IDS], np.int32)}).values())[0]
    u_out = list(run(unet, unet_feeds(emb)).values())[0]
    pred = randn_latents(42) - SIGMA * u_out
    v_in = [i.name for i in vae.get_inputs()][0]
    img = list(run(vae, {v_in: pred / VAE_SCALE}).values())[0]
    to_png(img, out_png)
    log("  wrote %s" % out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--image", action="store_true",
                    help="also generate sample images fp16 vs int8")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = lambda m: print(m, flush=True)

    report = {"models": {}, "sizes_mb": {}}
    for sub, out_name in (("unet", "unet_int8.onnx"),
                          ("text_encoder", "text_encoder_int8.onnx")):
        src = os.path.join(args.src_dir, sub, "model.onnx")
        dst = os.path.join(args.out_dir, out_name)
        log("[quantize] %s" % sub)
        quantize(src, dst, log)
        report["sizes_mb"][sub] = round(os.path.getsize(dst) / 1048576, 1)

    vae_src = os.path.join(args.src_dir, "vae_decoder", "model.onnx")
    vae_dst = os.path.join(args.out_dir, "vae_decoder.onnx")
    shutil.copyfile(vae_src, vae_dst)   # VAE stays fp16 (small, conv-heavy)
    report["sizes_mb"]["vae_decoder"] = round(os.path.getsize(vae_dst) / 1048576, 1)

    log("[verify] text_encoder")
    feeds = {"input_ids": np.array([TEST_IDS], np.int32)}
    report["models"]["text_encoder"] = verify(
        os.path.join(args.src_dir, "text_encoder", "model.onnx"),
        os.path.join(args.out_dir, "text_encoder_int8.onnx"), feeds, log)

    log("[verify] unet (few minutes on CPU)")
    emb = list(run(make_session(os.path.join(
        args.out_dir, "text_encoder_int8.onnx")), feeds).values())[0]
    report["models"]["unet"] = verify(
        os.path.join(args.src_dir, "unet", "model.onnx"),
        os.path.join(args.out_dir, "unet_int8.onnx"),
        unet_feeds(emb), log)

    report["sizes_mb"]["TOTAL"] = round(sum(report["sizes_mb"].values()), 1)
    log("[sizes] %s" % json.dumps(report["sizes_mb"]))

    if args.image:
        log("[samples] generating fp16 vs int8 images")
        base = (make_session(os.path.join(args.src_dir, "text_encoder", "model.onnx")),
                make_session(os.path.join(args.src_dir, "unet", "model.onnx")),
                make_session(vae_src))
        generate(base, os.path.join(args.out_dir, "sample_fp16.png"), log)
        del base
        lite = (make_session(os.path.join(args.out_dir, "text_encoder_int8.onnx")),
                make_session(os.path.join(args.out_dir, "unet_int8.onnx")),
                make_session(vae_dst))
        generate(lite, os.path.join(args.out_dir, "sample_int8.png"), log)

    ok = all(m["ok"] for m in report["models"].values())
    report["pass"] = ok
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log("[done] pass=%s total=%.0f MB (fp16 engine is ~2400 MB)" %
        (ok, report["sizes_mb"]["TOTAL"]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
