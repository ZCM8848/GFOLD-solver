"""导出 TOF-Net 到 ONNX 并用 onnxruntime 推理。

导出的 ONNX 模型是自包含的：输入 14 维原始物理量，内部完成 min-max 归一化、
sigmoid、TF 反标准化，直接输出 (p_feasible, tf 秒)。

用法：
    python -m inference.onnx --ckpt models/tofnet.pt          # 导出 + 校验 + 延迟对比

产物：
    models/tofnet.onnx          # 自包含 ONNX 模型
    models/tofnet.onnx.json     # 契约快照（features / threshold / tf_mean / tf_std）
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from common.model import TOFNet
from inference.predictor import TOFNetPredictor


class _ExportWrapper(nn.Module):
    """把归一化 + 双头 + sigmoid/反标准化 打包进一个可导出的前向。"""

    def __init__(self, model, lo, hi, tf_mean, tf_std):
        super().__init__()
        self.model = model
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.register_buffer("tf_mean", torch.as_tensor([tf_mean], dtype=torch.float32))
        self.register_buffer("tf_std", torch.as_tensor([tf_std], dtype=torch.float32))

    def forward(self, x):
        xn = (x - self.lo) / (self.hi - self.lo)
        logits, tf_std = self.model(xn)
        p = torch.sigmoid(logits)
        tf = tf_std * self.tf_std + self.tf_mean
        return p, tf


def export(ckpt_path, out_path, opset=18):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TOFNet(in_features=ckpt["in_features"], hidden=tuple(ckpt["hidden"]),
                   activation=ckpt["activation"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    features = ckpt["features"]
    lo = [ckpt["norm_bounds"][f][0] for f in features]
    hi = [ckpt["norm_bounds"][f][1] for f in features]
    wrapper = _ExportWrapper(model, lo, hi, ckpt["tf_mean"], ckpt["tf_std"])
    wrapper.eval()

    dummy = torch.zeros(1, ckpt["in_features"])
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["features"], output_names=["p_feasible", "tf"],
        dynamic_axes={"features": {0: "batch"}}, opset_version=opset,
    )

    meta = {"features": features, "threshold": ckpt["threshold"],
            "tf_mean": ckpt["tf_mean"], "tf_std": ckpt["tf_std"],
            "norm_bounds": ckpt["norm_bounds"]}
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_path


class OnnxPredictor:
    """onnxruntime 推理器，接口与 TOFNetPredictor 对齐。"""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._in = self.sess.get_inputs()[0].name
        self._out = [o.name for o in self.sess.get_outputs()]
        try:
            with open(onnx_path + ".json", encoding="utf-8") as f:
                meta = json.load(f)
            self.threshold = float(meta["threshold"])
            self.features = meta["features"]
        except FileNotFoundError:
            self.threshold = 0.5
            self.features = None

    def predict(self, features):
        x = np.asarray(features, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        p, tf = self.sess.run(self._out, {self._in: x})
        p = np.asarray(p).ravel()
        tf = np.asarray(tf).ravel()
        if single:
            return float(p[0]), float(tf[0])
        return p, tf

    def decide(self, features, threshold=None):
        p, tf = self.predict(features)
        th = self.threshold if threshold is None else threshold
        return p >= th, p, tf


def _latency_us(fn, feats, n=2000, warm=100):
    for f in feats[:min(warm, len(feats))]:
        fn(f)
    t0 = time.perf_counter()
    for f in feats[:n]:
        fn(f)
    return (time.perf_counter() - t0) / n * 1e6


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="导出 TOF-Net 到 ONNX")
    ap.add_argument("--ckpt", default="models/tofnet.pt")
    ap.add_argument("--out", default="models/tofnet.onnx")
    ap.add_argument("--n-check", type=int, default=200, help="校验样本数")
    args = ap.parse_args()

    print(f"[export] {args.ckpt} -> {args.out}")
    export(args.ckpt, args.out)

    torch_pred = TOFNetPredictor(args.ckpt, device="cpu")
    onnx_pred = OnnxPredictor(args.out)

    # 用数据里的可行样本校验一致性
    from visualization._common import load_data
    X, yf, yt, fm, meta = load_data("data")
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(yf.astype(bool))[0], args.n_check, replace=False)

    dp, dt = [], []
    for i in idx:
        p1, t1 = torch_pred.predict(X[i])
        p2, t2 = onnx_pred.predict(X[i])
        dp.append(abs(p1 - p2))
        dt.append(abs(t1 - t2))
    print(f"[check] {args.n_check} 样本  max|Δp|={max(dp):.2e}  max|Δtf|={max(dt):.2e}s")

    # 延迟对比（单样本，CPU）
    feats = [X[i] for i in idx]
    ut = _latency_us(lambda f: torch_pred.predict(f), feats)
    uo = _latency_us(lambda f: onnx_pred.predict(f), feats)
    print(f"[latency] torch CPU {ut:7.2f} us   onnxruntime {uo:7.2f} us   "
          f"speedup {ut / uo:.1f}x")


if __name__ == "__main__":
    main()
