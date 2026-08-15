"""Benchmark: G-FOLD 内部 TOF 寻优 vs TOF-Net 预测 + 固定 TF 求解。

对同一批 held-out 可行样本分别计时：
  1. 完整 TOF 搜索（time_of_flight=None，内部粗扫 + 黄金分割）
  2. TOF-Net 预测 TF（含前向耗时）+ 固定 TF 单次求解
报告耗时分布与加速比。

用法：
    python -m training.benchmark --n 300
"""

import argparse
import sys
import time

import numpy as np

import gfold

from generation.gfold_glue import features_to_config
from inference.predictor import TOFNetPredictor
from visualization._common import load_data


def _stats(ms):
    a = np.array(ms)
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)), "max": float(a.max()),
        "min": float(a.min()),
    }


def _fmt(s):
    return (f"mean {s['mean']:8.1f}  median {s['median']:8.1f}  "
            f"p90 {s['p90']:8.1f}  max {s['max']:8.1f}  min {s['min']:8.1f} ms")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="G-FOLD vs TOF-Net 求解耗时 benchmark")
    ap.add_argument("--ckpt", default="models/tofnet.pt")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--n", type=int, default=300, help="可行样本数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pred = TOFNetPredictor(args.ckpt, device=args.device)
    X, yf, yt, fm, meta = load_data(args.datadir)
    rng = np.random.default_rng(args.seed)
    feas_idx = np.where(yf.astype(bool))[0]
    sel = rng.choice(feas_idx, args.n, replace=False)

    # 预热（避免冷启动影响首个样本计时）
    for i in sel[:3]:
        gfold.solve(features_to_config(X[i], meta, None))
    if pred.device.type == "cuda":
        torch = __import__("torch")
        torch.cuda.synchronize()

    full_ms, pred_ms, fixed_ms, total_ms = [], [], [], []
    fixed_ok = 0
    for i in sel:
        feat = X[i]

        t0 = time.perf_counter()
        gfold.solve(features_to_config(feat, meta, None))
        full_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _, tf = pred.predict(feat)
        pred_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        try:
            gfold.solve(features_to_config(feat, meta, tf))
            fixed_ok += 1
        except ValueError:
            pass
        fixed_ms.append((time.perf_counter() - t0) * 1000.0)
        total_ms.append(pred_ms[-1] + fixed_ms[-1])

    sf = _stats(full_ms)
    sp = _stats(pred_ms)
    sfix = _stats(fixed_ms)
    stot = _stats(total_ms)

    print("=" * 62)
    print(f"Benchmark: G-FOLD 内部寻优 vs TOF-Net+固定求解  (n={args.n}, "
          f"predictor device={pred.device})")
    print("=" * 62)
    print(f"[A] 完整 TOF 搜索        {_fmt(sf)}")
    print(f"[B] TOF-Net 固定求解     {_fmt(sfix)}")
    print(f"    其中预测前向         {_fmt(sp)}")
    print(f"    TOF-Net 端到端(预测+求解)  {_fmt(stot)}")
    print("-" * 62)
    print(f"加速比 (median): {sf['median']/sfix['median']:.1f}x   "
          f"(mean): {sf['mean']/sfix['mean']:.1f}x")
    print(f"加速比 (端到端 median): {sf['median']/stot['median']:.1f}x")
    print(f"固定求解成功率: {fixed_ok}/{args.n} ({fixed_ok/args.n*100:.1f}%)")
    print("=" * 62)


if __name__ == "__main__":
    main()
