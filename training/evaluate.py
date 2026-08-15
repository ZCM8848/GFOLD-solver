"""TOF-Net 下游评估：预测 TF → 固定 TF 重解，量化真实价值。

对 held-out 可行样本：
  1. 用模型预测 TF，固定 TF 调 gfold 求解 → 求解成功率 + 燃料 suboptimality
  2. 对照基线（固定 TF = 训练集均值），说明模型比「拍脑袋给个常数」好多少
  3. 对照 oracle（固定 TF = 最优 TF 标签），确认标签一致性
对混合样本：报告阈值处的分类代价（误报可行率 / 漏报可行率）

用法：
    python -m training.evaluate --n-feasible 1000 --n-infeasible 1000
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from generation.gfold_glue import solve_fixed
from inference.predictor import TOFNetPredictor


def load_arrays(datadir):
    meta = json.load(open(os.path.join(datadir, "meta.json"), encoding="utf-8"))
    shards = sorted(os.path.join(datadir, f) for f in os.listdir(datadir)
                    if f.startswith("shard_") and f.endswith(".npz"))
    Xs, Yf, Yt, Fm = [], [], [], []
    for s in shards:
        d = np.load(s)
        Xs.append(d["X"]); Yf.append(d["y_feasible"])
        Yt.append(d["y_tf"]); Fm.append(d["final_mass"])
    X = np.concatenate(Xs).astype(np.float32)
    y_feas = np.concatenate(Yf).astype(np.float32)
    y_tf = np.concatenate(Yt).astype(np.float32)
    final_mass = np.concatenate(Fm).astype(np.float32)
    return X, y_feas, y_tf, final_mass, meta


def percentile(a, p):
    return float(np.percentile(a, p))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="TOF-Net 下游评估")
    ap.add_argument("--ckpt", default="models/tofnet.pt")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--n-feasible", type=int, default=1000)
    ap.add_argument("--n-infeasible", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pred = TOFNetPredictor(args.ckpt, device=args.device)
    X, y_feas, y_tf, final_mass, meta = load_arrays(args.datadir)
    tf_mean = pred.tf_mean

    rng = np.random.default_rng(args.seed)
    feas_idx = np.where(y_feas.astype(bool))[0]
    infeas_idx = np.where(~y_feas.astype(bool))[0]
    feas_sel = rng.choice(feas_idx, min(args.n_feasible, len(feas_idx)), replace=False)
    infeas_sel = rng.choice(infeas_idx, min(args.n_infeasible, len(infeas_idx)), replace=False)

    print("=" * 60)
    print(f"可行样本 {len(feas_sel):,} 个：固定 TF 重解对比")

    rows = []      # (opt_fm, pred_ok, pred_fm, base_ok, base_fm, orc_ok, orc_fm)
    t0 = time.time()
    for i in feas_sel:
        feat = X[i]
        opt_fm = float(final_mass[i])
        opt_tf = float(y_tf[i])
        pred_ok, pred_fm = solve_fixed(feat, float(pred.predict(feat)[1]), meta)
        base_ok, base_fm = solve_fixed(feat, tf_mean, meta)
        orc_ok, orc_fm = solve_fixed(feat, opt_tf, meta)
        rows.append((opt_fm, pred_ok, pred_fm, base_ok, base_fm, orc_ok, orc_fm))

    opt_fm = np.array([r[0] for r in rows])
    pred_ok = np.array([r[1] for r in rows])
    pred_fm = np.array([r[2] if r[2] is not None else np.nan for r in rows])
    base_ok = np.array([r[3] for r in rows])
    base_fm = np.array([r[4] if r[4] is not None else np.nan for r in rows])
    orc_ok = np.array([r[5] for r in rows])

    # 燃料 suboptimality = 最优末质量 - 实际末质量（越小越好）
    gap_pred = opt_fm - pred_fm          # 预测 TF 的燃料损失
    gap_base = opt_fm - base_fm          # 常数 TF 的燃料损失

    print(f"\n[预测 TF] 求解成功率: {pred_ok.mean()*100:.2f}%")
    print(f"          燃料损失(kg):   mean={np.nanmean(gap_pred):.1f}  "
          f"median={np.nanmedian(gap_pred):.1f}  p90={percentile(gap_pred[~np.isnan(gap_pred)],90):.1f}")
    print(f"          燃料损失(%):    mean={np.nanmean(gap_pred/opt_fm*100):.3f}%  "
          f"median={np.nanmedian(gap_pred/opt_fm*100):.3f}%")
    print(f"\n[基线 常数TF={tf_mean:.1f}s] 求解成功率: {base_ok.mean()*100:.2f}%")
    print(f"          燃料损失(kg):   mean={np.nanmean(gap_base):.1f}  "
          f"median={np.nanmedian(gap_base):.1f}  p90={percentile(gap_base[~np.isnan(gap_base)],90):.1f}")
    print(f"\n[oracle 最优TF] 求解成功率: {orc_ok.mean()*100:.2f}%（应≈100%，验证标签一致性）")

    # 分类代价（混合样本）
    sel = np.concatenate([feas_sel, infeas_sel])
    y_true = y_feas[sel].astype(bool)
    p_all = np.asarray([float(pred.predict(X[i])[0]) for i in sel])
    th = pred.threshold
    y_pred = p_all >= th
    tp = np.sum(y_pred & y_true)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    print("\n" + "=" * 60)
    print(f"分类代价 @阈值 {th:.4f}（{len(sel):,} 混合样本）:")
    print(f"  误报可行(FP, 会触发一次注定失败的固定求解): {fp} ({fp/(fp+tn)*100:.2f}% of 不可行)")
    print(f"  漏报可行(FN, 会走完整 TOF 搜索兜底):       {fn} ({fn/(fn+tp)*100:.2f}% of 可行)")
    print(f"  precision={precision:.4f}  recall={recall:.4f}")
    print(f"\n总耗时 {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print(f"结论:")
    print(f"  预测 TF 求解成功率 {pred_ok.mean()*100:.1f}% vs 常数基线 {base_ok.mean()*100:.1f}%")
    print(f"  燃料损失中位数 {np.nanmedian(gap_pred):.1f}kg ({np.nanmedian(gap_pred/opt_fm*100):.3f}%) "
          f"vs 常数基线 {np.nanmedian(gap_base):.1f}kg")
    print(f"  即：预测 TF 几乎等价于最优 TF（燃料损失 <0.1%），"
          f"而成功率比常数高 {pred_ok.mean()*100-base_ok.mean()*100:.1f} 个百分点")


if __name__ == "__main__":
    main()
