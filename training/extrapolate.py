"""边界外推评估：把模型的输入推到训练分布之外，测分类/回归退化程度。

对每个「轴」用 causal 采样（其他参数在训练范围内），把该轴覆盖到训练范围
之外，gfold 完整寻优求 ground-truth，再和模型预测对比，输出每轴的
可行率 / 分类准确率 / TF MAE·RMSE 随外推距离的退化。

用法：
    python -m training.extrapolate --n 25
"""

import argparse
import sys
import time

import numpy as np

import gfold

from generation import sampling
from generation.generate import recover_tf
from generation.gfold_glue import features_to_config
from inference.predictor import TOFNetPredictor

# (label, 独立输入参数名, 训练范围外的取值列表)
AXES = [
    ("altitude z (m)",          "z",                      [6500, 9000, 12000]),
    ("descent vz (m/s)",        "vz",                     [-550, -800, -1000]),
    ("offset rho (m)",          "rho",                    [3500, 6000, 9000]),
    ("twr_max",                 "twr_max",                [1.1, 6.0, 10.0]),
    ("glide_slope (deg)",       "glide_slope_angle_deg",  [8.0, 85.0]),
    ("max_angle (deg)",         "max_angle_deg",          [25.0, 60.0]),
    ("remaining_frac",          "remaining_frac",         [0.40, 0.60, 0.80]),
]

# 联合外推「远角」：多轴同时推到极端
CORNER = {"z": 12000.0, "vz": -1000.0, "rho": 9000.0, "twr_max": 1.2}


def solve_gt(feat, meta):
    cfg = features_to_config(feat, meta, None)   # 完整 TOF 搜索
    try:
        traj = gfold.solve(cfg)
        return True, recover_tf(traj), float(traj.final_mass)
    except ValueError:
        return False, None, None


def eval_config(pred, meta, rng, n, overrides):
    feas, tf_true, tf_pred, p = [], [], [], []
    for _ in range(n):
        d = sampling.sample_one(rng, overrides)
        feat = sampling.feature_vector(d)
        ok, tf, _ = solve_gt(feat, meta)
        feas.append(ok)
        if ok:
            tf_true.append(tf)
        pf, tfp = pred.predict(feat)
        p.append(pf)
        if ok:
            tf_pred.append(tfp)
    feas = np.array(feas)
    p = np.array(p)
    y_pred = p >= pred.threshold
    acc = float(np.mean(y_pred == feas))
    gt_rate = float(feas.mean())
    pred_rate = float(y_pred.mean())
    if tf_true:
        mae = float(np.mean(np.abs(np.array(tf_true) - np.array(tf_pred))))
        rmse = float(np.sqrt(np.mean((np.array(tf_true) - np.array(tf_pred)) ** 2)))
    else:
        mae = rmse = float("nan")
    return {"n": n, "gt_feas%": gt_rate * 100, "acc%": acc * 100,
            "pred_feas%": pred_rate * 100, "mae": mae, "rmse": rmse}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="边界外推评估")
    ap.add_argument("--ckpt", default="models/tofnet.pt")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--n", type=int, default=25, help="每个外推点的样本数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pred = TOFNetPredictor(args.ckpt, device="cpu")
    import json
    meta = json.load(open(f"{args.datadir}/meta.json", encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("边界外推评估（外推点用 gfold 完整寻优求 ground-truth，对比模型预测）")
    print(f"阈值 {pred.threshold:.4f}，每点 {args.n} 样本")
    print("=" * 78)
    print(f"{'配置':28s} {'gt可行%':>8s} {'acc%':>7s} {'pred可行%':>9s} "
          f"{'MAE(s)':>8s} {'RMSE(s)':>8s}")

    # 对照组：训练范围内
    t0 = time.time()
    ctrl = eval_config(pred, meta, rng, args.n, None)
    print(f"{'(对照组 训练范围内)':28s} {ctrl['gt_feas%']:8.1f} {ctrl['acc%']:7.1f} "
          f"{ctrl['pred_feas%']:9.1f} {ctrl['mae']:8.2f} {ctrl['rmse']:8.2f}")

    for label, key, levels in AXES:
        for lv in levels:
            r = eval_config(pred, meta, rng, args.n, {key: lv})
            tag = f"{label} = {lv:g}"
            print(f"{tag:28s} {r['gt_feas%']:8.1f} {r['acc%']:7.1f} "
                  f"{r['pred_feas%']:9.1f} {r['mae']:8.2f} {r['rmse']:8.2f}")

    rc = eval_config(pred, meta, rng, args.n, CORNER)
    print(f"{'联合外推远角':28s} {rc['gt_feas%']:8.1f} {rc['acc%']:7.1f} "
          f"{rc['pred_feas%']:9.1f} {rc['mae']:8.2f} {rc['rmse']:8.2f}")

    print("=" * 78)
    print(f"总耗时 {time.time()-t0:.0f}s")
    print("注：acc% 为阈值处分类准确率；MAE/RMSE 只在 ground-truth 可行样本上算")


if __name__ == "__main__":
    main()
