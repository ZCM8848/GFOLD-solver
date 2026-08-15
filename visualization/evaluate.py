"""评估可视化：预测 vs 真实散点、燃料 suboptimality、混淆矩阵、PR 曲线。

用法：
    python -m visualization.evaluate --n-feasible 500 --n-infeasible 500
"""

import argparse
import sys
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generation.gfold_glue import solve_fixed
from inference.predictor import TOFNetPredictor
from visualization._common import load_data, save


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="评估可视化")
    ap.add_argument("--ckpt", default="models/tofnet.pt")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--n-feasible", type=int, default=500)
    ap.add_argument("--n-infeasible", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pred = TOFNetPredictor(args.ckpt, device="cpu")
    X, yf, yt, fm, meta = load_data(args.datadir)
    tf_mean = pred.tf_mean
    th = pred.threshold

    rng = np.random.default_rng(args.seed)
    feas_idx = np.where(yf.astype(bool))[0]
    infeas_idx = np.where(~yf.astype(bool))[0]
    feas_sel = rng.choice(feas_idx, min(args.n_feasible, len(feas_idx)), replace=False)
    infeas_sel = rng.choice(infeas_idx, min(args.n_infeasible, len(infeas_idx)), replace=False)

    print(f"求解 {len(feas_sel)} 可行样本 ×2（预测 TF + 常数基线）...")
    t0 = time.time()
    opt_tf, pred_tf, base_gap, pred_gap = [], [], [], []
    for i in feas_sel:
        feat = X[i]
        opt = float(yt[i]); opt_fm = float(fm[i])
        pf = float(pred.predict(feat)[1])
        ok_p, fmp = solve_fixed(feat, pf, meta)
        ok_b, fmb = solve_fixed(feat, tf_mean, meta)
        opt_tf.append(opt); pred_tf.append(pf)
        pred_gap.append(opt_fm - fmp if ok_p else np.nan)
        base_gap.append(opt_fm - fmb if ok_b else np.nan)

    opt_tf = np.array(opt_tf); pred_tf = np.array(pred_tf)
    pred_gap = np.array(pred_gap); base_gap = np.array(base_gap)

    # 分类
    sel = np.concatenate([feas_sel, infeas_sel])
    y_true = yf[sel].astype(bool)
    p_all = np.array([float(pred.predict(X[i])[0]) for i in sel])
    y_pred = p_all >= th
    tp = int(np.sum(y_pred & y_true)); fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true)); tn = int(np.sum(~y_pred & ~y_true))

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "预测 TF vs 最优 TF", "燃料损失分布（预测 vs 常数基线）",
        "混淆矩阵 @阈值", "PR 曲线"])

    # 散点
    fig.add_trace(go.Scatter(x=opt_tf, y=pred_tf, mode="markers",
                             marker=dict(size=4, opacity=0.5), name="样本"), 1, 1)
    lo, hi = opt_tf.min(), opt_tf.max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="lime", dash="dash"), name="理想"), 1, 1)
    fig.update_xaxes(title_text="最优 TF (s)", row=1, col=1)
    fig.update_yaxes(title_text="预测 TF (s)", row=1, col=1)

    # 燃料损失直方图
    fig.add_trace(go.Histogram(x=pred_gap[~np.isnan(pred_gap)], nbinsx=40,
                               name="预测 TF", marker_color="cyan"), 1, 2)
    fig.add_trace(go.Histogram(x=base_gap[~np.isnan(base_gap)], nbinsx=40,
                               name="常数基线", marker_color="red",
                               opacity=0.6), 1, 2)
    fig.update_xaxes(title_text="燃料损失 (kg)", row=1, col=2)
    fig.update_yaxes(title_text="计数", row=1, col=2)

    # 混淆矩阵
    cm = np.array([[tn, fp], [fn, tp]])
    fig.add_trace(go.Heatmap(z=cm, x=["预测不可行", "预测可行"],
                             y=["真实不可行", "真实可行"],
                             colorscale="Blues", texttemplate="%{z}",
                             showscale=False), 2, 1)

    # PR 曲线
    order = np.argsort(p_all)[::-1]
    ps = p_all[order]; ts = y_true[order]
    ctp = np.cumsum(ts); cfp = np.cumsum(~ts)
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    recall = ctp / max(ts.sum(), 1)
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                             name="PR 曲线"), 2, 2)
    fig.update_xaxes(title_text="recall", row=2, col=2)
    fig.update_yaxes(title_text="precision", row=2, col=2)

    fig.update_layout(height=800, title=f"评估可视化（求解耗时 {time.time()-t0:.0f}s）")
    save(fig, "evaluate.html")


if __name__ == "__main__":
    main()
