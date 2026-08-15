"""数据集 EDA：2D 热力图（可行率 + 平均 TF）+ 统计。

对关键特征对画「可行率热力图」与「平均 TF 热力图」，叠加可行性边界等值线；
另输出 TF 直方图、特征-TF 相关性、可行率饼图。
--surface 额外输出平均 TF 的 3D 曲面。

用法：
    python -m visualization.dataset_eda
    python -m visualization.dataset_eda --max-samples 200000
    python -m visualization.dataset_eda --surface
"""

import argparse
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization._common import (FEATURES, derived_features, hist2d_mean,
                                   load_data, save)

PAIRS = [
    ("vz", "z", "下降速度 vz (m/s)", "高度 z (m)"),
    ("rho", "glide_slope_angle_deg", "水平偏移 ρ (m)", "glide slope (°)"),
    ("fuel", "speed", "剩余燃料 (kg)", "速度模 (m/s)"),
    ("twr", "max_thrust_pct", "初始 TWR", "最大节流比"),
]

FEAT_LOOKUP = {n: i for i, n in enumerate(FEATURES)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="数据集 EDA")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--max-samples", type=int, default=200000)
    ap.add_argument("--surface", action="store_true", help="额外输出 TF 3D 曲面")
    args = ap.parse_args()

    X, yf, yt, fm, meta = load_data(args.datadir)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X), min(args.max_samples, len(X)), replace=False)
    X, yf, yt = X[sel], yf[sel], yt[sel]

    vals = {n: X[:, i] for i, n in enumerate(FEATURES)}
    vals.update(derived_features(X))

    # ---- 热力图网格 ----
    fig = make_subplots(
        rows=len(PAIRS), cols=2,
        subplot_titles=[t for p in PAIRS for t in (f"可行率: {p[2]} vs {p[3]}",
                                                    f"平均 TF: {p[2]} vs {p[3]}")],
        vertical_spacing=0.06, horizontal_spacing=0.12)

    for r, (xn, yn, xl, yl) in enumerate(PAIRS, start=1):
        x = vals[xn]; y = vals[yn]
        # 可行率
        g, xc, yc = hist2d_mean(x, y, yf)
        fig.add_trace(go.Heatmap(z=g, x=xc, y=yc, colorscale="RdYlGn",
                                 zmin=0, zmax=1,
                                 colorbar=dict(title="可行率", x=0.455)),
                      row=r, col=1)
        fig.update_xaxes(title_text=xl, row=r, col=1)
        fig.update_yaxes(title_text=yl, row=r, col=1)
        # 平均 TF（仅可行）
        m = yf.astype(bool)
        g2, xc2, yc2 = hist2d_mean(x[m], y[m], yt[m])
        fig.add_trace(go.Heatmap(z=g2, x=xc2, y=yc2, colorscale="Viridis",
                                 colorbar=dict(title="TF (s)", x=1.02)),
                      row=r, col=2)
        fig.update_xaxes(title_text=xl, row=r, col=2)
        fig.update_yaxes(title_text=yl, row=r, col=2)

    fig.update_layout(height=320 * len(PAIRS), title="数据集 EDA：可行率与平均 TF")
    save(fig, "dataset_eda_heatmaps.html")

    # ---- 统计 ----
    feas = yf.astype(bool)
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=["TF 分布（可行）",
                                                         "特征 vs TF 相关系数"],
                         column_widths=[0.4, 0.6])
    fig2.add_trace(go.Histogram(x=yt[feas], nbinsx=60,
                                marker_color="cyan"), 1, 1)
    fig2.update_xaxes(title_text="TF (s)", row=1, col=1)
    fig2.update_yaxes(title_text="计数", row=1, col=1)

    corr_names = FEATURES + ["rho", "speed", "twr"]
    corr_vals = []
    for n in corr_names:
        v = vals[n] if n in vals else (X[:, FEAT_LOOKUP[n]] if n in FEAT_LOOKUP else None)
        c = np.corrcoef(v[feas], yt[feas])[0, 1]
        corr_vals.append(c)
    order = np.argsort(np.abs(corr_vals))
    fig2.add_trace(go.Bar(x=[corr_vals[i] for i in order],
                          y=[corr_names[i] for i in order],
                          orientation="h", marker_color="orange"), 1, 2)
    fig2.update_xaxes(title_text="Pearson 相关系数", row=1, col=2)
    fig2.update_layout(height=500, title="数据集统计")
    save(fig2, "dataset_eda_stats.html")

    if args.surface:
        fig3 = make_subplots(rows=2, cols=2,
                             subplot_titles=[f"TF 曲面: {p[2]} vs {p[3]}" for p in PAIRS],
                             specs=[[{"type": "surface"}, {"type": "surface"}],
                                    [{"type": "surface"}, {"type": "surface"}]])
        for i, (xn, yn, xl, yl) in enumerate(PAIRS):
            x = vals[xn]; y = vals[yn]; m = yf.astype(bool)
            g, xc, yc = hist2d_mean(x[m], y[m], yt[m], nx=30, ny=30)
            fig3.add_trace(go.Surface(z=g, x=xc, y=yc, colorscale="Viridis"),
                           row=i // 2 + 1, col=i % 2 + 1)
        fig3.update_layout(height=800, title="平均 TF 3D 曲面")
        save(fig3, "dataset_eda_tf_surface.html")


if __name__ == "__main__":
    main()
