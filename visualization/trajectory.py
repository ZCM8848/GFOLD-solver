"""交互式 3D 轨迹 + 时间序列可视化。

对一个样本求解 gfold，画出：
- 3D 位置轨迹（按推力模着色）+ 着陆点 + glide-slope 锥（可选）
- 时间序列面板：高度 / 速度模 / 推力（带节流上下限带）/ 质量 / 归一化节流

用法：
    python -m visualization.trajectory --sample-index 12345
    python -m visualization.trajectory --random --seed 0
    python -m visualization.trajectory --random --tof 25.0      # 固定 TOF 求解
"""

import argparse
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generation.gfold_glue import features_to_config, solve_search
from visualization._common import derived_features, load_data, save


def _cone_mesh(glide_deg, zmax, n=40):
    g = np.radians(glide_deg)
    z = np.linspace(0, zmax, n)
    theta = np.linspace(0, 2 * np.pi, n)
    Z, T = np.meshgrid(z, theta)
    R = Z * np.tan(g)
    return R * np.cos(T), R * np.sin(T), Z


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="轨迹可视化")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--sample-index", type=int, default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tof", type=float, default=None, help="固定 TOF；缺省=完整搜索最优")
    ap.add_argument("--no-cone", action="store_true", help="不画 glide-slope 锥")
    args = ap.parse_args()

    X, yf, yt, fm, meta = load_data(args.datadir)

    if args.sample_index is not None:
        idx = args.sample_index
    else:
        rng = np.random.default_rng(args.seed)
        feas = np.where(yf.astype(bool))[0]
        idx = int(rng.choice(feas))
    feat = X[idx]
    names = ["x", "y", "z", "vx", "vy", "vz", "dry_mass", "fuel",
             "real_max_thrust", "min_thrust_pct", "max_thrust_pct",
             "fuel_consumption", "glide_slope_angle_deg", "max_angle_deg"]

    cfg = features_to_config(feat, meta, args.tof)
    if args.tof is None:
        ok, traj = solve_search(feat, meta)
    else:
        import gfold
        try:
            traj = gfold.solve(cfg)
            ok = traj.status in ("Solved", "AlmostSolved")
        except ValueError as e:
            print(f"求解失败: {e}")
            return
    if not ok:
        print("样本不可行/求解失败，换个样本试试")
        return

    pos = np.asarray(traj.positions)
    vel = np.asarray(traj.velocities)
    thr = np.asarray(traj.thrusts)
    nthr = np.asarray(traj.normalized_thrusts)
    mass = np.exp(np.asarray(traj.z_values))
    tp = np.asarray(traj.time_points)

    real_thrust = feat[8]; min_pct = feat[9]; max_pct = feat[10]
    tmin = real_thrust * min_pct
    tmax = real_thrust * max_pct
    glide = feat[12]
    speed = np.linalg.norm(vel, axis=1)

    # ---- 3D 轨迹 ----
    cone_xyz = _cone_mesh(glide, pos[:, 2].max() * 1.1)
    cone_trace = go.Surface(x=cone_xyz[0], y=cone_xyz[1], z=cone_xyz[2],
                            colorscale="Greys", opacity=0.15, showscale=False,
                            name="glide-slope 锥")

    traj3d = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines",
        line=dict(color=thr, colorscale="Inferno", width=6,
                  cmin=0.0, cmax=float(real_thrust),
                  colorbar=dict(title="推力 N")),
        name="轨迹",
    )
    target = go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                          marker=dict(size=5, color="lime"), name="着陆点")

    fig3d = go.Figure(data=[cone_trace, traj3d, target] if not args.no_cone
                      else [traj3d, target])
    fig3d.update_layout(scene=dict(aspectmode="data",
                                   xaxis_title="x (m)", yaxis_title="y (m)",
                                   zaxis_title="z (m)"),
                        title=f"样本 #{idx} 着陆轨迹 (tf={tp[-1]:.2f}s)")
    save(fig3d, "trajectory_3d.html")

    # ---- 时间序列面板 ----
    figts = make_subplots(rows=3, cols=2, subplot_titles=[
        "高度 (m)", "速度模 (m/s)", "推力 (N) 与节流带",
        "质量 (kg)", "归一化节流 (0-1)", "推力加速度 |u| (m/s²)"])
    u = np.linalg.norm(np.asarray(traj.u_values), axis=1)

    figts.add_trace(go.Scatter(x=tp, y=pos[:, 2], line=dict(color="cyan")), 1, 1)
    figts.add_trace(go.Scatter(x=tp, y=speed, line=dict(color="orange")), 1, 2)
    figts.add_trace(go.Scatter(x=tp, y=thr, line=dict(color="red")), 2, 1)
    figts.add_hline(y=tmax, line_dash="dash", line_color="red", row=2, col=1)
    figts.add_hline(y=tmin, line_dash="dash", line_color="red", row=2, col=1)
    figts.add_trace(go.Scatter(x=tp, y=mass, line=dict(color="magenta")), 2, 2)
    figts.add_trace(go.Scatter(x=tp, y=nthr, line=dict(color="lime")), 3, 1)
    figts.add_trace(go.Scatter(x=tp, y=u, line=dict(color="yellow")), 3, 2)
    figts.update_layout(height=900, title=f"样本 #{idx} 时间序列",
                        showlegend=False)
    save(figts, "trajectory_series.html")

    print(f"样本 #{idx} 参数:")
    for n, v in zip(names, feat):
        print(f"  {n:20s} {v:.4g}")


if __name__ == "__main__":
    main()
