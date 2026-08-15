"""可视化共享工具：plotly 风格、数据加载、HTML 落盘。"""

import json
import os

import numpy as np
import plotly.io as pio

FIGDIR = "figures"
pio.templates.default = "plotly_dark"

FEATURES = ["x", "y", "z", "vx", "vy", "vz", "dry_mass", "fuel",
            "real_max_thrust", "min_thrust_pct", "max_thrust_pct",
            "fuel_consumption", "glide_slope_angle_deg", "max_angle_deg"]

G = 9.81


def load_data(datadir="data"):
    meta = json.load(open(os.path.join(datadir, "meta.json"), encoding="utf-8"))
    shards = sorted(os.path.join(datadir, f) for f in os.listdir(datadir)
                    if f.startswith("shard_") and f.endswith(".npz"))
    Xs, Yf, Yt, Fm = [], [], [], []
    for s in shards:
        d = np.load(s)
        Xs.append(d["X"]); Yf.append(d["y_feasible"])
        Yt.append(d["y_tf"]); Fm.append(d["final_mass"])
    X = np.concatenate(Xs).astype(np.float32)
    yf = np.concatenate(Yf).astype(np.float32)
    yt = np.concatenate(Yt).astype(np.float32)
    fm = np.concatenate(Fm).astype(np.float32)
    return X, yf, yt, fm, meta


def derived_features(X):
    """返回常用派生量 dict（rho 水平偏移、speed 速度模、twr 初始推重比）。"""
    rho = np.hypot(X[:, 0], X[:, 1])
    speed = np.linalg.norm(X[:, 3:6], axis=1)
    wet = X[:, 6] + X[:, 7]
    twr = X[:, 8] / (wet * G)
    return {"rho": rho, "speed": speed, "twr": twr}


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.write_html(path)
    print(f"[save] {path}")
    return path


def hist2d_mean(x, y, z, nx=40, ny=40):
    """按 (x,y) 分箱，返回每箱 z 的均值（plotly 需 z[y][x]，已转置）。"""
    H, xe, ye = np.histogram2d(x, y, bins=[nx, ny], weights=z)
    C, _, _ = np.histogram2d(x, y, bins=[nx, ny])
    grid = np.divide(H, C, out=np.full_like(H, np.nan), where=C > 0)
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2
    return grid.T, xc, yc
