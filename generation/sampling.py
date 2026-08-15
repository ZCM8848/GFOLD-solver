"""因果一致采样（纯函数，无 gfold 依赖，可独立单测）。

质量链：dry_mass -> dry_frac -> wet_full -> remaining_frac -> fuel/wet_mass
推力链：twr_max -> real_max_thrust = twr_max * wet_mass * g
"""

import numpy as np

from common import config as cfg


def sample_one(rng: np.random.Generator) -> dict:
    """采样一个因果自洽的参数集。返回物理量 dict（含派生量 wet_mass）。"""
    R = cfg.RANGES

    # 质量链
    dry_mass = float(rng.uniform(*R["dry_mass"]))
    dry_frac = float(rng.uniform(*R["dry_frac"]))
    wet_full = dry_mass / dry_frac
    remaining_frac = float(rng.uniform(*R["remaining_frac"]))
    fuel = (wet_full - dry_mass) * remaining_frac
    wet_mass = dry_mass + fuel

    # 推力链
    twr_max = float(rng.uniform(*R["twr_max"]))
    real_max_thrust = twr_max * wet_mass * cfg.KERBIN_G
    max_thrust_pct = float(rng.uniform(*R["max_thrust_pct"]))
    min_thrust_pct = float(rng.uniform(*R["min_thrust_pct"]))

    # Isp -> 燃料消耗率
    isp = float(rng.uniform(*R["isp"]))
    fuel_consumption = 1.0 / (isp * cfg.G0)

    # 约束
    max_angle_deg = float(rng.uniform(*R["max_angle_deg"]))
    glide_slope_angle_deg = float(rng.uniform(*R["glide_slope_angle_deg"]))

    # 状态（位置：高度 + 水平偏移方向均匀；速度：三分量独立）
    z = float(rng.uniform(*R["z"]))
    rho = float(rng.uniform(*R["rho"]))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    vx = float(rng.uniform(*R["vx"]))
    vy = float(rng.uniform(*R["vy"]))
    vz = float(rng.uniform(*R["vz"]))

    return {
        "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "dry_mass": dry_mass,
        "fuel": fuel,
        "wet_mass": wet_mass,
        "real_max_thrust": real_max_thrust,
        "min_thrust_pct": min_thrust_pct,
        "max_thrust_pct": max_thrust_pct,
        "fuel_consumption": fuel_consumption,
        "glide_slope_angle_deg": glide_slope_angle_deg,
        "max_angle_deg": max_angle_deg,
    }


def feature_vector(d: dict) -> list:
    """按 FEATURES 顺序抽取特征向量（原始物理量，未归一化）。"""
    return [d[f] for f in cfg.FEATURES]
