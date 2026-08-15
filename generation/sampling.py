"""因果一致采样（纯函数，无 gfold 依赖，可独立单测）。

质量链：dry_mass -> dry_frac -> wet_full -> remaining_frac -> fuel/wet_mass
推力链：twr_max -> real_max_thrust = twr_max * wet_mass * g
"""

import numpy as np

from common import config as cfg


def sample_one(rng: np.random.Generator, overrides: dict = None) -> dict:
    """采样一个因果自洽的参数集。返回物理量 dict（含派生量 wet_mass）。

    overrides: 覆盖独立输入参数（如 {"z": 9000.0}）用于边界外推采样，
    派生量仍按因果链重算，保证物理自洽。
    """
    R = cfg.RANGES
    ov = overrides or {}

    def u(name):
        return float(ov[name]) if name in ov else float(rng.uniform(*R[name]))

    # 质量链
    dry_mass = u("dry_mass")
    dry_frac = u("dry_frac")
    wet_full = dry_mass / dry_frac
    remaining_frac = u("remaining_frac")
    fuel = (wet_full - dry_mass) * remaining_frac
    wet_mass = dry_mass + fuel

    # 推力链
    twr_max = u("twr_max")
    real_max_thrust = twr_max * wet_mass * cfg.KERBIN_G
    max_thrust_pct = u("max_thrust_pct")
    min_thrust_pct = u("min_thrust_pct")

    # Isp -> 燃料消耗率
    isp = u("isp")
    fuel_consumption = 1.0 / (isp * cfg.G0)

    # 约束
    max_angle_deg = u("max_angle_deg")
    glide_slope_angle_deg = u("glide_slope_angle_deg")

    # 状态（位置：高度 + 水平偏移方向均匀；速度：三分量独立）
    z = u("z")
    rho = u("rho")
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    vx = u("vx")
    vy = u("vy")
    vz = u("vz")

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
