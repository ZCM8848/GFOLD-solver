"""gfold 胶水层：14 维特征向量 <-> gfold.Config 与求解。

供评估、轨迹可视化等复用，避免 gfold 配置逻辑在多处漂移。
特征顺序见 common.config.FEATURES。
"""

import gfold


def features_to_config(feat, meta, tof):
    """14 维特征向量 -> 固定 TOF 的 gfold.Config（构造式注入）。"""
    x, y, z, vx, vy, vz = feat[0:6]
    dry_mass = float(feat[6])
    fuel = float(feat[7])
    real_max_thrust = float(feat[8])
    min_pct = float(feat[9])
    max_pct = float(feat[10])
    fuel_cons = float(feat[11])
    glide = float(feat[12])
    max_angle = float(feat[13])
    wet_mass = dry_mass + fuel

    return gfold.Config(
        gfold.Spacecraft(
            wet_mass=wet_mass, fuel=fuel, real_max_thrust=real_max_thrust,
            min_thrust_pct=min_pct, max_thrust_pct=max_pct,
            max_velocity=meta["max_velocity"],
            initial_position=[x, y, z], initial_velocity=[vx, vy, vz],
            target_position=meta["target_position"],
            target_velocity=meta["target_velocity"],
            fuel_consumption=fuel_cons,
        ),
        gfold.Environment(
            gravity=meta["gravity"],
            glide_slope_angle_deg=glide, max_angle_deg=max_angle,
        ),
        gfold.Solver(n=meta["n"], time_of_flight=tof, tof_min=None, tof_max=None),
    )


def solve_fixed(feat, tof, meta):
    """固定 TOF 求解，返回 (ok, final_mass)。失败返回 (False, None)。"""
    cfg = features_to_config(feat, meta, tof)
    try:
        traj = gfold.solve(cfg)
        return True, float(traj.final_mass)
    except ValueError:
        return False, None


def solve_search(feat, meta):
    """完整 TOF 搜索求解，返回 (ok, traj_or_None)。"""
    cfg = features_to_config(feat, meta, None)
    try:
        traj = gfold.solve(cfg)
        return True, traj
    except ValueError:
        return False, None
