"""TOF-Net 全局配置：三段（生成 / 训练 / 推理）共享的唯一契约。

特征顺序、采样范围、归一化边界在此定义一次，生成时序列化进
``data/meta.json``，训练与推理读同一份，杜绝特征漂移。
"""

# 常量
G0 = 9.80665          # 标准重力（Isp 换算用）
KERBIN_G = 9.81       # Kerbin 表面重力 m/s^2

SCHEMA_VERSION = 1

# 特征 schema（顺序即契约，勿改动顺序；增删需同步 SCHEMA_VERSION）
FEATURES = [
    "x", "y", "z",
    "vx", "vy", "vz",
    "dry_mass",
    "fuel",
    "real_max_thrust",
    "min_thrust_pct",
    "max_thrust_pct",
    "fuel_consumption",
    "glide_slope_angle_deg",
    "max_angle_deg",
]
N_FEATURES = len(FEATURES)

# 独立采样范围（生成期因果链的入口，其余量为派生量）
# name -> (low, high)
RANGES = {
    "dry_mass": (12000.0, 26000.0),     # 干质量 kg（3.75m 一级 ~18t）
    "dry_frac": (0.04, 0.12),           # 干质量占比 dry/wet_full（质量比 8~25）
    "remaining_frac": (0.03, 0.30),     # 着陆点火时剩余燃料比例
    "twr_max": (1.5, 5.0),              # 初始全油门推重比
    "max_thrust_pct": (0.95, 1.00),     # 最大节流比
    "min_thrust_pct": (0.30, 0.50),     # 最小节流比（现实 ~40%）
    "isp": (200.0, 360.0),              # 比冲 s
    "max_angle_deg": (5.0, 15.0),       # 推力指向锥半角 deg
    "glide_slope_angle_deg": (15.0, 75.0),  # 位置锥半角 deg
    "z": (300.0, 6000.0),               # 初始高度 m
    "rho": (0.0, 3000.0),               # 初始水平偏移 m
    "vx": (-200.0, 200.0),
    "vy": (-200.0, 200.0),
    "vz": (-500.0, -20.0),              # 向下
}

# min-max 归一化边界（先验已知，生成/训练/推理共用同一套）
# name -> (low, high)
NORM_BOUNDS = {
    "x": (-3000.0, 3000.0),
    "y": (-3000.0, 3000.0),
    "z": (0.0, 6000.0),
    "vx": (-200.0, 200.0),
    "vy": (-200.0, 200.0),
    "vz": (-500.0, 0.0),
    "dry_mass": (12000.0, 26000.0),
    "fuel": (2000.0, 200000.0),
    "real_max_thrust": (2.0e5, 1.1e7),
    "min_thrust_pct": (0.30, 0.50),
    "max_thrust_pct": (0.95, 1.00),
    "fuel_consumption": (1.0 / (360.0 * G0), 1.0 / (200.0 * G0)),
    "glide_slope_angle_deg": (15.0, 75.0),
    "max_angle_deg": (5.0, 15.0),
}


def normalize(features):
    """按 NORM_BOUNDS 做 min-max 归一化到 [0, 1]。（返回 list[float]）"""
    out = []
    for name, v in zip(FEATURES, features):
        lo, hi = NORM_BOUNDS[name]
        out.append((v - lo) / (hi - lo))
    return out


# 固定求解器 / 环境参数（不采样）
N = 100                                   # 离散节点数
MAX_VELOCITY = 600.0                      # 速度模上限，保证初始点不违反
GRAVITY = [0.0, 0.0, -KERBIN_G]           # z-up，Kerbin
TARGET_POSITION = [0.0, 0.0, 0.0]         # 目标原点
TARGET_VELOCITY = [0.0, 0.0, 0.0]

# 生成默认值
DEFAULT_N_SAMPLES = 100_000
DEFAULT_SHARD_SIZE = 10_000
DEFAULT_SEED = 0
