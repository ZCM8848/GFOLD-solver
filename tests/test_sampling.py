"""采样因果链与归一化边界一致性的单元测试。"""

import numpy as np
import pytest

from common import config as cfg
from generation import sampling


def test_throttle_order():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d = sampling.sample_one(rng)
        assert d["min_thrust_pct"] < d["max_thrust_pct"]


def test_effective_twr_above_one():
    """有效最大推力必须能克服重力（保证推力侧不成为不可行来源）。"""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        d = sampling.sample_one(rng)
        twr = d["real_max_thrust"] / (d["wet_mass"] * cfg.KERBIN_G)
        assert twr * d["max_thrust_pct"] > 1.0


def test_mass_chain_consistency():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        d = sampling.sample_one(rng)
        assert d["wet_mass"] == pytest.approx(d["dry_mass"] + d["fuel"])
        assert d["fuel"] > 0
        assert d["dry_mass"] > 0


def test_fuel_consumption_maps_to_isp():
    rng = np.random.default_rng(3)
    for _ in range(500):
        d = sampling.sample_one(rng)
        isp = 1.0 / (d["fuel_consumption"] * cfg.G0)
        lo, hi = cfg.RANGES["isp"]
        assert lo <= isp <= hi


def test_features_within_norm_bounds():
    """钉住 RANGES 与 NORM_BOUNDS 的一致性：采样值必须落在归一化边界内。"""
    rng = np.random.default_rng(4)
    for _ in range(5000):
        d = sampling.sample_one(rng)
        f = sampling.feature_vector(d)
        assert len(f) == cfg.N_FEATURES
        for name, v in zip(cfg.FEATURES, f):
            lo, hi = cfg.NORM_BOUNDS[name]
            assert lo <= v <= hi, f"{name}={v} 超出 [{lo},{hi}]"


def test_reproducible_given_seed():
    a = sampling.sample_one(np.random.default_rng(123))
    b = sampling.sample_one(np.random.default_rng(123))
    assert a == b


def test_override_sampling():
    """overrides 覆盖独立输入，派生量仍按因果链自洽。"""
    rng = np.random.default_rng(0)
    d = sampling.sample_one(rng, overrides={"z": 9000.0, "vz": -700.0})
    assert d["z"] == 9000.0
    assert d["vz"] == -700.0
    assert d["wet_mass"] == pytest.approx(d["dry_mass"] + d["fuel"])
