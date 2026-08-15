"""归一化往返一致性测试。"""

import pytest

from common import config as cfg


def test_normalize_bounds_map_to_unit_interval():
    lo = [cfg.NORM_BOUNDS[f][0] for f in cfg.FEATURES]
    hi = [cfg.NORM_BOUNDS[f][1] for f in cfg.FEATURES]
    assert cfg.normalize(lo) == pytest.approx([0.0] * cfg.N_FEATURES)
    assert cfg.normalize(hi) == pytest.approx([1.0] * cfg.N_FEATURES)


def test_normalize_is_affine():
    f = [cfg.NORM_BOUNDS[name][0] + cfg.NORM_BOUNDS[name][1]
         for name in cfg.FEATURES]
    mid = cfg.normalize([(lo + hi) / 2 for lo, hi in
                         (cfg.NORM_BOUNDS[n] for n in cfg.FEATURES)])
    assert mid == pytest.approx([0.5] * cfg.N_FEATURES)
