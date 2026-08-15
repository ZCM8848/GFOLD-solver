"""钉住 recover_tf 对 gfold time_points 语义的依赖。"""

import pytest

import gfold

from generation.generate import recover_tf


def test_recover_tf_matches_fixed_tof_n100():
    cfg = gfold.Config(
        gfold.Spacecraft(), gfold.Environment(),
        gfold.Solver(n=100, time_of_flight=44.63, tof_min=None, tof_max=None),
    )
    traj = gfold.solve(cfg)
    assert recover_tf(traj) == pytest.approx(44.63, abs=1e-6)


def test_recover_tf_matches_fixed_tof_n50():
    cfg = gfold.Config(
        gfold.Spacecraft(), gfold.Environment(),
        gfold.Solver(n=50, time_of_flight=44.63, tof_min=None, tof_max=None),
    )
    traj = gfold.solve(cfg)
    assert recover_tf(traj) == pytest.approx(44.63, abs=1e-6)
