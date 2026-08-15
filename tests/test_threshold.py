"""select_threshold 边界情况测试。"""

import numpy as np
import pytest

from training.train import select_threshold


def test_perfect_separation():
    probs = np.array([0.9, 0.8, 0.3, 0.2])
    feas = np.array([True, True, False, False])
    th, f = select_threshold(probs, feas, beta=1.0)
    assert f == pytest.approx(1.0)
    assert 0.3 <= th <= 0.8


def test_all_negative_no_crash():
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    feas = np.array([False, False, False, False])
    th, f = select_threshold(probs, feas, beta=1.0)
    assert np.isfinite(th)


def test_all_positive_no_crash():
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    feas = np.array([True, True, True, True])
    th, f = select_threshold(probs, feas, beta=1.0)
    assert np.isfinite(th)


def test_beta_below_one_favors_precision():
    # 一个高概率负样本 + 一个低概率正样本：beta=0.5 应选更高阈值（更偏 precision）
    probs = np.array([0.9, 0.2, 0.1, 0.05])
    feas = np.array([False, True, True, True])
    th_05, _ = select_threshold(probs, feas, beta=0.5)
    th_10, _ = select_threshold(probs, feas, beta=1.0)
    assert th_05 >= th_10
