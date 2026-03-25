"""Tests for Rao spacing and Rayleigh circular statistics (``core.circular_stats``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.circular_stats import (
    _rao_critical_values,
    _rao_table_row_index,
    rao_spacing_test,
    rayleigh_test,
)


class TestRaoTableHelpers:
    def test_row_index_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            _rao_table_row_index(3)

    @pytest.mark.parametrize(
        ("n", "expected_row"),
        [
            (4, 1),
            (10, 7),
            (30, 27),
            (31, 27),
            (32, 27),
            (33, 28),
            (100, 33),
            (2000, 43),
        ],
    )
    def test_row_index_mapping(self, n: int, expected_row: int) -> None:
        assert _rao_table_row_index(n) == expected_row

    def test_critical_values_unknown_row_uses_largest(self) -> None:
        u = _rao_critical_values(999)
        assert u == _rao_critical_values(43)


class TestRaoSpacingTest:
    def test_too_few_angles_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            rao_spacing_test(np.array([0.0, 1.0, 2.0]))

    def test_uniform_quadrants(self) -> None:
        out = rao_spacing_test(np.array([0.0, 90.0, 180.0, 270.0]))
        assert out["n"] == 4
        assert out["U"] == pytest.approx(0.0)
        assert out["significant"] is False
        assert out["p_value"] == "> 0.10"

    def test_2d_input_is_raveled(self) -> None:
        a = np.array([[0.0, 90.0], [180.0, 270.0]])
        out = rao_spacing_test(a)
        assert out["n"] == 4

    def test_wrap_and_sort(self) -> None:
        out = rao_spacing_test(np.array([-10.0, 350.0, 90.0, 170.0]))
        assert out["n"] == 4
        assert "U" in out and "p_value" in out


class TestRayleighTest:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            rayleigh_test(np.array([]))

    def test_uniform_on_circle_low_r(self) -> None:
        angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        out = rayleigh_test(angles)
        assert out["n"] == 12
        assert out["r"] < 0.2
        assert out["p_value"] > 0.05

    def test_tight_cluster_high_r(self) -> None:
        center = 0.3
        angles = np.random.default_rng(0).normal(center, 0.05, size=30)
        out = rayleigh_test(angles)
        assert out["r"] > 0.9
        assert out["p_value"] < 0.01

    def test_n_gt_50_uses_short_p_formula(self) -> None:
        angles = np.linspace(0, 2 * math.pi, 60, endpoint=False)
        out = rayleigh_test(angles)
        assert out["n"] == 60
        assert 0.0 <= out["p_value"] <= 1.0

    def test_2d_input_raveled(self) -> None:
        out = rayleigh_test(np.array([[0.0], [math.pi / 2]]))
        assert out["n"] == 2
