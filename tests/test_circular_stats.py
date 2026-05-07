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

# En dash (U+2013), same as ``p_value`` strings in ``rao_spacing_test``.
_EN = "\u2013"


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
            (37, 28),
            (38, 29),
            (42, 29),
            (43, 30),
            (47, 30),
            (48, 31),
            (62, 31),
            (63, 32),
            (87, 32),
            (88, 33),
            (100, 33),
            (125, 33),
            (126, 34),
            (175, 34),
            (176, 35),
            (250, 35),
            (251, 36),
            (350, 36),
            (351, 37),
            (450, 37),
            (451, 38),
            (550, 38),
            (551, 39),
            (650, 39),
            (651, 40),
            (750, 40),
            (751, 41),
            (850, 41),
            (851, 42),
            (950, 42),
            (951, 43),
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

    def test_identical_angles_flags_rejection(self) -> None:
        out = rao_spacing_test(np.zeros(8))
        assert out["U"] > 300.0
        assert out["significant"] is True
        assert out["p_value"] == "< 0.001"

    def test_2d_input_is_raveled(self) -> None:
        a = np.array([[0.0, 90.0], [180.0, 270.0]])
        out = rao_spacing_test(a)
        assert out["n"] == 4

    def test_wrap_and_sort(self) -> None:
        out = rao_spacing_test(np.array([-10.0, 350.0, 90.0, 170.0]))
        assert out["n"] == 4
        assert "U" in out and "p_value" in out

    @pytest.mark.parametrize(
        ("angles", "expected_p"),
        [
            (
                np.array([80.05400156, 83.93890403, 85.61343119, 70.89509776]),
                f"0.001{_EN}0.01",
            ),
            (
                np.array([335.94132899, 308.50075145, 299.17665529, 312.64366824]),
                f"0.01{_EN}0.05",
            ),
            (
                np.array([212.29210324, 241.03198121, 240.88589861, 188.29791919]),
                f"0.05{_EN}0.10",
            ),
            (
                np.array(
                    [
                        72.04410608,
                        102.97287205,
                        69.37785702,
                        66.13046044,
                        84.41640955,
                        67.14665021,
                        57.45546007,
                        95.5375559,
                    ]
                ),
                f"0.001{_EN}0.01",
            ),
            (
                np.array(
                    [
                        322.56824591,
                        117.79625267,
                        342.09789341,
                        322.48479507,
                        350.66220465,
                        346.05477035,
                        114.08056131,
                        337.76591354,
                    ]
                ),
                f"0.05{_EN}0.10",
            ),
        ],
    )
    def test_p_value_brackets(self, angles: np.ndarray, expected_p: str) -> None:
        out = rao_spacing_test(angles)
        assert out["p_value"] == expected_p


class TestRayleighTest:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            rayleigh_test(np.array([]))

    def test_all_angles_identical_produces_expected_pvalue(self) -> None:
        theta = np.zeros(5)

        result = rayleigh_test(theta)

        assert result["n"] == 5
        assert result["r"] == pytest.approx(1.0, abs=1e-9)
        assert result["Z"] == pytest.approx(5.0, abs=1e-9)
        expected_p = math.exp(-5.0) * (
            1 + (2 * 5.0 - 5.0**2) / (4 * 5) - (24 * 5.0 - 132 * 5.0**2 + 76 * 5.0**3 - 9 * 5.0**4) / (288 * 5**2)
        )
        assert result["p_value"] == pytest.approx(expected_p, rel=1e-9)

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
