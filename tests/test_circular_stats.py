import math

import numpy as np
import pytest

from core.circular_stats import rao_spacing_test, rayleigh_test


def test_rao_spacing_uniform_angles_not_significant():
    angles = np.array([0.0, 90.0, 180.0, 270.0])

    result = rao_spacing_test(angles)

    assert result["n"] == 4
    assert result["significant"] is False
    assert result["p_value"] == "> 0.10"
    assert result["U"] == pytest.approx(0.0, abs=1e-9)


def test_rao_spacing_identical_angles_flags_rejection():
    angles = np.zeros(8)

    result = rao_spacing_test(angles)

    assert result["U"] > 300.0
    assert result["significant"] is True
    assert result["p_value"] == "< 0.001"


def test_rao_spacing_requires_minimum_sample_size():
    with pytest.raises(ValueError, match="requires at least 4 angles"):
        rao_spacing_test(np.array([0.0, 90.0, 180.0]))


def test_rayleigh_test_all_angles_identical_produces_expected_pvalue():
    theta = np.zeros(5)

    result = rayleigh_test(theta)

    assert result["n"] == 5
    assert result["r"] == pytest.approx(1.0, abs=1e-9)
    assert result["Z"] == pytest.approx(5.0, abs=1e-9)
    expected_p = math.exp(-5.0) * (
        1 + (2 * 5.0 - 5.0**2) / (4 * 5) - (24 * 5.0 - 132 * 5.0**2 + 76 * 5.0**3 - 9 * 5.0**4) / (288 * 5**2)
    )
    assert result["p_value"] == pytest.approx(expected_p, rel=1e-9)


def test_rayleigh_test_requires_non_empty_input():
    with pytest.raises(ValueError, match="requires at least one angle"):
        rayleigh_test(np.array([]))
