"""Tests for the `zebra_utils` module.

Use 'pytest tests/test_zebra_utils.py::test_name' to run a single test.

Use 'make test' to run all tests.
"""

import pytest

from zebra_puzzles.zebra_utils import round_using_std


@pytest.mark.parametrize(
    argnames=["value", "std", "value_rounded", "std_rounded"],
    argvalues=[
        # Test a value that is not rounded
        (0.4, 0.3, "0.4", "0.3"),
        # Test a value and standard deviation that is rounded
        (0.464, 0.31, "0.5", "0.3"),
        # Test rounding of a negative number
        (-0.464, 0.31, "-0.5", "0.3"),
        # Test rounding that requires trailing zeros
        (-0.460, 0.001, "-0.460", "0.001"),
        # Test default rounding when the standard deviation is unknown (0)
        (0.464, 0, "0.46", "0"),
        # Test default rounding when the standard deviation is unknown and the value has a trailing zero
        (0.40, 0, "0.40", "0"),
        # Test rounding of a number much smaller than the standard deviation
        (0.0002, 0.3, "0.0", "0.3"),
        # Test rounding of a number where a trailing zero should be re-added
        (0.4, 0.05, "0.40", "0.05"),
    ],
)
def test_rounding(value, std, value_rounded, std_rounded) -> None:
    """Test the rounding function."""
    # Test a value that is not rounded
    value_rounded, std_rounded = round_using_std(value=value, std=std)
    assert value_rounded == value_rounded
    assert std_rounded == std_rounded
