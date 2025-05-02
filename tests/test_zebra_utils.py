"""Tests for the `zebra_utils` module.

Use 'pytest tests/test_zebra_utils.py::test_name' to run a single test.

Use 'make test' to run all tests.
"""

from zebra_puzzles.zebra_utils import round_using_std


def test_rounding() -> None:
    """Test the rounding function."""
    # Test a value that is not rounded
    value_rounded, std_rounded = round_using_std(value=0.4, std=0.3)
    assert value_rounded == "0.4"
    assert std_rounded == "0.3"

    # Test a value and standard deviation that is rounded
    value_rounded, std_rounded = round_using_std(value=0.464, std=0.31)
    assert value_rounded == "0.5"
    assert std_rounded == "0.3"

    # Test rounding of a negative number
    value_rounded, std_rounded = round_using_std(value=-0.464, std=0.31)
    assert value_rounded == "-0.5"
    assert std_rounded == "0.3"

    # Test rounding that requires trailing zeros
    value_rounded, std_rounded = round_using_std(value=-0.460, std=0.001)
    assert value_rounded == "-0.460"
    assert std_rounded == "0.001"

    # Test default rounding when the standard deviation is unknown (0)
    value_rounded, std_rounded = round_using_std(value=0.464, std=0)
    assert value_rounded == "0.46"
    assert std_rounded == "0"

    # Test default rounding when the standard deviation is unknown and the value has a trailing zero
    value_rounded, std_rounded = round_using_std(value=0.40, std=0)
    assert value_rounded == "0.40"
    assert std_rounded == "0"

    # Test rounding of a number much smaller than the standard deviation
    value_rounded, std_rounded = round_using_std(value=0.0002, std=0.3)
    assert value_rounded == "0.0"
    assert std_rounded == "0.3"
