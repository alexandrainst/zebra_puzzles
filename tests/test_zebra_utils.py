"""Tests for the `zebra_utils` module.

Use 'pytest tests/test_zebra_utils.py::test_name' to run a single test.

Use 'make test' to run all tests.
"""

import pytest

from zebra_puzzles.zebra_utils import round_using_std, validate_language_config


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


# --- validate_language_config ---


def _valid_config() -> dict:
    """Return a minimal valid config for validate_language_config."""
    return dict(
        attribute_cases=["nom", "is", "is_not"],
        red_herring_attribute_cases=["nom", "is"],
        clue_cases_dict={"found_at": ["nom"], "next_to": ["nom", "none"]},
        red_herring_cases_dict={"same_herring": ["nom", "none"]},
        attributes={"color": {"red": ["the red one", "is red", "is not red"]}},
        red_herring_attributes={
            "glasses": ["the person with glasses", "wears glasses"]
        },
    )


def test_validate_language_config_valid() -> None:
    """A well-formed config raises no errors."""
    validate_language_config(**_valid_config())


@pytest.mark.parametrize(
    argnames=["attribute_cases", "match"],
    argvalues=[
        # Missing "is_not"
        (["nom", "is"], "is_not"),
        # Missing "is"
        (["nom", "is_not"], "missing required entries"),
        # Missing both
        (["nom"], "missing required entries"),
    ],
)
def test_attribute_cases_missing_required(attribute_cases, match) -> None:
    """attribute_cases must contain both 'is' and 'is_not'."""
    cfg = _valid_config()
    cfg["attribute_cases"] = attribute_cases
    cfg["attributes"] = {
        "color": {"red": [f"desc{i}" for i in range(len(attribute_cases))]}
    }
    with pytest.raises(ValueError, match=match):
        validate_language_config(**cfg)


def test_red_herring_attribute_cases_missing_is() -> None:
    """red_herring_attribute_cases must contain 'is'."""
    cfg = _valid_config()
    cfg["red_herring_attribute_cases"] = ["nom"]
    cfg["red_herring_attributes"] = {
        "glasses": ["the person with glasses", "wears glasses"]
    }
    with pytest.raises(ValueError, match="red_herring_attribute_cases must contain"):
        validate_language_config(**cfg)


def test_clue_cases_dict_unknown_case() -> None:
    """clue_cases_dict may not reference cases absent from attribute_cases."""
    cfg = _valid_config()
    cfg["clue_cases_dict"] = {"found_at": ["unknown_case"]}
    with pytest.raises(ValueError, match="unknown_case"):
        validate_language_config(**cfg)


def test_red_herring_cases_dict_unknown_case() -> None:
    """red_herring_cases_dict may not reference cases absent from red_herring_attribute_cases."""
    cfg = _valid_config()
    cfg["red_herring_cases_dict"] = {"same_herring": ["unknown_rh_case"]}
    with pytest.raises(ValueError, match="unknown_rh_case"):
        validate_language_config(**cfg)


@pytest.mark.parametrize(
    argnames=["descs", "match"],
    argvalues=[
        # Too few descriptions
        (["the red one", "is red"], "has 2 description"),
        # Too many descriptions
        (["the red one", "is red", "is not red", "extra"], "has 4 description"),
    ],
)
def test_attributes_wrong_description_count(descs, match) -> None:
    """Each attribute value must have exactly len(attribute_cases) descriptions."""
    cfg = _valid_config()
    cfg["attributes"] = {"color": {"red": descs}}
    with pytest.raises(ValueError, match=match):
        validate_language_config(**cfg)


@pytest.mark.parametrize(
    argnames=["descs", "match"],
    argvalues=[
        # Too few descriptions
        (["the glasses person"], "has 1 description"),
        # Too many descriptions
        (["the glasses person", "wears glasses", "extra"], "has 3 description"),
    ],
)
def test_red_herring_attributes_wrong_description_count(descs, match) -> None:
    """Each red herring attribute must have exactly len(red_herring_attribute_cases) descriptions."""
    cfg = _valid_config()
    cfg["red_herring_attributes"] = {"glasses": descs}
    with pytest.raises(ValueError, match=match):
        validate_language_config(**cfg)
