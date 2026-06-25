"""Validate a language config file for the zebra puzzle generator.

Usage:
    uv run .claude/skills/add-language/validate_config.py language=<lang_code>/<theme_name>

Example:
    uv run .claude/skills/add-language/validate_config.py language=fr/maisons
"""

import hydra
from omegaconf import DictConfig

from zebra_puzzles.zebra_utils import validate_language_config


@hydra.main(config_path="../../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Validate a language config file for the zebra puzzle generator."""
    attribute_cases = list(config.language.attribute_cases)
    red_herring_attribute_cases = list(config.language.red_herring_attribute_cases)

    validate_language_config(
        attribute_cases=attribute_cases,
        red_herring_attribute_cases=red_herring_attribute_cases,
        clue_cases_dict=config.language.clue_cases_dict,
        red_herring_cases_dict=config.language.red_herring_cases_dict,
        attributes=config.language.attributes,
        red_herring_attributes=config.language.red_herring_attributes,
    )

    n_attr_categories = len(config.language.attributes)
    n_attr_values = sum(len(v) for v in config.language.attributes.values())
    n_rh = len(config.language.red_herring_attributes)
    n_facts = len(config.language.red_herring_facts)

    print(f"Config for '{config.language.theme}' is valid.")
    print(f"  attribute_cases:            {attribute_cases}")
    print(f"  red_herring_attribute_cases: {red_herring_attribute_cases}")
    print(f"  Attributes: {n_attr_values} values across {n_attr_categories} categories")
    print(f"  Red herring attributes: {n_rh}")
    print(f"  Red herring facts:      {n_facts}")


if __name__ == "__main__":
    main()
