"""Main script.

Usage:
    uv run src/scripts/build_dataset.py <config_key>=<config_value> ...
"""

import hydra
from omegaconf import DictConfig

from zebra_puzzles.pipeline import build_dataset


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main script.

    Generates a dataset of zebra puzzles.

    Args:
        config: Config file.
    """
    n_puzzles = config.n_puzzles
    n_objects = config.n_objects
    n_attributes = config.n_attributes
    attributes = config.language.attributes
    prompt_templates = config.language.prompt_templates
    prompt_and = config.language.prompt_and
    clues_dict = config.language.clues_dict
    theme = config.language.theme

    n_red_herring_clues = config.n_red_herring_clues
    red_herring_clues_dict = config.language.red_herring_clues_dict
    red_herring_attributes = config.language.red_herring_attributes
    red_herring_facts = config.language.red_herring_facts

    build_dataset(
        attributes=attributes,
        clues_dict=clues_dict,
        prompt_templates=prompt_templates,
        prompt_and=prompt_and,
        n_objects=n_objects,
        n_attributes=n_attributes,
        n_puzzles=n_puzzles,
        theme=theme,
        n_red_herring_clues=n_red_herring_clues,
        red_herring_clues_dict=red_herring_clues_dict,
        red_herring_attributes=red_herring_attributes,
        red_herring_facts=red_herring_facts,
    )


if __name__ == "__main__":
    main()
