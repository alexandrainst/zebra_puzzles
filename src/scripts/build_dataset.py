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
    attributes = config.attributes
    prompt_template = config.prompt_template
    n_objects = config.n_objects
    n_attributes = config.n_attributes
    clues_dict = config.clues_dict

    build_dataset(
        attributes=attributes,
        clues_dict=clues_dict,
        prompt_template=prompt_template,
        n_objects=n_objects,
        n_attributes=n_attributes,
        n_puzzles=n_puzzles,
    )


if __name__ == "__main__":
    main()
