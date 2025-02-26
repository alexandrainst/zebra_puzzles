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
        config:
            Config file.
    """
    N_puzzles = config.N_puzzles
    theme = config.theme
    language = config.language
    rules_included = config.rules_included
    N_objects = config.N_objects
    N_attributes = config.N_attributes

    build_dataset(
        theme=theme, 
        language=language, 
        rules_included=rules_included, 
        n_objects=n_objects, 
        n_attributes=n_attributes, 
        n_puzzles=n_puzzles,
    )


if __name__ == "__main__":
    main()
