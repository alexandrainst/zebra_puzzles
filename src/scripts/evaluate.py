"""Evaluation script.

Usage:
    uv run src/scripts/evaluate.py <config_key>=<config_value> ...
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from zebra_puzzles.evaluation import evaluate_all


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main script.

    Evaluates a dataset of zebra puzzles.

    Args:
        config: Config file.
    """
    n_puzzles = config.n_puzzles
    n_objects = config.n_objects
    n_attributes = config.n_attributes
    model = config.model
    theme = config.language.theme
    generate_new_responses = config.generate_new_responses
    n_red_herring_clues = config.n_red_herring_clues

    # Get sorted names of all prompt files in the data folder
    file_paths = sorted(list(Path("data").glob("*[!_solution].txt")))

    evaluate_all(
        n_puzzles=n_puzzles,
        n_objects=n_objects,
        n_attributes=n_attributes,
        file_paths=file_paths,
        model=model,
        theme=theme,
        generate_new_responses=generate_new_responses,
        n_red_herring_clues=n_red_herring_clues,
    )


if __name__ == "__main__":
    main()
