"""Evaluation script.

This script should run after build_dataset.py.

Usage:
    uv run src/scripts/evaluate.py <config_key>=<config_value> ...
"""

import hydra
from omegaconf import DictConfig

from zebra_puzzles.eval_pipeline import evaluate_all


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
    n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated
    data_folder = config.data_folder

    evaluate_all(
        n_puzzles=n_puzzles,
        n_objects=n_objects,
        n_attributes=n_attributes,
        model=model,
        theme=theme,
        generate_new_responses=generate_new_responses,
        n_red_herring_clues=n_red_herring_clues,
        n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
        data_folder_str=data_folder,
    )


if __name__ == "__main__":
    main()
