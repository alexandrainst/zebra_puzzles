"""Script to analyse and plot performance of the LMM's.

This script should run after build_dataset.py and evaluate.py.

Usage:
    uv run src/scripts/plot_performance.py <config_key>=<config_value> ...
"""

import hydra
from omegaconf import DictConfig

from zebra_puzzles.plots import plot_results


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main script.

    Evaluates a dataset of zebra puzzles.

    Args:
        config: Config file.
    """
    n_puzzles = config.n_puzzles
    model = config.model
    theme = config.language.theme
    n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated
    data_folder = config.data_folder

    plot_results(
        n_puzzles=n_puzzles,
        model=model,
        theme=theme,
        n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
        data_folder_str=data_folder,
    )


if __name__ == "__main__":
    main()
