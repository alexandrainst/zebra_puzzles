"""Script to analyse and plot performance of the LMM's.

This script should run after build_dataset.py and evaluate.py.

Usage:
    uv run src/scripts/plot_performance.py <config_key>=<config_value> ...
"""

import hydra
from omegaconf import DictConfig

from zebra_puzzles.performance.plot_pipeline import plot_results


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main script.

    Plots and compares LLM performance on a dataset of zebra puzzles.

    Args:
        config: Config file.
    """
    n_puzzles = config.n_puzzles
    theme = config.language.theme
    data_folder = config.data_folder
    clue_types = list(config.clue_weights.keys())
    red_herring_clue_types = list(config.red_herring_clue_weights.keys())
    n_red_herring_clues_generated = config.n_red_herring_clues

    plot_results(
        n_puzzles=n_puzzles,
        theme=theme,
        data_folder_str=data_folder,
        clue_types=clue_types,
        red_herring_clue_types=red_herring_clue_types,
        n_red_herring_clues_generated=n_red_herring_clues_generated,
    )


if __name__ == "__main__":
    main()
