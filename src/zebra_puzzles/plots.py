"""Module for plotting the results of LLM's trying to solve zebra puzzles."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zebra_puzzles.file_utils import (
    get_puzzle_dimensions_from_filename,
    get_score_file_paths,
    load_scores,
)


def plot_results(
    n_puzzles: int,
    model: str,
    theme: str,
    n_red_herring_clues_evaluated: int,
    data_folder: str,
) -> None:
    """Plot the results of the LLM's trying to solve zebra puzzles.

    Args:
        n_puzzles: Number of puzzles evaluated.
        n_objects: Number of objects.
        n_attributes: Number of attributes.
        model: LLM model name.
        theme: Theme name.
        n_red_herring_clues_evaluated: Number of red herring clues evaluated.
        data_folder: Path to the data folder.

    TODO: Consider just plotting everything in a folder instead of specifying n_puzzles, model etc.
    """
    # ----- Import results from score files -----#

    # Get the paths of the score files
    score_file_paths = get_score_file_paths(
        data_folder=data_folder,
        model=model,
        n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
        theme=theme,
        n_puzzles=n_puzzles,
    )

    # Check the puzzle dimensions in score filenames
    n_objects_list, n_attributes_list = get_puzzle_dimensions_from_filename(
        score_file_paths=score_file_paths
    )

    # Define the score types to search for in the score files
    score_types = ["puzzle score", "cell score", "best permuted cell score"]

    # Load the scores from the score files
    mean_scores_array, std_mean_scores_array = load_scores(
        score_file_paths=score_file_paths,
        n_objects_list=n_objects_list,
        n_attributes_list=n_attributes_list,
        score_types=score_types,
    )

    # ----- Plot the results -----#

    # Prepare path for plots
    plot_path = Path(f"{data_folder}/plots/{model}/{n_red_herring_clues_evaluated}rh/")

    # Make heatmaps of mean scores
    plot_heatmaps(
        scores_array=mean_scores_array, score_types=score_types, plot_path=plot_path
    )

    # TODO: More plots e.g. score vs. n_red_herrings_evaluated


def plot_heatmaps(
    scores_array: np.ndarray, score_types: list[str], plot_path: Path
) -> None:
    """Plot heatmaps of the mean scores.

    Args:
        scores_array: Array of mean scores.
        score_types: List of score types as strings.
        plot_path: Path to save the plots.

    NOTE: Consider using subplots instead of saving separate figures for each score type.
    TODO: Add performance labels to cells
    """
    for score_type, score_type_array in zip(score_types, scores_array):
        # Set the figure size
        plt.figure(figsize=(10, 8))

        # Fill untested cells with grey
        empty_cells = np.zeros_like(score_type_array)
        empty_cells[score_type_array == -1] = 1
        empty_cells[score_type_array != -1] = 0
        plt.imshow(empty_cells, cmap="Greys", alpha=0.5)

        # Plot the mean scores
        score_type_array_not_empty = np.ma.masked_where(
            score_type_array == -1, score_type_array
        )
        plt.imshow(
            score_type_array_not_empty,
            cmap="viridis",
            aspect="equal",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        plt.colorbar()

        # Set the title and labels
        plt.title(f"Mean {score_type}")
        plt.xlabel("#Attributes")
        plt.ylabel("#Objects")

        # Set the ticks and tick labels
        n_objects_max = score_type_array.shape[0]
        n_attributes_max = score_type_array.shape[1]
        plt.xticks(
            ticks=np.arange(n_attributes_max),
            labels=[str(i + 1) for i in range(n_attributes_max)],
        )

        plt.yticks(
            ticks=np.arange(n_objects_max),
            labels=[str(i + 2) for i in range(n_objects_max)],
        )

        # Adjust the layout
        plt.tight_layout()
        # plt.show()

        # Save the plot
        plot_path.mkdir(parents=True, exist_ok=True)
        score_type = score_type.replace(" ", "_")
        plot_filename = f"mean_{score_type}.png"
        plt.savefig(plot_path / plot_filename, dpi=300, bbox_inches="tight")
        plt.close()
