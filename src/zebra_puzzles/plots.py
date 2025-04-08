"""Module for plotting the results of LLM's trying to solve zebra puzzles."""

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

    # Load the scores from the score files
    mean_scores_array, std_mean_scores_array = load_scores(
        score_file_paths=score_file_paths,
        n_objects_list=n_objects_list,
        n_attributes_list=n_attributes_list,
    )

    # Plot heatmap of the results

    # Prepare path for plots
