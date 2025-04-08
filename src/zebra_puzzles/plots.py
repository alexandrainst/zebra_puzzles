"""Module for plotting the results of LLM's trying to solve zebra puzzles."""

import re
from pathlib import Path

import numpy as np


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
    n_objects_list, n_attributes_list = get_puzzle_dimensions(
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


def get_score_file_paths(
    data_folder: str,
    model: str,
    n_red_herring_clues_evaluated: int,
    theme: str,
    n_puzzles: int,
) -> list[Path]:
    """Get the paths of the score files.

    Args:
        data_folder: Path to the data folder.
        model: LLM model name.
        n_red_herring_clues_evaluated: Number of red herring clues evaluated.
        theme: Theme name.
        n_puzzles: Number of puzzles evaluated.

    Returns:
        List of score file paths.
    """
    scores_path = f"{data_folder}/scores/{model}/{n_red_herring_clues_evaluated}rh/"

    # Get sorted names of all score files in the data folder that are evaluated on the correct number of puzzles
    score_file_paths = sorted(
        list(
            Path(scores_path).glob(
                f"puzzle_scores_{model}_{theme}*_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"
            )
        )
    )

    return score_file_paths


def get_puzzle_dimensions(score_file_paths: list[Path]) -> tuple[list[int], list[int]]:
    """Get the dimensions of the puzzles from the score file paths.

    Args:
        score_file_paths: List of score file paths.

    Returns:
        A tuple (n_objects_list, n_attributes_list) where:
            n_objects_list: List of the number of objects in the puzzles evaluated in each score file.
            n_attributes_list: List of the number of attributes in the puzzles evaluated in each score file.
    """
    # Get the dimensions of the puzzles
    n_objects_list = []
    n_attributes_list = []

    for score_file_path in score_file_paths:
        file_name = score_file_path.name

        # Get the numbers on each side of the "x" in the filename
        match = re.search(r"_(\d+)x(\d+)_", file_name)
        if match:
            n_objects = int(match.group(1))
            n_attributes = int(match.group(2))
        else:
            raise ValueError(f"Could not find dimensions in {file_name}")

        n_objects_list.append(n_objects)
        n_attributes_list.append(n_attributes)

    return n_objects_list, n_attributes_list


def load_scores(
    score_file_paths: list[Path],
    n_objects_list: list[int],
    n_attributes_list: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the scores from the score files.

    Args:
        score_file_paths: List of score file paths.
        n_objects_list: List of the number of objects in the puzzles evaluated in each score file.
        n_attributes_list: List of the number of attributes in the puzzles evaluated in each score file.

    Returns:
        A tuple (mean_scores_array, std_mean_scores_array) where:
            mean_scores_array: Array of mean scores for each score file.
            std_mean_scores_array: Array of standard deviations of the mean scores for each score file.
    """
    # Get the maximum number of objects and attributes
    n_objects_max = max(n_objects_list)
    n_attributes_max = max(n_attributes_list)

    # Prepare array of scores
    score_types = ["puzzle score", "cell score", "best permuted cell score"]
    mean_scores_array = (
        np.ones((len(score_types), n_objects_max, n_attributes_max)) * -1
    )
    std_mean_scores_array = (
        np.ones((len(score_types), n_objects_max, n_attributes_max)) * -1
    )

    for score_file_path, n_objects_in_file, n_attributes_in_file in zip(
        score_file_paths, n_objects_list, n_attributes_list
    ):
        # Load the score file
        with open(score_file_path, "r") as f:
            scores_str = f.read()

        for i_score_type, score_type in enumerate(score_types):
            # Get the number after "puzzle score:"
            score_str = scores_str.split(f"{score_type.capitalize()}:\n\tMean: ")[1]
            mean_str = score_str.split(" ")[0]
            mean_std_str = score_str.split("± ")[1].split(" ")[0]

            mean = float(mean_str)
            std_mean = float(mean_std_str)

            # Add the score to the array
            mean_scores_array[
                i_score_type, n_objects_in_file - 1, n_attributes_in_file - 1
            ] = mean
            std_mean_scores_array[
                i_score_type, n_objects_in_file - 1, n_attributes_in_file - 1
            ] = std_mean

    return mean_scores_array, std_mean_scores_array
