"""Module for the pipeline of plotting the results of LLM's trying to solve zebra puzzles."""

from pathlib import Path

import numpy as np

from zebra_puzzles.eval_comparisons import compare_all_eval_types
from zebra_puzzles.file_utils import (
    get_clue_type_file_paths,
    get_clue_type_frequencies,
    get_evaluated_params,
    get_puzzle_dimensions_from_filename,
    get_score_file_paths,
    load_scores,
)
from zebra_puzzles.plots import plot_heatmaps


def plot_results(n_puzzles: int, theme: str, data_folder_str: str) -> None:
    """Plot the results of the LLM's trying to solve zebra puzzles.

    Args:
        n_puzzles: Number of puzzles evaluated.
        theme: Theme name.
        data_folder_str: Path to the data folder as a string.

    TODO: More plots e.g. clue type histograms, clue type difficulty etc.
    TODO: Analyze when o3-mini fails to solve the puzzle. There seems to be a shift in puzzle numbers in files vs. in the score file.
    """
    # Convert the data folder string to a Path object
    data_folder = Path(data_folder_str)

    # ----- Import results from score files -----#
    model_names, n_red_herring_values = get_evaluated_params(data_folder=data_folder)

    # Define the score types to search for in the score files
    score_types = ["puzzle score", "cell score", "best permuted cell score"]

    (
        mean_scores_all_eval_array,
        std_mean_scores_all_eval_array,
        n_objects_max_all_eval,
        n_attributes_max_all_eval,
    ) = load_scores_and_plot_results_for_each_evaluation(
        data_folder=data_folder,
        theme=theme,
        n_puzzles=n_puzzles,
        model_names=model_names,
        n_red_herring_values=n_red_herring_values,
        score_types=score_types,
    )

    # ----- Compare the mean scores of different evaluations -----#
    compare_all_eval_types(
        model_names=model_names,
        mean_scores_all_eval_array=mean_scores_all_eval_array,
        std_mean_scores_all_eval_array=std_mean_scores_all_eval_array,
        n_red_herring_values=n_red_herring_values,
        n_objects_max_all_eval=n_objects_max_all_eval,
        n_attributes_max_all_eval=n_attributes_max_all_eval,
        data_folder=data_folder,
        score_types=score_types,
        n_puzzles=n_puzzles,
    )


def load_scores_and_plot_results_for_each_evaluation(
    data_folder: Path,
    theme: str,
    n_puzzles: int,
    model_names: list[str],
    n_red_herring_values: list[int],
    score_types: list[str],
) -> tuple[
    list[list[np.ndarray]], list[list[np.ndarray]], list[list[int]], list[list[int]]
]:
    """Load the scores from the score files and plot the results for each evaluation.

    Args:
        data_folder: Path to the data folder.
        theme: Theme name.
        n_puzzles: Number of puzzles evaluated.
        model_names: List of model names.
        n_red_herring_values: Number of red herring clues evaluated.
        score_types: List of score types as strings.

    Returns:
        A tuple (mean_scores_all_eval_array, std_mean_scores_all_eval_array, n_objects_max_all_eval, n_attributes_max_all_eval) where:
            mean_scores_all_eval_array: List of mean scores arrays. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
            std_mean_scores_all_eval_array: List of standard deviation arrays. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
            n_objects_max_all_eval: List of lists of the maximum number of objects in puzzles for each evaluation. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
            n_attributes_max_all_eval: List of lists of the maximum number of attributes in puzzles for each evaluation. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
    """
    mean_scores_all_eval_array = []
    std_mean_scores_all_eval_array = []
    n_objects_max_all_eval = []
    n_attributes_max_all_eval = []

    n_red_herring_clues_evaluated_max = max(n_red_herring_values)

    for n_red_herring_clues_evaluated in n_red_herring_values:
        mean_scores_all_models_array = []
        std_mean_scores_all_models_array = []
        n_objects_max_all_models = []
        n_attributes_max_all_models = []

        for model in model_names:
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
            mean_scores_array, std_mean_scores_array, std_scores_array = load_scores(
                score_file_paths=score_file_paths,
                n_objects_list=n_objects_list,
                n_attributes_list=n_attributes_list,
                score_types=score_types,
                n_puzzles=n_puzzles,
            )

            # Get the paths of the clue type files
            reduced_flag = (
                n_red_herring_clues_evaluated < n_red_herring_clues_evaluated_max
            )

            clue_type_file_paths = get_clue_type_file_paths(
                data_folder=data_folder,
                n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
                theme=theme,
                n_puzzles=n_puzzles,
                reduced_flag=reduced_flag,
            )

            # Load the clue type frequencies
            clue_type_frequencies = get_clue_type_frequencies(
                clue_type_file_paths=clue_type_file_paths
            )

            # ----- Plot the results -----#

            # Prepare path for plots
            plot_path = Path(
                f"{data_folder}/plots/{model}/{n_red_herring_clues_evaluated}rh/"
            )

            # Make heatmaps of mean scores
            plot_heatmaps(
                scores_array=mean_scores_array,
                score_types=score_types,
                plot_path=plot_path,
                n_red_herring_clues_evaluated_str=str(n_red_herring_clues_evaluated),
                std_scores_array=std_scores_array,
                single_model=True,
                model=model,
                n_puzzles=n_puzzles,
            )

            # Plot the clue type frequencies
            if clue_type_frequencies is not None:
                pass

            # TODO: Perhaps optional: Make a grid of histograms of clue type frequencies
            # TODO: Compute the difficulty of each clue type
            #                For example as the mean score of puzzles weighted by the number of times the clue type was used.
            #                Or normalise the scores to compare the difficulty of clues for the puzzle size. Then, the normalised scores can be compared across different puzzle sizes.
            # Save values across all models
            mean_scores_all_models_array.append(mean_scores_array)
            std_mean_scores_all_models_array.append(std_mean_scores_array)
            n_objects_max_all_models.append(max(n_objects_list))
            n_attributes_max_all_models.append(max(n_attributes_list))

        # Save values across all values of n_red_herring_clues_evaluated
        mean_scores_all_eval_array.append(mean_scores_all_models_array)
        std_mean_scores_all_eval_array.append(std_mean_scores_all_models_array)
        n_objects_max_all_eval.append(n_objects_max_all_models)
        n_attributes_max_all_eval.append(n_attributes_max_all_models)

    return (
        mean_scores_all_eval_array,
        std_mean_scores_all_eval_array,
        n_objects_max_all_eval,
        n_attributes_max_all_eval,
    )
