"""Module for the pipeline of plotting the results of LLM's trying to solve zebra puzzles."""

import logging
from pathlib import Path

import numpy as np

from zebra_puzzles.file_utils import (
    get_evaluated_params,
    get_puzzle_dimensions_from_filename,
    get_score_file_paths,
)
from zebra_puzzles.load_data import load_scores
from zebra_puzzles.performance.clue_analysis import (
    estimate_clue_type_difficulty_for_all_puzzle_sizes,
)
from zebra_puzzles.performance.eval_comparisons import compare_all_eval_types
from zebra_puzzles.performance.plots import (
    plot_clue_type_difficulty,
    plot_clue_type_frequencies,
    plot_heatmaps,
)

log = logging.getLogger(__name__)


def plot_results(
    n_puzzles: int,
    theme: str,
    data_folder_str: str,
    clue_types: list[str],
    red_herring_clue_types: list[str],
    n_red_herring_clues_generated: int,
) -> None:
    """Plot the results of the LLM's trying to solve zebra puzzles.

    Generate plots for each LLM evaluation, and compare the mean scores of different evaluations when possible.

    Args:
        n_puzzles: Number of puzzles evaluated.
        theme: Theme name.
        data_folder_str: Path to the data folder as a string.
        clue_types: List of possible non red herring clue types.
        red_herring_clue_types: List of possible red herring clue types.
        n_red_herring_clues_generated: Number of red herring clues generated in the original puzzles.

    NOTE: More plots can be added e.g. score vs. n_clues etc.
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
        clue_types=clue_types,
        red_herring_clue_types=red_herring_clue_types,
        n_red_herring_clues_generated=n_red_herring_clues_generated,
    )

    # ----- Compare the mean scores of different evaluations -----#
    if len(model_names) > 1 or len(n_red_herring_values) > 1:
        compare_all_eval_types(
            model_names=model_names,
            mean_scores_all_eval_array=mean_scores_all_eval_array,
            std_mean_scores_all_eval_array=std_mean_scores_all_eval_array,
            n_red_herring_values=n_red_herring_values,
            n_objects_max_all_eval=n_objects_max_all_eval,
            n_attributes_max_all_eval=n_attributes_max_all_eval,
            data_folder=data_folder,
            theme=theme,
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
    clue_types: list[str],
    red_herring_clue_types: list[str],
    n_red_herring_clues_generated: int,
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
        clue_types: List of possible non red herring clue types.
        red_herring_clue_types: List of possible red herring clue types.
        n_red_herring_clues_generated: Number of red herring clues generated in the original puzzles.

    Returns:
        A tuple (mean_scores_all_eval_array, std_mean_scores_all_eval_array, n_objects_max_all_eval, n_attributes_max_all_eval) where:
            mean_scores_all_eval_array: List of mean scores arrays. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
            std_mean_scores_all_eval_array: List of standard deviation arrays. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
            n_objects_max_all_eval: List of lists of the maximum number of objects in puzzles for each evaluation. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
            n_attributes_max_all_eval: List of lists of the maximum number of attributes in puzzles for each evaluation. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.

    NOTE: Consider splitting this into two functions: one for loading the scores and one for plotting the results.
    """
    mean_scores_all_eval_array = []
    std_mean_scores_all_eval_array = []
    n_objects_max_all_eval = []
    n_attributes_max_all_eval = []
    clue_type_file_paths_all_eval = []

    n_red_herring_clues_evaluated_max = max(n_red_herring_values)
    all_possible_clue_types = clue_types + red_herring_clue_types

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

            # ----- Plot the results -----#

            # Prepare path for plots
            plot_path = (
                data_folder
                / "plots"
                / theme
                / model
                / f"{n_red_herring_clues_evaluated}rh"
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

            # Save values across all models
            mean_scores_all_models_array.append(mean_scores_array)
            std_mean_scores_all_models_array.append(std_mean_scores_array)
            n_objects_max_all_models.append(max(n_objects_list))
            n_attributes_max_all_models.append(max(n_attributes_list))

        # ---- Plot the distribution of clue types -----#
        # Do it after all models have been evaluated to know the maximum number of objects and attributes

        (
            clue_type_file_paths_all_sizes,
            clue_type_frequencies_all_sizes,
            all_clue_types,
        ) = plot_clue_type_frequencies(
            data_folder=data_folder,
            n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
            n_red_herring_clues_generated=n_red_herring_clues_generated,
            theme=theme,
            n_puzzles=n_puzzles,
            n_objects_max_all_models=n_objects_max_all_models,
            n_attributes_max_all_models=n_attributes_max_all_models,
            clue_types=clue_types,
            all_possible_clue_types=all_possible_clue_types,
        )

        # ---- Plot the clue type difficulties for each model but only for max n_red_herring_clues_evaluated -----#

        if (
            n_red_herring_clues_evaluated == n_red_herring_clues_evaluated_max
            and n_puzzles > 1
        ):
            for i, model in enumerate(model_names):
                # Estimate the difficulty of each clue type for a specific model
                clue_type_difficulties_all_sizes = (
                    estimate_clue_type_difficulty_for_all_puzzle_sizes(
                        clue_type_frequencies_all_sizes=clue_type_frequencies_all_sizes,
                        all_possible_clue_types=all_possible_clue_types,
                        n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
                        model=model,
                        data_folder=data_folder,
                        theme=theme,
                        n_puzzles=n_puzzles,
                    )
                )
                if len(clue_type_difficulties_all_sizes) > 0:
                    # Make a grid of bar plots of clue type difficulty
                    plot_clue_type_difficulty(
                        clue_type_difficulties_all_sizes=clue_type_difficulties_all_sizes,
                        n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
                        data_folder=data_folder,
                        theme=theme,
                        n_puzzles=n_puzzles,
                        clue_types=clue_types,
                        all_clue_types=all_clue_types,
                        model=model,
                    )
                else:
                    log.warning(
                        f"No clue type difficulties found for model {model} with {n_red_herring_clues_evaluated} red herring clues evaluated and theme {theme}."
                    )

        # Save values across all values of n_red_herring_clues_evaluated
        mean_scores_all_eval_array.append(mean_scores_all_models_array)
        std_mean_scores_all_eval_array.append(std_mean_scores_all_models_array)
        n_objects_max_all_eval.append(n_objects_max_all_models)
        n_attributes_max_all_eval.append(n_attributes_max_all_models)
        clue_type_file_paths_all_eval.append(clue_type_file_paths_all_sizes)

    return (
        mean_scores_all_eval_array,
        std_mean_scores_all_eval_array,
        n_objects_max_all_eval,
        n_attributes_max_all_eval,
    )
