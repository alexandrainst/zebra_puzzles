"""Module for plotting the results of LLM's trying to solve zebra puzzles."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zebra_puzzles.eval_comparisons import compare_all_eval_types
from zebra_puzzles.file_utils import (
    get_evaluated_params,
    get_puzzle_dimensions_from_filename,
    get_score_file_paths,
    load_scores,
)
from zebra_puzzles.zebra_utils import round_using_std


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


def plot_heatmaps(
    scores_array: np.ndarray,
    score_types: list[str],
    plot_path: Path,
    n_red_herring_clues_evaluated_str: str,
    std_scores_array: np.ndarray,
    single_model: bool,
    model: str,
    n_puzzles: int,
) -> None:
    """Plot heatmaps of the mean scores.

    Args:
        scores_array: Array of mean scores.
        score_types: List of score types as strings.
        plot_path: Path to save the plots.
        n_red_herring_clues_evaluated_str: Number of red herring clues evaluated as a string.
        std_scores_array: Array of sample standard deviations of scores.
        single_model: Boolean indicating if the scores are from a single model.
        model: Name of the model or models as a string.
        n_puzzles: Number of puzzles evaluated.
        compare_mode: Mode of comparison, either "models" or "red_herrings".

    NOTE: Consider using subplots instead of saving separate figures for each score type.
    NOTE: Consider using i_not_evaluated_by_both instead of looking for -999 in the scores.
    """
    for score_type, score_type_array, std_score_type_array in zip(
        score_types, scores_array, std_scores_array
    ):
        # Set the figure size
        fig, ax = plt.subplots(figsize=(10, 8))

        # Fill untested cells with grey
        empty_cells = np.ones_like(score_type_array)
        empty_cells[score_type_array != -999] = 0
        ax.imshow(empty_cells, cmap="Greys", alpha=0.5)

        # Plot the mean scores
        score_type_array_not_empty = np.ma.masked_where(
            score_type_array == -999, score_type_array
        )
        image = ax.imshow(
            score_type_array_not_empty,
            cmap="Greens",
            aspect="equal",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        # Make a colorbar
        fig.colorbar(mappable=image, orientation="vertical", fraction=0.037, pad=0.04)

        # Set the title and labels
        title = choose_heatmap_title(
            single_model=single_model,
            score_type=score_type,
            n_red_herring_clues_evaluated_str=n_red_herring_clues_evaluated_str,
            model=model,
        )

        ax.set_title(title)
        ax.set_xlabel("# Attributes")
        ax.set_ylabel("# Objects")

        # Set the ticks and tick labels
        n_objects_max = score_type_array.shape[0]
        n_attributes_max = score_type_array.shape[1]
        ax.set_xticks(
            ticks=np.arange(n_attributes_max),
            labels=[str(i + 1) for i in range(n_attributes_max)],
        )

        ax.set_yticks(
            ticks=np.arange(n_objects_max),
            labels=[str(i + 2) for i in range(n_objects_max)],
        )

        # Annotate the cells with the mean scores (except for puzzle scores)

        for i in range(n_objects_max):
            for j in range(n_attributes_max):
                if score_type_array[i, j] != -999:
                    # If we are showing puzzle scores for a single evaluation, do not show the standard deviations, as the Bernoulli standard deviations can appear confusing
                    if score_type == "puzzle score" and single_model:
                        ax.text(
                            j,
                            i,
                            f"{score_type_array[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="black",
                        )
                    else:
                        # Round to the correct number significant digits
                        score_rounded, std_rounded = round_using_std(
                            value=score_type_array[i, j], std=std_score_type_array[i, j]
                        )
                        ax.text(
                            j,
                            i,
                            f"{score_rounded} ± {std_rounded}",
                            ha="center",
                            va="center",
                            color="black",
                        )

        # Adjust the layout
        fig.tight_layout()

        # Save the plot
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_filename = f"mean_{score_type}_{model}_{n_red_herring_clues_evaluated_str}rh_{n_puzzles}_puzzles.png"
        plot_filename = plot_filename.replace(" ", "_")
        plt.savefig(plot_path / plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def choose_heatmap_title(
    single_model: bool,
    score_type: str,
    n_red_herring_clues_evaluated_str: str,
    model: str,
) -> str:
    """Choose the title for the heatmap.

    Args:
        single_model: Boolean indicating if the scores are from a single model.
        score_type: Type of score as a string.
        n_red_herring_clues_evaluated_str: Number of red herring clues evaluated as a string.
        model: Name of the model or models as a string.

    Returns:
        title: Title for the heatmap.
    """
    if single_model:
        if not score_type == "puzzle score":
            title = f"{score_type.capitalize()}s with {n_red_herring_clues_evaluated_str} red herrings incl. sample std. dev. for model {model}"
        else:
            title = f"{score_type.capitalize()}s with {n_red_herring_clues_evaluated_str} red herrings for model {model}"
    else:
        title = f"Difference in mean {score_type} with {n_red_herring_clues_evaluated_str.replace('vs', '-')} red herrings for model {model.replace('vs', '-')} incl. std. error"
    return title
