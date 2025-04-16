"""Module for plotting the results of LLM's trying to solve zebra puzzles."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zebra_puzzles.file_utils import (
    get_evaluated_params,
    get_puzzle_dimensions_from_filename,
    get_score_file_paths,
    load_scores,
    save_dataset,
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


def compare_all_eval_types(
    model_names: list[str],
    mean_scores_all_eval_array: list[list[np.ndarray]],
    std_mean_scores_all_eval_array: list[list[np.ndarray]],
    n_red_herring_values: list[int],
    n_objects_max_all_eval: list[list[int]],
    n_attributes_max_all_eval: list[list[int]],
    data_folder: Path,
    score_types: list[str],
    n_puzzles: int,
) -> None:
    """Compare the mean scores of different evaluations.

    Args:
        model_names: List of model names.
        mean_scores_all_eval_array: List of list of mean scores arrays. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
        std_mean_scores_all_eval_array: List of standard deviation arrays. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
        n_red_herring_values: Number of red herring clues evaluated.
        n_objects_max_all_eval: List of lists of the maximum number of objects in puzzles for each evaluation. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
        n_attributes_max_all_eval: List of lists of the maximum number of attributes in puzzles for each evaluation. The outer list is for different n_red_herring_clues_evaluated and the inner list is for different models.
        data_folder: Path to the data folder.
        score_types: List of score types as strings.
        n_puzzles: Number of puzzles evaluated with each size.
    """
    # ----- Compare the mean scores of different models -----#

    # Compare the mean scores of different models (same n_red_herring_clues_evaluated)
    for i, n_red_herring_clues_evaluated in enumerate(n_red_herring_values):
        mean_scores_all_eval_array_n_red_herrings_i = mean_scores_all_eval_array[i]
        std_mean_scores_all_eval_array_n_red_herrings_i = (
            std_mean_scores_all_eval_array[i]
        )
        n_objects_max_all_eval_n_red_herrings_i = n_objects_max_all_eval[i]
        n_attributes_max_all_eval_n_red_herrings_i = n_attributes_max_all_eval[i]

        compare_eval_type(
            model_names=model_names,
            mean_scores_some_eval_array=mean_scores_all_eval_array_n_red_herrings_i,
            std_mean_scores_some_eval_array=std_mean_scores_all_eval_array_n_red_herrings_i,
            n_red_herring_values=[str(n_red_herring_clues_evaluated)],
            data_folder=data_folder,
            score_types=score_types,
            n_objects_max_some_eval=n_objects_max_all_eval_n_red_herrings_i,
            n_attributes_max_some_eval=n_attributes_max_all_eval_n_red_herrings_i,
            n_puzzles=n_puzzles,
        )

    # ----- Compare the mean scores of different n_red_herring_clues_evaluated -----#

    # For each model, compare mean scores for different n_red_herring_clues_evaluated
    for i, model in enumerate(model_names):
        # Get the parameters for a specific model for all n_red_herring_clues_evaluated
        mean_scores_all_eval_array_model_i = [
            red_herring_eval[i] for red_herring_eval in mean_scores_all_eval_array
        ]
        std_mean_scores_all_eval_array_model_i = [
            red_herring_eval[i] for red_herring_eval in std_mean_scores_all_eval_array
        ]
        n_objects_max_all_eval_model_i = [
            red_herring_eval[i] for red_herring_eval in n_objects_max_all_eval
        ]
        n_attributes_max_all_eval_model_i = [
            red_herring_eval[i] for red_herring_eval in n_attributes_max_all_eval
        ]

        compare_eval_type(
            model_names=[model],
            mean_scores_some_eval_array=mean_scores_all_eval_array_model_i,
            std_mean_scores_some_eval_array=std_mean_scores_all_eval_array_model_i,
            n_red_herring_values=[str(n) for n in n_red_herring_values],
            data_folder=data_folder,
            score_types=score_types,
            n_objects_max_some_eval=n_objects_max_all_eval_model_i,
            n_attributes_max_some_eval=n_attributes_max_all_eval_model_i,
            n_puzzles=n_puzzles,
        )


def compare_eval_type(
    model_names: list[str],
    mean_scores_some_eval_array: list[np.ndarray],
    std_mean_scores_some_eval_array: list[np.ndarray],
    n_red_herring_values: list[str],
    data_folder: Path,
    score_types: list[str],
    n_objects_max_some_eval: list[int],
    n_attributes_max_some_eval: list[int],
    n_puzzles: int,
) -> None:
    """Compare the mean scores of different evaluations of a specific type.

    This could be evaluations using different models or a different number of red herring clues.

    We assume that we only need to specify the maximum number of objects and attributes for each model.

    Args:
        model_names: List of model names.
        mean_scores_some_eval_array: List of mean scores arrays.
        std_mean_scores_some_eval_array: List of standard deviation arrays.
        n_red_herring_values: Number of red herring clues evaluated.
        data_folder: Path to the data folder.
        score_types: List of score types as strings.
        n_objects_max_some_eval: List of the maximum number of objects in puzzles for each evaluated model.
        n_attributes_max_some_eval: List of the maximum number of attributes in puzzles for each evaluated model.
        n_puzzles: Number of puzzles evaluated with each size.
    """
    # Check if the number of models is greater than 1
    if len(model_names) < 2:
        if len(n_red_herring_values) < 2:
            raise ValueError(
                "At least two models or two different n_red_herring_values are required for comparison."
            )
        # Choose each combination of two models
        compare_mode = "red_herrings"
        eval_idx_1, eval_idx_2 = np.triu_indices(len(n_red_herring_values), k=1)
        eval_names = n_red_herring_values
    else:
        if len(n_red_herring_values) > 1:
            raise ValueError(
                "Only one model can be compared across different n_red_herring_values."
            )
        # Choose each combination of two models
        compare_mode = "models"
        eval_idx_1, eval_idx_2 = np.triu_indices(len(model_names), k=1)
        eval_names = model_names

    # Iterate over all combinations of models
    for i in eval_idx_1:
        for j in eval_idx_2:
            # Get parameters for the two models where they overlap in n_objects and n_attributes
            (
                eval_i_scores,
                eval_j_scores,
                eval_i_std_mean_scores,
                eval_j_std_mean_scores,
                eval_i_name,
                eval_j_name,
            ) = load_score_overlap(
                eval_names=eval_names,
                mean_scores_some_eval_array=mean_scores_some_eval_array,
                std_mean_scores_some_eval_array=std_mean_scores_some_eval_array,
                n_objects_max_some_eval=n_objects_max_some_eval,
                n_attributes_max_some_eval=n_attributes_max_some_eval,
                i=i,
                j=j,
            )

            # Compute the difference in mean scores
            scores_diff, std_score_diff, i_not_evaluated_by_both = compute_scores_diff(
                eval_i_scores=eval_i_scores,
                eval_j_scores=eval_j_scores,
                eval_i_std_mean_scores=eval_i_std_mean_scores,
                eval_j_std_mean_scores=eval_j_std_mean_scores,
            )

            if compare_mode == "red_herrings":
                full_model_name = model_names[0]
                full_red_herring_name = f"{eval_i_name} vs {eval_j_name}"
                # Prepare path for plots
                plot_path = Path(
                    f"{data_folder}/plots/{full_model_name.replace(' ', '_')}/rh_comparison/"
                )

            else:
                full_model_name = f"{eval_i_name} vs {eval_j_name}"
                full_red_herring_name = n_red_herring_values[0]

                # Prepare path for plots
                plot_path = Path(
                    f"{data_folder}/plots/{full_model_name.replace(' ', '_')}/"
                )

            # Make heatmaps of differences in mean scores
            plot_heatmaps(
                scores_array=scores_diff,
                score_types=score_types,
                plot_path=plot_path,
                n_red_herring_clues_evaluated_str=full_red_herring_name,
                std_scores_array=std_score_diff,
                single_model=False,
                model=full_model_name,
                n_puzzles=n_puzzles,
            )

            create_comparison_txt(
                scores_diff=scores_diff,
                full_model_name=full_model_name,
                full_red_herring_name=full_red_herring_name,
                plot_path=plot_path,
                i_not_evaluated_by_both=i_not_evaluated_by_both,
                n_puzzles=n_puzzles,
            )


def load_score_overlap(
    eval_names: list[str],
    mean_scores_some_eval_array: list[np.ndarray],
    std_mean_scores_some_eval_array: list[np.ndarray],
    n_objects_max_some_eval: list[int],
    n_attributes_max_some_eval: list[int],
    i: int,
    j: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Load the scores of two evaluations and limit them to the minimum number of objects and attributes.

    An evaluation can be a model or a number of red herring clues.

    Args:
        eval_names: List of evaluation names.
        mean_scores_some_eval_array: List of mean scores arrays.
        std_mean_scores_some_eval_array: List of standard deviation arrays.
        n_objects_max_some_eval: List of the maximum number of objects in puzzles for each evaluation.
        n_attributes_max_some_eval: List of the maximum number of attributes in puzzles for each evaluation.
        i: Index of the first evaluation.
        j: Index of the second evaluation.

    Returns:
        A tuple (eval_i_scores, eval_j_scores, eval_i_std_mean_scores, eval_j_std_mean_scores, eval_i, eval_j) where:
            eval_i_scores: Mean scores of the first evaluation.
            eval_j_scores: Mean scores of the second evaluation.
            eval_i_std_mean_scores: Standard deviations of the mean scores of the first evaluation.
            eval_j_std_mean_scores: Standard deviations of the mean scores of the second evaluation.
            eval_i: Name of the first evaluation.
            eval_j: Name of the second evaluation.
    """
    # Get the evaluation specific parameters
    (eval_i, eval_i_scores, eval_i_std_mean_scores, n_objects_max_i, n_attributes_i) = (
        eval_names[i],
        mean_scores_some_eval_array[i],
        std_mean_scores_some_eval_array[i],
        n_objects_max_some_eval[i],
        n_attributes_max_some_eval[i],
    )
    (eval_j, eval_j_scores, eval_j_std_mean_scores, n_objects_max_j, n_attributes_j) = (
        eval_names[j],
        mean_scores_some_eval_array[j],
        std_mean_scores_some_eval_array[j],
        n_objects_max_some_eval[j],
        n_attributes_max_some_eval[j],
    )

    # Limit the number of objects and attributes to the minimum of the maxima of the two evaluations
    n_objects = min(n_objects_max_i, n_objects_max_j)
    n_attributes = min(n_attributes_i, n_attributes_j)
    eval_i_scores = eval_i_scores[:n_attributes, : n_objects - 1]
    eval_j_scores = eval_j_scores[:n_attributes, : n_objects - 1]
    eval_i_std_mean_scores = eval_i_std_mean_scores[:n_attributes, : n_objects - 1]
    eval_j_std_mean_scores = eval_j_std_mean_scores[:n_attributes, : n_objects - 1]
    return (
        eval_i_scores,
        eval_j_scores,
        eval_i_std_mean_scores,
        eval_j_std_mean_scores,
        eval_i,
        eval_j,
    )


def compute_scores_diff(
    eval_i_scores: np.ndarray,
    eval_j_scores: np.ndarray,
    eval_i_std_mean_scores: np.ndarray,
    eval_j_std_mean_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the difference in mean scores of two evaluations.

    Args:
        eval_i_scores: Mean scores of the first evaluation.
        eval_j_scores: Mean scores of the second evaluation.
        eval_i_std_mean_scores: Standard deviations of the mean scores of the first evaluation.
        eval_j_std_mean_scores: Standard deviations of the mean scores of the second evaluation.

    Returns:
        A tuple (scores_diff, std_score_diff, i_not_evaluated_by_both) where:
            scores_diff: Difference in mean scores of the two evaluations.
            std_score_diff: Standard deviation of the difference in mean scores.
            i_not_evaluated_by_both: Boolean array indicating cells not evaluated by both evaluations.
    """
    # Compute the difference in mean scores where the two evaluations have the same n_objects and n_attributes
    scores_diff = eval_i_scores - eval_j_scores

    # Compute the standard deviation of the difference of mean scores
    # The formula follows from the law of error propagation assuming the scores are independent (but they are in fact evaluated on the same puzzles)
    std_score_diff = np.sqrt(eval_i_std_mean_scores**2 + eval_j_std_mean_scores**2)

    # Define the cells that are not evaluated by one of the evaluations
    i_not_evaluated_by_both = np.logical_or(
        eval_i_scores == -999, eval_j_scores == -999
    )

    # If a cell is not evaluated by one of the evaluations, set it to -999
    scores_diff[i_not_evaluated_by_both] = -999
    std_score_diff[i_not_evaluated_by_both] = -999

    return scores_diff, std_score_diff, i_not_evaluated_by_both


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
    TODO: Correct number of significant digits in the text annotations.
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
        if single_model:
            if not score_type == "puzzle score":
                title = f"{score_type.capitalize()}s with {n_red_herring_clues_evaluated_str} red herrings incl. sample std. dev. for model {model}"
            else:
                title = f"{score_type.capitalize()}s with {n_red_herring_clues_evaluated_str} red herrings for model {model}"
        else:
            title = f"Difference in mean {score_type} with {n_red_herring_clues_evaluated_str.replace('vs', '-')} red herrings for model {model.replace('vs', '-')} incl. std. error"

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
                    if score_type == "puzzle score" and "vs" not in model:
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


def create_comparison_txt(
    scores_diff: np.ndarray,
    full_model_name: str,
    full_red_herring_name: str,
    plot_path: Path,
    i_not_evaluated_by_both: np.ndarray,
    n_puzzles: int,
):
    """Create a text file with the comparison results.

    Args:
        scores_diff: Array of score differences.
        full_model_name: String describing the model name configuration(s).
        full_red_herring_name: String describing the configuration(s) on the number of red herring clues.
        plot_path: Path to save the text file.
        i_not_evaluated_by_both: Boolean array indicating cells not evaluated by both models.
        n_puzzles: Number of puzzles evaluated of each size.

    """
    # Compute the mean score difference
    non_empty_scores_diff = scores_diff[~i_not_evaluated_by_both]
    n_non_empty_cells = len(non_empty_scores_diff)

    score_diff_all_cells = np.mean(non_empty_scores_diff)

    if n_puzzles > 1:
        # Compute the standard deviation of the score difference
        std_score_diff_all_cells = np.std(non_empty_scores_diff, ddof=1) / np.sqrt(
            n_non_empty_cells
        )

        # Compute the t-statistic (number of standard deviations between the means) across all cells
        # Note that this might average out differences in performance on puzzles of different sizes
        if std_score_diff_all_cells != 0:
            t_statistic_all_cells = score_diff_all_cells / std_score_diff_all_cells
        else:
            t_statistic_all_cells = -999

    else:
        std_score_diff_all_cells = 0.0

    # Round to the correct number significant digits
    score_diff_all_cells, std_score_diff_all_cells = round_using_std(
        value=score_diff_all_cells, std=std_score_diff_all_cells
    )

    # Save the overall results
    filename = (
        f"comparison_{full_model_name}_{full_red_herring_name}_{n_puzzles}_puzzles.txt"
    )
    filename = filename.replace(" ", "_")

    comparison_str = f"Model {full_model_name} with {full_red_herring_name} red herring clues on puzzle sizes evaluated by both. {n_puzzles} puzzles are evaluated for each size.\n"
    comparison_str += f"\n\nMean score difference: {score_diff_all_cells}"
    if n_puzzles > 1:
        comparison_str += (
            f"\nStandard deviation of the difference: {std_score_diff_all_cells}"
        )
        comparison_str += f"\n\nt-statistic: {t_statistic_all_cells:.2f} (number of standard deviations between the means)"

    # Save the comparison results to a text file
    save_dataset(data=comparison_str, filename=filename, folder=plot_path)
