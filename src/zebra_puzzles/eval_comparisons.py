"""Module for comparing the performance of different evaluation configurations on zebra puzzles."""

from pathlib import Path

import numpy as np

from zebra_puzzles.file_utils import save_dataset
from zebra_puzzles.plots import plot_heatmaps
from zebra_puzzles.zebra_utils import round_using_std


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
    # Choose the evaluation pairs for comparison
    eval_idx_1, eval_idx_2, compare_mode, eval_names = choose_eval_pairs(
        model_names=model_names, n_red_herring_values=n_red_herring_values
    )

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

            # Prepare names and paths
            if compare_mode == "red_herrings":
                full_model_name = model_names[0]
                full_red_herring_name = f"{eval_i_name} vs {eval_j_name}"

                plot_path = Path(
                    f"{data_folder}/plots/{full_model_name.replace(' ', '_')}/rh_comparison/"
                )

            else:
                full_model_name = f"{eval_i_name} vs {eval_j_name}"
                full_red_herring_name = n_red_herring_values[0]

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


def choose_eval_pairs(
    model_names: list[str], n_red_herring_values: list[str]
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Choose the evaluation pairs for comparison.

    Args:
        model_names: List of model names.
        n_red_herring_values: Number of red herring clues evaluated.

    Returns:
        A tuple (eval_idx_1, eval_idx_2, compare_mode, eval_names) where:
            eval_idx_1: Indices of the first evaluation.
            eval_idx_2: Indices of the second evaluation.
            compare_mode: Mode of comparison, either "models" or "red_herrings".
            eval_names: Names of the evaluations to be compared.
    """
    # Check if we are comparing models or red herring clues
    if len(model_names) == 1:
        if len(n_red_herring_values) < 2:
            raise Warning(
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

    return eval_idx_1, eval_idx_2, compare_mode, eval_names


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
    # Compute the comparison statistics
    score_diff_all_cells, std_score_diff_all_cells, t_statistic_all_cells = (
        compute_comparison_stats(
            scores_diff=scores_diff,
            i_not_evaluated_by_both=i_not_evaluated_by_both,
            n_puzzles=n_puzzles,
        )
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


def compute_comparison_stats(
    scores_diff: np.ndarray, i_not_evaluated_by_both: np.ndarray, n_puzzles: int
) -> tuple[float, float, float]:
    """Compute the comparison statistics.

    Args:
        scores_diff: Array of score differences.
        i_not_evaluated_by_both: Boolean array indicating cells not evaluated by both models.
        n_puzzles: Number of puzzles evaluated of each size.

    Returns:
        A tuple (score_diff_all_cells, std_score_diff_all_cells, t_statistic_all_cells) where:
            score_diff_all_cells: Mean score difference across all cells.
            std_score_diff_all_cells: Standard deviation of the mean score difference across all cells.
            t_statistic_all_cells: t-statistic for the score difference across all cells.
    """
    # Compute the mean score difference
    non_empty_scores_diff = scores_diff[~i_not_evaluated_by_both]
    n_non_empty_cells = len(non_empty_scores_diff)

    score_diff_all_cells = np.mean(non_empty_scores_diff)

    if n_puzzles > 1:
        # Compute the standard deviation of the mean score difference
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
    return score_diff_all_cells, std_score_diff_all_cells, t_statistic_all_cells
