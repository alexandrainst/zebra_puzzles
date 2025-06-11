"""Module for comparing responses to solutions."""

import itertools
import logging

import numpy as np
from pydantic import BaseModel

from zebra_puzzles.zebra_utils import bernoulli_std, capitalize, round_using_std

log = logging.getLogger(__name__)


def compute_metrics(
    scores_all_types: list[np.ndarray], score_types: list[str], n_puzzles: int
) -> dict[str, tuple[str | float, ...]]:
    """Compute the metrics.

    For each score type e.g. cell score, a dictionary of metrics is computed. This dictionary includes a string describing the rounded metrics.

    Assumes that the scores are normally distributed, except the puzzle score. Also assumes that the maximum length of the string describing each metric is 100 characters.

    The puzzle score is assumed to follow a Bernoulli distribution.

    Args:
        scores_all_types: Tuple of scores as numpy arrays. Each element contains the scores for a specific score type.
        score_types: List of score type names as strings.
        n_puzzles: Number of puzzles as an integer.

    Returns:
        Metrics as a dictionary of with the score type as the key, and the values being a tuple of ndarrays. The tuple contains the rounded metrics for the score type and a string describing the metrics for the score type.

    NOTE: More metrics could be added e.g. from sklearn.metrics
    """
    # Number of score types
    n_metrics = len(score_types)

    # Initialize metrics
    mean_scores = np.zeros(n_metrics, dtype=float)
    if n_puzzles > 1:
        std_scores = np.zeros(n_metrics, dtype=float)
        std_mean_scores = np.zeros(n_metrics, dtype=float)

    # Initialize strings describing metrics for each score type
    # U100 is a Unicode string with a maximum length of 100 characters
    score_strings = np.zeros(n_metrics, dtype="U100")

    for i, scores in enumerate(scores_all_types):
        # Take the mean
        mean_scores[i] = float(np.mean(scores))

        if n_puzzles > 1:
            if score_types[i] == "puzzle score":
                # Take the standard deviations of the sample and of the mean for a Bernoulli distribution
                n_successes = int(mean_scores[i] * n_puzzles)
                std_scores[i], std_mean_scores[i] = bernoulli_std(
                    n_trials=n_puzzles, n_successes=n_successes
                )
            else:
                # Take the standard deviation
                std_scores[i] = float(np.std(scores, ddof=1))

                # Take the standard deviation of the mean
                std_mean_scores[i] = std_scores[i] / np.sqrt(float(n_puzzles))

            # Round to significant digits
            std_scores[i] = np.format_float_positional(
                std_scores[i], precision=1, fractional=False
            )

            mean_scores_str_i, std_mean_scores_str_i = round_using_std(
                value=mean_scores[i], std=std_mean_scores[i]
            )
            # Rarely, the floats will still remove a trailing zero, so we use the strings as well
            # TODO: If the other parts of 'metrics' are used, we should check their rounding 
            mean_scores[i], std_mean_scores[i] = mean_scores_str_i, std_mean_scores_str_i

            # Describe the score with a string
            score_str = f"\tMean: {mean_scores_str_i} ± {std_mean_scores_str_i} (1σ)"
            score_str += f"\n\tSample standard deviation: {std_scores[i]}"
            score_strings[i] = score_str
        else:
            # Round mean to 2 significant digits
            mean_precision = 2
            mean_scores[i] = np.format_float_positional(
                mean_scores[i], precision=mean_precision, fractional=False
            )

            # Describe the score with a string
            score_strings[i] = f"\tMean: {mean_scores[i]}"

    # Make a dictionary of metrics and score strings for each score type
    if n_puzzles > 1:
        metrics: dict[str, tuple[str | float, ...]] = {
            score_type: (
                mean_scores[i],
                std_scores[i],
                std_mean_scores[i],
                score_strings[i],
            )
            for i, score_type in enumerate(score_types)
        }
    else:
        metrics = {
            score_type: (mean_scores[i], score_strings[i])
            for i, score_type in enumerate(score_types)
        }

    return metrics


def compare_output_to_solution(
    output: BaseModel, solution: BaseModel, n_objects: int, n_attributes: int
) -> tuple[int, float, float]:
    """Compare the output to the solution.

    The puzzle score is 1 for a correct solution and 0 for an incorrect solution.
    The cell score is the proportion of cells that are correct.
    The best permuted cell score is the best cell score after trying all permutations of the objects in the response. This will give a high score if the LLM coupled the attributes correctly, but misunderstood the order of the objects.

    Args:
        output: The output in OutputFormat format.
        solution: The solution in OutputFormat format.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        A tuple (puzzle_score, cell_score), where:
            puzzle_score: A puzzle-level score as an integer.
            cell_score: A cell-level score as a float.
            best_permuted_cell_score: The best cell-level score as a float after trying all permutations of the objects in the response.
    """
    # Convert the output and solution to dictionaries
    try:
        output_dict = dict(output)
    except Exception as output_error:
        log.error(
            f"Error converting output to dictionary. Output: {output}\nError: {output_error}"
        )
        output_dict = {"error": str(output)}
    solution_dict = dict(solution)

    # Compare the full output to the solution

    if output_dict == solution_dict:
        puzzle_score = 1
        cell_score = 1.0
        best_permuted_cell_score = 1.0
    else:
        # Compare all cells
        cell_score = compute_cell_score(
            output=output_dict,
            solution=solution_dict,
            n_objects=n_objects,
            n_attributes=n_attributes,
        )

        # Check if the puzzle is solved after stripping whitespace in cells
        if cell_score == 1:
            puzzle_score = 1
            best_permuted_cell_score = 1.0
        else:
            puzzle_score = 0

            # Evaluate every permutation of the objects in the response
            best_permuted_cell_score = compute_best_permuted_cell_score(
                output=output_dict,
                solution=solution_dict,
                n_objects=n_objects,
                n_attributes=n_attributes,
            )

    return puzzle_score, cell_score, best_permuted_cell_score


def compute_cell_score(
    output: dict[str, list],
    solution: dict[str, list],
    n_objects: int,
    n_attributes: int,
) -> float:
    """Compute the cell score.

    Args:
        output: The output as a dictionary of objects and their attributes.
        solution: The solution as a dictionary of objects and their attributes.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        The cell score as a float.
    """
    # Compare each cell
    cell_score: float = 0.0
    for attributes_output, attributes_solution in zip(
        output.values(), solution.values()
    ):
        for attribute_output, attribute_solution in zip(
            attributes_output, attributes_solution
        ):
            if attribute_output.strip() == attribute_solution.strip():
                cell_score += 1.0

    # Normalise the cell score
    cell_score /= float(n_objects * n_attributes)

    return cell_score


def compute_best_permuted_cell_score(
    output: dict[str, list],
    solution: dict[str, list],
    n_objects: int,
    n_attributes: int,
) -> float:
    """Compute the best permuted cell score.

    Args:
        output: The output as a dictionary of objects and their attributes.
        solution: The solution as a dictionary of objects and their attributes.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        The best permuted cell score as a float.
    """
    best_permuted_cell_score = 0.0
    objects = list(output.keys())

    # Create all permutations of the objects where each object appears exactly once

    object_permutations = list(itertools.permutations(objects))

    # Evaluate each permutation
    for object_permutation in object_permutations:
        # Create a new output with the objects permuted
        output_permuted = {object: output[object] for object in object_permutation}

        # Compare the permuted output to the solution
        permuted_cell_score = compute_cell_score(
            output=output_permuted,
            solution=solution,
            n_objects=n_objects,
            n_attributes=n_attributes,
        )

        # Update the best permuted cell score
        if permuted_cell_score > best_permuted_cell_score:
            best_permuted_cell_score = permuted_cell_score

    return best_permuted_cell_score


def format_scores(
    scores_all_types: list[np.ndarray],
    score_types: list[str],
    metrics: dict[str, tuple],
    n_puzzles: int,
) -> str:
    """Format the scores.

    This creates a string describing the overall metrics and the scores of each puzzle.

    Args:
        scores_all_types: Tuple of scores as numpy arrays. Each element contains the scores for a specific score type.
        score_types: List of score type names as strings.
        n_puzzles: Number of puzzles as an integer.
        metrics: Metrics as a dictionary of with the score type as the key, and the values being a tuple of ndarrays. The tuple contains the rounded metrics for the score type and a string describing the metrics for the score type.

    Returns:
        A formatted string of the scores.
    """
    # --- Describe overall metrics ---#

    score_str = "Puzzle Scores\n"
    score_str += "-------------\n"
    score_str += "Metrics\n\n"
    if n_puzzles > 1:
        score_str += "Uncertainty is given as one standard deviation (1σ), corresponding to a 68% confidence interval. The 95% confidence interval is approximately ±2σ.\n\n"

    # Complete the string describing all metrics
    metrics_str = ""
    for score_type in score_types:
        metrics_str += f"{capitalize(score_type)}:\n"
        metrics_str += metrics[score_type][-1]
        metrics_str += "\n\n"

    metrics_str = metrics_str[:-1]

    score_str += metrics_str

    # --- Describe scores of individual puzzles ---#

    score_str += "\n-------------\n"
    score_str += "Single puzzle scores\n"

    for i in range(n_puzzles):
        score_str += f"\nPuzzle {i}: "
        for score_type, scores in zip(score_types, scores_all_types):
            score_str += f"\t{score_type}: {scores[i]:.2f}"

    return score_str
