"""Module for analysing the statistics of clue types."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from zebra_puzzles.load_data import (
    load_clue_type_frequencies,
    load_individual_puzzle_scores,
)


def get_all_clue_type_frequencies(
    clue_type_file_paths_all_sizes: dict[str, list[Path]],
) -> tuple[
    dict[str, dict[int, dict[str, int]]],
    dict[str, dict[int, int]],
    dict[str, dict[int, dict[str, float]]],
]:
    """Get the frequencies of each clue type from the clue type files.

    Loads the clue type frequencies from the clue type files for all puzzle sizes.

    We also compute the normalised frequencies which sum to 1 in each puzzle, and we store the number of clues in each puzzle.

    Args:
        clue_type_file_paths_all_sizes: List of paths to the clue type files.

    Returns:
        A tuple (clue_type_frequencies_all_sizes, n_clues_all_sizes, clue_type_frequencies_all_sizes_normalised) where:
            clue_type_frequencies_all_sizes: Dictionary of dictionaries of dictionaries of clue type frequencies.
                The outer dictionary is for each puzzle size, the middle dictionary is for a puzzle index, and the inner dictionary is for each clue type.
            n_clues_all_sizes: Dictionary of dictionaries of the number of clues for each puzzle size.
                The outer dictionary is for each puzzle size, and the inner dictionary is for a puzzle index.
            clue_type_frequencies_all_sizes_normalised: Dictionary of dictionaries of dictionaries of normalised clue type frequencies. The format matches clue_type_frequencies_all_sizes.
    """
    clue_type_frequencies_all_sizes: dict[str, dict[int, dict[str, int]]] = {}
    clue_type_frequencies_one_size: dict[int, dict[str, int]] = {}

    n_clues_all_sizes: dict[str, dict[int, int]] = {}
    n_clues_one_size: dict[int, int] = {}

    clue_type_frequencies_all_sizes_normalised: dict[
        str, dict[int, dict[str, float]]
    ] = {}
    clue_type_frequencies_one_size_normalised: dict[int, dict[str, float]] = {}

    # Loop though all the puzzle sizes
    for puzzle_size, clue_type_file_paths in clue_type_file_paths_all_sizes.items():
        # Loop through all the clue type files
        for clue_type_file_path in clue_type_file_paths:
            # Get the puzzle index from the clue type file name
            puzzle_index = int(clue_type_file_path.stem.split("_")[2])

            # Load the clue type frequencies from the file
            clue_type_frequencies = load_clue_type_frequencies(
                clue_type_file_path=clue_type_file_path
            )

            # Get the number of clues in the puzzle
            n_clues = sum(clue_type_frequencies.values())

            # Normalise
            clue_type_frequencies_normalised = normalise_dict_values(
                dict_original=clue_type_frequencies, sum_values=n_clues
            )

            # Add this puzzle to the dictionaries
            clue_type_frequencies_one_size[puzzle_index] = clue_type_frequencies
            n_clues_one_size[puzzle_index] = n_clues
            clue_type_frequencies_one_size_normalised[puzzle_index] = (
                clue_type_frequencies_normalised
            )

        # Add the clue type frequencies for this puzzle size to the dictionary of all sizes
        clue_type_frequencies_all_sizes[puzzle_size] = clue_type_frequencies_one_size
        n_clues_all_sizes[puzzle_size] = n_clues_one_size
        clue_type_frequencies_all_sizes_normalised[puzzle_size] = (
            clue_type_frequencies_one_size_normalised
        )

        # Reset dictionaries for the next puzzle size
        clue_type_frequencies_one_size = {}
        n_clues_one_size = {}
        clue_type_frequencies_one_size_normalised = {}

    return (
        clue_type_frequencies_all_sizes,
        n_clues_all_sizes,
        clue_type_frequencies_all_sizes_normalised,
    )


def normalise_dict_values(
    dict_original: dict[str, int], sum_values: int
) -> dict[str, float]:
    """Normalise values in a dictionary.

    Args:
        dict_original: Dictionary of keys and integer values.
        sum_values: The sum of the values in the dictionary.

    Returns:
        A dictionary of normalised values.
    """
    # Normalise the frequencies
    dict_normalised: dict[str, float] = {
        key: freq / float(sum_values) for key, freq in dict_original.items()
    }

    # Check that the sum of dict_normalised is close to 1
    sum_normalised_dict = sum(dict_normalised.values())
    if abs(sum_normalised_dict - 1.0) > 0.0001:
        raise ValueError(
            f"The normalised dictionary values do not sum to 1. They sum to {sum_normalised_dict}."
        )

    return dict_normalised


def get_all_mean_clue_frequencies_per_puzzle_size(
    clue_type_frequencies_all_sizes: dict[str, dict[int, dict[str, int]]],
    clue_type_frequencies_all_sizes_normalised: dict[str, dict[int, dict[str, float]]],
    n_puzzles: int,
    clue_types: list[str],
    red_herring_clue_types: list[str],
) -> tuple[dict[str, dict[str, float]], list[str], float]:
    """Get the mean of the normalised frequencies of each clue type for all puzzle sizes.

    Args:
        clue_type_frequencies_all_sizes: Dictionary of dictionaries of dictionaries of clue type frequencies.
            The outer dictionary is for each puzzle size, the middle dictionary is for a puzzle index, and the inner dictionary is for each clue type.
        clue_type_frequencies_all_sizes_normalised: Dictionary of dictionaries of dictionaries of normalised clue type frequencies.
            The format matches clue_type_frequencies_all_sizes.
        n_puzzles: The number of puzzles for each puzzle size.
        clue_types: List of non red herring clue types.
        red_herring_clue_types: List of red herring clue types.

    Returns:
        A tuple (clue_type_frequencies_normalised_mean_all_sizes, all_clue_types, max_mean_normalised_frequency), where:
            clue_type_frequencies_normalised_mean_all_sizes: Dictionary of dictionaries of mean normalised clue type frequencies for each puzzle size.
            all_clue_types: List of all clue types.
            max_mean_normalised_frequency: Maximum mean normalised frequency across all puzzle sizes.
    """
    clue_type_frequencies_normalised_mean_all_sizes: dict[str, dict[str, float]] = {}
    all_clue_types: list[str] = []

    # Get the mean of the normalised frequencies of each clue type for all puzzle sizes
    # and make a list of all clue types
    for puzzle_size in clue_type_frequencies_all_sizes.keys():
        clue_type_frequencies_normalised_mean_one_size = get_mean_clue_frequencies_for_one_puzzle_size(
            clue_type_frequencies_all_sizes_normalised=clue_type_frequencies_all_sizes_normalised,
            puzzle_size=puzzle_size,
            n_puzzles=n_puzzles,
        )
        clue_type_frequencies_normalised_mean_all_sizes[puzzle_size] = (
            clue_type_frequencies_normalised_mean_one_size
        )
        all_clue_types.extend(clue_type_frequencies_normalised_mean_one_size.keys())

    all_clue_types = sorted(set(all_clue_types))

    all_possible_clue_types = clue_types + red_herring_clue_types

    # Sort all clue types by all_possible_clue_types
    all_clue_types.sort(key=lambda x: all_possible_clue_types.index(x))

    # Get the maximum frequency for each clue type across all puzzle sizes
    max_mean_normalised_frequency = max(
        [
            max(clue_type_frequencies_normalised_mean_all_sizes[puzzle_size].values())
            for puzzle_size in clue_type_frequencies_normalised_mean_all_sizes.keys()
        ]
    )

    return (
        clue_type_frequencies_normalised_mean_all_sizes,
        all_clue_types,
        max_mean_normalised_frequency,
    )


def get_mean_clue_frequencies_for_one_puzzle_size(
    clue_type_frequencies_all_sizes_normalised: dict[str, dict[int, dict[str, float]]],
    puzzle_size: str,
    n_puzzles: int,
) -> dict[str, float]:
    """Get the mean of the normalised frequencies of each clue type for puzzles of a specific size.

    Args:
        clue_type_frequencies_all_sizes_normalised: Dictionary of dictionaries of dictionaries of normalised clue type frequencies.
            The outer dictionary is for each puzzle size, the middle dictionary is for a puzzle index, and the inner dictionary is for each clue type.
        puzzle_size: String describing the puzzle size.
        n_puzzles: The number of puzzles for each puzzle size.

    Returns:
        A dictionary with clue types as keys and the mean of the normalised frequencies as the values.
    """
    # Take the mean of the clue type frequencies across all puzzles of this size
    clue_type_frequencies_normalised_mean_one_size: dict[str, float] = {}
    for clue_type_frequencies_normalised in clue_type_frequencies_all_sizes_normalised[
        puzzle_size
    ].values():
        # Sum the normalised frequencies of all puzzles of the chosen size
        for clue_type, freq_norm in clue_type_frequencies_normalised.items():
            if clue_type not in clue_type_frequencies_normalised_mean_one_size:
                clue_type_frequencies_normalised_mean_one_size[clue_type] = 0.0
            clue_type_frequencies_normalised_mean_one_size[clue_type] += freq_norm

    # Divide the sum for each clue type by the number of puzzles
    for (
        clue_type,
        freq_norm_sum,
    ) in clue_type_frequencies_normalised_mean_one_size.items():
        clue_type_frequencies_normalised_mean_one_size[clue_type] = (
            freq_norm_sum / float(n_puzzles)
        )

    sum_all_normalised_frequencies = sum(
        clue_type_frequencies_normalised_mean_one_size.values()
    )
    if abs(sum_all_normalised_frequencies - 1.0) > 0.0001:
        raise ValueError(
            f"The normalised frequencies do not sum to 1. They sum to {sum_all_normalised_frequencies}."
        )

    return clue_type_frequencies_normalised_mean_one_size


def estimate_clue_type_difficulty_for_all_puzzle_sizes(
    clue_type_frequencies_all_sizes: dict[str, dict[int, dict[str, int]]],
    clue_types: list[str],
    red_herring_clue_types: list[str],
    n_red_herring_clues_evaluated: int,
    model: str,
    theme: str,
    n_puzzles: int,
    data_folder: Path,
) -> dict[str, dict[str, float]]:
    """Estimate the difficulty of each clue type for each puzzle size.

    To estimate the clue type difficulties for a size, the size needs to contain multiple puzzles with different clue type frequencies and different scores.

    Args:
        clue_type_frequencies_all_sizes: Dictionary of dictionaries of dictionaries of clue type frequencies.
            The outer dictionary is for each puzzle size, the middle dictionary is for a puzzle index, and the inner dictionary is for each clue type.
        clue_types: List of non red herring clue types.
        red_herring_clue_types: List of red herring clue types.
        n_red_herring_clues_evaluated: Number of red herring clues evaluated.
        model: LLM model used to evaluate the puzzles.
        theme: Theme of the puzzle.
        n_puzzles: The number of puzzles for each puzzle size.
        data_folder: Path to the data folder.

    Returns:
        A dictionary of dictionaries of clue difficulties as floats.
            The outer dictionary is for each puzzle size, the inner dictionary is for each clue type.


    NOTE: We can consider other methods for computing the difficulty of each clue type.
    NOTE: It would be very useful to have a measure of the uncertainty of the difficulty estimates. Perhaps by using scipy.

    """
    clue_type_difficulties_all_sizes: dict[str, dict[str, float]] = {}

    all_possible_clue_types = clue_types + red_herring_clue_types

    n_non_evaluated_puzzles = 0
    n_identical_frequencies = 0
    n_identical_scores = 0

    puzzle_sizes = sorted(clue_type_frequencies_all_sizes.keys())

    # Select a puzzle size
    for puzzle_size in puzzle_sizes:
        # Get the frequencies of each clue type for this puzzle size
        clue_type_frequencies_one_size = clue_type_frequencies_all_sizes[puzzle_size]
        puzzle_indices_one_size = sorted(clue_type_frequencies_one_size.keys())

        # Check if the combination of clue types and their frequencies are identical for all puzzles of this size
        all_identical_frequencies_flag = check_identical_frequencies(
            puzzle_indices_one_size=puzzle_indices_one_size,
            clue_type_frequencies_one_size=clue_type_frequencies_one_size,
            all_possible_clue_types=all_possible_clue_types,
        )

        # If the frequencies are identical for all puzzles of this size, skip this size
        if all_identical_frequencies_flag:
            n_identical_frequencies += 1
            continue

        # Load the list of scores incl. all n_puzzles
        scores_individual_puzzles = load_individual_puzzle_scores(
            data_folder=data_folder,
            score_type="cell_score",
            model=model,
            n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
            n_puzzles=n_puzzles,
            theme=theme,
            puzzle_size=puzzle_size,
        )

        # If we tried to load scores for a size that was not evaluated by the model, skip this size
        if scores_individual_puzzles[0] == -999.0:
            n_non_evaluated_puzzles += 1
            continue

        # If the scores are identical for all puzzles of this size, skip this size
        if len(set(scores_individual_puzzles.values())) == 1:
            n_identical_scores += 1
            continue

        # Calculate the difficulty of each clue type
        # Use linear regression to model score as a function of the clue type frequencies in each puzzle
        clue_difficulties_dict = estimate_clue_type_difficulty_for_one_puzzle_size(
            clue_type_frequencies_one_size=clue_type_frequencies_one_size,
            scores_individual_puzzles=scores_individual_puzzles,
            n_puzzles=n_puzzles,
            all_possible_clue_types=all_possible_clue_types,
            puzzle_indices_one_size=puzzle_indices_one_size,
        )

        # Append the difficulties for this puzzle size to the list
        # Use the negative of the importances as the difficulty
        clue_type_difficulties_all_sizes[puzzle_size] = clue_difficulties_dict

    print(
        f"Out of {len(puzzle_sizes)} puzzle sizes for model {model}, clue difficulty estimation has been skipped for {n_identical_frequencies} sizes with identical clue type frequencies, {n_identical_scores} sizes with identical scores, and {n_non_evaluated_puzzles} sizes with no evaluated puzzles."
    )
    if (
        len(clue_type_difficulties_all_sizes)
        != len(puzzle_sizes)
        - n_identical_frequencies
        - n_identical_scores
        - n_non_evaluated_puzzles
    ):
        raise Warning(
            f"The number of puzzle sizes with estimated clue difficulties is not equal to the number of puzzle sizes minus the number of skipped sizes. {len(clue_type_difficulties_all_sizes)} != {len(puzzle_sizes)} - {n_identical_frequencies} - {n_identical_scores} - {n_non_evaluated_puzzles}"
        )

    return clue_type_difficulties_all_sizes


def estimate_clue_type_difficulty_for_one_puzzle_size(
    clue_type_frequencies_one_size: dict[int, dict[str, int]],
    scores_individual_puzzles: dict[int, float],
    n_puzzles: int,
    all_possible_clue_types: list[str],
    puzzle_indices_one_size: list[int],
) -> dict[str, float]:
    """Estimate the difficulty of each clue type for a specific puzzle size.

    This function uses linear regression to model scores as a function of the clue type frequencies in each puzzle.
    The coefficients of the linear regression model are used to estimate the difficulty of each clue type.
    The higher the coefficient, the more it is correlated with a high score, and the easier the clue type is.

    Args:
        clue_type_frequencies_one_size: Dictionary of dictionaries of clue type frequencies for the given size.
            The outer dictionary is for each puzzle index, and the inner dictionary is for each clue type.
        scores_individual_puzzles: Dictionary of scores for each puzzle index.
        n_puzzles: The number of puzzles for this size.
        all_possible_clue_types: List of all possible clue types.
        puzzle_indices_one_size: List of puzzle indices for the given size.

    Returns:
        A dictionary of clue difficulties as floats.
    """
    # --- Fit a linear regression model to predict scores from frequencies ---#
    regression_model = LinearRegression()

    # Create the feature matrix (frequency of each clue type in each puzzle)
    X = np.zeros((n_puzzles, len(all_possible_clue_types)))
    for i, clue_type in enumerate(all_possible_clue_types):
        for j, puzzle_index in enumerate(puzzle_indices_one_size):
            X[j, i] = clue_type_frequencies_one_size[puzzle_index].get(clue_type, 0)

    # Create the target vector
    y = np.zeros((n_puzzles, 1))

    for j, puzzle_index in enumerate(puzzle_indices_one_size):
        y[j] = scores_individual_puzzles[puzzle_index]

    # Fit the regression model to the data
    regression_model.fit(X, y)

    # --- Interpret the fit ---#

    # Estimate feature importance in the linear regression model
    # The higher the coefficient, the more important the feature is for predicting the target variable
    clue_importances = regression_model.coef_[0]

    # Scale the clue importances so the absolute values sum to 1
    clue_importances_normalised = clue_importances / np.sum(abs(clue_importances))

    # Take the negative of the importances as the difficulty
    clue_difficulties = -clue_importances_normalised

    # Make a dictionary of clue importances
    clue_difficulties_dict = {
        clue_type: float(clue_difficulties[i])
        for i, clue_type in enumerate(all_possible_clue_types)
    }

    return clue_difficulties_dict


def check_identical_frequencies(
    puzzle_indices_one_size: list[int],
    clue_type_frequencies_one_size: dict[int, dict[str, int]],
    all_possible_clue_types: list[str],
) -> bool:
    """Check if the frequencies of each clue type are identical for all puzzles of a given size.

    Args:
        puzzle_indices_one_size: List of puzzle indices for the given size.
        clue_type_frequencies_one_size: Dictionary of dictionaries of clue type frequencies for the given size.
        all_possible_clue_types: List of all possible clue types.

    Returns:
        A boolean indicating whether the frequencies are identical for all puzzles of this size.
    """
    for puzzle_index in puzzle_indices_one_size[1:]:
        # For each clue type, check if the frequencies are identical to the first puzzle
        n_identical_frequencies_one_size = 0
        n_identical_frequencies_one_puzzle = 0

        # Check if the frequencies in this puzzle are identical to the first puzzle
        for clue_type in all_possible_clue_types:
            if clue_type_frequencies_one_size[puzzle_index].get(
                clue_type, 0
            ) != clue_type_frequencies_one_size[puzzle_indices_one_size[0]].get(
                clue_type, 0
            ):
                n_identical_frequencies_one_puzzle += 1

            # Check whether all frequencies are identical for this puzzle size
            if n_identical_frequencies_one_puzzle == len(all_possible_clue_types):
                n_identical_frequencies_one_size += 1

        # Check whether all frequencies are identical for this puzzle size
        all_identical_frequencies_flag = n_identical_frequencies_one_size == len(
            puzzle_indices_one_size
        )

    return all_identical_frequencies_flag
