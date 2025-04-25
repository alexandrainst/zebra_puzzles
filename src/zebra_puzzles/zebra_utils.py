"""Utility module for generating and evaluating zebra puzzles."""

import json
from pathlib import Path
from random import choices, sample, shuffle
from typing import Any, Type

import numpy as np
from pydantic import BaseModel, create_model
from sklearn.linear_model import LinearRegression

from zebra_puzzles.file_utils import load_individual_puzzle_scores


def generate_solution(
    attributes: dict[str, dict[str, list[str]]], n_objects: int, n_attributes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the solution to a zebra puzzle.

    Chooses categories and assigns attribute values to each object in the solution. Uses 1-based object indices.

    Args:
        attributes: Attributes as a dictionary of dictionaries.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        A tuple (solution, chosen_categories, chosen_attributes, chosen_attributes_descs), where:
            solution: A solution to a zebra puzzle as a matrix of object indices and chosen attributes. The dimensions are n_objects x (1 + n_attributes).
            chosen_categories: Categories chosen for the solution as a ndarray of strings of length n_attributes.
            chosen_attributes: Attribute values chosen for the solution as a matrix of strings. The dimensions are n_objects x n_attributes.
            chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix of strings. 3 versions are provided per description for different sentence structures. The dimensions are 3 x n_objects x n_attributes.
    """
    # Get the possible categories
    all_categories = np.array(list(attributes.keys()))

    # Choose a category for each attribute while maintaining the order of the categories
    chosen_cat_indices = sorted(
        np.array(sample(range(len(all_categories)), k=n_attributes))
    )
    chosen_categories = all_categories[chosen_cat_indices]

    # Choose attribute values for each category
    chosen_attributes = np.array(
        [sample(list(attributes[cat].keys()), k=n_objects) for cat in chosen_categories]
    )

    # Find the attribute descriptions for each attribute in each category
    chosen_attributes_descs = np.array(
        [
            [attributes[cat][key] for key in chosen_attributes[i]]
            for i, cat in enumerate(chosen_categories)
        ]
    )

    # Transpose the attribute matrices
    chosen_attributes = chosen_attributes.T
    chosen_attributes_descs = chosen_attributes_descs.T

    # Add a column of 1-based object indices to the solution
    solution = np.hstack(
        (np.array([list(range(1, n_objects + 1))]).T, chosen_attributes)
    )

    return solution, chosen_categories, chosen_attributes, chosen_attributes_descs


def format_solution_as_json(solution: np.ndarray) -> str:
    """Format the solution as a json dictionary.

    Args:
        solution: Solution to the zebra puzzle as a matrix of object indices and chosen attributes.

    Returns:
        The solution as a json dictionary
    """
    solution_dict = {f"object_{row[0].item()}": row[1:].tolist() for row in solution}
    solution_json = json.dumps(solution_dict, indent=4, ensure_ascii=False)
    return solution_json


def create_solution_template(n_objects: int, chosen_categories: np.ndarray) -> str:
    """Create a solution template for a zebra puzzle.

    For example:
    {
    "object_1": ["attribute_1", "attribute_2"],
    "object_2": ["attribute_1", "attribute_2"]
    }

    Assumes the maximum string length is 100 characters.

    Args:
        n_objects: Number of objects in the puzzle.
        chosen_categories: Categories chosen for the solution.

    Returns:
        The solution template as a string.
    """
    # U100 is a Unicode string with a maximum length of 100 characters
    example_solution = np.zeros((n_objects, len(chosen_categories) + 1), dtype="U100")
    for i in range(n_objects):
        example_solution[i, 0] = f"{i + 1}"
        for j, cat in enumerate(chosen_categories):
            example_solution[i, j + 1] = f"{cat}_{i + 1}"

    solution_template = format_solution_as_json(example_solution)

    return solution_template


def describe_random_attributes(
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    i_objects: list[int],
    n_attributes: int,
    diff_cat: bool = False,
    desc_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get a random attribute description for an object.

    Choose a random attribute for each object with indices given by i_objects. Looks up attributes from chosen_attributes in the attributes dict.

    The attributes are sorted by category to be presented in the preferred order.

    Assumes the maximum string length is 100 characters.

    Args:
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        i_objects: The index of the object to select an attribute from.
        n_attributes: Number of attributes per object.
        diff_cat: If True, the output attributes must belong to different categories.
        desc_index: The index of the description to use for the last object in the clue if more than one object is described.

    Returns:
        A tuple (random_attributes, random_attributes_desc), where:
            random_attributes: A list of strings contraining one random attribute per object.
            random_attributes_desc: A list of strings using the attributes to describe the objects.
    """
    # Number of objects in the clue
    n_clue_objects = len(i_objects)

    if diff_cat:
        i_attributes = sample(list(range(n_attributes)), k=n_clue_objects)
    else:
        i_attributes = choices(list(range(n_attributes)), k=n_clue_objects)

    # Keep the order of the categories
    i_attributes.sort()

    # Initialize the random attributes as type 'object' to avoid setting a maximum string length
    # U100 is a Unicode string with a maximum length of 100 characters
    random_attributes = np.empty((n_clue_objects), dtype="U100")
    random_attributes_desc = np.empty((n_clue_objects), dtype="U100")

    for i, (i_obj, i_attr) in enumerate(zip(i_objects, i_attributes)):
        random_attributes[i] = chosen_attributes[i_obj][i_attr]
        if i == len(i_objects) - 1 and n_clue_objects > 1:
            random_attributes_desc[i] = chosen_attributes_descs[desc_index][i_obj][
                i_attr
            ]
        else:
            random_attributes_desc[i] = chosen_attributes_descs[0][i_obj][i_attr]

    return random_attributes, random_attributes_desc


def generate_output_format_class(n_objects: int) -> Type[BaseModel]:
    """Generate the OutputFormat class based on the number of objects.

    The OutputFormat class is a dynamically generated Pydantic model that represents the output format of the LLM.

    The format will be:
        object_1: list[str]
        object_2: list[str]
        ...

    Args:
        n_objects: Number of objects in the puzzle.

    Returns:
        A dynamically generated OutputFormat class.
    """
    fields: dict[str, Any] = {
        f"object_{i + 1}": (list[str], ...) for i in range(n_objects)
    }

    OutputFormat = create_model("OutputFormat", **fields)

    return OutputFormat


def shuffle_clues(
    chosen_clues: list[str],
    chosen_red_herring_clues: list[str],
    chosen_clue_types: list[str],
    chosen_red_herring_clue_types: list[str],
) -> tuple[list[str], str, str]:
    """Shuffle the clues and red herrings and return the indices of the red herrings.

    The clue types are also shuffled and returned as a string.

    Args:
        chosen_clues: Chosen clues for the zebra puzzle as a list of strings.
        chosen_red_herring_clues: Chosen red herring clues for the zebra puzzle as a list of strings.
        chosen_clue_types: Chosen clue types for the zebra puzzle as a list of strings.
        chosen_red_herring_clue_types: Chosen red herring clue types for the zebra puzzle as a list of strings.

    Returns:
        A tuple (chosen_clues, red_herring_indices_str, chosen_clue_types_str), where:
            chosen_clues: Shuffled clues for the zebra puzzle as a list of strings incl. red herrings.
            red_herring_indices_str: String of indices of the red herrings in the shuffled list of clues.
            chosen_clue_types_str: String of comma-separated clue types chosen for the puzzle.
    """
    # Combine clues and red herrings
    chosen_clues = chosen_clues + chosen_red_herring_clues
    chosen_clue_types = chosen_clue_types + chosen_red_herring_clue_types

    # Shuffle the clues and red herrings
    i_shuffled = list(range(len(chosen_clues)))
    shuffle(i_shuffled)
    chosen_clues = [chosen_clues[i] for i in i_shuffled]
    chosen_clue_types = [chosen_clue_types[i] for i in i_shuffled]

    # Get the new indices of the red herrings
    i_red_herrings = [
        new_i
        for new_i, old_i in enumerate(i_shuffled)
        if old_i >= len(chosen_clues) - len(chosen_red_herring_clues)
    ]
    red_herring_indices_str = ", ".join([str(i) for i in i_red_herrings])

    chosen_clue_types_str = ", ".join(chosen_clue_types)

    return chosen_clues, red_herring_indices_str, chosen_clue_types_str


def round_using_std(value: float, std: float) -> tuple[str, str]:
    """Round a value to match a standard deviation.

    Assumes the value is not much larger than 1.

    Args:
        value: The value to round as a float.
        std: The standard deviation to match as a float.

    Returns:
        A tuple (value, std) where:
            value: The rounded value as a string.
            std: The rounded standard deviation as a string.
    """
    std_rounded = np.format_float_positional(std, precision=1, fractional=False)

    # If the standard deviation is 0, we get the same score for all puzzles. In this case, just use 2 significant digits.
    if std_rounded == "0.":
        value_precision = 2
    else:
        # Get the number of decimal places in the standard deviation
        value_precision = len(str(std_rounded).split(".")[1])

    # Round the value to the same number of decimal places as the standard deviation
    value_rounded = np.format_float_positional(
        value, precision=value_precision, fractional=True
    )

    # Include trailing zeros
    if std_rounded == "0.":
        # Set n_decimal_places to the value_precision minus the number of non-zero digits in the value before the decimal point
        digits_before_decimal = value_rounded.split(".")[0]
        n_nonzero_digits_before_decimal = len(
            [d for d in digits_before_decimal if d != "0"]
        )
        n_decimal_places = 2 - n_nonzero_digits_before_decimal
    else:
        # Get n_decimal_places as the number of decimal places in std_rounded
        n_decimal_places = len(std_rounded.split(".")[1])
    n_trailing_zeros = n_decimal_places - len(value_rounded.split(".")[1])
    if n_trailing_zeros > 0:
        value_rounded += "0" * n_trailing_zeros

    # Turn 1. into 1 and 0. into 0
    if value_rounded[-1] == ".":
        value_rounded = value_rounded[:-1]
    if std_rounded[-1] == ".":
        std_rounded = std_rounded[:-1]

    return value_rounded, std_rounded


def bernoulli_std(n_trials: int, n_successes: int) -> tuple[float, float]:
    """Calculate the standard deviation of success and probability of success in a bernoulli trial.

    We assume puzzle scores are independent Bernoulli trials, each with the same probability of success.

    Args:
        n_trials: Number of trials.
        n_successes: Number of successes.

    Returns:
        A tuple (std_one_trial, std_p), where:
            std_one_trial: The standard deviation of the bernoulli distribution (of 0's and 1's)
            std_p: The standard error of the outcomes i.e. the standard deviation of the probability of success.
    """
    # Calculate the probability of success
    p = n_successes / n_trials

    # Calculate the error of the bernoulli distribution (of 0's and 1's)
    std_one_trial = np.sqrt(p * (1 - p))

    # Calculate the error of the mean (p)
    std_p = np.sqrt(p * (1 - p) / n_trials)

    return std_one_trial, std_p


# --- Plot utility functions ---#


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


def estimate_clue_type_difficulty(
    clue_type_frequencies_all_sizes: dict[str, dict[int, dict[str, int]]],
    clue_types: list[str],
    red_herring_clue_types: list[str],
    n_red_herring_clues_evaluated: int,
    model: str,
    theme: str,
    n_puzzles: int,
    data_folder: Path,
) -> dict[str, dict[str, float]]:
    """Estimate the difficulty of each clue type.

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

    # TODO: Compute the difficulty of each clue type
    #                For example as the mean score of puzzles weighted by the number of times the clue type was used.
    #                Or normalise the scores to compare the difficulty of clues for the puzzle size. Then, the normalised scores can be compared across different puzzle sizes.
    #                Or use linear regression to estimate the difficulty of each clue type.

    NOTE: Consider using scipy instead of sklearn for linear regression to get the standard deviation of the coefficients.
    NOTE: Consider if we should fit to clue type frequencies or normalised clue type frequencies.
    NOTE: Consider if the normalisation of difficulties should be done differently.
    TODO: Handle extreme values and NaN values in the difficulties.
    """
    clue_type_difficulties_all_sizes: dict[str, dict[str, float]] = {}

    all_possible_clue_types = clue_types + red_herring_clue_types

    # Select a puzzle size
    for puzzle_size in clue_type_frequencies_all_sizes.keys():
        # Get the frequencies of each clue type for this puzzle size
        clue_type_frequencies = clue_type_frequencies_all_sizes[puzzle_size]
        clue_types_one_size = clue_type_frequencies.keys()

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
            continue

        # Calculate the difficulty of each clue type
        # Use linear regression to model score as a function of the clue type frequencies in each puzzle

        # Create a linear regression model
        regression_model = LinearRegression()

        # Create the feature matrix (frequency of each clue type in each puzzle)
        X = np.zeros((n_puzzles, len(all_possible_clue_types)))
        for i, clue_type in enumerate(all_possible_clue_types):
            for j, puzzle_index in enumerate(clue_types_one_size):
                X[j, i] = clue_type_frequencies[puzzle_index].get(clue_type, 0)
        # Create the target vector
        y = np.zeros((n_puzzles, 1))

        for j, puzzle_index in enumerate(list(clue_type_frequencies.keys())):
            y[j] = scores_individual_puzzles[puzzle_index]

        # Fit the regression model to the data
        regression_model.fit(X, y)

        # Estimate feature importance in the linear regression model
        # The higher the coefficient, the more important the feature is for predicting the target variable
        clue_importances = regression_model.coef_[0]

        # TODO: Get the standard deviation of each clue type's importance

        # Normalise the clue importances to sum to 1
        clue_importances_normalised = clue_importances / np.sum(clue_importances)

        # Make a dictionary of clue importances
        clue_importances_normalised_dict = {
            clue_type: float(clue_importances_normalised[i])
            for i, clue_type in enumerate(all_possible_clue_types)
        }

        # Append the difficulties for this puzzle size to the list
        clue_type_difficulties_all_sizes[puzzle_size] = clue_importances_normalised_dict

    return clue_type_difficulties_all_sizes
