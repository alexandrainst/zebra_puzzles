"""Utility module for generating zebra puzzles."""

from random import sample
from typing import Dict, List, Tuple


def define_clues(clues_included: str) -> List:
    """Define clue types for the puzzle.

    Args:
        clues_included: A string descriping which clue types to include.

    Returns:
        clues: List of included clue types.

    NOTE: In the future, we can support more clues and selection of clues in the config file.
    TODO: Implement clue functions.
    """
    if clues_included == "all":
        clues = [
            "found_at",
            "not_at",
            "next_to",
            "not_next_to",
            "left_of",
            "right_of",
            "not_left_of",
            "not_right_of",
            "same_house",
            "not_same_house",
            "between",
            "not_between",
        ]
    else:
        raise ValueError("Unsupported clues '{clues_included}'")
    return clues


def generate_solution(
    attributes: Dict[str, Dict[str, str]], n_objects: int, n_attributes: int
) -> Tuple[List[List], List, List[List]]:
    """Generate the solution to a zebra puzzle.

    Args:
        attributes: Attributes as a dictionary of dictionaries.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        solution: A solution to a zebra puzzle as a list of lists representing the matrix of object indices and chosen attributes. This matrix is n_attributes x n_objects.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
    """
    # Choose a category for each attribute
    chosen_categories = sample(list(attributes.keys()), k=n_attributes)

    # Choose attribute values for each category
    chosen_attributes = [
        sample(list(attributes[cat].keys()), k=n_objects) for cat in chosen_categories
    ]

    # Add the object indices to the solution
    indices = [str(x) for x in list(range(n_objects))]
    solution = [indices] + chosen_attributes

    return solution, chosen_categories, chosen_attributes


def generate_clues(
    solution: List[List],
    clues: List,
    chosen_categories: List,
    chosen_attributes: List[List],
) -> str:
    """Generate a zebra puzzle.

    Args:
        solution: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_attributes x n_objects.
        clues: Possible clues to include in the clues as a list of tuples. Each tuple contains the clue name and function. TODO: Edit this description when the clues are implemented.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.

    Returns:
        chosen_clues: Clues for the zebra puzzle as a string.

    TODO: Implement the generation of the clues.
    """
    chosen_clues = (
        "1. This is an example. \n 2. This is the second part of the example."
    )
    return chosen_clues
