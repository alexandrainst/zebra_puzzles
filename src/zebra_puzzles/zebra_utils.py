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


def complete_clue(
    clue: str,
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
) -> str:
    """Complete the chosen clue type with random parts of the solution to create a full clue.

    TODO: Consider how the clues will be evaluted. We should probably save more than a string.
    TODO: Move the clue descriptions to the config file.

    Args:
        clue: Chosen clue type as a string.
        n_objects: Number of objects in the puzzle as an int.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.

    Returns:
        full_clue: Full clue as a string.
    """
    if clue == "found_at":
        i_object = sample(range(n_objects), 1)[0]
        attribute = sample(chosen_attributes[i_object], 1)[0]
        attribute_desc = attributes[attribute]
        full_clue = "Personen der {attribute_desc} er ved hus nummer {i}.".format(
            attribute_desc=attribute_desc, i=i_object
        )
    else:
        raise ValueError("Unsupported clue '{clue}'")

    return full_clue


def generate_solution(
    attributes: Dict[str, Dict[str, str]], n_objects: int, n_attributes: int
) -> Tuple[List[List], List, List[List]]:
    """Generate the solution to a zebra puzzle.

    Args:
        attributes: Attributes as a dictionary of dictionaries.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        solution: A solution to a zebra puzzle as a list of lists representing the matrix of object indices and chosen attributes. This matrix is n_objects x n_attributes.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
    """
    # Choose a category for each attribute
    chosen_categories = sample(list(attributes.keys()), k=n_attributes)

    # Choose attribute values for each category
    chosen_attributes = [
        sample(list(attributes[cat].keys()), k=n_objects) for cat in chosen_categories
    ]

    # Transpose the attribute matrix
    chosen_attributes = [
        [row[i] for row in chosen_attributes] for i in range(n_attributes)
    ]

    solution = [[str(i)] + row for i, row in enumerate(chosen_attributes)]

    return solution, chosen_categories, chosen_attributes


def choose_clues(
    solution: List[List],
    clues: List,
    chosen_categories: List,
    chosen_attributes: List[List],
) -> str:
    """Generate a zebra puzzle.

    Args:
        solution: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        clues: Possible clues to include in the clues as a list of tuples. Each tuple contains the clue name and function. TODO: Edit this description when the clues are implemented.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.

    Returns:
        chosen_clues: Clues for the zebra puzzle as a string.

    TODO: Implement the generation of the clues.
    """
    chosen_clues = "1. This is an example. \n2. This is the second part of the example."
    return chosen_clues
