"""Utility module for generating zebra puzzles."""

from random import sample
from typing import Dict, List, Tuple


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


def define_clues(clues_included: str) -> List:
    """Define clue types for the puzzle.

    Args:
        clues_included: A string descriping which clue types to include.

    Returns:
        clues: List of included clue types.

    NOTE: In the future, we can support more clues and selection of clues in the config file.
    TODO: Implement clue functions. not_at, next_to, not_next_to, left_of, right_of, not_left_of, not_right_of, same_house, not_same_house, between, not_between
    """
    if clues_included == "all":
        clues = ["found_at", "not_at"]
    else:
        raise ValueError("Unsupported clues '{clues_included}'")

    return clues


def complete_clue(
    clue: str,
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
) -> str:
    """Complete the chosen clue type with random parts of the solution to create a full clue.

    TODO: Consider how the clues will be evaluted. We should probably save more than a string.
    TODO: Move the clue descriptions to the config file.

    Args:
        clue: Chosen clue type as a string.
        n_objects: Number of objects in the puzzle as an int.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution.

    Returns:
        full_clue: Full clue as a string.
    """
    if clue == "found_at":
        # Choose a random object
        i_object = sample(range(n_objects), 1)[0]

        # Choose a random attribute and the corresponding category
        i_attribute, attribute = sample(
            list(enumerate(chosen_attributes[i_object])), 1
        )[0]
        chosen_category = chosen_categories[i_attribute]

        # Get the attribute description
        attribute_desc = attributes[chosen_category][attribute]

        # Create the full clue
        full_clue = "Personen der {attribute_desc} er ved hus nummer {i}.".format(
            attribute_desc=attribute_desc, i=i_object
        )
    elif clue == "not_at":
        full_clue = "This is an example clue of type not_at."
    else:
        raise ValueError("Unsupported clue '{clue}'")

    return full_clue


def save_dataset(data: str, filename: str, folder: str = "data") -> None:
    """Save a zebra puzzle dataset.

    Args:
        data: Data to save.
        filename: Name of the file.
        folder: Folder to save the file in.

    TODO: Consider preferred format.
    """
    with open(folder + "/" + filename, "w") as file:
        file.write(data)
