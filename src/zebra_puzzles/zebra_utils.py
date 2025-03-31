"""Utility module for generating and evaluating zebra puzzles."""

import json
from random import choices, sample, shuffle
from typing import Any, Type

import numpy as np
from pydantic import BaseModel, create_model


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
        A tuple (chosen_clues, i_red_herrings_str, chosen_clue_types_str), where:
            chosen_clues: Shuffled clues for the zebra puzzle as a list of strings incl. red herrings.
            i_red_herrings_str: String of indices of the red herrings in the shuffled list of clues.
            chosen_clue_types_str: Shuffled clue types for the zebra puzzle as a string.

    """
    # Combine clues and red herrings
    chosen_clues = chosen_clues + chosen_red_herring_clues
    chosen_clue_types = chosen_clue_types + chosen_red_herring_clue_types

    # Shuffle the clues and red herrings
    i_shuffled = list(range(len(chosen_clues)))
    shuffle(i_shuffled)
    chosen_clues = [chosen_clues[i] for i in i_shuffled]
    chosen_clue_types = [chosen_clue_types[i] for i in i_shuffled]

    # Get the indices of the red herrings
    i_red_herrings = [
        i for i in i_shuffled if i >= len(chosen_clues) - len(chosen_red_herring_clues)
    ]
    i_red_herrings_str = ", ".join([str(i) for i in i_red_herrings])

    chosen_clue_types_str = ", ".join(chosen_clue_types)

    return chosen_clues, i_red_herrings_str, chosen_clue_types_str
