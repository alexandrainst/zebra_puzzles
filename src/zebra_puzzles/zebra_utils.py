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
        chosen_categories: Categories chosen for the solution as a list.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
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


def complete_clue(
    clue: str,
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
    clues_dict: Dict[str, str],
) -> str:
    """Complete the chosen clue type with random parts of the solution to create a full clue.

    TODO: Consider how the clues will be evaluted. We should probably include more information in the dict such as a lambda function.
    TODO: Include more clue types. For example not_at, next_to, not_next_to, left_of, right_of, not_left_of, not_right_of, same_house, not_same_house, between, not_between
    NOTE: The current implementation does not allow objects to have non-unique attributes

    Args:
        clue: Chosen clue type as a string.
        n_objects: Number of objects in the puzzle as an int.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.

    Returns:
        full_clue: Full clue as a string.
    """
    clue_description = clues_dict[clue]

    if clue == "found_at":
        # Choose a random object
        i_object = sample(list(range(n_objects)), 1)[0]

        # Choose an attribute
        attribute_desc = describe_random_attribute(
            attributes=attributes,
            chosen_attributes=chosen_attributes,
            chosen_categories=chosen_categories,
            i_object=i_object,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc=attribute_desc, i_object=i_object
        )
    elif clue == "not_at":
        # Choose two random objects - one for the attribute and one not connected to this attribute
        i_object, i_other_object = sample(list(range(n_objects)), 2)

        # Choose an attribute of the first object
        attribute_desc = describe_random_attribute(
            attributes=attributes,
            chosen_attributes=chosen_attributes,
            chosen_categories=chosen_categories,
            i_object=i_object,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc=attribute_desc, i_other_object=i_other_object
        )
    else:
        raise ValueError("Unsupported clue '{clue}'")

    return full_clue


def describe_random_attribute(
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
    i_object: int,
) -> str:
    """Choose a random attribute.

    Consider replacing this function by an array of chosen attribute descriptions or making chosen_attributes a dict.

    Args:
        attributes: Attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution.
        i_object: The index of the object to select an attribute from.

    Returns:
        attribute_desc: A string using the attribute to describe an object.

    """
    # Choose a random attribute and the corresponding category
    i_attribute, attribute = sample(list(enumerate(chosen_attributes[i_object])), 1)[0]

    chosen_category = chosen_categories[i_attribute]

    # Get the attribute description
    attribute_desc = attributes[chosen_category][attribute]

    return attribute_desc


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


def complete_prompt(
    chosen_clues: List[str],
    n_objects: int,
    chosen_categories: List[str],
    chosen_attributes: List[List],
    prompt_template: str,
) -> str:
    """Complete the prompt with the chosen clues.

    Args:
        chosen_clues: Chosen clues for the zebra puzzle as a list of strings.
        n_objects: Number of objects in the puzzle.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
        prompt_template: Template for the prompt.

    Returns:
        prompt: The full prompt for the zebra puzzle as a string.
    """
    chosen_clues = [f"{i + 1}. {clue}" for i, clue in enumerate(chosen_clues)]

    if len(chosen_clues) > 1:
        chosen_clues_str = "\n".join(chosen_clues)
    else:
        chosen_clues_str = chosen_clues[0]

    prompt = prompt_template.format(
        n_objects=n_objects,
        chosen_categories=chosen_categories,
        chosen_attributes=chosen_attributes,
        chosen_clues_str=chosen_clues_str,
    )
    return prompt
