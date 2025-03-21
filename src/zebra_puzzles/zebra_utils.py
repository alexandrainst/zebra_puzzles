"""Utility module for generating zebra puzzles."""

from random import sample, shuffle

import numpy as np


def generate_solution(
    attributes: dict[str, dict[str, str]], n_objects: int, n_attributes: int
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
    chosen_clues: list[str],
    chosen_red_herring_clues: list[str],
    n_objects: int,
    n_attributes: int,
    chosen_categories: np.ndarray,
    chosen_attributes: np.ndarray,
    prompt_templates: list[str],
    prompt_and: str,
) -> str:
    """Complete the prompt with the chosen clues.

    Formats the clues as a numbered list with each hint starting with a capital letter. This is combined with the prompt template from the config file to create the full prompt.

    Assumes commas are used similarly across the supported languages.
    Assumes each hint should start with a capital letter.

    Args:
        chosen_clues: Chosen clues for the zebra puzzle as a list of strings.
        chosen_red_herring_clues: Chosen red herring clues for the zebra puzzle as a list of strings.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
        prompt_templates: List of templates for the prompt.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".

    Returns:
        The full prompt for the zebra puzzle as a string.

    """
    # Mix and shuffle clues and red herrings
    chosen_clues = chosen_clues + chosen_red_herring_clues
    shuffle(chosen_clues)

    # Format clues
    if len(chosen_clues) > 1:
        # Format chosen_clues as a numbered list
        chosen_clues = [
            f"{i + 1}. {clue[0].upper()}{clue[1:]}"
            for i, clue in enumerate(chosen_clues)
        ]
    else:
        chosen_clues = [f"{clue[0].upper()}{clue[1:]}" for clue in chosen_clues]

    if len(chosen_clues) > 1:
        chosen_clues_str = "\n".join(chosen_clues)
    else:
        chosen_clues_str = chosen_clues[0]

    # Format chosen_categories as a comma separated list
    chosen_categories_str = format_list_in_prompt(
        list_of_strings=chosen_categories, prompt_and=prompt_and, oxford_comma=False
    )

    # Transpose chosen_attributes
    chosen_attributes = chosen_attributes.T

    # Sort the attributes
    chosen_attributes = np.array([sorted(x) for x in chosen_attributes])

    # Comma seprate the attributes in each category and combine with the category title
    chosen_attributes_strs = [
        f"{cat}: {format_list_in_prompt(list_of_strings=chosen_attributes[i], prompt_and=prompt_and, oxford_comma=False)}"
        for i, cat in enumerate(chosen_categories)
    ]

    if n_attributes > 1:
        # Use uppercase for the first letter of each attribute string
        chosen_attributes_strs = [
            f"{x[0].upper()}{x[1:]}." for x in chosen_attributes_strs
        ]

    # Combine the attribute strings
    chosen_attributes_str = "\n".join(chosen_attributes_strs)

    # Choose a prompt template
    if n_attributes > 1:
        prompt_template = prompt_templates[0]
    else:
        prompt_template = prompt_templates[1]

    # Combine the prompt template with puzzle information
    prompt = prompt_template.format(
        n_objects=n_objects,
        chosen_categories_str=chosen_categories_str,
        chosen_attributes_str=chosen_attributes_str,
        chosen_clues_str=chosen_clues_str,
    )
    return prompt


def format_list_in_prompt(
    list_of_strings: np.ndarray, prompt_and: str, oxford_comma: bool = False
):
    """Format a list for a prompt.

    Args:
        list_of_strings: Array of strings to format.
        prompt_and: String to use for separating the last two elements, e.g. "and".
        oxford_comma: Option to include an Oxford comma.

    Returns:
        Formatted list as a string.
    """
    if len(list_of_strings) == 1:
        formatted_list = list_of_strings[0]
    elif len(list_of_strings) == 2:
        formatted_list = f"{list_of_strings[0]} {prompt_and} {list_of_strings[1]}"
    else:
        formatted_list = ", ".join(list_of_strings[:-1])
        if oxford_comma:
            formatted_list += ", "
        formatted_list += f" {prompt_and} {list_of_strings[-1]}"

    return formatted_list
