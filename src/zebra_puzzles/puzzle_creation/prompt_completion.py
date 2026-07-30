"""Module for completing the puzzle prompts."""

import numpy as np

from zebra_puzzles.zebra_utils import capitalize, create_solution_template


def complete_prompt(
    chosen_clues: list[str],
    n_objects: int,
    n_attributes: int,
    chosen_categories: np.ndarray,
    chosen_attributes: np.ndarray,
    prompt_templates: list[str],
    prompt_and: str,
    prompt_replacements: dict[str, str],
    prompt_comma: str = ", ",
    prompt_and_spacing: bool = True,
    prompt_full_stop: str = ".",
    prompt_colon: str = ": ",
) -> str:
    """Complete the prompt with the chosen clues.

    Formats the clues as a numbered list with each hint starting with a capital letter. This is combined with the prompt template from the config file to create the full prompt.

    Assumes each hint should start with a capital letter.

    Args:
        chosen_clues: Chosen clues incl. red herrings for the zebra puzzle as a list of strings.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
        prompt_templates: List of templates for the prompt.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".
        prompt_replacements: Dictionary of strings to replace in the prompt.
        prompt_comma: String to use for separating non-final elements in a list, e.g. ", ".
        prompt_and_spacing: Whether to surround prompt_and with spaces when joining a list. Languages
            that do not put spaces between words (e.g. Chinese) should set this to False.
        prompt_full_stop: Sentence-ending punctuation appended to each category's attribute list line,
            e.g. "." or "。".
        prompt_colon: Separator between a category title and its attribute list, e.g. ": " or "：".

    Returns:
        The full prompt for the zebra puzzle as a string.
    """
    # Format clues
    if len(chosen_clues) > 1:
        # Format chosen_clues as a numbered list
        chosen_clues = [
            f"{i + 1}. {capitalize(clue)}" for i, clue in enumerate(chosen_clues)
        ]
    else:
        chosen_clues = [f"{capitalize(clue)}" for clue in chosen_clues]

    if len(chosen_clues) > 1:
        chosen_clues_str = "\n".join(chosen_clues)
    else:
        chosen_clues_str = chosen_clues[0]

    # Format chosen_categories as a comma separated list
    chosen_categories_str = format_list_in_prompt(
        list_of_strings=chosen_categories,
        prompt_and=prompt_and,
        oxford_comma=False,
        prompt_comma=prompt_comma,
        prompt_and_spacing=prompt_and_spacing,
    )

    # Transpose chosen_attributes
    chosen_attributes = chosen_attributes.T

    # Sort the attributes
    chosen_attributes = np.array([sorted(x) for x in chosen_attributes])

    # Comma separate the attributes in each category and combine with the category title
    chosen_attributes_strs = [
        f"{cat}{prompt_colon}{format_list_in_prompt(list_of_strings=chosen_attributes[i], prompt_and=prompt_and, oxford_comma=False, prompt_comma=prompt_comma, prompt_and_spacing=prompt_and_spacing)}"
        for i, cat in enumerate(chosen_categories)
    ]

    if n_attributes > 1:
        # Use uppercase for the first letter of each attribute string
        chosen_attributes_strs = [
            f"{capitalize(x)}{prompt_full_stop}" for x in chosen_attributes_strs
        ]

    # Combine the attribute strings
    chosen_attributes_str = "\n".join(chosen_attributes_strs)

    # Create a solution template
    solution_template = create_solution_template(
        n_objects=n_objects, chosen_categories=chosen_categories
    )
    # Choose a prompt template
    if n_attributes > 1:
        prompt_template = prompt_templates[0]
    else:
        prompt_template = prompt_templates[1]

    # Insert a sentinel in place of the solution template so that prompt_replacements
    # are applied only to the narrative text, not the JSON example.
    _sentinel = "\x00"
    prompt = prompt_template.format(
        n_objects=n_objects,
        chosen_categories_str=chosen_categories_str,
        chosen_attributes_str=chosen_attributes_str,
        chosen_clues_str=chosen_clues_str,
        solution_template=_sentinel,
    )

    # Replace text in the prompt (solution template excluded)
    for key, value in prompt_replacements.items():
        prompt = prompt.replace(key, value)

    # Insert the solution template
    prompt = prompt.replace(_sentinel, solution_template)

    return prompt


def format_list_in_prompt(
    list_of_strings: np.ndarray,
    prompt_and: str,
    oxford_comma: bool = False,
    prompt_comma: str = ", ",
    prompt_and_spacing: bool = True,
):
    """Format a list for a prompt.

    Args:
        list_of_strings: Array of strings to format.
        prompt_and: String to use for separating the last two elements, e.g. "and".
        oxford_comma: Option to include an Oxford comma.
        prompt_comma: String to use for separating non-final elements, e.g. ", ".
        prompt_and_spacing: Whether to surround prompt_and with spaces. Languages that do not put
            spaces between words (e.g. Chinese) should set this to False.

    Returns:
        Formatted list as a string.
    """
    and_str = f" {prompt_and} " if prompt_and_spacing else prompt_and

    if len(list_of_strings) == 1:
        formatted_list = list_of_strings[0]
    elif len(list_of_strings) == 2:
        formatted_list = f"{list_of_strings[0]}{and_str}{list_of_strings[1]}"
    else:
        formatted_list = prompt_comma.join(list_of_strings[:-1])
        if oxford_comma:
            formatted_list += prompt_comma
        formatted_list += f"{and_str}{list_of_strings[-1]}"

    return formatted_list
