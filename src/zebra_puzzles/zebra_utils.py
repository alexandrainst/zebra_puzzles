"""Utility module for generating zebra puzzles."""

from random import sample


def generate_solution(
    attributes: dict[str, dict[str, str]], n_objects: int, n_attributes: int
) -> tuple[list[list], list, list[list]]:
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

    solution = [[str(i + 1)] + row for i, row in enumerate(chosen_attributes)]

    return solution, chosen_categories, chosen_attributes


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
    n_objects: int,
    chosen_categories: list[str],
    chosen_attributes: list[list],
    prompt_template: str,
) -> str:
    """Complete the prompt with the chosen clues.

    Assumes commas are used similarly across the supported languages.

    Args:
        chosen_clues: Chosen clues for the zebra puzzle as a list of strings.
        n_objects: Number of objects in the puzzle.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
        prompt_template: Template for the prompt.

    Returns:
        prompt: The full prompt for the zebra puzzle as a string.


    TODO: Improve the prompt here and in the config file.
    TODO: Add support for puzzles with a single category and/or object
    """
    chosen_clues = [f"{i + 1}. {clue}" for i, clue in enumerate(chosen_clues)]

    if len(chosen_clues) > 1:
        chosen_clues_str = "\n".join(chosen_clues)
    else:
        chosen_clues_str = chosen_clues[0]

    chosen_categories_part1 = ", ".join(chosen_categories[:-1])
    chosen_categories_part2 = chosen_categories[-1]

    # Transpose chosen_attributes
    chosen_attributes = list(map(list, zip(*chosen_attributes)))

    # Flatten chosen_attributes
    chosen_attributes_flat = [y for x in chosen_attributes for y in x]

    # Format chosen_attributes as a comma separated list
    chosen_attributes_part1 = ", ".join(chosen_attributes_flat[:-1])
    chosen_attributes_part2 = chosen_attributes_flat[-1]

    prompt = prompt_template.format(
        n_objects=n_objects,
        chosen_categories_part1=chosen_categories_part1,
        chosen_categories_part2=chosen_categories_part2,
        chosen_attributes_part1=chosen_attributes_part1,
        chosen_attributes_part2=chosen_attributes_part2,
        chosen_clues_str=chosen_clues_str,
    )
    return prompt
