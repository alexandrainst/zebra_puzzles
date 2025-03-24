"""Utility module for generating and evaluating zebra puzzles."""

import os
from pathlib import Path
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


def clean_folder(folder: str, keep_files: list[str]) -> None:
    """Clean a folder by deleting outdated files.

    Creates the folder if it does not exist.

    Args:
        folder: Folder to clean.
        keep_files: List of files to keep in the folder.
    """
    # Create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # Get a list of files in the folder
    existing_files = os.listdir(folder)

    # Get a list of files to delete
    files_to_delete = [file for file in existing_files if file not in keep_files]

    if len(files_to_delete) > 0:
        useroutput = input(
            f"\nDo you want to delete the following outdated files in the folder '{folder}'?\n\n{files_to_delete}\n\n(y/n): "
        )
        if useroutput == "y":
            for file in files_to_delete:
                os.remove(os.path.join(folder, file))
            print("Old files were deleted.")
        else:
            print("Old files were not deleted.")


def save_dataset(data: str, filename: str, folder: str = "data") -> None:
    """Save a file.

    Args:
        data: Data to save.
        filename: Name of the file.
        folder: Folder to save the file in.

    """
    with open(folder + "/" + filename, "w") as file:
        file.write(data)


def prepare_data_folders(
    n_puzzles: int,
    theme: str,
    n_objects: int,
    n_attributes: int,
    n_red_herring_clues: int,
) -> tuple[list[str], list[str], str, str]:
    """Prepare the data folders for the dataset.

    Args:
        n_puzzles: Number of puzzles to generate.
        theme: Theme of the puzzles.
        n_objects: Number of objects a the puzzle.
        n_attributes: Number of attributes of each object.
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.

    Returns:
        A tuple (prompt_filenames, solution_filenames, puzzle_folder, solution_folder), where:
            prompt_filenames: List of prompt file names.
            solution_filenames: List of solution file names.
            puzzle_folder: Folder for the prompt files.
            solution_folder: Folder for the solution files

    """
    # Create data file names
    prompt_filenames = ["zebra_puzzle_{}.txt".format(i) for i in range(n_puzzles)]
    solution_filenames = [
        str(file.split(".")[0]) + "_solution.txt" for file in prompt_filenames
    ]

    # Define folders
    subfolder = f"{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh"
    puzzle_folder = f"data/{subfolder}/puzzles"
    solution_folder = f"data/{subfolder}/solutions"

    # Clean folders
    clean_folder(folder=puzzle_folder, keep_files=prompt_filenames)
    clean_folder(folder=solution_folder, keep_files=solution_filenames)

    return prompt_filenames, solution_filenames, puzzle_folder, solution_folder


def prepare_eval_folders(
    theme: str,
    n_objects: int,
    n_attributes: int,
    n_red_herring_clues: int,
    model: str,
    n_puzzles: int,
    generate_new_responses: bool,
) -> tuple[list[Path], list[Path], list[str], str, str, str]:
    """Prepare the folders for the evaluation.

    Args:
        theme: The theme of the puzzles.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        n_red_herring_clues: Number of red herring clues included in the puzzles as an integer.
        model: The model to use for the evaluation as a string.
        n_puzzles: Number of puzzles to evaluate as an integer.
        generate_new_responses: Whether to generate new responses or use existing ones.

    Returns:
        A tuple (puzzle_paths, solution_paths, response_filenames, response_folder, score_filename, score_folder), where:
            puzzle_paths: Paths to the puzzles.
            solution_paths: Paths to the solutions.
            response_filenames: Names of the response files.
            response_folder: Folder to save the responses in.
            score_filename: Name of the score file.
            score_folder: Folder to save the scores in.
    """
    # Define the subfolders for puzzles, solutions, responses, and evaluations
    puzzle_subfolder = f"{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh"
    eval_subfolder = f"{puzzle_subfolder}/{model}"

    # Get sorted names of all prompt files in the data folder
    puzzle_paths = sorted(list(Path(f"data/{puzzle_subfolder}/puzzles").glob("*.txt")))

    solution_paths = [
        puzzle_path.parent.parent.joinpath("solutions") for puzzle_path in puzzle_paths
    ]

    # Create reponse file names
    response_filenames = [
        f"{file_path.stem}_response.json" for file_path in puzzle_paths
    ]

    score_filename = f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_red_herring_clues}_rh_{n_puzzles}_puzzles.txt"

    # Define evaluation folders
    response_folder = f"responses/{eval_subfolder}"
    score_folder = f"scores/{eval_subfolder}"

    if generate_new_responses:
        # Clean or create reponses folder
        clean_folder(folder=response_folder, keep_files=response_filenames)

    # Create the score folder if it does not exist
    os.makedirs(score_folder, exist_ok=True)

    return (
        puzzle_paths,
        solution_paths,
        response_filenames,
        response_folder,
        score_filename,
        score_folder,
    )


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

    # Create a solution template
    solution_template = create_solution_template(
        n_objects=n_objects, chosen_categories=chosen_categories
    )
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
        solution_template=solution_template,
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


def format_solution(solution: np.ndarray) -> str:
    """Format the solution as a json dictionary.

    Args:
        solution: Solution to the zebra puzzle as a matrix of object indices and chosen attributes.

    Returns:
        The solution as a string representing a json dictionary
    """
    solution_json = "{\n"

    for row in solution.astype(str):
        row_object = row[0]
        row_attributes = '", "'.join(row[1:])
        solution_json += f'"object_{row_object}": ["{row_attributes}"],\n'

    solution_json += "}"

    # Delete last comma
    solution_json = solution_json.replace(",\n}", "\n}")

    return solution_json


def create_solution_template(n_objects: int, chosen_categories: np.ndarray) -> str:
    """Create a solution template for a zebra puzzle.

    For example:
    {
    "object_1": ["attribute_1", "attribute_2"],
    "object_2": ["attribute_1", "attribute_2"]
    }


    Args:
        n_objects: Number of objects in the puzzle.
        chosen_categories: Categories chosen for the solution.

    Returns:
        The solution template as a string.
    """
    example_solution = np.zeros((n_objects, len(chosen_categories) + 1), dtype="U100")
    for i in range(n_objects):
        example_solution[i, 0] = f"{i + 1}"
        for j, cat in enumerate(chosen_categories):
            example_solution[i, j + 1] = f"{cat}_{i + 1}"

    solution_template = format_solution(example_solution)

    return solution_template
