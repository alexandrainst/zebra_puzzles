"""Pipeline module for generating and saving zebra puzzles."""

from tqdm import tqdm

from zebra_puzzles.file_utils import prepare_data_folders, save_dataset
from zebra_puzzles.puzzle_creation.clue_selection import choose_clues
from zebra_puzzles.puzzle_creation.prompt_completion import complete_prompt
from zebra_puzzles.puzzle_creation.red_herring_selection import choose_red_herrings
from zebra_puzzles.zebra_utils import (
    format_solution_as_json,
    generate_solution,
    shuffle_clues,
)


def run_pipeline(
    n_objects: int,
    n_attributes: int,
    attributes: dict[str, dict[str, list[str]]],
    clues_dict: dict[str, str],
    clue_weights: dict[str, float],
    clue_cases_dict: dict[str, list[str]],
    prompt_templates: list[str],
    prompt_and: str,
    prompt_replacements: dict[str, str],
    n_red_herring_clues: int,
    red_herring_clues_dict: dict[str, str],
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, list[str]],
    red_herring_clue_weights: dict[str, float],
    red_herring_cases_dict: dict[str, list[str]],
) -> tuple[str, str, str, str]:
    """Run the pipeline to generate one zebra puzzle.

    Generates a solution, chooses clues, and creates a prompt for the zebra puzzle.

    Args:
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        attributes: Possible attributes as a dictionary of dictionaries.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and descriptions of each clue type.
        clue_weights: Weights for clue selection as a dictionary containing a title and a weight for each clue type.
        clue_cases_dict: A dictionary containing the clue type as a key and a list of grammatical cases for clue attributes as values.
        prompt_templates: List of templates for the prompt.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".
        prompt_replacements: Dictionary of strings to replace in the prompt.
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the puzzle as a dictionary of fact titles and a list of description strings.
        red_herring_clue_weights: Weights for red herring clue selection as a dictionary containing a title and a weight for each clue type.
        red_herring_cases_dict: A dictionary containing the red herring clue type as a key and a list of grammatical cases for clue attributes as values.

    Returns:
        A tuple (prompt, solution_str, red_herring_indices_str, chosen_clue_types_str), where:
            prompt: The full prompt for the zebra puzzle as a string.
            solution_str: The solution as a string.
            red_herring_indices_str: String of comma-separated indices of the red herring clues in the shuffled list of clues.
            chosen_clue_types_str: String of comma-separated clue types chosen for the puzzle.

    NOTE: Consider if enumeration should be removed when we only have one clue.
    """
    solution, chosen_categories, chosen_attributes, chosen_attributes_descs = (
        generate_solution(
            attributes=attributes, n_objects=n_objects, n_attributes=n_attributes
        )
    )

    chosen_clues, chosen_clue_types = choose_clues(
        solution=solution,
        chosen_attributes=chosen_attributes,
        chosen_attributes_descs=chosen_attributes_descs,
        n_objects=n_objects,
        n_attributes=n_attributes,
        clues_dict=clues_dict,
        clue_weights=clue_weights,
        clue_cases_dict=clue_cases_dict,
    )

    chosen_red_herring_clues, chosen_red_herring_clue_types = choose_red_herrings(
        n_red_herring_clues=n_red_herring_clues,
        red_herring_clues_dict=red_herring_clues_dict,
        red_herring_attributes=red_herring_attributes,
        red_herring_facts=red_herring_facts,
        red_herring_clue_weights=red_herring_clue_weights,
        red_herring_cases_dict=red_herring_cases_dict,
        chosen_attributes=chosen_attributes,
        chosen_attributes_descs=chosen_attributes_descs,
        n_objects=n_objects,
        n_attributes=n_attributes,
    )

    chosen_clues, red_herring_indices_str, chosen_clue_types_str = shuffle_clues(
        chosen_clues=chosen_clues,
        chosen_red_herring_clues=chosen_red_herring_clues,
        chosen_clue_types=chosen_clue_types,
        chosen_red_herring_clue_types=chosen_red_herring_clue_types,
    )

    prompt = complete_prompt(
        chosen_clues=chosen_clues,
        n_objects=n_objects,
        n_attributes=n_attributes,
        chosen_categories=chosen_categories,
        chosen_attributes=chosen_attributes,
        prompt_templates=prompt_templates,
        prompt_and=prompt_and,
        prompt_replacements=prompt_replacements,
    )

    solution_json = format_solution_as_json(solution=solution)

    return prompt, solution_json, red_herring_indices_str, chosen_clue_types_str


def build_dataset(
    n_objects: int,
    n_attributes: int,
    attributes: dict[str, dict[str, list[str]]],
    clues_dict: dict[str, str],
    clue_weights: dict[str, float],
    clue_cases_dict: dict[str, list[str]],
    prompt_templates: list[str],
    prompt_and: str,
    prompt_replacements: dict[str, str],
    n_puzzles: int,
    theme: str,
    n_red_herring_clues: int,
    red_herring_clues_dict: dict[str, str],
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, list[str]],
    red_herring_clue_weights: dict[str, float],
    red_herring_cases_dict: dict[str, list[str]],
    data_folder_str: str,
) -> None:
    """Build a dataset of zebra puzzles.

    Generates a specified number of zebra puzzles. Saves prompts, solutions, indices to the red herring clues and clue types in separate files in the data folder.

    Args:
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        attributes: Possible attributes as a dictionary of dictionaries.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and descriptions of each clue type.
        clue_weights: Weights for clue selection as a dictionary containing a title and a weight for each clue type.
        clue_cases_dict: A dictionary containing the clue type as a key and a list of grammatical cases for clue attributes as values.
        prompt_templates: List of templates for the prompt.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".
        prompt_replacements: Dictionary of strings to replace in the prompt.
        n_puzzles: Number of puzzles to generate.
        theme: Theme of the puzzles.
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the puzzle as a dictionary of fact titles and a list of description strings.
        red_herring_clue_weights: Weights for red herring clue selection as a dictionary containing a title and a weight for each clue type.
        red_herring_cases_dict: A dictionary containing the red herring clue type as a key and a list of grammatical cases for clue attributes as values.
        data_folder_str: Folder to save the dataset in as a string.
    """
    (
        prompt_filenames,
        clue_type_filenames,
        red_herring_filenames,
        solution_filenames,
        puzzle_folder,
        clue_type_folder,
        red_herring_folder,
        solution_folder,
    ) = prepare_data_folders(
        n_puzzles=n_puzzles,
        theme=theme,
        n_objects=n_objects,
        n_attributes=n_attributes,
        n_red_herring_clues=n_red_herring_clues,
        data_folder_str=data_folder_str,
    )

    for i in tqdm(
        range(n_puzzles),
        total=n_puzzles,
        desc="Building dataset",
        unit="puzzle",
        colour="#5599ff",
        ascii="░█",
    ):
        prompt, solution_json, red_herring_indices_str, chosen_clue_types_str = (
            run_pipeline(
                n_objects=n_objects,
                n_attributes=n_attributes,
                attributes=attributes,
                clues_dict=clues_dict,
                clue_weights=clue_weights,
                clue_cases_dict=clue_cases_dict,
                prompt_templates=prompt_templates,
                prompt_and=prompt_and,
                prompt_replacements=prompt_replacements,
                n_red_herring_clues=n_red_herring_clues,
                red_herring_clues_dict=red_herring_clues_dict,
                red_herring_attributes=red_herring_attributes,
                red_herring_facts=red_herring_facts,
                red_herring_clue_weights=red_herring_clue_weights,
                red_herring_cases_dict=red_herring_cases_dict,
            )
        )
        save_dataset(data=prompt, filename=prompt_filenames[i], folder=puzzle_folder)
        save_dataset(
            data=solution_json, filename=solution_filenames[i], folder=solution_folder
        )
        save_dataset(
            data=red_herring_indices_str,
            filename=red_herring_filenames[i],
            folder=red_herring_folder,
        )
        save_dataset(
            data=chosen_clue_types_str,
            filename=clue_type_filenames[i],
            folder=clue_type_folder,
        )
