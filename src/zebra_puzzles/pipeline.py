"""Pipeline module for generating and saving zebra puzzles."""

from tqdm import tqdm

from zebra_puzzles.clue_selection import choose_clues
from zebra_puzzles.red_herring_selection import choose_red_herrings
from zebra_puzzles.zebra_utils import (
    complete_prompt,
    format_solution,
    generate_solution,
    prepare_data_folders,
    save_dataset,
)


def run_pipeline(
    n_objects: int,
    n_attributes: int,
    attributes: dict[str, dict[str, str]],
    clues_dict: dict[str, str],
    prompt_templates: list[str],
    prompt_and: str,
    n_red_herring_clues: int,
    red_herring_clues_dict: dict[str, str],
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, str],
) -> tuple[str, str]:
    """Run the pipeline to generate one zebra puzzle.

    Args:
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        attributes: Possible attributes as a dictionary of dictionaries.
        prompt_templates: List of templates for the prompt.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the puzzle as a list of strings.
        verbose: Option to print the prompt and solution as a boolean.
        eval: Option to evaluate the prompt as a boolean.

    Returns:
        A tuple (prompt, solution_str), where:
            prompt: The full prompt for the zebra puzzle as a string.
            solution_str: The solution as a string.

    TODO: Consider if enumeration should be removed when we only have one clue.
    """
    solution, chosen_categories, chosen_attributes, chosen_attributes_descs = (
        generate_solution(
            attributes=attributes, n_objects=n_objects, n_attributes=n_attributes
        )
    )

    chosen_clues = choose_clues(
        solution=solution,
        chosen_attributes=chosen_attributes,
        chosen_attributes_descs=chosen_attributes_descs,
        n_objects=n_objects,
        n_attributes=n_attributes,
        clues_dict=clues_dict,
    )

    chosen_red_herring_clues = choose_red_herrings(
        n_red_herring_clues=n_red_herring_clues,
        red_herring_clues_dict=red_herring_clues_dict,
        red_herring_attributes=red_herring_attributes,
        red_herring_facts=red_herring_facts,
        chosen_attributes=chosen_attributes,
        chosen_attributes_descs=chosen_attributes_descs,
        n_objects=n_objects,
        n_attributes=n_attributes,
    )

    prompt = complete_prompt(
        chosen_clues=chosen_clues,
        chosen_red_herring_clues=chosen_red_herring_clues,
        n_objects=n_objects,
        n_attributes=n_attributes,
        chosen_categories=chosen_categories,
        chosen_attributes=chosen_attributes,
        prompt_templates=prompt_templates,
        prompt_and=prompt_and,
    )

    solution_json = format_solution(solution=solution)

    return prompt, solution_json


def build_dataset(
    n_objects: int,
    n_attributes: int,
    attributes: dict[str, dict[str, str]],
    clues_dict: dict[str, str],
    prompt_templates: list[str],
    prompt_and: str,
    n_puzzles: int,
    theme: str,
    n_red_herring_clues: int,
    red_herring_clues_dict: dict[str, str],
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, str],
) -> None:
    """Build a dataset of zebra puzzles.

    Saves prompts and solutions in separate files in the data folder.

    Args:
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        attributes: Possible attributes as a dictionary of dictionaries.
        prompt_templates: List of templates for the prompt.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".
        n_puzzles: Number of puzzles to generate.
        theme: Theme of the puzzles.
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the puzzle as a list of strings.

    NOTE: Consider only saving the puzzle and solution instead of the whole prompt.
    """
    prompt_filenames, solution_filenames, puzzle_folder, solution_folder = (
        prepare_data_folders(
            n_puzzles=n_puzzles,
            theme=theme,
            n_objects=n_objects,
            n_attributes=n_attributes,
            n_red_herring_clues=n_red_herring_clues,
        )
    )

    for i in tqdm(
        range(n_puzzles),
        total=n_puzzles,
        desc="Building dataset",
        unit="puzzle",
        colour="#5599ff",
        ascii="░█",
    ):
        prompt, solution_json = run_pipeline(
            n_objects=n_objects,
            n_attributes=n_attributes,
            attributes=attributes,
            clues_dict=clues_dict,
            prompt_templates=prompt_templates,
            prompt_and=prompt_and,
            n_red_herring_clues=n_red_herring_clues,
            red_herring_clues_dict=red_herring_clues_dict,
            red_herring_attributes=red_herring_attributes,
            red_herring_facts=red_herring_facts,
        )
        save_dataset(data=prompt, filename=prompt_filenames[i], folder=puzzle_folder)
        save_dataset(
            data=solution_json, filename=solution_filenames[i], folder=solution_folder
        )
