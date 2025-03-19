"""Pipeline module for generating and saving zebra puzzles."""

from tqdm import tqdm

from zebra_puzzles.clue_selection import choose_clues
from zebra_puzzles.zebra_utils import (
    clean_folder,
    complete_prompt,
    format_solution,
    generate_solution,
    save_dataset,
)


def run_pipeline(
    n_objects: int,
    n_attributes: int,
    attributes: dict[str, dict[str, str]],
    clues_dict: dict[str, str],
    prompt_template: str,
    prompt_and: str,
    verbose=False,
    eval=False,
) -> tuple[str, str]:
    """Run the pipeline to generate one zebra puzzle.

    Args:
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        attributes: Possible attributes as a dictionary of dictionaries.
        prompt_template: Template for the prompt as a string.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".
        verbose: Option to print the prompt and solution as a boolean.
        eval: Option to evaluate the prompt as a boolean.

    Returns:
        A tuple (prompt, solution_str), where:
            prompt: The full prompt for the zebra puzzle as a string.
            solution_str: The solution as a string.

    TODO: Implement evaluation.
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

    prompt = complete_prompt(
        chosen_clues=chosen_clues,
        n_objects=n_objects,
        chosen_categories=chosen_categories,
        chosen_attributes=chosen_attributes,
        prompt_template=prompt_template,
        prompt_and=prompt_and,
    )

    solution_json = format_solution(solution=solution)

    if verbose:
        print("*** Prompt *** \n", prompt)
        print("*** Solution *** \n", solution_json)

    if eval:
        pass

    return prompt, solution_json


def build_dataset(
    n_objects: int,
    n_attributes: int,
    attributes: dict[str, dict[str, str]],
    clues_dict: dict[str, str],
    prompt_template: str,
    prompt_and: str,
    n_puzzles: int,
) -> None:
    """Build a dataset of zebra puzzles.

    Saves prompts and solutions in separate files in the data folder.

    Args:
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        attributes: Possible attributes as a dictionary of dictionaries.
        prompt_template: Template for the prompt.
        prompt_and: String to use for separating the last two elements in a list, e.g. "and".
        n_puzzles: Number of puzzles to generate.

    NOTE: Consider only saving the puzzle and solution instead of the whole prompt.
    """
    # Create data file names
    prompt_filenames = ["zebra_puzzle_{}.txt".format(i) for i in range(n_puzzles)]
    solution_filenames = [
        str(file.split(".")[0]) + "_solution.txt" for file in prompt_filenames
    ]

    data_filenames = prompt_filenames + solution_filenames

    # Clean data folder
    clean_folder(folder="data", keep_files=data_filenames)

    for i in tqdm(range(n_puzzles)):
        prompt, solution_json = run_pipeline(
            n_objects=n_objects,
            n_attributes=n_attributes,
            attributes=attributes,
            clues_dict=clues_dict,
            prompt_template=prompt_template,
            prompt_and=prompt_and,
            verbose=False,
            eval=False,
        )
        save_dataset(data=prompt, filename=data_filenames[i], folder="data")
        save_dataset(data=solution_json, filename=solution_filenames[i], folder="data")
