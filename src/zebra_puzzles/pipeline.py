"""Pipeline module for generating and saving zebra puzzles."""

from typing import Dict, Tuple

from zebra_puzzles.clue_selection import choose_clues
from zebra_puzzles.zebra_utils import complete_prompt, generate_solution, save_dataset


def run_pipeline(
    n_objects: int,
    n_attributes: int,
    attributes: Dict[str, Dict[str, str]],
    clues_dict: Dict[str, str],
    prompt_template: str,
    verbose=False,
    eval=False,
) -> Tuple[str, str]:
    """Run the pipeline to generate one zebra puzzle.

    Args:
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        attributes: Possible attributes as a dictionary of dictionaries.
        prompt_template: Template for the prompt.
        verbose: Print the prompt and solution.
        eval: Evaluate the prompt.

    Returns:
        A tuple (prompt, solution_str) with the prompt and the solution as a string.

    TODO: Implement evaluation.
    TODO: Consider if enumeration should be removed when we only have one clue.
    TODO: Consider using the clue descriptions in complete_prompt() only.
    """
    # clues = define_clues(clues_included=clues_included)

    solution, chosen_categories, chosen_attributes = generate_solution(
        attributes=attributes, n_objects=n_objects, n_attributes=n_attributes
    )
    chosen_clues = choose_clues(
        solution=solution,
        chosen_categories=chosen_categories,
        chosen_attributes=chosen_attributes,
        n_objects=n_objects,
        attributes=attributes,
        clues_dict=clues_dict,
    )

    prompt = complete_prompt(
        chosen_clues=chosen_clues,
        n_objects=n_objects,
        chosen_categories=chosen_categories,
        chosen_attributes=chosen_attributes,
        prompt_template=prompt_template,
    )

    solution_str = "\n".join([" ".join(row) for row in solution])

    if verbose:
        print("*** Prompt *** \n", prompt)
        print("*** Solution *** \n", solution_str)

    if eval:
        pass

    return prompt, solution_str


def build_dataset(
    n_objects: int,
    n_attributes: int,
    attributes: Dict[str, Dict[str, str]],
    clues_dict: Dict[str, str],
    prompt_template: str,
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
        n_puzzles: Number of puzzles to generate.

    NOTE: Consider only saving the puzzle and solution instead of the whole prompt.
    """
    for i in range(n_puzzles):
        prompt, solution_str = run_pipeline(
            n_objects=n_objects,
            n_attributes=n_attributes,
            attributes=attributes,
            clues_dict=clues_dict,
            prompt_template=prompt_template,
            verbose=False,
            eval=False,
        )
        save_dataset(
            data=prompt, filename="zebra_puzzle_{}.txt".format(i), folder="data"
        )
        save_dataset(
            data=solution_str,
            filename="zebra_puzzle_{}_solution.txt".format(i),
            folder="data",
        )
