"""Pipeline module for generating and saving zebra puzzles."""

from typing import Tuple

from zebra_puzzles.zebra_utils import (
    complete_prompt,
    define_attributes,
    define_rules,
    generate_puzzle,
    generate_solution,
)


def run_pipeline(
    theme, language, rules_included, N_objects, N_attributes, verbose=False, eval=False
) -> Tuple[str, str]:
    """Run the pipeline to generate one zebra puzzle.

    Args: verbose and eval flags. If verbose, print the prompt and solution. If eval, evaluate the prompt.

    Returns: A prompt and solution as strings.

    TODO: Implement evaluation.
    """
    attributes = define_attributes(theme, language)
    rules = define_rules(rules_included)
    solution, chosen_categories, chosen_attributes = generate_solution(
        attributes, N_objects, N_attributes
    )
    puzzle = generate_puzzle(solution, rules, chosen_categories, chosen_attributes)
    prompt = complete_prompt(
        language, theme, puzzle, chosen_attributes, chosen_categories, N_objects
    )

    print("solution", solution)

    solution_str = "\n".join([" ".join(row) for row in solution])

    if verbose:
        print("*** Prompt *** \n", prompt)
        print("*** Solution *** \n", solution_str)

    if eval:
        pass

    return prompt, solution_str


def save_dataset(data, filename, folder="data") -> None:
    """Save a zebra puzzle dataset.

    Args: Data to save, filename, and folder.
    TODO: Consider preferred format.
    """
    with open(folder + "/" + filename, "w") as file:
        file.write(data)


def build_dataset(
    theme, language, rules_included, N_objects, N_attributes, N_puzzles
) -> None:
    """Build a dataset of zebra puzzles.

    Saves prompts and solutions in separate files in the data folder.

    Args: Number of puzzles to generate.

    NOTE: Consider only saving the puzzle and solution instead of the whole prompt.
    """
    for i in range(N_puzzles):
        prompt, solution_str = run_pipeline(
            theme,
            language,
            rules_included,
            N_objects,
            N_attributes,
            verbose=False,
            eval=False,
        )
        save_dataset(prompt, filename="zebra_puzzle_{}.txt".format(i), folder="data")
        save_dataset(
            solution_str,
            filename="zebra_puzzle_{}_solution.txt".format(i),
            folder="data",
        )
