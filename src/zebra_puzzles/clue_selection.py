"""Module for selecting clues for a zebra puzzle."""

from random import sample
from typing import Dict, List

from zebra_puzzles.zebra_solver import solver
from zebra_puzzles.zebra_utils import complete_clue


def choose_clues(
    solution: List[List],
    clues: List,
    chosen_categories: List[str],
    chosen_attributes: List[List],
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
) -> List[str]:
    """Generate a zebra puzzle.

    If the solver identifies a different solution than the expected one, it will raise a warning and change the solution to the one found by the solver.

    Args:
        solution: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        clues: Possible clues to include in the clues as a list of tuples. Each tuple contains the clue name and function. TODO: Edit this description when the clues are implemented.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
        n_objects: Number of objects in the puzzle.
        attributes: Possible attributes as a dictionary of dictionaries.

    Returns:
        chosen_clues: Clues for the zebra puzzle as a string.

    TODO: Implement the generation of the clues.
    """
    solution_attempt: List[List] = []
    solved = False
    chosen_clues: List[str] = []
    while not solved:
        # Add a random clue

        new_clue = choose_random_clue(
            clues=clues,
            n_objects=n_objects,
            attributes=attributes,
            chosen_attributes=chosen_attributes,
            chosen_categories=chosen_categories,
        )

        # Try to solve the puzzle

        current_clues = chosen_clues + [new_clue]
        new_solution_attempt, completeness = solver(chosen_clues=current_clues)

        # Check if solution attempt has changed and if it has, save the clue
        if new_solution_attempt != solution_attempt:
            solution_attempt = new_solution_attempt
            chosen_clues.append(new_clue)

        # Check if the solution is complete. If it is, check if the solution attempt is the same as the solution

        if completeness == 1:
            solved = True
            if solution_attempt != solution:
                # Change the solution to the solution attempt and raise a warning
                solution = solution_attempt
                raise Warning(
                    "The solver has found a solution that is not the expected one: \nFound \n{solution_attempt} \nExpected \n{solution}"
                )

            # Try removing each clue and see if the solution is still found
            for i, clue in enumerate(chosen_clues):
                new_solution_attempt, completeness = solver(
                    chosen_clues=chosen_clues[:i] + chosen_clues[i + 1 :]
                )
                if new_solution_attempt == solution:
                    chosen_clues.pop(i)

        # TODO: Remove this after testing
        solved = True

    # TODO: Delete the following example when clues are chosen above
    chosen_clues = ["This is an example.", "This is the second part of the example."]

    return chosen_clues


def choose_random_clue(
    clues: List,
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
) -> str:
    """Choose a random clue from the list of possible clues.

    Args:
        clues: List of possible clues as strings.
        n_objects: Number of objects in the puzzle as an int.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution

    Returns:
        full_clue: Full clue as a string.

    #TODO: Change the output of complete_clue to reflect the needed input of the zebra solver.
    """
    clue = sample(clues, 1)[0]

    full_clue = complete_clue(
        clue=clue,
        n_objects=n_objects,
        attributes=attributes,
        chosen_attributes=chosen_attributes,
        chosen_categories=chosen_categories,
    )

    return full_clue
