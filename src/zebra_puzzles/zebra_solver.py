"""Module for solving a zebra puzzle."""

from typing import List, Tuple


def solver(chosen_clues: List[str]) -> Tuple[List[List], float]:
    """Solve a zebra puzzle.

    Args:
        chosen_clues: Clues for the zebra puzzle as a list of strings.
        new_clue: New clue to add to the solver as a string.

    Returns:
        solution_attempt: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        completeness: Completeness of the solution as a float.

    #TODO: Implement the solver
    """
    # Solve the puzzle

    solution_attempt: List[List] = [["0", "", ""], ["1", "", ""]]

    # Measure completeness of the solution
    completeness = 0

    return solution_attempt, completeness
