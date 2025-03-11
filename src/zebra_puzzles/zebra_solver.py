"""Module for solving a zebra puzzle."""

import numpy as np


def solver(chosen_clues: list[str]) -> tuple[np.ndarray, float]:
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

    solution_attempt: np.ndarray = np.array([["0", "", ""], ["1", "", ""]])

    # Measure completeness of the solution
    completeness = 0

    return solution_attempt, completeness
