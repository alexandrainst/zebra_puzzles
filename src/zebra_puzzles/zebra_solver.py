"""Module for solving a zebra puzzle.

Inspired by https://stackoverflow.com/questions/318888/solving-who-owns-the-zebra-programmatically
"""

from typing import List, Tuple


def solver(
    chosen_clues: List[str], chosen_attributes: List[List]
) -> Tuple[List[List], float]:
    """Solve a zebra puzzle.

    Args:
        chosen_clues: Clues for the zebra puzzle as a list of strings.
        new_clue: New clue to add to the solver as a string.
        chosen_attributes: Attribute values chosen for the solution. They should be sorted by category, but the order of attributes should be independent of the solution (random or sorted).

    Returns:
        solution_attempt: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        completeness: Completeness of the solution as a float.

    #TODO: Implement the solver
    """
    # Define the puzzle

    # Solve the puzzle

    solution_attempt: List[List] = [["0", "", ""], ["1", "", ""]]

    # Measure completeness of the solution
    completeness = 0

    return solution_attempt, completeness
