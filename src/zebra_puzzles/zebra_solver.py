"""Module for solving a zebra puzzle.

Inspired by https://stackoverflow.com/questions/318888/solving-who-owns-the-zebra-programmatically
"""

from typing import List, Tuple

from constraint import AllDifferentConstraint, Problem


def solver(
    constraints: List[Tuple], chosen_attributes: List[List], n_objects: int
) -> Tuple[List[List], float]:
    """Solve a zebra puzzle.

    Args:
        constraints: Constraints for the zebra puzzle as a list of Tuples. Each tuple contains a constraint function and a list of directly affected variables. Each constaint corresponds to one clue.
        chosen_attributes: Attribute values chosen for the solution. They should be sorted by category, but the order of attributes should be independent of the solution (random or sorted).
        n_objects: Number of objects in the puzzle.

    Returns:
        solution_attempt: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        completeness: Completeness of the solution as a float.

    # TODO: Implement the solver
    # NOTE: We could remove the uniqueness constraint
    """
    # Define the puzzle
    problem = Problem()

    # Flatten attributes
    chosen_attributes_flat = [y for x in chosen_attributes for y in x]

    problem.addVariables(chosen_attributes_flat, range(1, n_objects))

    # All properties must be unique
    for vars_ in chosen_attributes:
        problem.addConstraint(AllDifferentConstraint(), vars_)

    # Add clues
    for constraint, constraint_var in constraints:
        problem.addConstraint(constraint, constraint_var)

    # Solve the puzzle
    solution_attempt: List[List] = [["0", "", ""], ["1", "", ""]]

    # Measure completeness of the solution
    completeness = 0

    return solution_attempt, completeness
