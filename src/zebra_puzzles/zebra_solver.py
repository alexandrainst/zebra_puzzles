"""Module for solving a zebra puzzle."""

import numpy as np
from constraint import AllDifferentConstraint, OptimizedBacktrackingSolver, Problem


def solver(
    constraints: list[tuple], chosen_attributes: np.ndarray, n_objects: int
) -> tuple[list[dict[str, int]], float]:
    """Solve a zebra puzzle.

    Args:
        constraints: Constraints for the zebra puzzle as a list of tuples. Each constaint corresponds to one clue. Each tuple (constraint_function, variables) contains:
            constraint_function: A constraint function that the variables must satisfy.
            variables: Attributes that the constraint applies to.
        chosen_attributes: Attribute values chosen for the solution. They should be sorted by category, but the order of attributes should be independent of the solution (random or sorted).
        n_objects: Number of objects in the puzzle.

    Returns:
        A tuple (solution_attempt, completeness), where:
            solution_attempt: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
            completeness: Completeness of the solution as a float.
    # NOTE: We could remove the uniqueness constraint
    """
    # ---- Define the puzzle ----#
    solver = OptimizedBacktrackingSolver()
    problem = Problem(solver)

    # Define attributes
    chosen_attributes_flat = chosen_attributes.flatten().tolist()
    problem.addVariables(chosen_attributes_flat, list(range(1, n_objects + 1)))

    # All properties must be unique
    for attributes_in_category in chosen_attributes.tolist():
        problem.addConstraint(AllDifferentConstraint(), attributes_in_category)

    # Add clues
    for constraint, constraint_var in constraints:
        problem.addConstraint(constraint, constraint_var.tolist())

    # ---- Solve the puzzle ----#
    solutions = problem.getSolutions()

    # Measure completeness of the solution.
    # NOTE: This can be improved by measuring the overlap between all found solutions

    if len(solutions) > 0:
        completeness = 1.0 / float(len(solutions))
    else:
        print("solutions:", solutions)
        print("constraints:", constraints)
        raise ValueError("This puzzle has no solution")

    return solutions, completeness


def format_solution(
    solution_dict: dict[str, int], n_objects: int, n_attributes: int
) -> np.ndarray:
    """Change solution format from dict to list.

    Assumes the maximum string length is 100 characters.

    Args:
        solution_dict: Solution as a dict of attributes and which object they are connected to.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        Solution as a matrix in a numpy array.
    """
    solution_list = np.empty((n_objects, n_attributes + 1), dtype="U100")
    for i_object in range(1, n_objects + 1):
        solution_list[i_object - 1, :] = [str(i_object)] + [
            k for k, v in solution_dict.items() if v == i_object
        ]

    return solution_list
