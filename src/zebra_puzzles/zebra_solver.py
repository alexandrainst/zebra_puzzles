"""Module for solving a zebra puzzle."""

from typing import Dict, List, Tuple

from constraint import AllDifferentConstraint, OptimizedBacktrackingSolver, Problem


def solver(
    constraints: List[Tuple], chosen_attributes: List[List], n_objects: int
) -> Tuple[List[Dict[str, int]], float]:
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
    # ---- Define the puzzle ----#
    solver = OptimizedBacktrackingSolver()
    problem = Problem(solver)

    # Define attributes
    chosen_attributes_flat = [y for x in chosen_attributes for y in x]
    problem.addVariables(chosen_attributes_flat, list(range(1, n_objects + 1)))

    # All properties must be unique
    for attributes_in_category in chosen_attributes:
        problem.addConstraint(AllDifferentConstraint(), attributes_in_category)

    # Add cluesSolve
    for constraint, constraint_var in constraints:
        problem.addConstraint(constraint, constraint_var)

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


def format_solution(solution_dict: Dict[str, int], n_objects: int) -> List[List]:
    """Change solution format from dict to list.

    Args:
        solution_dict: Solution as a dict of attributues and which object they are connected to.
        n_objects: Number of objects in the puzzle.

    Return:
        solution_list: Solution as a list of lists.
    """
    solution_list = []
    for i_object in range(1, n_objects + 1):
        solution_list.append(
            [str(i_object)] + [k for k, v in solution_dict.items() if v == i_object]
        )

    return solution_list
