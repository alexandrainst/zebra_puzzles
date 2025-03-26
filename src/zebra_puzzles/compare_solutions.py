"""Module for comparing a response to the solution."""

import itertools

from pydantic import BaseModel


def compare_solutions(
    output: BaseModel, solution: BaseModel, n_objects: int, n_attributes: int
) -> tuple[int, float, float]:
    """Compare the output to the solution.

    The puzzle score is 1 for a correct solution and 0 for an incorrect solution.
    The cell score is the proportion of cells that are correct.
    The best permuted cell score is the best cell score after trying all permutations of the objects in the response. This will give a high score if the LLM coupled the attributes correctly, but misunderstood the order of the objects.

    Args:
        output: The output in OutputFormat format.
        solution: The solution in OutputFormat format.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        A tuple (puzzle_score, cell_score), where:
            puzzle_score: A puzzle-level score as an integer.
            cell_score: A cell-level score as a float.
            best_permuted_cell_score: The best cell-level score as a float after trying all permutations of the objects in the response.
    """
    # Convert the output and solution to dictionaries
    output_dict = dict(output)
    solution_dict = dict(solution)

    # Compare the full output to the solution

    if output_dict == solution_dict:
        puzzle_score = 1
        cell_score = 1.0
        best_permuted_cell_score = 1.0
    else:
        # Compare all cells
        cell_score = compute_cell_score(
            output=output_dict,
            solution=solution_dict,
            n_objects=n_objects,
            n_attributes=n_attributes,
        )

        # Check if the puzzle is solved after stripping whitespace in cells
        if cell_score == 1:
            puzzle_score = 1
            best_permuted_cell_score = 1.0
        else:
            puzzle_score = 0

            # Evaluate every permutation of the objects in the response
            best_permuted_cell_score = compute_best_permuted_cell_score(
                output=output_dict,
                solution=solution_dict,
                n_objects=n_objects,
                n_attributes=n_attributes,
            )

    return puzzle_score, cell_score, best_permuted_cell_score


def compute_cell_score(
    output: dict[str, list],
    solution: dict[str, list],
    n_objects: int,
    n_attributes: int,
) -> float:
    """Compute the cell score.

    Args:
        output: The output as a dictionary of objects and their attributes.
        solution: The solution as a dictionary of objects and their attributes.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        The cell score as a float.
    """
    # Compare each cell
    cell_score: float = 0.0
    for attributes_output, attributes_solution in zip(
        output.values(), solution.values()
    ):
        for attribute_output, attribute_solution in zip(
            attributes_output, attributes_solution
        ):
            if attribute_output.strip() == attribute_solution.strip():
                cell_score += 1.0

    # Normalise the cell score
    cell_score /= float(n_objects * n_attributes)

    return cell_score


def compute_best_permuted_cell_score(
    output: dict[str, list],
    solution: dict[str, list],
    n_objects: int,
    n_attributes: int,
) -> float:
    """Compute the best permuted cell score.

    Args:
        output: The output as a dictionary of objects and their attributes.
        solution: The solution as a dictionary of objects and their attributes.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        The best permuted cell score as a float.
    """
    best_permuted_cell_score = 0.0
    objects = list(output.keys())

    # Create all permutations of the objects where each object appears exactly once

    object_permutations = list(itertools.permutations(objects))

    # Evaluate each permutation
    for object_permutation in object_permutations:
        # Create a new output with the objects permuted
        output_permuted = {object: output[object] for object in object_permutation}

        # Compare the permuted output to the solution
        permuted_cell_score = compute_cell_score(
            output=output_permuted,
            solution=solution,
            n_objects=n_objects,
            n_attributes=n_attributes,
        )

        # Update the best permuted cell score
        if permuted_cell_score > best_permuted_cell_score:
            best_permuted_cell_score = permuted_cell_score

    return best_permuted_cell_score
