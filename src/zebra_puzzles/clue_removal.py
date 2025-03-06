"""Module for removing redundant clues."""

from typing import List, Tuple

from zebra_puzzles.zebra_solver import solver


def remove_redundant_clues_part1(
    new_clue: str,
    chosen_clues: List[str],
    clue_par: Tuple[str, List[int], List[str]],
    clue_pars: List,
    clue_type: str,
    clue_types: List[str],
) -> bool:
    """Use simple rules to check if a suggested clue is redundant.

    This is to avoid using the solver for every clue suggestion.

    Args:
        new_clue: The clue to check as a string.
        chosen_clues: Chosen clues for the zebra puzzle as a list of strings.
        clue_par: Clue parameters for the new clue.
        clue_pars: List of clue parameters for the puzzle solver.
        clue_type: Clue type for the new clue.
        clue_types: List of clue types.

    Returns:
        redundant: Boolean indicating if the clue is redundant

    """
    # Check if the clue has already been chosen (same clue parameters)
    if new_clue in chosen_clues:
        return True

    # Check if a clue of the same meaning has already been chosen
    if clue_type in clue_types:
        if clue_type in ("same_object", "not_same_object"):
            for clue_type_j, i_objects_j, attributes_j in clue_pars:
                if clue_type == clue_type_j:
                    if sorted(i_objects_j) == sorted(clue_par[1]) and sorted(
                        attributes_j
                    ) == sorted(clue_par[2]):
                        return True

    # Check if not_at is used after found_at with the same attribute
    if clue_type == "not_at" and "found_at" in clue_types:
        for clue_type_j, i_objects_j, attributes_j in clue_pars:
            if clue_type_j == "found_at":
                if attributes_j == clue_par[2]:
                    return True

    return False


def remove_redundant_clues_part2(
    constraints: List,
    chosen_clues: List[str],
    chosen_attributes_sorted: List[List],
    n_objects: int,
) -> Tuple[List[str], List]:
    """Remove redundant clues and constraints.

    Tries removing each clue and see if the solution is still found.
    Starts from the end of the list for easier iteration through a list we are removing elements from.

    Args:
        constraints: List of constraints for the puzzle solver.
        chosen_clues: Clues for the zebra puzzle as a list of strings.
        chosen_attributes_sorted: List of lists of attribute values chosen for the solution after sorting each category to avoid spoiling the solution.
        n_objects: Number of objects in the puzzle.

    Returns:
        chosen_clues: Non-redundant clues for the zebra puzzle as a list of strings.
        constraints: Non-redundant constraints for the puzzle solver.

    """
    for i in range(len(constraints) - 1, -1, -1):
        _, completeness = solver(
            constraints=constraints[:i] + constraints[i + 1 :],
            chosen_attributes=chosen_attributes_sorted,
            n_objects=n_objects,
        )
        if completeness == 1:
            del chosen_clues[i]
            del constraints[i]

    return chosen_clues, constraints
