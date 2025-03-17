"""Module for removing redundant clues."""

import numpy as np

from zebra_puzzles.zebra_solver import solver


def remove_redundant_clues_with_rules(
    new_clue: str,
    old_clues: list[str],
    old_constraints: list,
    new_clue_parameters: tuple[str, list[int], np.ndarray],
    old_clue_parameters: list,
    new_clue_type: str,
    old_clue_types: list[str],
    prioritise_old_clues: bool = False,
) -> tuple[bool, list[str], list, list, list]:
    """Remove redundant clues and constraints.

    Check if a suggested clue is redundant and remove it or existing clues if it is, depending on prioritise_old_clues.

    Args:
        new_clue: The suggested clue to check as a string.
        old_clues: Chosen clues for the zebra puzzle as a list of strings.
        old_constraints: List of constraints for the puzzle solver.
        new_clue_parameters: A tuple (clue_type, i_clue_objects, clue_attributes) containing clue parameters for the suggested clue, where:
            clue_type: Type of the clue as a string.
            i_clue_objects: List of object indices in the clue as integers.
            clue_attributes: Array of attribute values as strings for the clue.
        old_clue_parameters: List of all previously chosen clue parameters as described above for new_clue_parameters.
        new_clue_type: Clue type for the suggested clue.
        old_clue_types: List of all previously chosen clue types.
        prioritise_old_clues: Boolean indicating if the new clue should be rejected if it includes all information of an existing clue. This will reduce a bias towards more specific clues and result in more clues per puzzle. Otherwise, the old less specific clue will be removed.

    Returns:
        A tuple (redundant, old_clues, old_constraints, old_clue_parameters, old_clue_types), where:
            redundant: Boolean indicating if the suggested clue is redundant.
            old_clues: Non-redundant old clues for the zebra puzzle as a list of strings.
            old_constraints: List of non-redundant old constraints for the puzzle solver.
            old_clue_parameters: List of non-redundant old clue parameters. This list contains tuples as described above for new_clue_parameters.
            old_clue_types: List of non-redundant old clue types.
    """
    redundant, clues_to_remove = is_clue_redundant(
        new_clue=new_clue,
        old_clues=old_clues,
        new_clue_parameters=new_clue_parameters,
        old_clue_parameters=old_clue_parameters,
        new_clue_type=new_clue_type,
        old_clue_types=old_clue_types,
        prioritise_old_clues=prioritise_old_clues,
    )

    if clues_to_remove != []:
        # Sort the list of clues to remove from last to first and only include unique ones
        clues_to_remove = sorted(list(set(clues_to_remove)), reverse=True)

        for i in clues_to_remove:
            del old_clues[i]
            del old_constraints[i]
            del old_clue_parameters[i]
            del old_clue_types[i]

    return redundant, old_clues, old_constraints, old_clue_parameters, old_clue_types


def is_clue_redundant(
    new_clue: str,
    old_clues: list[str],
    new_clue_parameters: tuple[str, list[int], np.ndarray],
    old_clue_parameters: list[tuple[str, list[int], np.ndarray]],
    new_clue_type: str,
    old_clue_types: list[str],
    prioritise_old_clues: bool = False,
) -> tuple[bool, list[int]]:
    """Use simple rules to check if a suggested clue is redundant.

    This is to avoid using the solver for every clue suggestion and thereby speed up the clue selection process.

    NOTE: More checks could be added e.g. "same_object" and "not_same_object" with 1 identical attribute and secondary attributes of the same category.
    NOTE: Consider adapting for non-unique attributes
    TODO: Combine checks for fewer loops

    Args:
        new_clue: The suggested clue to check as a string.
        old_clues: Chosen clues for the zebra puzzle as a list of strings.
        new_clue_parameters: A tuple (clue_type, i_clue_objects, clue_attributes) containing clue parameters for the suggested clue, where:
            clue_type: Type of the clue as a string.
            i_clue_objects: List of object indices in the clue as integers.
            clue_attributes: Array of attribute values as strings for the clue.
        old_clue_parameters: List of all previously chosen clue parameters as described above for new_clue_parameters.
        new_clue_type: Clue type for the suggested clue.
        old_clue_types: List of all previously chosen clue types.
        prioritise_old_clues: Boolean indicating if the new clue should be rejected if it includes all information of an old clue. This will reduce a bias towards more specific clues and result in more clues per puzzle. Otherwise, the old less specific clue will be added in clues_to_remove.

    Returns:
        A tuple (redundant, clues_to_remove), where:
            redundant: Boolean indicating if the clue is redundant
            clues_to_remove: List of indices of clues to remove if the new clue is more specific than an existing clue. This is always empty if prioritise_old_clues is False.

    """
    clues_to_remove = []

    # ---- Check if the clue has already been chosen ----#
    if new_clue in old_clues:
        return True, []

    # ---- Check if not_at is used after found_at with the same attribute (but not the same objects) ----#
    if new_clue_type == "not_at" and "found_at" in old_clue_types:
        for clue_type_j, _, attributes_j in old_clue_parameters:
            if clue_type_j == "found_at" and attributes_j == new_clue_parameters[2]:
                return True, []

    elif new_clue_type == "found_at" and "not_at" in old_clue_types:
        for i, (clue_type_j, _, attributes_j) in enumerate(old_clue_parameters):
            if clue_type_j == "not_at" and attributes_j == new_clue_parameters[2]:
                if prioritise_old_clues:
                    return True, []
                else:
                    clues_to_remove.append(i)
                    # We can stop here because none of the following checks will be true
                    return False, clues_to_remove

    # ---- Check if between clues exclude not_same_object ----#
    elif new_clue_type == "not_same_object":
        # Go through the list of chosen clues
        for clue_type_j, i_objects_j, attributes_j in old_clue_parameters:
            # Check if the new clue type and an existing clue type are a pair in redundant_clues
            if clue_type_j in {"between", "not_between"}:
                # Combine objects and attributes in the clue pairwise
                combined_obj_attributes = {
                    f"{x}{y}" for x, y in zip(i_objects_j, attributes_j)
                }
                combined_obj_attributes_new = {
                    f"{x}{y}"
                    for x, y in zip(new_clue_parameters[1], new_clue_parameters[2])
                }

                # Check if the combination of objects and attributes are the included in the existing clue
                if combined_obj_attributes_new.issubset(combined_obj_attributes):
                    return True, []

    elif new_clue_type in {"between", "not_between"}:
        # Go through the list of chosen clues
        for i, (clue_type_j, i_objects_j, attributes_j) in enumerate(
            old_clue_parameters
        ):
            # Check if the new clue type and an existing clue type are a pair in redundant_clues
            if clue_type_j == "not_same_object":
                # Combine objects and attributes in the clue pairwise
                combined_obj_attributes = {
                    f"{x}{y}" for x, y in zip(i_objects_j, attributes_j)
                }
                combined_obj_attributes_new = {
                    f"{x}{y}"
                    for x, y in zip(new_clue_parameters[1], new_clue_parameters[2])
                }

                # Check if the combination of objects and attributes are the included in the existing clue
                if combined_obj_attributes.issubset(combined_obj_attributes_new):
                    if prioritise_old_clues:
                        return True, []
                    else:
                        clues_to_remove.append(i)

    # ---- Check if clues containing the same objects and attributes are redundant ----#

    # List the clues where if the first clue is already chosen, the second clue is redundant if they contain the same objects and attributes
    redundant_clues = {
        ("same_object", "same_object"),
        ("not_same_object", "not_same_object"),
        ("left_of", "right_of"),
        ("left_of", "not_same_object"),
        ("right_of", "left_of"),
        ("right_of", "not_same_object"),
        ("just_left_of", "just_right_of"),
        ("just_left_of", "right_of"),
        ("just_left_of", "left_of"),
        ("just_left_of", "next_to"),
        ("just_left_of", "not_same_object"),
        ("just_right_of", "just_left_of"),
        ("just_right_of", "left_of"),
        ("just_right_of", "right_of"),
        ("just_right_of", "next_to"),
        ("just_right_of", "not_same_object"),
        ("next_to", "not_same_object"),
        ("one_between", "not_same_object"),
        ("multiple_between", "not_same_object"),
        ("between", "not_between"),
    }

    # Sort the new objects and attributes outside the following loop as they could be compared several times
    sorted_new_objects = sorted(new_clue_parameters[1])
    sorted_new_attributes = sorted(new_clue_parameters[2])

    # Go through the list of chosen clues
    for i, (clue_type_j, i_objects_j, attributes_j) in enumerate(old_clue_parameters):
        # Check if the new clue adds no new information
        if (
            (clue_type_j, new_clue_type) in redundant_clues
            and sorted(i_objects_j) == sorted_new_objects
            and sorted(attributes_j) == sorted_new_attributes
        ):
            return True, []

        # Check if an existing clue adds is less specific than the new clue
        if (
            (clue_type_j, new_clue_type) in redundant_clues
            and sorted(i_objects_j) == sorted_new_objects
            and sorted(attributes_j) == sorted_new_attributes
        ):
            if prioritise_old_clues:
                return True, []
            else:
                clues_to_remove.append(i)

    # Otherwise, the clue might not be redundant
    return False, clues_to_remove


def remove_redundant_clues_with_solver(
    chosen_constraints: list,
    chosen_clues: list[str],
    chosen_attributes_sorted: np.ndarray,
    n_objects: int,
) -> tuple[list[str], list]:
    """Remove redundant clues and constraints.

    Tries removing each clue and see if the solution is still found.
    Starts from the end of the list for easier iteration through a list we are removing elements from.

    Args:
        chosen_constraints: Constraints for the zebra puzzle as a list of tuples. Each constaint corresponds to one clue. Each tuple (constraint_function, variables) contains:
            constraint_function: A constraint function that the variables must satisfy.
            variables: Attributes that the constraint applies to.
        chosen_clues: Clues for the zebra puzzle as a list of strings.
        chosen_attributes_sorted: Matrix of attribute values chosen for the solution after sorting each category to avoid spoiling the solution.
        n_objects: Number of objects in the puzzle.

    Returns:
        A tuple (chosen_clues, constraints), where:
            chosen_clues: Non-redundant clues for the zebra puzzle as a list of strings.
            constraints: Non-redundant constraints for the puzzle solver.

    """
    for i in range(len(chosen_constraints) - 1, -1, -1):
        _, completeness = solver(
            constraints=chosen_constraints[:i] + chosen_constraints[i + 1 :],
            chosen_attributes=chosen_attributes_sorted,
            n_objects=n_objects,
        )
        if completeness == 1:
            del chosen_clues[i]
            del chosen_constraints[i]

    return chosen_clues, chosen_constraints
