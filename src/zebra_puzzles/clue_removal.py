"""Module for removing redundant clues."""

from zebra_puzzles.zebra_solver import solver


def remove_redundant_clues_part1(
    new_clue: str,
    chosen_clues: list[str],
    clue_par: tuple[str, list[int], list[str]],
    clue_pars: list,
    clue_type: str,
    clue_types: list[str],
    prioritise_old_clues: bool = False,
) -> tuple[bool, list[int]]:
    """Use simple rules to check if a suggested clue is redundant.

    This is to avoid using the solver for every clue suggestion and thereby speed up the clue selection process.

    NOTE: More checks could be added e.g. "same_object" and "not_same_object" with 1 identical attribute and secondary attributes of the same category.
    NOTE: Consider adapting for non-unique attributes
    TODO: Combine checks for fewer loops

    Args:
        new_clue: The clue to check as a string.
        chosen_clues: Chosen clues for the zebra puzzle as a list of strings.
        clue_par: Clue parameters for the new clue.
        clue_pars: List of clue parameters for the puzzle solver.
        clue_type: Clue type for the new clue.
        clue_types: List of clue types.
        prioritise_old_clues: Boolean indicating if the new clue should be rejected if it includes all information of an existing clue. This will reduce a bias towards more specific clues and result in more clues per puzzle. Otherwise, the old less specific clue will be added in clues_to_remove.

    Returns:
        redundant: Boolean indicating if the clue is redundant
        clues_to_remove: List of indices of clues to remove if the new clue is more specific than an existing clue. This is always empty if prioritise_old_clues is False.

    """
    clues_to_remove = []

    # ---- Check if the clue has already been chosen ----#
    if new_clue in chosen_clues:
        return True, []

    # ---- Check if not_at is used after found_at with the same attribute (but not the same objects) ----#
    if clue_type == "not_at" and "found_at" in clue_types:
        for clue_type_j, _, attributes_j in clue_pars:
            if clue_type_j == "found_at" and attributes_j == clue_par[2]:
                return True, []

    elif clue_type == "found_at" and "not_at" in clue_types:
        for i, (clue_type_j, _, attributes_j) in enumerate(clue_pars):
            if clue_type_j == "not_at" and attributes_j == clue_par[2]:
                if prioritise_old_clues:
                    return True, []
                else:
                    clues_to_remove.append(i)
                    # We can stop here because none of the following checks will be true
                    return False, clues_to_remove

    # ---- Check if between clues exclude not_same_object ----#
    elif clue_type == "not_same_object":
        # Go through the list of chosen clues
        for clue_type_j, i_objects_j, attributes_j in clue_pars:
            # Check if the new clue type and an existing clue type are a pair in redundant_clues
            if clue_type_j in {"between", "not_between"}:
                # Combine pairwise
                combined_obj_attributes = {
                    f"{x}{y}" for x, y in zip(i_objects_j, attributes_j)
                }
                combined_obj_attributes_new = {
                    f"{x}{y}" for x, y in zip(clue_par[1], clue_par[2])
                }

                # Check if the combination of objects and attributes are the included in the existing clue
                if combined_obj_attributes_new.issubset(combined_obj_attributes):
                    return True, []

    elif clue_type in {"between", "not_between"}:
        # Go through the list of chosen clues
        for i, (clue_type_j, i_objects_j, attributes_j) in enumerate(clue_pars):
            # Check if the new clue type and an existing clue type are a pair in redundant_clues
            if clue_type_j == "not_same_object":
                # Combine pairwise
                combined_obj_attributes = {
                    f"{x}{y}" for x, y in zip(i_objects_j, attributes_j)
                }
                combined_obj_attributes_new = {
                    f"{x}{y}" for x, y in zip(clue_par[1], clue_par[2])
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
        ("n_between", "not_same_object"),
        ("between", "not_between"),
    }

    # Sort the new objects and attributes outside the following loop as they could be compared several times
    sorted_new_objects = sorted(clue_par[1])
    sorted_new_attributes = sorted(clue_par[2])

    # Go through the list of chosen clues
    for i, (clue_type_j, i_objects_j, attributes_j) in enumerate(clue_pars):
        # Check if the new clue adds no new information
        if (clue_type_j, clue_type) in redundant_clues:
            # Check if the objects and attributes are the same
            if (
                sorted(i_objects_j) == sorted_new_objects
                and sorted(attributes_j) == sorted_new_attributes
            ):
                return True, []

        # Check if an existing clue adds is less specific than the new clue
        if (clue_type_j, clue_type) in redundant_clues:
            if (
                sorted(i_objects_j) == sorted_new_objects
                and sorted(attributes_j) == sorted_new_attributes
            ):
                if prioritise_old_clues:
                    return True, []
                else:
                    clues_to_remove.append(i)

    # Otherwise, the clue might not be redundant
    return False, clues_to_remove


def remove_redundant_clues_part2(
    constraints: list,
    chosen_clues: list[str],
    chosen_attributes_sorted: list[list],
    n_objects: int,
) -> tuple[list[str], list]:
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
