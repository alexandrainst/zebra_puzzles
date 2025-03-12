"""Module for selecting clues for a zebra puzzle."""

from random import choices, randint, sample, shuffle

import numpy as np
from constraint import InSetConstraint, NotInSetConstraint

from zebra_puzzles.clue_removal import (
    remove_redundant_clues_part1,
    remove_redundant_clues_part2,
)
from zebra_puzzles.zebra_solver import format_solution, solver


def choose_clues(
    solution: np.ndarray,
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    n_objects: int,
    n_attributes: int,
    clues_dict: dict[str, str],
) -> list[str]:
    """Generate a zebra puzzle.

    If the solver identifies a different solution than the expected one, it will raise a warning and change the solution to the one found by the solver.

    Args:
        solution: Solution to the zebra puzzle as a matrix of strings containing object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.

    Returns:
        Clues for the zebra puzzle as a list of strings.

    """
    # Exclude clues that cannot be used for this puzzle. We assume all puzzles are have at least 2 houses.
    if n_objects <= 2:
        if any(
            [
                i in clues_dict
                for i in ["not_next_to", "next_to", "left_of", "right_of", "between"]
            ]
        ):
            raise ValueError(
                "Too few objects for the chosen clues. Please adjust the config file."
            )
    if n_attributes == 1:
        if any([i in clues_dict for i in ["not_same_object", "same_object"]]):
            raise ValueError(
                "Too few attributes for the chosen clues. Please adjust the config file."
            )

    # Transpose and sort the attributes
    chosen_attributes_sorted = chosen_attributes.T
    chosen_attributes_sorted = np.array([sorted(x) for x in chosen_attributes_sorted])

    solutions: list[dict[str, int]] = []
    solved: bool = False
    chosen_clues: list[str] = []
    chosen_constraints: list[tuple] = []
    chosen_clue_parameters: list = []
    chosen_clue_types: list[str] = []

    max_iter = 100
    i_iter = 0
    while not solved:
        # Generate a random clue
        new_clue_type = sample(sorted(clues_dict), 1)[0]
        new_clue, new_constraint, new_clue_parameters = create_clue(
            clue=new_clue_type,
            n_objects=n_objects,
            n_attributes=n_attributes,
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            clues_dict=clues_dict,
        )

        # Check if the clue is obviously redundant before using the solver to save runtime
        redundant, clues_to_remove = remove_redundant_clues_part1(
            new_clue=new_clue,
            old_clues=chosen_clues,
            new_clue_parameters=new_clue_parameters,
            old_clue_parameters=chosen_clue_parameters,
            new_clue_type=new_clue_type,
            old_clue_types=chosen_clue_types,
            prioritise_old_clues=True,
        )
        if redundant:
            continue
        elif clues_to_remove != []:
            # Sort the list of clues to remove from last to first and only include unique ones
            clues_to_remove = sorted(list(set(clues_to_remove)), reverse=True)

            for i in clues_to_remove:
                del chosen_clues[i]
                del chosen_constraints[i]
                del chosen_clue_parameters[i]
                del chosen_clue_types[i]

        current_constraints = chosen_constraints + [new_constraint]

        # Try to solve the puzzle
        new_solutions, completeness = solver(
            constraints=current_constraints,
            chosen_attributes=chosen_attributes_sorted,
            n_objects=n_objects,
        )

        # Check if solution attempt has changed and if it has, save the clue
        if new_solutions != solutions:
            solutions = new_solutions
            chosen_clues.append(new_clue)
            chosen_constraints.append(new_constraint)
            chosen_clue_parameters.append(new_clue_parameters)
            chosen_clue_types.append(new_clue_type)

        # Check if the solution is complete and the clues are non-redundant

        if completeness == 1:
            solved = True

            # Check if the solver found an unexpected solution. This should not be possible.
            solution_attempt = format_solution(
                solution_dict=solutions[0],
                n_objects=n_objects,
                n_attributes=n_attributes,
            )

            if [sorted(x) for x in solution_attempt] != [sorted(x) for x in solution]:
                # Change the solution to the solution attempt and raise a warning
                solution_old = solution
                solution = solution_attempt
                raise Warning(
                    "The solver has found a solution that is not the expected one: \nFound \n{solution_attempt} \nExpected \n{solution}".format(
                        solution_attempt=solution_attempt, solution=solution_old
                    )
                )

            # Remove redundant clues
            chosen_clues, constraints = remove_redundant_clues_part2(
                chosen_constraints=chosen_constraints,
                chosen_clues=chosen_clues,
                chosen_attributes_sorted=chosen_attributes_sorted,
                n_objects=n_objects,
            )

        i_iter += 1
        if i_iter >= max_iter:
            solved = True
            print("solution:", solution)
            print("chosen clues so far:", chosen_clues)
            print("current_constraints:", current_constraints)
            raise StopIteration("Used too many attempts to solve the puzzle.")

    return chosen_clues


def create_clue(
    clue: str,
    n_objects: int,
    n_attributes: int,
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    clues_dict: dict[str, str],
) -> tuple[str, tuple, tuple[str, list[int], np.ndarray]]:
    """Create a clue of a chosen type using random parts of the solution.

    NOTE: More clue types can be included. For example: odd_pos, even_pos, either
    NOTE: The current implementation does not allow objects to have non-unique attributes

    Args:
        clue: Chosen clue type as a string.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        chosen_categories: Categories chosen for the solution.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.

    Returns:
        A tuple (full_clue, constraint, clue_par), where:
            full_clue: Full clue as a string.
            constraint: Tuple consisting of a constraint function and a list of variables directly affected by the constraint.
            clue_par: List containing the clue type, the object indices described in the clue and the attribute names.
    """
    clue_description = clues_dict[clue]

    if clue in ("found_at", "not_at"):
        if clue == "found_at":
            # Choose a random object
            i_object = sample(list(range(n_objects)), 1)[0]
            i_clue_object = i_object
        elif clue == "not_at":
            # Choose two random objects - one for the attribute and one not connected to this attribute
            i_object, i_clue_object = sample(list(range(n_objects)), 2)

        # Choose an attribute
        clue_attributes, attribute_desc = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=[i_object],
            n_attributes=n_attributes,
        )

        # Save the clue object index for the clue_par list
        i_objects = [i_clue_object]

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc=attribute_desc[0], i_object=i_clue_object + 1
        )

        if clue == "found_at":
            constraint = (InSetConstraint([i_clue_object + 1]), clue_attributes)
        elif clue == "not_at":
            constraint = (NotInSetConstraint([i_clue_object + 1]), clue_attributes)

    elif clue in ("same_object", "not_same_object"):
        if clue == "same_object":
            # Choose a random object
            i_object = sample(list(range(n_objects)), 1)[0]
            i_objects = [i_object, i_object]
        elif clue == "not_same_object":
            # Choose two random objects
            i_objects = sample(list(range(n_objects)), 2)

        # Choose two unique attributes
        clue_attributes, clue_attribute_descs = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
            diff_cat=True,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc_1=clue_attribute_descs[0],
            attribute_desc_2=clue_attribute_descs[1],
        )

        if clue == "same_object":
            constraint = (lambda a, b: a == b, clue_attributes)
        elif clue == "not_same_object":
            constraint = (lambda a, b: a != b, clue_attributes)

    elif clue in (
        "next_to",
        "not_next_to",
        "just_left_of",
        "just_right_of",
        "left_of",
        "right_of",
    ):
        if clue == "not_next_to":
            # Choose two objects that are not next to each other or identical
            i_objects = [0, 0]
            while abs(i_objects[0] - i_objects[1]) <= 1:
                i_objects = sample(list(range(n_objects)), 2)
        elif clue in ("left_of", "right_of"):
            # Choose two random objects
            i_objects = sample(list(range(n_objects)), 2)
            i_objects = sorted(i_objects)
            if clue == "right_of":
                i_objects = i_objects[::-1]
        else:
            # Choose a random object to the left of another
            i_object = sample(list(range(n_objects - 1)), 1)[0]

            # Choose the object on the right and shuffle the order
            i_objects = [i_object, i_object + 1]
            if clue == "next_to":
                shuffle(i_objects)
            elif clue == "just_right_of":
                i_objects = i_objects[::-1]

        # Choose two random attributes
        clue_attributes, clue_attributes_desc = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc_1=clue_attributes_desc[0],
            attribute_desc_2=clue_attributes_desc[1],
        )

        if clue == "next_to":
            constraint = (lambda a, b: abs(a - b) == 1, clue_attributes)
        elif clue == "just_left_of":
            constraint = (lambda a, b: b - a == 1, clue_attributes)
        elif clue == "just_right_of":
            constraint = (lambda a, b: a - b == 1, clue_attributes)
        elif clue == "not_next_to":
            constraint = (lambda a, b: abs(a - b) > 1, clue_attributes)
        elif clue == "left_of":
            constraint = (lambda a, b: b - a > 0, clue_attributes)
        elif clue == "right_of":
            constraint = (lambda a, b: a - b > 0, clue_attributes)

    elif clue in ("between", "not_between"):
        # Choose three random objects
        i_objects = sample(list(range(n_objects)), 3)
        i_objects = sorted(i_objects)

        # Randomly choose the order in which to mention the first and last object
        if randint(0, 1):
            i_objects = i_objects[::-1]

        if clue == "not_between":
            # Randomly choose the order in which to mention the center object and another object
            i_objects_last2 = i_objects[-2:]
            shuffle(i_objects_last2)
            i_objects[-2:] = i_objects_last2

        # Choose three random attributes
        clue_attributes, clue_attributes_desc = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc_1=clue_attributes_desc[0],
            attribute_desc_2=clue_attributes_desc[1],
            attribute_desc_3=clue_attributes_desc[2],
        )

        if clue == "between":
            constraint = (lambda a, b, c: a < b < c or a > b > c, clue_attributes)
        else:
            constraint = (
                lambda a, b, c: not (b < a < c or b > a > c)
                and a != b
                and a != c
                and b != c,
                clue_attributes,
            )

    elif clue == "n_between":
        # Choose two random objects with a distance of at least 2
        n_between = 0
        while n_between < 2:
            i_objects = sample(list(range(n_objects)), 2)
            n_between = abs(i_objects[0] - i_objects[1])

        # Choose two random attributes
        clue_attributes, clue_attributes_desc = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc_1=clue_attributes_desc[0],
            attribute_desc_2=clue_attributes_desc[1],
            n_between=n_between,
        )

        constraint = (lambda a, b: abs(b - a) == n_between, clue_attributes)

    else:
        raise ValueError("Unsupported clue '{clue}'")

    # Save the clue parameters
    clue_par = (clue, i_objects, clue_attributes)

    return full_clue, constraint, clue_par


def describe_random_attributes(
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    i_objects: list[int],
    n_attributes: int,
    diff_cat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Get a random attribute description for an object.

    Choose a random attribute for each object with indices given by i_objects. Looks up attributes from chosen_attributes in the attributes dict.

    Assumes the maximum string length is 100 characters.

    Args:
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        i_objects: The index of the object to select an attribute from.
        n_attributes: Number of attributes per object.
        diff_cat: If True, the output attributes must belong to different categories.

    Returns:
        A tuple (random_attributes, random_attributes_desc), where:
            random_attributes: A list of strings contraining one random attribute per object.
            random_attributes_desc: A list of strings using the attributes to describe the objects.
    """
    if diff_cat:
        i_attributes = sample(list(range(n_attributes)), k=len(i_objects))
    else:
        i_attributes = choices(list(range(n_attributes)), k=len(i_objects))

    # Initialize the random attributes as type 'object' to avoid setting a maximum string length
    random_attributes = np.empty((len(i_objects)), dtype="U100")
    random_attributes_desc = np.empty((len(i_objects)), dtype="U100")

    for i, (i_obj, i_attr) in enumerate(zip(i_objects, i_attributes)):
        random_attributes[i] = chosen_attributes[i_obj][i_attr]
        random_attributes_desc[i] = chosen_attributes_descs[i_obj][i_attr]

    return random_attributes, random_attributes_desc
