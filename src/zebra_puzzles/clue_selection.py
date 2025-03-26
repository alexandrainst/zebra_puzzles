"""Module for selecting clues for a zebra puzzle."""

from random import randint, sample, shuffle

import numpy as np
from constraint import InSetConstraint, NotInSetConstraint

from zebra_puzzles.clue_removal import (
    remove_redundant_clues_with_rules,
    remove_redundant_clues_with_solver,
)
from zebra_puzzles.zebra_solver import format_solution_as_matrix, solver
from zebra_puzzles.zebra_utils import describe_random_attributes


def choose_clues(
    solution: np.ndarray,
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    n_objects: int,
    n_attributes: int,
    clues_dict: dict[str, str],
) -> list[str]:
    """Choose clues for a zebra puzzle.

    If the solver identifies a different solution than the expected one, it will raise a warning and change the solution to the one found by the solver.

    Args:
        solution: Solution to the zebra puzzle as a matrix of strings containing object indices and chosen attribute values. This matrix is n_objects x (n_attributes + 1).
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.

    Returns:
        Clues for the zebra puzzle as a list of strings.

    """
    applicable_clues_dict = exclude_clues(
        clues_dict=clues_dict, n_objects=n_objects, n_attributes=n_attributes
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
        new_clue_type = sample(sorted(applicable_clues_dict), 1)[0]
        new_clue, new_constraint, new_clue_parameters = create_clue(
            clue=new_clue_type,
            n_objects=n_objects,
            n_attributes=n_attributes,
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            clues_dict=clues_dict,
        )

        # Check if the clue is obviously redundant before using the solver to save runtime
        (
            redundant,
            chosen_clues,
            chosen_constraints,
            chosen_clue_parameters,
            chosen_clue_types,
        ) = remove_redundant_clues_with_rules(
            new_clue=new_clue,
            old_clues=chosen_clues,
            old_constraints=chosen_constraints,
            new_clue_parameters=new_clue_parameters,
            old_clue_parameters=chosen_clue_parameters,
            new_clue_type=new_clue_type,
            old_clue_types=chosen_clue_types,
            prioritise_old_clues=False,
        )
        if redundant:
            continue

        current_constraints = chosen_constraints + [new_constraint]

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
            test_original_solution(
                solutions=solutions,
                solution=solution,
                n_objects=n_objects,
                n_attributes=n_attributes,
                chosen_clues=chosen_clues,
            )

            # Remove redundant clues
            chosen_clues, chosen_constraints = remove_redundant_clues_with_solver(
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


def test_original_solution(
    solutions: list[dict[str, int]],
    solution: np.ndarray,
    n_objects: int,
    n_attributes: int,
    chosen_clues: list[str],
):
    """Test if the solver found the original solution or an unexpected one.

    Finding a new solution should not be possible and indicates a bug in the solver or the clue selection process. If this happens, an error is raised.

    Args:
        solutions: Solutions to the zebra puzzle found by the solver as a list of dictionaries containing object indices and chosen attribute values.
        solution: Expected solution to the zebra puzzle as a matrix of strings containing object indices and chosen attribute values. This matrix is n_objects x (n_attributes + 1).
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.
        chosen_clues: Clues for the zebra puzzle as a list of strings.

    """
    solution_attempt = format_solution_as_matrix(
        solution_dict=solutions[0], n_objects=n_objects, n_attributes=n_attributes
    )

    if [sorted(obj) for obj in solution_attempt] != [sorted(obj) for obj in solution]:
        raise ValueError(
            f"The solver has found a solution that is not the expected one: \nFound \n{solution_attempt} \nExpected \n{solution} \nChosen clues: \n{chosen_clues}"
        )


def exclude_clues(
    clues_dict: dict[str, str], n_objects: int, n_attributes: int
) -> dict[str, str]:
    """Exclude clue types that cannot be used for this puzzle. We assume all puzzles have at least 2 objects.

    Args:
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.

    Returns:
        Clues that can be used for this puzzle as a dictionary containing a title and a description of each.

    """
    applicable_clues_dict = {k: v for k, v in clues_dict.items()}

    for clue in clues_dict.keys():
        if (
            (n_objects <= 3 and clue in ["multiple_between"])
            or (
                n_objects <= 2
                and clue
                in [
                    "not_next_to",
                    "next_to",
                    "left_of",
                    "right_of",
                    "between",
                    "not_between",
                    "one_between",
                ]
            )
            or (n_attributes == 1 and clue in ["not_same_object", "same_object"])
        ):
            del applicable_clues_dict[clue]
    return applicable_clues_dict


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
            constraint: Tuple consisting of a constraint function and a list of variables (clue attributes) directly affected by the constraint.
            clue_par: A tuple (clue_type, i_clue_objects, clue_attributes), where:
                clue_type: The type of clue as a string.
                i_clue_objects: The object indices described in the clue as a list of integers.
                clue_attributes: The attribute names as an array of strings.
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
            desc_index = 1
        elif clue == "not_same_object":
            # Choose two random objects
            i_objects = sample(list(range(n_objects)), 2)
            desc_index = 2

        # Choose two unique attributes
        clue_attributes, clue_attribute_descs = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
            diff_cat=True,
            desc_index=desc_index,
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

    elif clue in ("one_between", "multiple_between"):
        # Choose two random objects with a distance of at least 2
        objects_are_chosen = False
        while not objects_are_chosen:
            i_objects = sample(list(range(n_objects)), 2)
            n_between = abs(i_objects[0] - i_objects[1]) - 1
            if (n_between == 1 and clue == "one_between") or (
                n_between > 1 and clue == "multiple_between"
            ):
                objects_are_chosen = True

        # Choose two random attributes
        clue_attributes, clue_attributes_desc = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
        )

        if clue == "one_between":
            # Create the full clue
            full_clue = clue_description.format(
                attribute_desc_1=clue_attributes_desc[0],
                attribute_desc_2=clue_attributes_desc[1],
            )
        else:
            # Create the full clue
            full_clue = clue_description.format(
                attribute_desc_1=clue_attributes_desc[0],
                attribute_desc_2=clue_attributes_desc[1],
                n_between=n_between,
            )

        constraint = (lambda a, b: abs(b - a) - 1 == n_between, clue_attributes)

    else:
        raise ValueError("Unsupported clue '{clue}'")

    # Save the clue parameters
    clue_par = (clue, i_objects, clue_attributes)

    return full_clue, constraint, clue_par
