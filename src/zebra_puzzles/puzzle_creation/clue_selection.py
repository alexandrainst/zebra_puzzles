"""Module for selecting clues for a zebra puzzle."""

import logging
from random import randint, sample, shuffle

import numpy as np
from constraint import InSetConstraint, NotInSetConstraint

from zebra_puzzles.clue_removal import (
    remove_redundant_clues_with_rules,
    remove_redundant_clues_with_solver,
)
from zebra_puzzles.puzzle_creation.zebra_solver import (
    raise_if_unexpected_solution_found,
    solver,
)
from zebra_puzzles.zebra_utils import describe_random_attributes

log = logging.getLogger(__name__)


def choose_clues(
    solution: np.ndarray,
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    n_objects: int,
    n_attributes: int,
    clues_dict: dict[str, str],
    clue_weights: dict[str, float],
    clue_cases_dict: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Choose clues for a zebra puzzle.

    If the solver identifies a different solution than the expected one, it will raise a warning and change the solution to the one found by the solver.

    Args:
        solution: Solution to the zebra puzzle as a matrix of strings containing object indices and chosen attribute values. This matrix is n_objects x (n_attributes + 1).
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and descriptions of each clue type.
        clue_weights: Weights for clue selection as a dictionary containing a title and a weight for each clue type.
        clue_cases_dict: A dictionary containing the clue type as a key and a list of grammatical cases for clue attributes as values.

    Returns:
        A tuple (chosen_clues, chosen_clue_types), where:
            chosen_clues: Clues for the zebra puzzle as a list of strings.
            chosen_clue_types: Types of clues chosen for the puzzle as a list of strings.
    """
    # Get the probability of selecting each applicable clue type
    applicable_clues_dict, clue_probabilities = get_clue_probabilities(
        clue_weights=clue_weights,
        clues_dict=clues_dict,
        n_objects=n_objects,
        n_attributes=n_attributes,
    )

    # Transpose and sort the attributes
    chosen_attributes_sorted = chosen_attributes.T
    chosen_attributes_sorted = np.array([sorted(x) for x in chosen_attributes_sorted])

    solutions: list[dict[str, int]] = []
    chosen_clues: list[str] = []
    chosen_constraints: list[tuple] = []
    chosen_clue_parameters: list = []
    chosen_clue_types: list[str] = []

    # Define the maximum number of attempts to create a solvable puzzle by adding clues
    max_iter = 100

    # Add clues until the puzzle is solved or the maximum number of attempts is reached
    for _ in range(max_iter):
        # Generate a random clue
        new_clue_type = str(
            np.random.choice(sorted(applicable_clues_dict), p=clue_probabilities)
        )
        new_clue, new_constraint, new_clue_parameters = create_clue(
            clue=new_clue_type,
            n_objects=n_objects,
            n_attributes=n_attributes,
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            clues_dict=clues_dict,
            clue_cases_dict=clue_cases_dict,
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
            # Check if the solver found an unexpected solution. This should not be possible.
            raise_if_unexpected_solution_found(
                solutions=solutions,
                solution=solution,
                n_objects=n_objects,
                n_attributes=n_attributes,
                chosen_clues=chosen_clues,
            )

            # Remove redundant clues
            chosen_clues, chosen_constraints, chosen_clue_types = (
                remove_redundant_clues_with_solver(
                    chosen_constraints=chosen_constraints,
                    chosen_clues=chosen_clues,
                    chosen_attributes_sorted=chosen_attributes_sorted,
                    n_objects=n_objects,
                    chosen_clue_types=chosen_clue_types,
                )
            )

            # Break the loop because the puzzle is solved
            break
    else:  # If the loop was not broken, it means the puzzle was not solved
        log.warning(
            f"Failed to solve the puzzle after maximum attempts.\nsolution: {solution}\nchosen clues so far: {chosen_clues}\ncurrent_constraints: {current_constraints}"
        )
        raise StopIteration("Used too many attempts to solve the puzzle.")

    return chosen_clues, chosen_clue_types


def get_clue_probabilities(
    clue_weights: dict[str, float],
    clues_dict: dict[str, str],
    n_objects: int,
    n_attributes: int,
) -> tuple[dict[str, str], np.ndarray]:
    """Get the applicable clues and their probabilities.

    Args:
        clue_weights: Weights for clue selection as a dictionary containing a title and a weight for each clue type.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and descriptions of each clue type.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.

    Returns:
        A tuple (applicable_clues_dict, clue_probabilities), where:
            applicable_clues_dict: Clue types that can be used for this puzzle as a dictionary containing a title and a description of each clue.
            clue_probabilities: Probabilities of selecting each applicable clue type as a numpy array.

    NOTE: Consider setting p=0 for excluded clue types instead of defining applicable_clues_dict
    """
    applicable_clues_dict = exclude_clues(
        clues_dict=clues_dict, n_objects=n_objects, n_attributes=n_attributes
    )

    # Select the weights for applicable clues
    applicable_clue_weights = {
        clue_type: clue_weights[clue_type]
        for clue_type in applicable_clues_dict.keys()
        if clue_type in clue_weights
    }

    # Normalise the clue weights
    clue_probabilities = np.array(list(applicable_clue_weights.values()))
    clue_probabilities = clue_probabilities / np.sum(clue_probabilities)

    return applicable_clues_dict, clue_probabilities


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
                    "just_left_of",
                    "just_right_of",
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
    clue_cases_dict: dict[str, list[str]],
) -> tuple[str, tuple, tuple[str, list[int], np.ndarray]]:
    """Create a clue of a chosen type using random parts of the solution.

    Args:
        clue: Chosen clue type as a string.
        n_objects: Number of objects in the puzzle as an integer.
        n_attributes: Number of attributes per object as an integer.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        chosen_categories: Categories chosen for the solution.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        clue_cases_dict: A dictionary containing the clue type as a key and a list of grammatical cases for clue attributes as values.

    Returns:
        A tuple (full_clue, constraint, clue_par), where:
            full_clue: Full clue as a string.
            constraint: Tuple consisting of a constraint function and a list of variables (clue attributes) directly affected by the constraint.
            clue_par: A tuple (clue_type, i_clue_objects, clue_attributes), where:
                clue_type: The type of clue as a string.
                i_clue_objects: The object indices described in the clue as a list of integers.
                clue_attributes: The attribute names as an array of strings.

    NOTE: More clue types can be included. For example: odd_pos, even_pos, either
    NOTE: The current implementation does not allow objects to have non-unique attributes
    NOTE: Half-herrings could be added, where the clue adds some information but also contains irrelevant information.
    """
    clue_description = clues_dict[clue]

    # Define the order of grammatical cases in clue descriptions
    case_to_index = {"nom": 0, "acc": 3, "dat": 4, "none": 999}

    # Choose desc indices based on clue type and grammatical case in clue_cases_dict
    cases = clue_cases_dict[clue]
    desc_indices: list[int] = [case_to_index[case] for case in cases]

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
            desc_indices=desc_indices,
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
            desc_index_none = 1
        elif clue == "not_same_object":
            # Choose two random objects
            i_objects = sample(list(range(n_objects)), 2)
            desc_index_none = 2

        # Replace 'none' in desc_indices with the chosen desc_index_none
        desc_indices[desc_indices.index(case_to_index["none"])] = desc_index_none

        # Choose two unique attributes
        clue_attributes, clue_attribute_descs = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
            diff_cat=True,
            desc_indices=desc_indices,
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
            desc_indices=desc_indices,
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
            desc_indices=desc_indices,
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
            desc_indices=desc_indices,
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
