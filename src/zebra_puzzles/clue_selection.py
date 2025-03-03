"""Module for selecting clues for a zebra puzzle."""

from random import sample
from typing import Dict, List, Tuple

from constraint import InSetConstraint, NotInSetConstraint

from zebra_puzzles.zebra_solver import solver


def choose_clues(
    solution: List[List],
    chosen_categories: List[str],
    chosen_attributes: List[List],
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    clues_dict: Dict[str, str],
) -> List[str]:
    """Generate a zebra puzzle.

    If the solver identifies a different solution than the expected one, it will raise a warning and change the solution to the one found by the solver.

    Args:
        solution: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
        n_objects: Number of objects in the puzzle.
        attributes: Possible attributes as a dictionary of dictionaries.

    Returns:
        chosen_clues: Clues for the zebra puzzle as a string.

    TODO: Implement the generation of more than a single clue.
    """
    # Transpose and sort the attributes
    chosen_attributes_sorted = list(map(list, zip(*chosen_attributes)))
    chosen_attributes_sorted = [sorted(x) for x in chosen_attributes_sorted]

    solution_attempt: List[List] = []
    solved = False
    chosen_clues: List[str] = []
    constraints: List[Tuple] = []
    while not solved:
        # Generate a random clue
        new_clue = sample(sorted(clues_dict), 1)[0]
        new_clue, constraint = complete_clue(
            clue=new_clue,
            n_objects=n_objects,
            attributes=attributes,
            chosen_attributes=chosen_attributes,
            chosen_categories=chosen_categories,
            clues_dict=clues_dict,
        )

        current_constraints = constraints + [constraint]

        # Try to solve the puzzle
        new_solution_attempt, completeness = solver(
            constraints=current_constraints,
            chosen_attributes=chosen_attributes_sorted,
            n_objects=n_objects,
        )

        # Check if solution attempt has changed and if it has, save the clue
        if new_solution_attempt != solution_attempt:
            solution_attempt = new_solution_attempt
            chosen_clues.append(new_clue)
            constraints.append(constraint)

        # Check if the solution is complete. If it is, check if the solution attempt is the same as the solution

        if completeness == 1:
            solved = True
            if solution_attempt != solution:
                # Change the solution to the solution attempt and raise a warning
                solution = solution_attempt
                raise Warning(
                    "The solver has found a solution that is not the expected one: \nFound \n{solution_attempt} \nExpected \n{solution}"
                )

            # Try removing each clue and see if the solution is still found
            for i, constraint in enumerate(constraints):
                new_solution_attempt, completeness = solver(
                    constraints=constraints[:i] + constraints[i + 1 :],
                    chosen_attributes=chosen_attributes_sorted,
                    n_objects=n_objects,
                )
                if new_solution_attempt == solution:
                    chosen_clues.pop(i)
                    constraints.pop(i)

        # TODO: Remove this after testing
        solved = True

    return chosen_clues


def complete_clue(
    clue: str,
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
    clues_dict: Dict[str, str],
) -> Tuple[str, Tuple]:
    """Complete the chosen clue type with random parts of the solution to create a full clue.

    TODO: Consider how the clues will be evaluted. We should probably include more information in the dict such as a lambda function.
    TODO: Include more clue types. For example not_at, next_to, not_next_to, left_of, right_of, not_left_of, not_right_of, same_object, not_same_object, between, not_between
    NOTE: The current implementation does not allow objects to have non-unique attributes

    Args:
        clue: Chosen clue type as a string.
        n_objects: Number of objects in the puzzle as an int.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution.
        clues_dict: Possible clue types to include in the puzzle as a dictionary containing a title and a description of each clue.

    Returns:
        full_clue: Full clue as a string.
        constraint: Tuple consisting of a constraint function and a list of variables directly affected by the constraint.
    """
    clue_description = clues_dict[clue]

    if clue == "found_at":
        # E.g. the person who paints lives in house no. 5.

        # Choose a random object
        i_object = sample(list(range(n_objects)), 1)[0]

        # Choose an attribute
        attribute, attribute_desc = describe_random_attribute(
            attributes=attributes,
            chosen_attributes=chosen_attributes,
            chosen_categories=chosen_categories,
            i_object=i_object,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc=attribute_desc, i_object=i_object + 1
        )

        constraint = (InSetConstraint([i_object]), [attribute])

    elif clue == "not_at":
        # E.g. the person who paints does not live in house no. 5.

        # Choose two random objects - one for the attribute and one not connected to this attribute
        i_object, i_other_object = sample(list(range(n_objects)), 2)

        # Choose an attribute of the first object
        attribute, attribute_desc = describe_random_attribute(
            attributes=attributes,
            chosen_attributes=chosen_attributes,
            chosen_categories=chosen_categories,
            i_object=i_object,
        )

        # Create the full clue
        full_clue = clue_description.format(
            attribute_desc=attribute_desc, i_other_object=i_other_object + 1
        )

        constraint = (NotInSetConstraint([i_other_object]), [attribute])

    else:
        raise ValueError("Unsupported clue '{clue}'")

    return full_clue, constraint


def describe_random_attribute(
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
    i_object: int,
) -> Tuple[str, str]:
    """Choose a random attribute.

    Consider replacing this function by an array of chosen attribute descriptions or making chosen_attributes a dict.

    Args:
        attributes: Attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution.
        i_object: The index of the object to select an attribute from.

    Returns:
        attribute_desc: A string using the attribute to describe an object.

    """
    # Choose a random attribute and the corresponding category
    i_attribute, attribute = sample(list(enumerate(chosen_attributes[i_object])), 1)[0]

    chosen_category = chosen_categories[i_attribute]

    # Get the attribute description
    attribute_desc = attributes[chosen_category][attribute]

    return attribute, attribute_desc
