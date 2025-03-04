"""Module for selecting clues for a zebra puzzle."""

from random import sample
from typing import Dict, List, Tuple

from constraint import InSetConstraint, NotInSetConstraint

from zebra_puzzles.zebra_solver import format_solution, solver


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
        chosen_clues: Clues for the zebra puzzle as a list of strings.

    """
    # Transpose and sort the attributes
    chosen_attributes_sorted = [list(i) for i in zip(*chosen_attributes)]
    chosen_attributes_sorted = [sorted(x) for x in chosen_attributes_sorted]

    solutions: List[Dict[str, int]] = []
    solved: bool = False
    chosen_clues: List[str] = []
    constraints: List[Tuple] = []

    max_iter = 100
    i_iter = 0
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
        new_solutions, completeness = solver(
            constraints=current_constraints,
            chosen_attributes=chosen_attributes_sorted,
            n_objects=n_objects,
        )

        # Check if solution attempt has changed and if it has, save the clue
        if new_solutions != solutions:
            solutions = new_solutions
            chosen_clues.append(new_clue)
            constraints.append(constraint)

        # Check if the solution is complete and the clues are non-redundant

        if completeness == 1:
            solved = True

            # Check if the solver found an unexpected solution. This should not be possible.
            solution_attempt = format_solution(
                solution_dict=solutions[0], n_objects=n_objects
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
            chosen_clues, constraints = remove_redundant_clues(
                constraints=constraints,
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


def remove_redundant_clues(
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

        constraint = (InSetConstraint([i_object + 1]), [attribute])

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

        constraint = (NotInSetConstraint([i_other_object + 1]), [attribute])

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
