"""Utility module for generating zebra puzzles."""

from random import sample
from typing import Dict, List, Tuple


def define_clues(clues_included: str) -> List:
    """Define clue types for the puzzle.

    Args:
        clues_included: A string descriping which clue types to include.

    Returns:
        clues: List of included clue types.

    NOTE: In the future, we can support more clues and selection of clues in the config file.
    TODO: Implement clue functions. not_at, next_to, not_next_to, left_of, right_of, not_left_of, not_right_of, same_house, not_same_house, between, not_between
    """
    if clues_included == "all":
        clues = ["found_at", "not_at"]
    else:
        raise ValueError("Unsupported clues '{clues_included}'")

    return clues


def complete_clue(
    clue: str,
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
) -> str:
    """Complete the chosen clue type with random parts of the solution to create a full clue.

    TODO: Consider how the clues will be evaluted. We should probably save more than a string.
    TODO: Move the clue descriptions to the config file.

    Args:
        clue: Chosen clue type as a string.
        n_objects: Number of objects in the puzzle as an int.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution.

    Returns:
        full_clue: Full clue as a string.
    """
    if clue == "found_at":
        # Choose a random object
        i_object = sample(range(n_objects), 1)[0]

        # Choose a random attribute and the corresponding category
        i_attribute, attribute = sample(
            list(enumerate(chosen_attributes[i_object])), 1
        )[0]
        chosen_category = chosen_categories[i_attribute]

        # Get the attribute description
        attribute_desc = attributes[chosen_category][attribute]

        # Create the full clue
        full_clue = "Personen der {attribute_desc} er ved hus nummer {i}.".format(
            attribute_desc=attribute_desc, i=i_object
        )
    elif clue == "not_at":
        full_clue = "This is an example clue of type not_at."
    else:
        raise ValueError("Unsupported clue '{clue}'")

    return full_clue


def generate_solution(
    attributes: Dict[str, Dict[str, str]], n_objects: int, n_attributes: int
) -> Tuple[List[List], List, List[List]]:
    """Generate the solution to a zebra puzzle.

    Args:
        attributes: Attributes as a dictionary of dictionaries.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        solution: A solution to a zebra puzzle as a list of lists representing the matrix of object indices and chosen attributes. This matrix is n_objects x n_attributes.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
    """
    # Choose a category for each attribute
    chosen_categories = sample(list(attributes.keys()), k=n_attributes)

    # Choose attribute values for each category
    chosen_attributes = [
        sample(list(attributes[cat].keys()), k=n_objects) for cat in chosen_categories
    ]

    # Transpose the attribute matrix
    chosen_attributes = [
        [row[i] for row in chosen_attributes] for i in range(n_attributes)
    ]

    solution = [[str(i)] + row for i, row in enumerate(chosen_attributes)]

    return solution, chosen_categories, chosen_attributes


def choose_random_clue(
    clues: List,
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
    chosen_attributes: List[List],
    chosen_categories: List[str],
) -> str:
    """Choose a random clue from the list of possible clues.

    Args:
        clues: List of possible clues as strings.
        n_objects: Number of objects in the puzzle as an int.
        attributes: Possible attributes as a dictionary of dictionaries.
        chosen_attributes: Attribute values chosen for the solution as a list of lists.
        chosen_categories: Categories chosen for the solution

    Returns:
        full_clue: Full clue as a string.

    #TODO: Change the output of complete_clue to reflect the needed input of the zebra solver.
    """
    clue = sample(clues, 1)[0]

    full_clue = complete_clue(
        clue=clue,
        n_objects=n_objects,
        attributes=attributes,
        chosen_attributes=chosen_attributes,
        chosen_categories=chosen_categories,
    )

    return full_clue


def solver(chosen_clues: List[str]) -> Tuple[List[List], float]:
    """Solve a zebra puzzle.

    Args:
        chosen_clues: Clues for the zebra puzzle as a list of strings.
        new_clue: New clue to add to the solver as a string.

    Returns:
        solution_attempt: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        completeness: Completeness of the solution as a float.

    #TODO: Implement the solver
    """
    # Solve the puzzle

    solution_attempt: List[List] = []

    # Measure completeness of the solution
    completeness = 0

    return solution_attempt, completeness


def choose_clues(
    solution: List[List],
    clues: List,
    chosen_categories: List[str],
    chosen_attributes: List[List],
    n_objects: int,
    attributes: Dict[str, Dict[str, str]],
) -> List[str]:
    """Generate a zebra puzzle.

    If the solver identifies a different solution than the expected one, it will raise a warning and change the solution to the one found by the solver.

    Args:
        solution: Solution to the zebra puzzle as a list of lists representing the solution matrix of object indices and chosen attribute values. This matrix is n_objects x n_attributes.
        clues: Possible clues to include in the clues as a list of tuples. Each tuple contains the clue name and function. TODO: Edit this description when the clues are implemented.
        chosen_categories: Categories chosen for the solution.
        chosen_attributes: Attribute values chosen for the solution.
        n_objects: Number of objects in the puzzle.
        attributes: Possible attributes as a dictionary of dictionaries.

    Returns:
        chosen_clues: Clues for the zebra puzzle as a string.

    TODO: Implement the generation of the clues.
    """
    solution_attempt: List[List] = []
    solved = False
    chosen_clues: List[str] = []
    while not solved:
        # Add a random clue

        new_clue = choose_random_clue(
            clues=clues,
            n_objects=n_objects,
            attributes=attributes,
            chosen_attributes=chosen_attributes,
            chosen_categories=chosen_categories,
        )

        # Try to solve the puzzle

        current_clues = chosen_clues + [new_clue]
        new_solution_attempt, completeness = solver(chosen_clues=current_clues)

        # Check if solution attempt has changed and if it has, save the clue
        if new_solution_attempt != solution_attempt:
            solution_attempt = new_solution_attempt
            chosen_clues.append(new_clue)

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
            for i, clue in enumerate(chosen_clues):
                new_solution_attempt, completeness = solver(
                    chosen_clues=chosen_clues[:i] + chosen_clues[i + 1 :]
                )
                if new_solution_attempt == solution:
                    chosen_clues.pop(i)

        # TODO: Remove this after testing
        solved = True

    # TODO: Delete the following example when clues are chosen above
    chosen_clues = ["This is an example.", "This is the second part of the example."]

    return chosen_clues
