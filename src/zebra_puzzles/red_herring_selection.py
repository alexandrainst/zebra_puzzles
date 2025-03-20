"""Module for selecting red herrings for a zebra puzzle."""

from random import sample

import numpy as np

from zebra_puzzles.clue_selection import describe_random_attributes


def choose_red_herrings(
    n_red_herring_clues: int,
    red_herring_clues_dict: dict[str, str],
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, str],
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    n_objects: int,
    n_attributes: int,
) -> list[str]:
    """Choose red herrings for a zebra puzzle.

    Args:
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the puzzle as a list of strings.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        Chosen red herring clues for the zebra puzzle as a list of strings.

    """
    chosen_clues: list[str] = []
    used_red_herrings: list[str] = []
    for _ in range(n_red_herring_clues):
        # Choose a red herring clue type
        clue_type = sample(sorted(red_herring_clues_dict), 1)[0]

        # Create a red herring clue
        clue, used_red_herrings = create_red_herring(
            clue_type=clue_type,
            red_herring_attributes=red_herring_attributes,
            red_herring_facts=red_herring_facts,
            used_red_herrings=used_red_herrings,
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            n_objects=n_objects,
            n_attributes=n_attributes,
            red_herring_clues_dict=red_herring_clues_dict,
        )

        chosen_clues.append(clue)

    return chosen_clues


def create_red_herring(
    clue_type: str,
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, str],
    used_red_herrings: list[str],
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    n_objects: int,
    n_attributes: int,
    red_herring_clues_dict: dict[str, str],
) -> tuple[str, list[str]]:
    """Complete a red herring clue.

    This can include attributes from the solution or red herring attributes, but no clues will add new information about the solution.

    Args:
        clue_type: Type of red herring clue as a string.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the clue as a dictionary of fact titles and descriptions.
        used_red_herrings: Attributes that have already been used in red herring clues as a list of strings.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings

    Returns:
        A tuple (full_clue, used_red_herring_attributes), where:
            clue: The completed red herring clue as a string.
            used_red_herring_attributes: Attributes that have already been used in red herring clues as a list of strings.

    TODO: Do not select something already in used_red_herrings

    """
    clue_description = red_herring_clues_dict[clue_type]

    if clue_type in ("fact", "object_fact"):
        # Choose a red herring fact
        fact_key = sample(
            [
                herring
                for herring in sorted(red_herring_facts)
                if herring not in used_red_herrings
            ],
            1,
        )[0]
        chosen_fact = red_herring_facts[fact_key]

        # Choose an object to describe
        i_objects = sample(list(range(n_objects)), 1)

        # Choose an attribute from the solution
        _, object_attributes_desc = describe_random_attributes(
            chosen_attributes=chosen_attributes,
            chosen_attributes_descs=chosen_attributes_descs,
            i_objects=i_objects,
            n_attributes=n_attributes,
        )

        used_red_herrings.append(fact_key)

        # Create the clue
        if clue_type == "fact":
            full_clue = clue_description.format(fact=chosen_fact)
        elif clue_type == "object_fact":
            full_clue = clue_description.format(
                attribute_desc=object_attributes_desc[0], fact=chosen_fact
            )

    elif clue_type in (
        "same_herring",
        "next_to_herring",
        "friends",
        "found_at",
        "not_at",
    ):
        # Choose an object to describe
        i_objects = sample(list(range(n_objects)), 1)

        if clue_type not in ("found_at", "not_at"):
            # Choose an attribute from the solution
            _, object_attributes_desc = describe_random_attributes(
                chosen_attributes=chosen_attributes,
                chosen_attributes_descs=chosen_attributes_descs,
                i_objects=i_objects,
                n_attributes=n_attributes,
            )

        # Choose a red herring attribute
        red_herring_attribute_key = sample(
            [
                herring
                for herring in sorted(red_herring_attributes)
                if herring not in used_red_herrings
            ],
            1,
        )[0]

        # Choose a description based on the sentence structure in the clue type
        # E.g. "har ikke en hund" vs. "ikke har en hund" in Danish
        if clue_type in ("found_at", "not_at", "next_to_herring", "friends"):
            desc_index = 0
        elif clue_type == "same_herring":
            desc_index = 1

        attribute_desc_herring: str = red_herring_attributes[red_herring_attribute_key][
            desc_index
        ]

        used_red_herrings.append(red_herring_attribute_key)

        # Create the clue
        if clue_type in ("found_at", "not_at"):
            full_clue = clue_description.format(
                attribute_desc_herring=attribute_desc_herring, i_object=i_objects[0] + 1
            )

        else:
            full_clue = clue_description.format(
                attribute_desc=object_attributes_desc[0],
                attribute_desc_herring=attribute_desc_herring,
            )

    elif clue_type == "double_herring":
        # Choose two red herring attributes
        red_herring_attribute_keys = sample(
            [
                herring
                for herring in sorted(red_herring_attributes)
                if herring not in used_red_herrings
            ],
            2,
        )
        attribute_desc_herring_1 = red_herring_attributes[
            red_herring_attribute_keys[0]
        ][0]
        attribute_desc_herring_2 = red_herring_attributes[
            red_herring_attribute_keys[1]
        ][1]

        for herring in red_herring_attribute_keys:
            used_red_herrings.append(herring)

        # Create the clue
        full_clue = clue_description.format(
            attribute_desc_herring_1=attribute_desc_herring_1,
            attribute_desc_herring_2=attribute_desc_herring_2,
        )

    else:
        raise ValueError(f"Invalid red herring clue type '{clue_type}'")

    return full_clue, used_red_herrings
