"""Module for selecting red herrings for a zebra puzzle."""

from random import sample

import numpy as np

from zebra_puzzles.puzzle_creation.clue_selection import (
    describe_random_attributes,
    get_clue_probabilities,
)


def choose_red_herrings(
    n_red_herring_clues: int,
    red_herring_clues_dict: dict[str, str],
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, list[str]],
    red_herring_clue_weights: dict[str, float],
    red_herring_cases_dict: dict[str, list[str]],
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    n_objects: int,
    n_attributes: int,
    case_to_index: dict[str, int],
    red_herring_subject_cases: dict[str, str] | None = None,
    same_herring_templates: dict[str, str] | None = None,
    double_herring_templates: dict[str, str] | None = None,
    attribute_case_to_index: dict[str, int] | None = None,
) -> tuple[list[str], list[str]]:
    """Choose red herrings for a zebra puzzle.

    Args:
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the puzzle as a dictionary of fact titles and a list of description strings.
        red_herring_clue_weights: Weights for red herring clue selection as a dictionary containing a title and a weight for each clue type.
        red_herring_cases_dict: A dictionary containing the red herring clue type as a key and a list of grammatical cases for clue attributes as values.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        case_to_index: Mapping from grammatical case names to red herring attribute description list indices.
        red_herring_subject_cases: Optional per-red-herring subject case override for same_herring/double_herring clues. Maps a red herring attribute key to the case name to use for the grammatical subject when that red herring is the predicate.
        same_herring_templates: Optional per-red-herring template override for same_herring clues. Maps a red herring attribute key to a clue template string to use instead of red_herring_clues_dict["same_herring"] when that red herring is the predicate.
        double_herring_templates: Optional per-red-herring template override for double_herring clues, analogous to same_herring_templates but keyed by the second (predicate) red herring.
        attribute_case_to_index: Mapping from grammatical case names to regular attribute description list indices (distinct from case_to_index, which uses the red herring case ordering). Required to resolve red_herring_subject_cases correctly for same_herring, since its subject is a regular attribute, not a red herring one — the two orderings can disagree on any case beyond "nom" (e.g. red herrings have no "is_not" slot, shifting every later index by one).

    Returns:
        A tuple (chosen_clues, chosen_clue_types), where:
            chosen_clues: The completed red herring clues as a list of strings.
            chosen_clue_types: The types of red herring clues as a list of strings.
    """
    # Get the probability of selecting each clue type
    _, clue_probabilities = get_clue_probabilities(
        clue_weights=red_herring_clue_weights,
        clues_dict=red_herring_clues_dict,
        n_objects=n_objects,
        n_attributes=n_attributes,
    )
    red_herring_clues_dict_keys = sorted(red_herring_clues_dict)
    clue_probabilities_values = [
        clue_probabilities[clue] for clue in red_herring_clues_dict_keys
    ]

    chosen_clues: list[str] = []
    chosen_clue_types: list[str] = []
    used_red_herrings: list[str] = []
    for _ in range(n_red_herring_clues):
        # Choose a red herring clue type
        clue_type = str(
            np.random.choice(red_herring_clues_dict_keys, p=clue_probabilities_values)
        )

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
            red_herring_cases_dict=red_herring_cases_dict,
            case_to_index=case_to_index,
            red_herring_subject_cases=red_herring_subject_cases,
            same_herring_templates=same_herring_templates,
            double_herring_templates=double_herring_templates,
            attribute_case_to_index=attribute_case_to_index,
        )

        chosen_clues.append(clue)
        chosen_clue_types.append(clue_type)

    return chosen_clues, chosen_clue_types


def create_red_herring(
    clue_type: str,
    red_herring_attributes: dict[str, list[str]],
    red_herring_facts: dict[str, list[str]],
    used_red_herrings: list[str],
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    n_objects: int,
    n_attributes: int,
    red_herring_clues_dict: dict[str, str],
    red_herring_cases_dict: dict[str, list[str]],
    case_to_index: dict[str, int],
    red_herring_subject_cases: dict[str, str] | None = None,
    same_herring_templates: dict[str, str] | None = None,
    double_herring_templates: dict[str, str] | None = None,
    attribute_case_to_index: dict[str, int] | None = None,
) -> tuple[str, list[str]]:
    """Complete a red herring clue.

    This can include attributes from the solution or red herring attributes, but no clues will add new information about the solution.

    Args:
        clue_type: Type of red herring clue as a string.
        red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
        red_herring_facts: Possible red herring facts to include in the clue as a dictionary of fact titles and a list of description strings.
        used_red_herrings: Attributes that have already been used in red herring clues as a list of strings.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.
        red_herring_clues_dict: Possible red herring clue types to include in the puzzle as a list of strings
        red_herring_cases_dict: A dictionary containing the red herring clue type as a key and a list of grammatical cases for clue attributes as values.
        case_to_index: Mapping from grammatical case names to attribute description list indices.
        red_herring_subject_cases: Optional per-red-herring subject case override for same_herring/double_herring clues. Maps a red herring attribute key to the case name to use for the subject when that red herring is the predicate.
        same_herring_templates: Optional per-red-herring template override for same_herring clues. Maps a red herring attribute key to a clue template string to use instead of red_herring_clues_dict["same_herring"] when that red herring is the predicate.
        double_herring_templates: Optional per-red-herring template override for double_herring clues, analogous to same_herring_templates but keyed by the second (predicate) red herring.
        attribute_case_to_index: Mapping from grammatical case names to regular attribute description list indices. Used (instead of case_to_index) to resolve red_herring_subject_cases for same_herring, since its subject is a regular attribute — the red herring case ordering can disagree with it on any case beyond "nom".

    Returns:
        A tuple (full_clue, used_red_herring_attributes), where:
            clue: The completed red herring clue as a string.
            used_red_herring_attributes: Attributes that have already been used in red herring clues as a list of strings.

    # NOTE: More red herring types could be added. For example, types corresponding to more of the normal clue types.
    """
    clue_description = red_herring_clues_dict[clue_type]

    # Define the order of grammatical cases in clue descriptions

    # Choose desc indices based on clue type and grammatical case in clue_cases_dict
    cases = red_herring_cases_dict[clue_type]
    desc_indices: list[int] = [case_to_index[case] for case in cases]

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
        chosen_fact_all_versions = red_herring_facts[fact_key]

        if clue_type == "object_fact":
            # Choose an object to describe
            i_objects = sample(list(range(n_objects)), 1)

            # Choose an attribute from the solution
            _, object_attributes_desc = describe_random_attributes(
                chosen_attributes=chosen_attributes,
                chosen_attributes_descs=chosen_attributes_descs,
                i_objects=i_objects,
                n_attributes=n_attributes,
                desc_indices=desc_indices,
            )
            if len(chosen_fact_all_versions) > 1:
                # If the fact has two descriptions, use the second one for object_fact
                chosen_fact = chosen_fact_all_versions[1]
            else:
                chosen_fact = chosen_fact_all_versions[0]
        else:
            chosen_fact = chosen_fact_all_versions[0]

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
        "herring_found_at",
        "herring_not_at",
    ):
        # Choose an object to describe
        i_objects = sample(list(range(n_objects)), 1)

        # For same_herring, choose the red herring attribute before fetching the subject: its
        # identity may override the subject's case and the clue template itself (e.g. Hindi's
        # possessive "X की Y है" constructions, which need an oblique-marked subject instead of
        # the default nominative, plus a custom template per red herring since the possessive
        # marker's gender varies by what's possessed).
        subject_desc_index = desc_indices[0]
        if clue_type == "same_herring":
            red_herring_attribute_key = sample(
                [
                    herring
                    for herring in sorted(red_herring_attributes)
                    if herring not in used_red_herrings
                ],
                1,
            )[0]
            if (
                red_herring_subject_cases
                and red_herring_attribute_key in red_herring_subject_cases
                and attribute_case_to_index is not None
            ):
                # The subject here is a regular attribute, so its case must be resolved via
                # attribute_case_to_index, not case_to_index (the red herring ordering) — the two
                # can disagree on any case beyond "nom" (red herrings have no "is_not" slot).
                subject_desc_index = attribute_case_to_index[
                    red_herring_subject_cases[red_herring_attribute_key]
                ]
            if (
                same_herring_templates
                and red_herring_attribute_key in same_herring_templates
            ):
                clue_description = same_herring_templates[red_herring_attribute_key]

        if clue_type not in ("herring_found_at", "herring_not_at"):
            # First object is a solution attribute, the second is a red herring attribute
            # Choose an attribute from the solution
            _, object_attributes_desc = describe_random_attributes(
                chosen_attributes=chosen_attributes,
                chosen_attributes_descs=chosen_attributes_descs,
                i_objects=i_objects,
                n_attributes=n_attributes,
                desc_indices=[subject_desc_index],
            )

        if clue_type != "same_herring":
            # Choose a red herring attribute
            red_herring_attribute_key = sample(
                [
                    herring
                    for herring in sorted(red_herring_attributes)
                    if herring not in used_red_herrings
                ],
                1,
            )[0]

        # Choose a red herring description based on the sentence structure in the clue type
        if clue_type == "same_herring":
            red_herring_desc_index = case_to_index["is"]
        elif clue_type in ("herring_found_at", "herring_not_at"):
            red_herring_desc_index = desc_indices[0]
        else:
            red_herring_desc_index = desc_indices[1]

        attribute_desc_herring: str = red_herring_attributes[red_herring_attribute_key][
            red_herring_desc_index
        ]

        used_red_herrings.append(red_herring_attribute_key)

        # Create the clue
        if clue_type in ("herring_found_at", "herring_not_at"):
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

        # The second (predicate) red herring's identity may override the subject's case and the
        # clue template itself, same as for same_herring above.
        subject_desc_index = desc_indices[0]
        predicate_herring_key = red_herring_attribute_keys[1]
        if (
            red_herring_subject_cases
            and predicate_herring_key in red_herring_subject_cases
        ):
            subject_desc_index = case_to_index[
                red_herring_subject_cases[predicate_herring_key]
            ]
        if (
            double_herring_templates
            and predicate_herring_key in double_herring_templates
        ):
            clue_description = double_herring_templates[predicate_herring_key]

        attribute_desc_herring_1 = red_herring_attributes[
            red_herring_attribute_keys[0]
        ][subject_desc_index]
        attribute_desc_herring_2 = red_herring_attributes[predicate_herring_key][
            case_to_index["is"]
        ]

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
