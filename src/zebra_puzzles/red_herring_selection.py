"""Module for selecting red herrings for a zebra puzzle."""

import numpy as np


def choose_red_herrings(
    red_herring_info: tuple[int, list[str], dict[str, dict[str, str]], list[str]],
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
) -> list[str]:
    """Choose red herrings for a zebra puzzle.

    Args:
        red_herring_info: Information about red herrings as a tuple (n_red_herring_clues, red_herring_clues, red_herring_attributes, red_herring_facts), where:
            n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
            red_herring_clues: Possible red herring clue types to include in the puzzle as a list of strings.
            red_herring_attributes: Possible red herring attributes as a dictionary of dictionaries.
            red_herring_facts: Possible red herring facts to include in the puzzle as a list of strings.
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.

    Returns:
        Chosen red herring clues for the zebra puzzle as a list of strings.

    """
    (
        n_red_herring_clues,
        red_herring_clues,
        red_herring_attributes,
        red_herring_facts,
    ) = red_herring_info

    chosen_red_herring_clues = []
    for i in range(n_red_herring_clues):
        chosen_red_herring_clues.append("This is a red herring clue.")

    return chosen_red_herring_clues
