"""Utility module for generating zebra puzzles."""

from random import sample
from typing import Dict, List, Tuple


def define_attributes(theme, language) -> Dict:
    """Define attributes for the puzzle.

    Args: Theme and language from the config file.
    Returns: Attributes as a dictionary of dictionaries.
        The outer dictionary has the attribute category as key, and the inner dictionary has the attribute value as key and the attribute description as value.
        The attribute description completes the phrase "Personen der..." in Danish.

    NOTE: More languages and themes can be added.
    """
    if theme == "houses":
        if language == "Dansk":
            attributes = {
                "Land": {
                    "Danmark": "danskeren",
                    "Sverige": "svenskeren",
                    "Letland": "personen fra Letland",
                    "Frankrig": "franskmanden",
                },
                "Hobby": {
                    "Klatring": "klatrer",
                    "Maling": "maler",
                    "Brætspil": "spiller brætspil",
                    "Tennis": "spiller tennis",
                },
                "Kæledyr": {
                    "Kat": "har en kat",
                    "Hund": "har en hund",
                    "Kanin": "har en kanin",
                    "Zebra": "har en zebra",
                },
            }
        else:
            raise ValueError("Unsupported language '{language}' for theme '{theme}'")
    else:
        raise ValueError("Unsupported theme '{theme}'")
    return attributes


def define_rules(rules_included) -> List[str]:
    """Define rules for the puzzle.

    Args: rule types to include/exclude from the config file.
    NOTE: In the future, we can support more rules and selection of rules in the config file.
    TODO: Implement rule functions.
    """
    if rules_included == "all":
        rules = [
            "Found_at",
            "Not_at",
            "Next_to",
            "Not_next_to",
            "Left_of",
            "Right_of",
            "Not_left_of",
            "Not_right_of",
            "Same_house",
            "Not_same_house",
            "Between",
            "Not_between",
        ]
    else:
        raise ValueError("Unsupported rules '{rules_included}'")
    return rules


def generate_solution(attributes, N_objects, N_attributes) -> Tuple[List, List, List]:
    """Generate the solution to a zebra puzzle.

    Args: Attributes as a dictionary of dictionaries.
    Returns: A solution to a zebra puzzle as a matrix of attribute values.
    """
    # Choose a category for each attribute
    chosen_categories = sample(list(attributes.keys()), k=N_attributes)

    # Choose attribute values for each category
    chosen_attributes = [
        sample(list(attributes[cat].keys()), k=N_objects) for cat in chosen_categories
    ]

    # Add the object indices to the solution
    indices = [str(x) for x in list(range(N_objects))]
    solution = [indices] + chosen_attributes

    return solution, chosen_categories, chosen_attributes


def generate_puzzle(solution, rules, chosen_categories, chosen_attributes) -> str:
    """Generate a zebra puzzle.

    Args: Solution to a zebra puzzle as a matrix of attribute values, rules, chosen categories, and chosen attributes (solution without indices).
    Returns: A zebra puzzle as a string.
    TODO: Implement the generation of the puzzle.
    """
    puzzle = "1. This is an example. \n 2. This is the second part of the example."
    return puzzle


def complete_prompt(
    language, theme, puzzle, chosen_attributes, chosen_categories, N_objects
) -> str:
    """Complete the prompt for the zebra puzzle.

    Args: Language, theme, puzzle, chosen attributes, chosen categories, and number of objects.
    Returns: A prompt for the zebra puzzle as a string.
    TODO: Improve the prompt incl. by adding a description of the output format.
    NOTE: We can add more themes and languages.
    """
    if language == "Dansk":
        prompt_intro = "Her følger en opgave. "
        if theme == "houses":
            prompt_main = """Der er {N_objects} huse på en række. I hvert hus bor en person med en bestemte egenskaber, som er forskellige fra de andre.
                             Disse egenskaber er i kategorierne {chosen_categories} og inkluderer {chosen_attributes}. Her følger en række hints:"""
            prompt_outro = "Hvem har hvilke egenskaber og bor i hvilket hus?"
        else:
            raise ValueError("Unsupported language '{language}' for theme '{theme}'")
    else:
        raise ValueError("Unsupported theme '{theme}'")

    prompt = prompt_intro + prompt_main + puzzle + prompt_outro
    return prompt
