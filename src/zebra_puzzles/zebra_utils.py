"""Utility module for generating and evaluating zebra puzzles."""

import json
import logging
import os
import time
from random import choices, sample, shuffle
from typing import Any, Type

import numpy as np
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError, create_model

log = logging.getLogger(__name__)


def generate_solution(
    attributes: dict[str, dict[str, list[str]]], n_objects: int, n_attributes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the solution to a zebra puzzle.

    Chooses categories and assigns attribute values to each object in the solution. Uses 1-based object indices.

    Args:
        attributes: Attributes as a dictionary of dictionaries.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        A tuple (solution, chosen_categories, chosen_attributes, chosen_attributes_descs), where:
            solution: A solution to a zebra puzzle as a matrix of object indices and chosen attributes. The dimensions are n_objects x (1 + n_attributes).
            chosen_categories: Categories chosen for the solution as a ndarray of strings of length n_attributes.
            chosen_attributes: Attribute values chosen for the solution as a matrix of strings. The dimensions are n_objects x n_attributes.
            chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix of strings. 3 versions are provided per description for different sentence structures. The dimensions are 3 x n_objects x n_attributes.
    """
    # Get the possible categories
    all_categories = np.array(list(attributes.keys()))

    # Choose a category for each attribute while maintaining the order of the categories
    chosen_cat_indices = sorted(
        np.array(sample(range(len(all_categories)), k=n_attributes))
    )
    chosen_categories = all_categories[chosen_cat_indices]

    # Choose attribute values for each category
    chosen_attributes = np.array(
        [sample(list(attributes[cat].keys()), k=n_objects) for cat in chosen_categories]
    )

    # Find the attribute descriptions for each attribute in each category
    chosen_attributes_descs = np.array(
        [
            [attributes[cat][key] for key in chosen_attributes[i]]
            for i, cat in enumerate(chosen_categories)
        ]
    )

    # Transpose the attribute matrices
    chosen_attributes = chosen_attributes.T
    chosen_attributes_descs = chosen_attributes_descs.T

    # Add a column of 1-based object indices to the solution
    solution = np.hstack(
        (np.array([list(range(1, n_objects + 1))]).T, chosen_attributes)
    )

    return solution, chosen_categories, chosen_attributes, chosen_attributes_descs


def format_solution_as_json(solution: np.ndarray) -> str:
    """Format the solution as a json dictionary.

    Args:
        solution: Solution to the zebra puzzle as a matrix of object indices and chosen attributes.

    Returns:
        The solution as a json dictionary
    """
    solution_dict = {f"object_{row[0].item()}": row[1:].tolist() for row in solution}
    solution_json = json.dumps(solution_dict, indent=4, ensure_ascii=False)
    return solution_json


def create_solution_template(n_objects: int, chosen_categories: np.ndarray) -> str:
    """Create a solution template for a zebra puzzle.

    For example:
    {
    "object_1": ["attribute_1", "attribute_2"],
    "object_2": ["attribute_1", "attribute_2"]
    }

    Assumes the maximum string length is 100 characters.

    Args:
        n_objects: Number of objects in the puzzle.
        chosen_categories: Categories chosen for the solution.

    Returns:
        The solution template as a string.
    """
    # U100 is a Unicode string with a maximum length of 100 characters
    example_solution = np.zeros((n_objects, len(chosen_categories) + 1), dtype="U100")
    for i in range(n_objects):
        example_solution[i, 0] = f"{i + 1}"
        for j, cat in enumerate(chosen_categories):
            example_solution[i, j + 1] = f"{cat}_{i + 1}"

    solution_template = format_solution_as_json(example_solution)

    return solution_template


def describe_random_attributes(
    chosen_attributes: np.ndarray,
    chosen_attributes_descs: np.ndarray,
    i_objects: list[int],
    n_attributes: int,
    desc_indices: list[int],
    diff_cat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Get a random attribute description for an object.

    Choose a random attribute for each object with indices given by i_objects. Looks up attributes from chosen_attributes in the attributes dict.

    The attributes are sorted by category to be presented in the preferred order.

    Assumes the maximum string length is 100 characters.

    Args:
        chosen_attributes: Attribute values chosen for the solution as a matrix.
        chosen_attributes_descs: Attribute descriptions for the chosen attributes as a matrix.
        i_objects: The index of the object to select an attribute from.
        n_attributes: Number of attributes per object.
        diff_cat: If True, the output attributes must belong to different categories.
        desc_indices: A list of indeces of the descriptions to use for each object in the clue.

    Returns:
        A tuple (random_attributes, random_attributes_desc), where:
            random_attributes: A list of strings contraining one random attribute per object.
            random_attributes_desc: A list of strings using the attributes to describe the objects.
    """
    # Number of objects in the clue
    n_clue_objects = len(i_objects)

    if diff_cat:
        i_attributes = sample(list(range(n_attributes)), k=n_clue_objects)
    else:
        i_attributes = choices(list(range(n_attributes)), k=n_clue_objects)

    # Keep the order of the categories
    i_attributes.sort()

    # Initialize the random attributes as type 'object' to avoid setting a maximum string length
    # U100 is a Unicode string with a maximum length of 100 characters
    random_attributes = np.empty((n_clue_objects), dtype="U100")
    random_attributes_desc = np.empty((n_clue_objects), dtype="U100")

    for i, (i_obj, i_attr, desc_index) in enumerate(
        zip(i_objects, i_attributes, desc_indices)
    ):
        random_attributes[i] = chosen_attributes[i_obj][i_attr]
        random_attributes_desc[i] = chosen_attributes_descs[desc_index][i_obj][i_attr]

    return random_attributes, random_attributes_desc


def generate_output_format_class(n_objects: int) -> Type[BaseModel]:
    """Generate the OutputFormat class based on the number of objects.

    The OutputFormat class is a dynamically generated Pydantic model that represents the output format of the LLM.

    The format will be:
        object_1: list[str]
        object_2: list[str]
        ...

    Args:
        n_objects: Number of objects in the puzzle.

    Returns:
        A dynamically generated OutputFormat class.
    """
    fields: dict[str, Any] = {
        f"object_{i + 1}": (list[str], ...) for i in range(n_objects)
    }

    OutputFormat = create_model("OutputFormat", **fields)

    return OutputFormat


def shuffle_clues(
    chosen_clues: list[str],
    chosen_red_herring_clues: list[str],
    chosen_clue_types: list[str],
    chosen_red_herring_clue_types: list[str],
) -> tuple[list[str], str, str]:
    """Shuffle the clues and red herrings and return the indices of the red herrings.

    The clue types are also shuffled and returned as a string.

    Args:
        chosen_clues: Chosen clues for the zebra puzzle as a list of strings.
        chosen_red_herring_clues: Chosen red herring clues for the zebra puzzle as a list of strings.
        chosen_clue_types: Chosen clue types for the zebra puzzle as a list of strings.
        chosen_red_herring_clue_types: Chosen red herring clue types for the zebra puzzle as a list of strings.

    Returns:
        A tuple (chosen_clues, red_herring_indices_str, chosen_clue_types_str), where:
            chosen_clues: Shuffled clues for the zebra puzzle as a list of strings incl. red herrings.
            red_herring_indices_str: String of indices of the red herrings in the shuffled list of clues.
            chosen_clue_types_str: String of comma-separated clue types chosen for the puzzle.
    """
    # Combine clues and red herrings
    chosen_clues = chosen_clues + chosen_red_herring_clues
    chosen_clue_types = chosen_clue_types + chosen_red_herring_clue_types

    # Shuffle the clues and red herrings
    i_shuffled = list(range(len(chosen_clues)))
    shuffle(i_shuffled)
    chosen_clues = [chosen_clues[i] for i in i_shuffled]
    chosen_clue_types = [chosen_clue_types[i] for i in i_shuffled]

    # Get the new indices of the red herrings
    i_red_herrings = [
        new_i
        for new_i, old_i in enumerate(i_shuffled)
        if old_i >= len(chosen_clues) - len(chosen_red_herring_clues)
    ]
    red_herring_indices_str = ", ".join([str(i) for i in i_red_herrings])

    chosen_clue_types_str = ", ".join(chosen_clue_types)

    return chosen_clues, red_herring_indices_str, chosen_clue_types_str


def round_using_std(value: float, std: float) -> tuple[str, str]:
    """Round a value to match a standard deviation.

    Assumes the value is not much larger than 1.

    Args:
        value: The value to round as a float.
        std: The standard deviation to match as a float.

    Returns:
        A tuple (value, std) where:
            value: The rounded value as a string.
            std: The rounded standard deviation as a string.
    """
    std_rounded = np.format_float_positional(std, precision=1, fractional=False)

    # If the standard deviation is 0, we get the same score for all puzzles. In this case, just use 2 significant digits.
    if std_rounded == "0.":
        value_precision = 2
    else:
        # Get the number of decimal places in the standard deviation
        value_precision = len(str(std_rounded).split(".")[1])

    # Round the value to the same number of decimal places as the standard deviation
    value_rounded = np.format_float_positional(
        value, precision=value_precision, fractional=True
    )

    # Include trailing zeros
    if std_rounded == "0.":
        # Set n_decimal_places to the value_precision minus the number of non-zero digits in the value before the decimal point
        digits_before_decimal = value_rounded.split(".")[0]
        n_nonzero_digits_before_decimal = len(
            [d for d in digits_before_decimal if d != "0"]
        )
        n_decimal_places = 2 - n_nonzero_digits_before_decimal
    else:
        # Get n_decimal_places as the number of decimal places in std_rounded
        n_decimal_places = len(std_rounded.split(".")[1])
    n_trailing_zeros = n_decimal_places - len(value_rounded.split(".")[1])
    if n_trailing_zeros > 0:
        value_rounded += "0" * n_trailing_zeros

    # Turn 1. into 1 and 0. into 0
    if value_rounded[-1] == ".":
        value_rounded = value_rounded[:-1]
    if std_rounded[-1] == ".":
        std_rounded = std_rounded[:-1]

    return value_rounded, std_rounded


def bernoulli_std(n_trials: int, n_successes: int) -> tuple[float, float]:
    """Calculate the standard deviation of success and probability of success in a bernoulli trial.

    We assume puzzle scores are independent Bernoulli trials, each with the same probability of success.

    Args:
        n_trials: Number of trials.
        n_successes: Number of successes.

    Returns:
        A tuple (std_one_trial, std_p), where:
            std_one_trial: The standard deviation of the bernoulli distribution (of 0's and 1's)
            std_p: The standard error of the outcomes i.e. the standard deviation of the probability of success.
    """
    # Calculate the probability of success
    p = n_successes / n_trials

    # Calculate the error of the bernoulli distribution (of 0's and 1's)
    std_one_trial = np.sqrt(p * (1 - p))

    # Calculate the error of the mean (p)
    std_p = np.sqrt(p * (1 - p) / n_trials)

    return std_one_trial, std_p


def capitalize(text: str) -> str:
    """Capitalize the first letter of a string, while leaving the rest unchanged.

    Args:
        text: The input string to capitalize.

    Returns:
        The input string with the first letter capitalized.
    """
    if not text:
        return text
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def query_llm(
    prompt: str, model: str, response_format: Type[BaseModel], n_objects: int
) -> BaseModel:
    """Query an LLM API.

    Args:
        prompt: The prompt to use for the evaluation.
        model: The model to use for the evaluation.
        response_format: The response format as a Pydantic model.
        n_objects: The number of objects in the puzzle.

    Returns:
        The output in OutputFormat format.
    """
    logging.getLogger("httpx").setLevel(logging.ERROR)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    error_response: str = ""

    # Generate LLM output
    try:
        response = client.beta.chat.completions.parse(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0,
            seed=42,
            response_format=response_format,
            max_completion_tokens=100_000,
            reasoning_effort="medium",
        )
    except BadRequestError as first_error:
        max_retries = 5
        wait_time = 5
        retries: int = 0
        # gpt-4o-mini
        if "'Unrecognized request argument supplied: reasoning_effort'" in str(
            first_error
        ):
            for retry in range(max_retries):
                try:
                    response = client.beta.chat.completions.parse(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=0,
                        seed=42,
                        response_format=response_format,
                        max_completion_tokens=16_384,
                    )
                except (
                    InternalServerError,
                    APIError,
                    APIConnectionError,
                    RateLimitError,
                ) as e:
                    log.warning(f"\nRetry no. {retry} due to:\n{e}")
                    # Wait before retrying
                    time.sleep(wait_time)
                    continue
                # Below are cases where we do not want to retry
                except APITimeoutError as e:
                    # Timeout error can indicate that the request was too complex, so this is a real failure of the model
                    error_response = str(e)
                    log.error(f"\nTimeout error:\n{error_response}")
                except Exception as e:
                    error_response = str(e)
                    log.error(f"\nAn unexpected error occurred:\n{error_response}")
                break
            else:  # If we reach this, it means we exhausted all retries
                error_response = (
                    f"Error after retrying {max_retries} times:\n{error_response}"
                )
                log.error(f"\nToo many errors occurred:\n{error_response}")
        # o3-mini
        elif "'temperature' is not supported" in str(first_error):
            for retry in range(max_retries):
                try:
                    response = client.beta.chat.completions.parse(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        seed=42,
                        response_format=response_format,
                        max_completion_tokens=100_000,
                        reasoning_effort="medium",
                    )
                except (
                    InternalServerError,
                    APIError,
                    APIConnectionError,
                    RateLimitError,
                ) as e:
                    retries += 1
                    log.info(f"\nRetry no. {retry} due to:\n{e}")
                    # Wait before retrying
                    time.sleep(wait_time)
                    continue
                # Below are cases where we do not want to retry
                except APITimeoutError as e:
                    # Timeout error can indicate that the request was too complex, so this is a real failure of the model
                    error_response = str(e)
                    log.error(f"\nTimeout error:\n{error_response}")
                except Exception as e:
                    error_response = str(e)
                    log.error(f"\nAn unexpected error occurred:\n{error_response}")

                break
            else:  # If we reach this, it means we exhausted all retries
                error_response = (
                    f"Error after retrying {max_retries} times:\n{error_response}"
                )
                log.error(f"\nToo many errors occurred:\n{error_response}")

    except Exception as e:
        error_response = str(e)
        log.error(
            f"\nAn unexpected error occurred during first API call:\n{error_response}"
        )

    # Reformat response
    try:
        # TODO: Figure out why the reponse is sometimes none for large puzzles and o3-mini
        output = response_format.model_validate(response.choices[0].message.parsed)
    except (ValidationError, AttributeError) as e:
        if error_response == "":
            error_response = f"{response}\nError message:{str(e)}"
        log.error(
            f"\nValidation error or attribute error occurred while parsing the response:\n{error_response}\n"
        )
        response_formatted: dict[str, Any] = {
            f"object_{i + 1}": [""] for i in range(n_objects)
        }
        response_formatted["object_1"] = [f"\nresponse:\n{error_response}"]

        output = response_format.model_validate(response_formatted)

    return output
