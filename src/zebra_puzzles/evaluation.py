"""Module for evaluation."""

import ast
import os
from pathlib import Path
from typing import Any, Iterator, Type

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, create_model

# Load environment variables to get the API key
load_dotenv()


def generate_output_format_class(n_objects: int, n_attributes: int) -> Type[BaseModel]:
    """Generate the OutputFormat class based on the number of objects.

    The OutputFormat class is a dynamically generated Pydantic model that represents the output format of the LLM.

    The format will be:
        object_1: list[str]
        object_2: list[str]
        ...

    Args:
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        A dynamically generated OutputFormat class.
    """
    fields: dict[str, Any] = {
        f"object_{i + 1}": (list[str], ...) for i in range(n_objects)
    }

    OutputFormat = create_model("OutputFormat", **fields)

    return OutputFormat


def evaluate_all(
    n_puzzles: int,
    n_objects: int,
    n_attributes: int,
    file_paths: Iterator[Path],
    model: str,
) -> float:
    """Evaluate a dataset of zebra puzzles.

    Args:
        n_puzzles: Number of puzzles to evaluate as an integer.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        file_paths: Iterator of file paths to the dataset files.
        model: The model to use for the evaluation as a string.

    Returns:
        The mean score of the dataset as a float.

    """
    scores = np.zeros(n_puzzles)
    for i, file_path in enumerate(file_paths):
        score = evaluate_single_puzzle(
            file_path=file_path,
            n_objects=n_objects,
            n_attributes=n_attributes,
            model=model,
        )
        scores[i] = score

    # Mean
    mean_score = float(np.mean(scores))

    print(f"Mean score: {mean_score}")

    return mean_score


def evaluate_single_puzzle(
    file_path: Path, n_objects: int, n_attributes: int, model: str
) -> float:
    """Evaluate a dataset of zebra puzzles.

    Args:
        file_path: Path to the dataset file.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an
        model: The model to use for the evaluation as a

    Returns:
        A score for the puzzle as a float.
    """
    # Load the prompt
    with file_path.open() as file:
        prompt = file.read()

    with file_path.with_stem(f"{file_path.stem}_solution").open() as file:
        solution = file.read()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Generate the dynamic OutputFormat class
    OutputFormat = generate_output_format_class(
        n_objects=n_objects, n_attributes=n_attributes
    )

    # Generate LLM output
    response = client.beta.chat.completions.parse(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0,
        seed=42,
        response_format=OutputFormat,
    )

    # Reformat response
    output = OutputFormat.model_validate(response.choices[0].message.parsed)

    # Change the format of solution to OutputFormat
    # Format the solution as a dict
    # TODO: Convert to OutputFormat in a nicer way
    solution.replace("\n", " ")
    # Delete last comma
    solution.replace(", }", "}")

    solution_dict: dict = ast.literal_eval(solution)
    solution_json = OutputFormat.model_validate(solution_dict)

    score = compare_solutions(output, solution_json)

    return score


def compare_solutions(output: BaseModel, solution: BaseModel) -> float:
    """Compare the output to the solution.

    Args:
        output: The output as a dictionary.
        solution: The solution as a dictionary.

    Returns:
        A score for the puzzle as a float.
    """
    # Extract solution arrays

    # Strip whitespace

    # Compare the output to the solution

    score = 0
    return score
