"""Module for evaluation."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Type

import numpy as np
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, create_model
from tqdm import tqdm

from zebra_puzzles.zebra_utils import clean_folder, save_dataset

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
    file_paths: list[Path],
    model: str,
    theme: str,
) -> None:
    """Evaluate a dataset of zebra puzzles.

    Args:
        n_puzzles: Number of puzzles to evaluate as an integer.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        file_paths: Iterator of file paths to the dataset files.
        model: The model to use for the evaluation as a string.
        theme: The theme of the puzzles

    """
    # Create reponse file names
    response_filenames = [f"{file_path.stem}_response.json" for file_path in file_paths]

    # Clean reponses folder
    clean_folder(folder="responses", keep_files=response_filenames)

    # Initialize scores
    puzzle_scores: np.ndarray = np.zeros(n_puzzles)
    cell_scores: np.ndarray = np.zeros(n_puzzles)

    # Evaluate each puzzle
    for i, file_path in tqdm(enumerate(file_paths), total=n_puzzles):
        puzzle_score, cell_score = evaluate_single_puzzle(
            file_path=file_path,
            n_objects=n_objects,
            n_attributes=n_attributes,
            model=model,
            response_filename=response_filenames[i],
        )
        puzzle_scores[i] = puzzle_score
        cell_scores[i] = cell_score

    # Compute summary metrics
    metrics = compute_metrics(puzzle_scores=puzzle_scores, cell_scores=cell_scores)

    # Save scores
    score_str = format_scores(
        puzzle_scores=puzzle_scores, cell_scores=cell_scores, metrics=metrics
    )

    filename = (
        f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_puzzles}.txt"
    )
    save_dataset(data=score_str, filename=filename, folder="scores")


def compute_metrics(
    puzzle_scores: np.ndarray, cell_scores: np.ndarray
) -> dict[str, float]:
    """Compute the metrics.

    Args:
        puzzle_scores: Puzzle scores as a numpy array.
        cell_scores: Cell scores as a numpy array.

    Returns:
        Metrics as a dictionary.
    """
    mean_puzzle_score = float(np.mean(puzzle_scores))
    mean_cell_score = float(np.mean(cell_scores))

    metrics = {
        "mean_puzzle_score": mean_puzzle_score,
        "mean_cell_score": mean_cell_score,
    }

    return metrics


def format_scores(
    puzzle_scores: np.ndarray, cell_scores: np.ndarray, metrics: dict[str, float]
) -> str:
    """Format the scores.

    Args:
        puzzle_scores: Puzzle scores as a numpy array.
        cell_scores: Cell scores as a numpy array.
        metrics: Metrics as a dictionary.

    Returns:
        A formatted string of the scores.
    """
    score_str = "Puzzle Scores\n"
    score_str += "-------------\n"
    score_str += "Metrics\n"

    metrics_str = json.dumps(metrics, indent=4)

    score_str += metrics_str

    score_str += "\n-------------\n"
    score_str += "Single puzzle scores\n"

    for i, (puzzle_score, cell_score) in enumerate(zip(puzzle_scores, cell_scores)):
        score_str += f"Puzzle {i}: {puzzle_score} {cell_score}\n"

    return score_str


def evaluate_single_puzzle(
    file_path: Path,
    n_objects: int,
    n_attributes: int,
    model: str,
    response_filename: str,
) -> tuple[float, float]:
    """Evaluate a dataset of zebra puzzles.

    Args:
        file_path: Path to the dataset file.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an
        model: The model to use for the evaluation as a
        response_filename: The name of the response file.

    Returns:
        A tuple (puzzle_score, cell_score), where:
            puzzle_score: A puzzle-level score as a float.
            cell_score: A cell-level score as a float.
    """
    logging.getLogger("httpx").setLevel(logging.ERROR)

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
    try:
        response = client.beta.chat.completions.parse(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0,
            seed=42,
            response_format=OutputFormat,
        )
    except BadRequestError as e:
        if "'temperature' is not supported" in str(e):
            response = client.beta.chat.completions.parse(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                seed=42,
                response_format=OutputFormat,
            )
        else:
            raise e

    # Reformat response
    output = OutputFormat.model_validate(response.choices[0].message.parsed)

    # Change the format of solution to OutputFormat

    solution_json = json.loads(solution)

    solution_json = OutputFormat.model_validate(solution_json)

    puzzle_score, cell_score = compare_solutions(output, solution_json)

    # Save the output
    output_str = json.dumps(output.model_dump(), indent=4)
    save_dataset(data=output_str, filename=response_filename, folder="responses")

    return puzzle_score, cell_score


def compare_solutions(output: BaseModel, solution: BaseModel) -> tuple[float, float]:
    """Compare the output to the solution.

    Args:
        output: The output as a dictionary.
        solution: The solution as a dictionary.

    Returns:
        A tuple (puzzle_score, cell_score), where:
            puzzle_score: A puzzle-level score as a float.
            cell_score: A cell-level score as a float.
    """
    # Extract solution arrays

    # Strip whitespace

    # Compare the output to the solution

    if output == solution:
        puzzle_score = 1
    else:
        puzzle_score = 0

    cell_score = puzzle_score

    # Sort and compare again

    return puzzle_score, cell_score
