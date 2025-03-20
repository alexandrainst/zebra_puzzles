"""Module for evaluation."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Type

import numpy as np
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError, create_model
from tqdm import tqdm

from zebra_puzzles.compare_solutions import compare_solutions
from zebra_puzzles.zebra_utils import clean_folder, save_dataset

# Load environment variables to get the API key
load_dotenv()


def evaluate_all(
    n_puzzles: int,
    n_objects: int,
    n_attributes: int,
    file_paths: list[Path],
    model: str,
    theme: str,
    generate_new_responses: bool = False,
) -> None:
    """Evaluate a dataset of zebra puzzles.

    Args:
        n_puzzles: Number of puzzles to evaluate as an integer.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        file_paths: Iterator of file paths to the dataset files.
        model: The model to use for the evaluation as a string.
        theme: The theme of the puzzles.
        generate_new_responses: Whether to generate new responses or use existing ones.

    TODO: Make the script more robust in cases where the expected responses are not found.
    """
    # Create reponse file names
    response_filenames = [f"{file_path.stem}_response.json" for file_path in file_paths]

    if generate_new_responses:
        # Clean reponses folder
        clean_folder(folder="responses", keep_files=response_filenames)

    # Initialize scores
    puzzle_scores: np.ndarray = np.zeros(n_puzzles)
    cell_scores: np.ndarray = np.zeros(n_puzzles)
    best_permuted_cell_scores: np.ndarray = np.zeros(n_puzzles)

    # Evaluate each puzzle
    for i, file_path in tqdm(
        enumerate(file_paths),
        total=n_puzzles,
        desc="Evaluating",
        unit="puzzle",
        colour="#5599ff",
        ascii="░█",
    ):
        puzzle_score, cell_score, best_permuted_cell_score = evaluate_single_puzzle(
            file_path=file_path,
            n_objects=n_objects,
            n_attributes=n_attributes,
            model=model,
            response_filename=response_filenames[i],
            generate_new_responses=generate_new_responses,
        )
        puzzle_scores[i] = puzzle_score
        cell_scores[i] = cell_score
        best_permuted_cell_scores[i] = best_permuted_cell_score

    scores_all_types = [puzzle_scores, cell_scores, best_permuted_cell_scores]
    score_types = ["puzzle score", "cell score", "best permuted cell score"]

    # Compute summary metrics
    metrics = compute_metrics(
        scores_all_types=scores_all_types, score_types=score_types, n_puzzles=n_puzzles
    )

    # Save scores
    score_str = format_scores(
        scores_all_types=scores_all_types,
        score_types=score_types,
        metrics=metrics,
        n_puzzles=n_puzzles,
    )

    filename = (
        f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_puzzles}.txt"
    )
    save_dataset(data=score_str, filename=filename, folder="scores")


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


def evaluate_single_puzzle(
    file_path: Path,
    n_objects: int,
    n_attributes: int,
    model: str,
    response_filename: str,
    generate_new_responses: bool,
) -> tuple[float, float, float]:
    """Evaluate a dataset of zebra puzzles.

    Args:
        file_path: Path to the dataset file.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an
        model: The model to use for the evaluation as a
        response_filename: The name of the response file.
        generate_new_responses: Whether to generate new responses or use existing ones.

    Returns:
        A tuple (puzzle_score, cell_score), where:
            puzzle_score: A puzzle-level score as a float.
            cell_score: A cell-level score as a float.
            best_permuted_cell_score: A cell-level score as a float after trying all permutations of the objects in the response.
    """
    # Generate the dynamic OutputFormat class
    OutputFormat = generate_output_format_class(n_objects=n_objects)

    if generate_new_responses:
        output = query_LLM(
            file_path=file_path, model=model, response_format=OutputFormat
        )

    else:
        # Load an existing response
        with Path(f"responses/{response_filename}").open() as file:
            response_str = file.read()

        output = json.loads(response_str)
        output = OutputFormat.model_validate(output)

    # Load the solution

    with file_path.with_stem(f"{file_path.stem}_solution").open() as file:
        solution = file.read()

    # Change the format of solution to OutputFormat

    solution_json = json.loads(solution)

    solution_json = OutputFormat.model_validate(solution_json)

    puzzle_score, cell_score, best_permuted_cell_score = compare_solutions(
        output=output,
        solution=solution_json,
        n_objects=n_objects,
        n_attributes=n_attributes,
    )

    # Save the output
    output_str = json.dumps(output.model_dump(), indent=4)
    save_dataset(data=output_str, filename=response_filename, folder="responses")

    return puzzle_score, cell_score, best_permuted_cell_score


def query_LLM(
    file_path: Path, model: str, response_format: Type[BaseModel]
) -> BaseModel:
    """Query an LLM API.

    Args:
        file_path: Path to the dataset file.
        model: The model to use for the evaluation.
        response_format: The response format as a Pydantic model.

    Returns:
        The output in OutputFormat format.

    """
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # Load the prompt
    with file_path.open() as file:
        prompt = file.read()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Generate LLM output
    try:
        response = client.beta.chat.completions.parse(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0,
            seed=42,
            response_format=response_format,
        )
    except BadRequestError as e:
        if "'temperature' is not supported" in str(e):
            response = client.beta.chat.completions.parse(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                seed=42,
                response_format=response_format,
            )
        else:
            raise e

    # Reformat response
    try:
        output = response_format.model_validate(response.choices[0].message.parsed)
    except ValidationError as e:
        print("response:\n", response)
        print(
            "\nresponse.choices[0].message.parsed:\n",
            response.choices[0].message.parsed,
        )
        print()
        raise e

    return output


def compute_metrics(
    scores_all_types: list[np.ndarray], score_types: list[str], n_puzzles: int
) -> dict[str, tuple]:
    """Compute the metrics.

    For each score type e.g. cell score, a dictionary of metrics is computed. This dictionary includes a string describing the rounded metrics.

    Args:
        scores_all_types: Tuple of scores as numpy arrays. Each element contains the scores for a specific score type.
        score_types: List of score type names as strings.
        n_puzzles: Number of puzzles as an integer.

    Returns:
        Metrics as a dictionary of with the score type as the key, and the values being a tuple of ndarrays. The tuple contains the rounded metrics for the score type and a string describing the metrics for the score type.

    TODO: Add more metrics e.g. from sklearn.metrics
    """
    # Number of score types
    n_metrics = len(score_types)

    # Initialize metrics
    mean_scores = np.zeros(n_metrics, dtype=float)
    std_scores = np.zeros(n_metrics, dtype=float)
    std_mean_scores = np.zeros(n_metrics, dtype=float)

    # Initialize strings describing metrics for each score type
    score_strings = np.zeros(n_metrics, dtype="U100")

    for i, scores in enumerate(scores_all_types):
        # Take the mean
        mean_scores[i] = float(np.mean(scores))

        # Take the standard deviation
        std_scores[i] = float(np.std(scores, ddof=1))

        # Compute the standard deviation of the mean
        std_mean_scores[i] = std_scores[i] / np.sqrt(float(n_puzzles))

        # Round to significant digits
        std_scores[i] = np.format_float_positional(
            std_scores[i], precision=1, fractional=False
        )
        std_mean_scores[i] = np.format_float_positional(
            std_mean_scores[i], precision=1, fractional=False
        )
        mean_precision = len(str(std_mean_scores[i]).split(".")[1])
        mean_scores[i] = np.format_float_positional(
            mean_scores[i], precision=mean_precision, fractional=False
        )

        # Describe the score with a string
        score_str = f"\tMean: {mean_scores[i]} ± {std_mean_scores[i]}"
        score_str += f"\n\tPopulation standard deviation: {std_scores[i]}"
        score_strings[i] = score_str

    # Make a dictionary of metrics and score strings for each score type
    metrics = {
        score_type: (
            mean_scores[i],
            std_scores[i],
            std_mean_scores[i],
            score_strings[i],
        )
        for i, score_type in enumerate(score_types)
    }

    return metrics


def format_scores(
    scores_all_types: list[np.ndarray],
    score_types: list[str],
    metrics: dict[str, tuple],
    n_puzzles: int,
) -> str:
    """Format the scores.

    Args:
        scores_all_types: Tuple of scores as numpy arrays. Each element contains the scores for a specific score type.
        score_types: List of score type names as strings.
        n_puzzles: Number of puzzles as an integer.
        metrics: Metrics as a dictionary of with the score type as the key, and the values being a tuple of ndarrays. The tuple contains the rounded metrics for the score type and a string describing the metrics for the score type.

    Returns:
        A formatted string of the scores.
    """
    # --- Describe overall metrics ---#

    score_str = "Puzzle Scores\n"
    score_str += "-------------\n"
    score_str += "Metrics\n\n"

    # Complete the string describing all metrics
    metrics_str = ""
    for score_type in score_types:
        metrics_str += f"{score_type.capitalize()}:\n"
        metrics_str += metrics[score_type][-1]
        metrics_str += "\n\n"

    metrics_str = metrics_str[:-1]

    score_str += metrics_str

    # --- Describe scores of individual puzzles ---#

    score_str += "\n-------------\n"
    score_str += "Single puzzle scores\n"

    for i in range(n_puzzles):
        score_str += f"\nPuzzle {i}: "
        for score_type, scores in zip(score_types, scores_all_types):
            score_str += f"\t{score_type}: {scores[i]:.2f}"

    return score_str
