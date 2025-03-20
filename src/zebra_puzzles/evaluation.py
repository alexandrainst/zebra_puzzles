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

from zebra_puzzles.zebra_utils import clean_folder, save_dataset

# Load environment variables to get the API key
load_dotenv()


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
    sorted_cell_scores: np.ndarray = np.zeros(n_puzzles)

    # Evaluate each puzzle
    for i, file_path in tqdm(
        enumerate(file_paths),
        total=n_puzzles,
        desc="Evaluating",
        unit="puzzle",
        colour="#5599ff",
        ascii="░█",
    ):
        puzzle_score, cell_score, sorted_cell_score = evaluate_single_puzzle(
            file_path=file_path,
            n_objects=n_objects,
            n_attributes=n_attributes,
            model=model,
            response_filename=response_filenames[i],
        )
        puzzle_scores[i] = puzzle_score
        cell_scores[i] = cell_score
        sorted_cell_scores[i] = sorted_cell_score

    # Compute summary metrics
    metrics = compute_metrics(
        scores_all_types=[puzzle_scores, cell_scores, sorted_cell_scores],
        score_types=["puzzle score", "cell score", "sorted cell score"],
        n_puzzles=n_puzzles,
    )

    # Save scores
    score_str = format_scores(
        scores_all_types=[puzzle_scores, cell_scores, sorted_cell_scores],
        score_types=["puzzle score", "cell score", "sorted cell score"],
        metrics=metrics,
        n_puzzles=n_puzzles,
    )

    filename = (
        f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_puzzles}.txt"
    )
    save_dataset(data=score_str, filename=filename, folder="scores")


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
        metrics_str += "\n"

    score_str += metrics_str

    # --- Describe scores of individual puzzles ---#

    score_str += "\n-------------\n"
    score_str += "Single puzzle scores\n"

    for i in range(n_puzzles):
        score_str += f"\nPuzzle {i}: "
        for score_type, scores in zip(score_types, scores_all_types):
            score_str += f"\t{score_type}: {scores[i]:.2f}"

    return score_str


def evaluate_single_puzzle(
    file_path: Path,
    n_objects: int,
    n_attributes: int,
    model: str,
    response_filename: str,
) -> tuple[float, float, float]:
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
            sorted_cell_score: A cell-level score as a float after sorting the output and solution by the first attribute in each object.
    """
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # Load the prompt
    with file_path.open() as file:
        prompt = file.read()

    with file_path.with_stem(f"{file_path.stem}_solution").open() as file:
        solution = file.read()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Generate the dynamic OutputFormat class
    OutputFormat = generate_output_format_class(n_objects=n_objects)

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
    try:
        output = OutputFormat.model_validate(response.choices[0].message.parsed)
    except ValidationError as e:
        print("response.choices[0].message:\n", response.choices[0].message)
        print(
            "\nresponse.choices[0].message.parsed:\n",
            response.choices[0].message.parsed,
        )
        print()
        raise e

    # Change the format of solution to OutputFormat

    solution_json = json.loads(solution)

    solution_json = OutputFormat.model_validate(solution_json)

    puzzle_score, cell_score, sorted_cell_score = compare_solutions(
        output=output,
        solution=solution_json,
        n_objects=n_objects,
        n_attributes=n_attributes,
    )

    # Save the output
    output_str = json.dumps(output.model_dump(), indent=4)
    save_dataset(data=output_str, filename=response_filename, folder="responses")

    return puzzle_score, cell_score, sorted_cell_score


def compare_solutions(
    output: BaseModel, solution: BaseModel, n_objects: int, n_attributes: int
) -> tuple[int, float, float]:
    """Compare the output to the solution.

    The puzzle score is 1 for a correct solution and 0 for an incorrect solution.
    The cell score is the proportion of cells that are correct.
    The sorted cell score is the cell score after sorting the output and solution by the first attribute in each object. This will give a high score if the LLM coupled the attributes correctly, but misunderstood the order of the objects.

    Args:
        output: The output in OutputFormat format.
        solution: The solution in OutputFormat format.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        A tuple (puzzle_score, cell_score), where:
            puzzle_score: A puzzle-level score as an integer.
            cell_score: A cell-level score as a float.
            sorted_cell_score: A cell-level score as a float after sorting the output and solution by the first attribute in each object.
    """
    # Extract solution arrays

    # Compare the full output to the solution

    if output == solution:
        puzzle_score = 1
        cell_score = 1.0
        sorted_cell_score = 1.0
    else:
        # Compare all cells
        cell_score = compute_cell_score(
            output=dict(output),
            solution=dict(solution),
            n_objects=n_objects,
            n_attributes=n_attributes,
        )

        # Check if the puzzle is solved after stripping whitespace in cells
        if cell_score == 1:
            puzzle_score = 1
            sorted_cell_score = 1.0
        else:
            puzzle_score = 0

            # Sort by the first attribute in each object
            output_sorted = dict(sorted(output, key=lambda x: x[1]))
            solution_sorted = dict(sorted(solution, key=lambda x: x[1]))

            # Compare the sorted output to the sorted solution
            sorted_cell_score = compute_cell_score(
                output_sorted, solution_sorted, n_objects, n_attributes
            )

    return puzzle_score, cell_score, sorted_cell_score


def compute_cell_score(
    output: dict[str, list],
    solution: dict[str, list],
    n_objects: int,
    n_attributes: int,
) -> float:
    """Compute the cell score.

    Args:
        output: The output in OutputFormat format.
        solution: The solution in OutputFormat format.
        n_objects: Number of objects in the puzzle.
        n_attributes: Number of attributes of each object.

    Returns:
        The cell score as a float.
    """
    # Compare each cell
    cell_score: float = 0.0
    for attributes_output, attributes_solution in zip(
        output.values(), solution.values()
    ):
        for attribute_output, attribute_solution in zip(
            attributes_output, attributes_solution
        ):
            if attribute_output.strip() == attribute_solution.strip():
                cell_score += 1.0

    # Normalise the cell score
    cell_score /= float(n_objects * n_attributes)

    return cell_score
