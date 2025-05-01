"""Module for evaluation."""

import json
import logging
import os
from pathlib import Path
from typing import Type

import numpy as np
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from zebra_puzzles.compare_solutions import compare_solutions
from zebra_puzzles.file_utils import load_puzzle, prepare_eval_folders, save_dataset
from zebra_puzzles.zebra_utils import (
    bernoulli_std,
    generate_output_format_class,
    round_using_std,
)

# Load environment variables to get the API key
load_dotenv()


def evaluate_all(
    n_puzzles: int,
    n_red_herring_clues: int,
    n_red_herring_clues_evaluated: int,
    n_objects: int,
    n_attributes: int,
    model: str,
    theme: str,
    generate_new_responses: bool,
    data_folder_str: str,
) -> None:
    """Evaluate a dataset of zebra puzzles.

    An LLM is used to evaluate each puzzle. Performance is evaluated by comparing the output of the LLM with the expected solution. Metrics are computed for each puzzle and saved in a file.

    Args:
        n_puzzles: Number of puzzles to evaluate as an integer.
        n_red_herring_clues: Number of red herring clues in the generated puzzles as an integer.
        n_red_herring_clues_evaluated: Number of red herring clues to be included in the evaluated puzzles as an integer. If this is smaller than the number of red herring clues used to generate the puzzles, the evaluation will be done on a subset of the red herring clues.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        model: The model to use for the evaluation as a string.
        theme: The theme of the puzzles as a string.
        generate_new_responses: A boolean describing whether to generate new responses or use existing ones.
        data_folder_str: The path to the folder containing the data as a string.

    TODO: Make the script more robust in cases where the expected responses are not found.
    """
    (
        puzzle_paths,
        solution_paths,
        reduced_puzzle_paths,
        reduced_clue_type_paths,
        response_filenames,
        response_folder,
        score_filename,
        score_folder,
    ) = prepare_eval_folders(
        theme=theme,
        n_objects=n_objects,
        n_attributes=n_attributes,
        n_red_herring_clues=n_red_herring_clues,
        n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
        model=model,
        n_puzzles=n_puzzles,
        generate_new_responses=generate_new_responses,
        data_folder_str=data_folder_str,
    )
    # Initialize scores
    puzzle_scores: np.ndarray = np.zeros(n_puzzles)
    cell_scores: np.ndarray = np.zeros(n_puzzles)
    best_permuted_cell_scores: np.ndarray = np.zeros(n_puzzles)

    # Evaluate each puzzle
    for i in tqdm(
        range(n_puzzles),
        total=n_puzzles,
        desc="Evaluating",
        unit="puzzle",
        colour="#5599ff",
        ascii="░█",
    ):
        puzzle_score, cell_score, best_permuted_cell_score = evaluate_single_puzzle(
            puzzle_file_path=puzzle_paths[i],
            solution_file_path=solution_paths[i],
            reduced_puzzle_file_path=reduced_puzzle_paths[i],
            reduced_clue_type_file_path=reduced_clue_type_paths[i],
            n_objects=n_objects,
            n_attributes=n_attributes,
            model=model,
            response_filename=response_filenames[i],
            generate_new_responses=generate_new_responses,
            response_folder_path=response_folder,
            n_red_herring_clues_evaluated=n_red_herring_clues_evaluated,
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

    save_dataset(data=score_str, filename=score_filename, folder=score_folder)


def evaluate_single_puzzle(
    puzzle_file_path: Path,
    solution_file_path: Path,
    reduced_puzzle_file_path: Path,
    reduced_clue_type_file_path: Path,
    n_objects: int,
    n_attributes: int,
    model: str,
    response_filename: str,
    response_folder_path: Path,
    generate_new_responses: bool,
    n_red_herring_clues_evaluated: int,
) -> tuple[float, float, float]:
    """Evaluate a dataset of zebra puzzles.

    An LLM is called to evaluate a puzzle. The response is saved and compared with the expected solution.

    Args:
        puzzle_file_path: Path to the prompt file.
        solution_file_path: Path to the solution file.
        reduced_puzzle_file_path: Path to the reduced puzzle file.
        reduced_clue_type_file_path: Path to the reduced clue type file.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an
        model: The model to use for the evaluation as a
        response_filename: The name of the response file.
        response_folder_path: The path to the folder to save the response file in.
        generate_new_responses: Whether to generate new responses or use existing ones.
        n_red_herring_clues_evaluated: Number of red herring clues included in the puzzles as an integer. If this is smaller than the number of red herring clues used to generate the puzzles, the evaluation will be done on a subset of the red herring clues.

    Returns:
        A tuple (puzzle_score, cell_score), where:
            puzzle_score: A puzzle-level score as a float.
            cell_score: A cell-level score as a float.
            best_permuted_cell_score: A cell-level score as a float after trying all permutations of the objects in the response.
    """
    # Generate the dynamic OutputFormat class
    OutputFormat = generate_output_format_class(n_objects=n_objects)

    if generate_new_responses:
        prompt = load_puzzle(
            puzzle_file_path=puzzle_file_path,
            reduced_puzzle_file_path=reduced_puzzle_file_path,
            reduced_clue_type_file_path=reduced_clue_type_file_path,
            n_red_herrings_to_keep=n_red_herring_clues_evaluated,
        )

        output = query_llm(prompt=prompt, model=model, response_format=OutputFormat)

    else:
        # Load an existing response
        response_file_path = response_folder_path / response_filename
        with open(response_file_path, "r") as file:
            response_str = file.read()

        output = json.loads(response_str)
        output = OutputFormat.model_validate(output)

    # Load the solution
    with solution_file_path.open() as file:
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
    output_str = json.dumps(output.model_dump(), indent=4, ensure_ascii=False)
    save_dataset(
        data=output_str, filename=response_filename, folder=response_folder_path
    )

    return puzzle_score, cell_score, best_permuted_cell_score


def query_llm(prompt: str, model: str, response_format: Type[BaseModel]) -> BaseModel:
    """Query an LLM API.

    Args:
        prompt: The prompt to use for the evaluation.
        model: The model to use for the evaluation.
        response_format: The response format as a Pydantic model.

    Returns:
        The output in OutputFormat format.
    """
    logging.getLogger("httpx").setLevel(logging.ERROR)

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
) -> dict[str, tuple[str | float, ...]]:
    """Compute the metrics.

    For each score type e.g. cell score, a dictionary of metrics is computed. This dictionary includes a string describing the rounded metrics.

    Assumes that the scores are normally distributed, except the puzzle score. Also assumes that the maximum length of the string describing each metric is 100 characters.

    The puzzle score is assumed to follow a Bernoulli distribution.

    Args:
        scores_all_types: Tuple of scores as numpy arrays. Each element contains the scores for a specific score type.
        score_types: List of score type names as strings.
        n_puzzles: Number of puzzles as an integer.

    Returns:
        Metrics as a dictionary of with the score type as the key, and the values being a tuple of ndarrays. The tuple contains the rounded metrics for the score type and a string describing the metrics for the score type.

    NOTE: More metrics could be added e.g. from sklearn.metrics
    """
    # Number of score types
    n_metrics = len(score_types)

    # Initialize metrics
    mean_scores = np.zeros(n_metrics, dtype=float)
    if n_puzzles > 1:
        std_scores = np.zeros(n_metrics, dtype=float)
        std_mean_scores = np.zeros(n_metrics, dtype=float)

    # Initialize strings describing metrics for each score type
    # U100 is a Unicode string with a maximum length of 100 characters
    score_strings = np.zeros(n_metrics, dtype="U100")

    for i, scores in enumerate(scores_all_types):
        # Take the mean
        mean_scores[i] = float(np.mean(scores))

        if n_puzzles > 1:
            if score_types[i] == "puzzle_score":
                # Take the standard deviations of the sample and of the mean for a Bernoulli distribution
                n_successes = int(mean_scores[i] * n_puzzles)
                std_scores[i], std_mean_scores[i] = bernoulli_std(
                    n_trials=n_puzzles, n_successes=n_successes
                )
            else:
                # Take the standard deviation
                std_scores[i] = float(np.std(scores, ddof=1))

                # Take the standard deviation of the mean
                std_mean_scores[i] = std_scores[i] / np.sqrt(float(n_puzzles))

            # Round to significant digits
            std_scores[i] = np.format_float_positional(
                std_scores[i], precision=1, fractional=False
            )

            mean_scores[i], std_mean_scores[i] = round_using_std(
                value=mean_scores[i], std=std_mean_scores[i]
            )

            # Describe the score with a string
            score_str = f"\tMean: {mean_scores[i]} ± {std_mean_scores[i]} (1σ)"
            score_str += f"\n\tSample standard deviation: {std_scores[i]}"
            score_strings[i] = score_str
        else:
            # Round mean to 2 significant digits
            mean_precision = 2
            mean_scores[i] = np.format_float_positional(
                mean_scores[i], precision=mean_precision, fractional=False
            )

            # Describe the score with a string
            score_strings[i] = f"\tMean: {mean_scores[i]}"

    # Make a dictionary of metrics and score strings for each score type
    if n_puzzles > 1:
        metrics: dict[str, tuple[str | float, ...]] = {
            score_type: (
                mean_scores[i],
                std_scores[i],
                std_mean_scores[i],
                score_strings[i],
            )
            for i, score_type in enumerate(score_types)
        }
    else:
        metrics = {
            score_type: (mean_scores[i], score_strings[i])
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

    This creates a string describing the overall metrics and the scores of each puzzle.

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
    if n_puzzles > 1:
        score_str += "Uncertainty is given as one standard deviation (1σ), corresponding to a 68% confidence interval. The 95% confidence interval is approximately ±2σ.\n\n"

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
