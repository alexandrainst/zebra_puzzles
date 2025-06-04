"""Module for evaluation."""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from zebra_puzzles.evaluation.compare_solutions import (
    compare_output_to_solution,
    compute_metrics,
    format_scores,
)
from zebra_puzzles.file_utils import prepare_eval_folders, save_dataset
from zebra_puzzles.load_data import load_puzzle, load_solution
from zebra_puzzles.zebra_utils import generate_output_format_class, query_llm

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

        output = query_llm(
            prompt=prompt,
            model=model,
            response_format=OutputFormat,
            n_objects=n_objects,
        )

    else:
        # Load an existing response
        response_file_path = response_folder_path / response_filename
        output = load_solution(
            solution_file_path=response_file_path, OutputFormat=OutputFormat
        )

    # Load the solution
    solution_json = load_solution(
        solution_file_path=solution_file_path, OutputFormat=OutputFormat
    )

    puzzle_score, cell_score, best_permuted_cell_score = compare_output_to_solution(
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
