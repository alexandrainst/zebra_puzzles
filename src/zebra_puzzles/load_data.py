"""Module for loading data from files."""

import json
import re
from pathlib import Path
from typing import Type

import numpy as np
from pydantic import BaseModel

from zebra_puzzles.clue_removal import remove_red_herrings
from zebra_puzzles.file_utils import save_dataset


def load_puzzle(
    puzzle_file_path: Path,
    reduced_puzzle_file_path: Path,
    reduced_clue_type_file_path: Path,
    n_red_herrings_to_keep: int,
) -> str:
    """Load a puzzle and reduce the number of red herrings.

    This function loads a puzzle from a file, and if n_red_herrings_to_keep us less than the number of red herrings in the puzzle, it removes some of them.
    It also saves the new puzzle file and clue types.

    Args:
        puzzle_file_path: Path to the puzzle file.
        reduced_puzzle_file_path: Path to the folder where the reduced puzzle file will be saved.
        reduced_clue_type_file_path: Path to the folder where the reduced clue type file will be saved.
        n_red_herrings_to_keep: Number of red herring clues to be included in the puzzle as an integer.

    Returns:
        The prompt as a string.
    """
    # Load the prompt
    with puzzle_file_path.open() as file:
        prompt = file.read()

    # Load the red herring indices
    red_herring_path = puzzle_file_path.parent.parent.joinpath("red_herrings").joinpath(
        puzzle_file_path.stem + "_red_herrings.txt"
    )

    with red_herring_path.open() as file:
        red_herring_indices_str = file.read()

    # Load the clue types
    clue_type_path = puzzle_file_path.parent.parent.joinpath("clue_types").joinpath(
        puzzle_file_path.stem + "_clue_types.txt"
    )
    with clue_type_path.open() as file:
        chosen_clue_types_str = file.read()

    # Remove some red herrings and save the new puzzle file and clue types
    prompt, chosen_clue_types_str, fewer_red_herrings_flag = remove_red_herrings(
        prompt=prompt,
        red_herring_indices_str=red_herring_indices_str,
        n_red_herrings_to_keep=n_red_herrings_to_keep,
        chosen_clue_types_str=chosen_clue_types_str,
    )

    if fewer_red_herrings_flag:
        # Save the new puzzle and clue types in the right folder e.g. 3rh instead of 5rh
        reduced_puzzle_filename = reduced_puzzle_file_path.stem + ".txt"
        reduced_puzzle_folder = reduced_puzzle_file_path.parent
        save_dataset(
            data=prompt, filename=reduced_puzzle_filename, folder=reduced_puzzle_folder
        )

        clue_type_filename = reduced_clue_type_file_path.stem + ".txt"
        reduced_clue_type_folder = reduced_clue_type_file_path.parent
        save_dataset(
            data=chosen_clue_types_str,
            filename=clue_type_filename,
            folder=reduced_clue_type_folder,
        )

    return prompt


def load_scores(
    score_file_paths: list[Path],
    n_objects_list: list[int],
    n_attributes_list: list[int],
    score_types: list[str],
    n_puzzles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the scores from the score files.

    The scores are stored in a 3D array with dimensions (n_score_types, n_objects_max-1, n_attributes_max). The number of objects is at least 2 in all puzzles, so we create n_objects_max-1 rows.

    Values are -999 if the score was not found in the file.

    If n_puzzles is 1, the standard deviation is not available, so it is set to 0.

    Args:
        score_file_paths: List of score file paths.
        n_objects_list: List of the number of objects in the puzzles evaluated in each score file.
        n_attributes_list: List of the number of attributes in the puzzles evaluated in each score file.
        score_types: List of score types to find in the score files.
        n_puzzles: Number of puzzles evaluated in each score file.

    Returns:
        A tuple (mean_scores_array_min_2_n_objects, std_mean_scores_array_min_2_n_objects, std_scores_array_min_2_n_objects) where:
            mean_scores_array_min_2_n_objects: Array of mean scores for each score type in each score file. The dimensions are n_score_types x n_objects_max-1 x n_attributes_max. Values are -999 if the score was not found.
            std_mean_scores_array_min_2_n_objects: Array of standard deviations of the mean scores for each score type in each score file. The dimensions are n_score_types x n_objects_max-1 x n_attributes_max. Values are -999 if the score was not found.
            std_scores_array_min_2_n_objects: Array of sample standard deviations of the scores for each score type in each score file. The dimensions are n_score_types x n_objects_max-1 x n_attributes_max. Values are -999 if the score was not found.
    """
    # Get the maximum number of objects and attributes
    n_objects_max = max(n_objects_list)
    n_attributes_max = max(n_attributes_list)

    # Prepare array of scores
    mean_scores_array, std_mean_scores_array, std_scores_array = [
        (np.ones((len(score_types), n_objects_max, n_attributes_max)) * -999)
        for _ in range(3)
    ]

    for score_file_path, n_objects_in_file, n_attributes_in_file in zip(
        score_file_paths, n_objects_list, n_attributes_list
    ):
        # Load the score file
        with open(score_file_path, "r") as f:
            scores_str = f.read()

        for i_score_type, score_type in enumerate(score_types):
            # Get the number after "puzzle score:"
            score_str = scores_str.split(f"{score_type.capitalize()}:\n\tMean: ")[1]
            if n_puzzles > 1:
                mean_str = score_str.split(" ")[0]
                mean_std_str = score_str.split("± ")[1].split(" ")[0]
                std_str = scores_str.split("Sample standard deviation: ")[1]
                std_str = std_str.split("\n")[0]
            else:
                mean_str = score_str.split("\n")[0]

                # The standard deviation is not available for a single puzzle
                mean_std_str = "0"
                std_str = "0"

            mean = float(mean_str)
            std_mean = float(mean_std_str)
            std = float(std_str)

            # Add the score to the array
            mean_scores_array[
                i_score_type, n_objects_in_file - 1, n_attributes_in_file - 1
            ] = mean
            std_mean_scores_array[
                i_score_type, n_objects_in_file - 1, n_attributes_in_file - 1
            ] = std_mean
            std_scores_array[
                i_score_type, n_objects_in_file - 1, n_attributes_in_file - 1
            ] = std

    # Remove the n_objects = 1 row
    mean_scores_array_min_2_n_objects = np.delete(mean_scores_array, 0, axis=1)
    std_mean_scores_array_min_2_n_objects = np.delete(std_mean_scores_array, 0, axis=1)
    std_scores_array_min_2_n_objects = np.delete(std_scores_array, 0, axis=1)

    return (
        mean_scores_array_min_2_n_objects,
        std_mean_scores_array_min_2_n_objects,
        std_scores_array_min_2_n_objects,
    )


def load_individual_puzzle_scores(
    data_folder: Path,
    score_type: str,
    model: str,
    n_red_herring_clues_evaluated: int,
    n_puzzles: int,
    theme: str,
    puzzle_size: str,
) -> dict[int, float]:
    """Load the individual puzzle scores from the score files.

    The scores from a specific score file and for a specific score type.

    Args:
        data_folder: Path to the data folder.
        score_type: Type of score to load.
        model: LLM model name.
        n_red_herring_clues_evaluated: Number of red herring clues evaluated.
        n_puzzles: Number of puzzles evaluated.
        theme: Theme name.
        puzzle_size: Size of the puzzle.

    Returns:
        The scores as a dictionary with the puzzle index as the key and the score as the value.
    """
    score_file_path = (
        data_folder
        / "scores"
        / model
        / f"{n_red_herring_clues_evaluated}rh"
        / f"puzzle_scores_{model}_{theme}_{puzzle_size}_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"
    )

    score_type = score_type.replace("_", " ")
    try:
        with open(score_file_path, "r") as f:
            scores_str = f.read()
            # Get every number after "{score_type}: " and before "\t"
            pattern = re.compile(rf"{score_type}:\s+(\d.+)\t")
            matches = pattern.findall(scores_str)
            scores_individual_puzzles = {
                i: float(match) for i, match in enumerate(matches)
            }
    except FileNotFoundError:
        # If the file is not found, return scores of -999.
        scores_individual_puzzles = {i: -999.0 for i in range(n_puzzles)}

    return scores_individual_puzzles


def load_clue_type_frequencies(clue_type_file_path: Path):
    """Load the clue type frequencies from a file.

    Args:
        clue_type_file_path: Path to the clue type file.
        clue_type_frequencies: Dictionary of clue type frequencies.

    Returns:
        clue_type_frequencies: Dictionary of clue type frequencies.
    """
    with open(clue_type_file_path, "r") as file:
        # Read the clue types from the file
        chosen_clue_types_str = file.read()

    # Split the string of clue types into a list
    chosen_clue_types = [
        clue_type.strip() for clue_type in chosen_clue_types_str.split(",")
    ]

    # Count the frequency of each clue type
    clue_type_frequencies: dict[str, int] = {}
    for clue_type in chosen_clue_types:
        if clue_type in clue_type_frequencies:
            clue_type_frequencies[clue_type] += 1
        else:
            clue_type_frequencies[clue_type] = 1

    return clue_type_frequencies


def load_solution(solution_file_path: Path, OutputFormat: Type[BaseModel]) -> BaseModel:
    """Load a puzzle solution or LLM response.

    Args:
        solution_file_path: Path to the solution or response file.
        OutputFormat: The output format as a Pydantic model.

    Returns:
        The solution in OutputFormat format.
    """
    with solution_file_path.open() as file:
        solution = file.read()

    # Change the format of solution to OutputFormat

    solution_json = json.loads(solution)
    solution_json = OutputFormat.model_validate(solution_json)

    return solution_json
