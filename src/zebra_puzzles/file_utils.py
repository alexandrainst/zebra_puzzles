"""Module for handling files and directories."""

import os
import re
from pathlib import Path

import numpy as np

from zebra_puzzles.clue_removal import remove_red_herrings


def clean_folder(folder: str, keep_files: list[str]) -> None:
    """Clean a folder by deleting outdated files.

    Creates the folder if it does not exist.

    Args:
        folder: Folder to clean.
        keep_files: List of files to keep in the folder.
    """
    # Create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # Get a list of files in the folder
    existing_files = os.listdir(folder)

    # Get a list of files to delete
    files_to_delete = [file for file in existing_files if file not in keep_files]

    if len(files_to_delete) > 0:
        useroutput = input(
            f"\nDo you want to delete the following outdated files in the folder '{folder}'?\n\n{files_to_delete}\n\n(y/n): "
        )
        if useroutput == "y":
            for file in files_to_delete:
                os.remove(os.path.join(folder, file))
            print("Old files were deleted.")
        else:
            print("Old files were not deleted.")


def save_dataset(data: str, filename: str, folder: str) -> None:
    """Save a file.

    Args:
        data: Data to save.
        filename: Name of the file.
        folder: Folder to save the file in.

    """
    with open(folder + "/" + filename, "w", encoding="utf-8") as file:
        file.write(data)


def load_puzzle(
    puzzle_path: Path,
    reduced_puzzle_path: Path,
    reduced_clue_type_path: Path,
    n_red_herrings_to_keep: int,
) -> str:
    """Load a puzzle and reduce the number of red herrings.

    This function loads a puzzle from a file, and if n_red_herrings_to_keep us less than the number of red herrings in the puzzle, it removes some of them.
    It also saves the new puzzle file and clue types.

    Args:
        puzzle_path: Path to the puzzle file.
        reduced_puzzle_path: Path to the folder where the reduced puzzle file will be saved.
        reduced_clue_type_path: Path to the folder where the reduced clue type file will be saved.
        n_red_herrings_to_keep: Number of red herring clues to be included in the puzzle as an integer.

    Returns:
        The prompt as a string.
    """
    # Load the prompt
    with puzzle_path.open() as file:
        prompt = file.read()

    # Load the red herring indices
    red_herring_path = puzzle_path.parent.parent.joinpath("red_herrings").joinpath(
        puzzle_path.stem + "_red_herrings.txt"
    )

    with red_herring_path.open() as file:
        red_herring_indices_str = file.read()

    # Load the clue types
    clue_type_path = puzzle_path.parent.parent.joinpath("clue_types").joinpath(
        puzzle_path.stem + "_clue_types.txt"
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
        reduced_puzzle_filename = reduced_puzzle_path.stem + ".txt"
        reduced_puzzle_folder = str(reduced_puzzle_path.parent)
        save_dataset(
            data=prompt, filename=reduced_puzzle_filename, folder=reduced_puzzle_folder
        )

        clue_type_filename = reduced_clue_type_path.stem + ".txt"
        reduced_clue_type_folder = str(reduced_clue_type_path.parent)
        save_dataset(
            data=chosen_clue_types_str,
            filename=clue_type_filename,
            folder=reduced_clue_type_folder,
        )

    return prompt


def prepare_data_folders(
    n_puzzles: int,
    theme: str,
    n_objects: int,
    n_attributes: int,
    n_red_herring_clues: int,
    data_folder: str,
) -> tuple[list[str], list[str], list[str], list[str], str, str, str, str]:
    """Prepare the data folders for the dataset.

    Args:
        n_puzzles: Number of puzzles to generate.
        theme: Theme of the puzzles.
        n_objects: Number of objects a the puzzle.
        n_attributes: Number of attributes of each object.
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        data_folder: Path to the data folder as a string.

    Returns:
        A tuple (prompt_filenames, clue_type_filenames, red_herring_filenames, solution_filenames, puzzle_folder, clue_type_folder, red_herring_folder, solution_folder), where:
            prompt_filenames: List of prompt file names.
            clue_type_filenames: List of clue type file names.
            red_herring_filenames: List of red herring file names.
            solution_filenames: List of solution file names.
            puzzle_folder: Folder for the prompt files.
            clue_type_folder: Folder for the clue type files.
            red_herring_folder: Folder for the red herring files.
            solution_folder: Folder for the solution files.
    """
    # Create data file names
    prompt_filenames = ["zebra_puzzle_{}.txt".format(i) for i in range(n_puzzles)]
    clue_type_filenames = [
        str(file.split(".")[0]) + "_clue_types.txt" for file in prompt_filenames
    ]
    red_herring_filenames = [
        str(file.split(".")[0]) + "_red_herrings.txt" for file in prompt_filenames
    ]
    solution_filenames = [
        str(file.split(".")[0]) + "_solution.json" for file in prompt_filenames
    ]

    # Define folders
    subfolder = (
        f"{data_folder}/{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh"
    )
    puzzle_folder = f"{subfolder}/puzzles"
    clue_type_folder = f"{subfolder}/clue_types"
    red_herring_folder = f"{subfolder}/red_herrings"
    solution_folder = f"{subfolder}/solutions"

    # Clean folders
    clean_folder(folder=puzzle_folder, keep_files=prompt_filenames)
    clean_folder(folder=clue_type_folder, keep_files=clue_type_filenames)
    clean_folder(folder=red_herring_folder, keep_files=red_herring_filenames)
    clean_folder(folder=solution_folder, keep_files=solution_filenames)

    return (
        prompt_filenames,
        clue_type_filenames,
        red_herring_filenames,
        solution_filenames,
        puzzle_folder,
        clue_type_folder,
        red_herring_folder,
        solution_folder,
    )


def prepare_eval_folders(
    theme: str,
    n_objects: int,
    n_attributes: int,
    n_red_herring_clues: int,
    n_red_herring_clues_evaluated: int,
    model: str,
    n_puzzles: int,
    generate_new_responses: bool,
    data_folder: str,
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[str], str, str, str]:
    """Prepare the folders for the evaluation.

    Args:
        theme: The theme of the puzzles.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        n_red_herring_clues: Number of red herring clues included in the generated version of the puzzles as an integer.
        n_red_herring_clues_evaluated: Number of red herring clues included in the evaluated version of the puzzles as an integer.
        model: The model to use for the evaluation as a string.
        n_puzzles: Number of puzzles to evaluate as an integer.
        generate_new_responses: Whether to generate new responses or use existing ones.
        data_folder: Path to the data folder as a string.

    Returns:
        A tuple (puzzle_paths, solution_paths, reduced_puzzle_paths, reduced_clue_type_paths, response_filenames, response_folder, score_filename, score_folder), where:
            puzzle_paths: Paths to the puzzles.
            solution_paths: Paths to the solutions.
            reduced_puzzle_paths: Paths to the puzzles after reducing the number of red herrings.
            reduced_clue_type_paths: Paths to the clue types after reducing the number of red herrings.
            response_filenames: Names of the response files.
            response_folder: Folder to save the responses in.
            score_filename: Name of the score file.
            score_folder: Folder to save the scores in.
    """
    # Define the subfolders for puzzles, solutions, responses, and evaluations
    puzzle_subfolder = f"{theme}/{n_objects}x{n_attributes}"

    # Get sorted names of all prompt files in the data folder
    puzzle_paths = sorted(
        list(
            Path(
                f"{data_folder}/{puzzle_subfolder}/{n_red_herring_clues}rh/puzzles"
            ).glob("*.txt")
        )
    )

    solution_paths = [
        puzzle_path.parent.parent.joinpath("solutions") for puzzle_path in puzzle_paths
    ]

    # Create reponse file names
    response_filenames = [
        f"{file_path.stem}_response.json" for file_path in puzzle_paths
    ]

    score_filename = f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"

    # Define evaluation folders
    response_folder = f"{data_folder}/{puzzle_subfolder}/{n_red_herring_clues_evaluated}rh/responses/{model}"
    score_folder = f"{data_folder}/scores/{model}/{n_red_herring_clues_evaluated}rh"
    reduced_puzzle_folder = f"{data_folder}/{puzzle_subfolder}/{n_red_herring_clues_evaluated}rh/reduced_puzzles"
    reduced_clue_type_folder = f"{data_folder}/{puzzle_subfolder}/{n_red_herring_clues_evaluated}rh/reduced_clue_types"

    # Get the paths of the reduced puzzles and clue types based on reduced_puzzle_folder
    reduced_puzzle_paths = [
        Path(reduced_puzzle_folder).joinpath(f"{file_path.stem}_reduced.txt")
        for file_path in puzzle_paths
    ]
    reduced_clue_type_paths = [
        Path(reduced_clue_type_folder).joinpath(
            f"{file_path.stem}_clue_types_reduced.txt"
        )
        for file_path in puzzle_paths
    ]

    if generate_new_responses:
        reduced_puzzle_filenames = [path.stem + ".txt" for path in reduced_puzzle_paths]
        reduced_clue_type_filenames = [
            path.stem + ".txt" for path in reduced_clue_type_paths
        ]

        # Clean or create reponses folder
        clean_folder(folder=response_folder, keep_files=response_filenames)
        clean_folder(folder=reduced_puzzle_folder, keep_files=reduced_puzzle_filenames)
        clean_folder(
            folder=reduced_clue_type_folder, keep_files=reduced_clue_type_filenames
        )

    # Create the score folder if it does not exist
    os.makedirs(score_folder, exist_ok=True)

    return (
        puzzle_paths,
        solution_paths,
        reduced_puzzle_paths,
        reduced_clue_type_paths,
        response_filenames,
        response_folder,
        score_filename,
        score_folder,
    )


def get_score_file_paths(
    data_folder: str,
    model: str,
    n_red_herring_clues_evaluated: int,
    theme: str,
    n_puzzles: int,
) -> list[Path]:
    """Get the paths of the score files.

    Args:
        data_folder: Path to the data folder.
        model: LLM model name.
        n_red_herring_clues_evaluated: Number of red herring clues evaluated.
        theme: Theme name.
        n_puzzles: Number of puzzles evaluated.

    Returns:
        List of score file paths.
    """
    scores_path = f"{data_folder}/scores/{model}/{n_red_herring_clues_evaluated}rh/"

    # Get sorted names of all score files in the data folder that are evaluated on the correct number of puzzles
    score_file_paths = sorted(
        list(
            Path(scores_path).glob(
                f"puzzle_scores_{model}_{theme}*_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"
            )
        )
    )

    return score_file_paths


def get_puzzle_dimensions_from_filename(
    score_file_paths: list[Path],
) -> tuple[list[int], list[int]]:
    """Get the dimensions of the puzzles from the score file paths.

    Args:
        score_file_paths: List of score file paths.

    Returns:
        A tuple (n_objects_list, n_attributes_list) where:
            n_objects_list: List of the number of objects in the puzzles evaluated in each score file.
            n_attributes_list: List of the number of attributes in the puzzles evaluated in each score file.
    """
    # Get the dimensions of the puzzles
    n_objects_list = []
    n_attributes_list = []

    for score_file_path in score_file_paths:
        file_name = score_file_path.name

        # Get the numbers on each side of the "x" in the filename
        match = re.search(r"_(\d+)x(\d+)_", file_name)
        if match:
            n_objects = int(match.group(1))
            n_attributes = int(match.group(2))
        else:
            raise ValueError(f"Could not find dimensions in {file_name}")

        n_objects_list.append(n_objects)
        n_attributes_list.append(n_attributes)

    return n_objects_list, n_attributes_list


def load_scores(
    score_file_paths: list[Path],
    n_objects_list: list[int],
    n_attributes_list: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the scores from the score files.

    Args:
        score_file_paths: List of score file paths.
        n_objects_list: List of the number of objects in the puzzles evaluated in each score file.
        n_attributes_list: List of the number of attributes in the puzzles evaluated in each score file.

    Returns:
        A tuple (mean_scores_array, std_mean_scores_array) where:
            mean_scores_array: Array of mean scores for each score file.
            std_mean_scores_array: Array of standard deviations of the mean scores for each score file.
    """
    # Get the maximum number of objects and attributes
    n_objects_max = max(n_objects_list)
    n_attributes_max = max(n_attributes_list)

    # Prepare array of scores
    score_types = ["puzzle score", "cell score", "best permuted cell score"]
    mean_scores_array = (
        np.ones((len(score_types), n_objects_max, n_attributes_max)) * -1
    )
    std_mean_scores_array = (
        np.ones((len(score_types), n_objects_max, n_attributes_max)) * -1
    )

    for score_file_path, n_objects_in_file, n_attributes_in_file in zip(
        score_file_paths, n_objects_list, n_attributes_list
    ):
        # Load the score file
        with open(score_file_path, "r") as f:
            scores_str = f.read()

        for i_score_type, score_type in enumerate(score_types):
            # Get the number after "puzzle score:"
            score_str = scores_str.split(f"{score_type.capitalize()}:\n\tMean: ")[1]
            mean_str = score_str.split(" ")[0]
            mean_std_str = score_str.split("± ")[1].split(" ")[0]

            mean = float(mean_str)
            std_mean = float(mean_std_str)

            # Add the score to the array
            mean_scores_array[
                i_score_type, n_objects_in_file - 1, n_attributes_in_file - 1
            ] = mean
            std_mean_scores_array[
                i_score_type, n_objects_in_file - 1, n_attributes_in_file - 1
            ] = std_mean

    return mean_scores_array, std_mean_scores_array
