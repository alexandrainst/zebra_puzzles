"""Module for handling files and directories."""

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)


def clean_folder(folder_path: Path, keep_files: list[str]) -> None:
    """Clean a folder by deleting outdated files.

    Creates the folder if it does not exist.

    Args:
        folder_path: Path to the folder to clean.
        keep_files: List of files to keep in the folder.
    """
    # Convert the folder path to a string
    folder = str(folder_path)

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
            log.info(f"Deleted outdated files in {folder}:\n{files_to_delete}\n")
        else:
            log.info(f"Old files were not deleted in {folder}.")


def save_dataset(data: str, filename: str, folder: Path) -> None:
    """Save a file.

    Args:
        data: Data to save.
        filename: Name of the file.
        folder: Path of the folder to save the file in.
    """
    file_path = folder / filename

    with file_path.open("w", encoding="utf-8") as file:
        file.write(data)


def prepare_data_folders(
    n_puzzles: int,
    theme: str,
    n_objects: int,
    n_attributes: int,
    n_red_herring_clues: int,
    data_folder_str: str,
) -> tuple[list[str], list[str], list[str], list[str], Path, Path, Path, Path]:
    """Prepare the data folders for the dataset.

    Args:
        n_puzzles: Number of puzzles to generate.
        theme: Theme of the puzzles.
        n_objects: Number of objects a the puzzle.
        n_attributes: Number of attributes of each object.
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.
        data_folder_str: Path to the data folder as a string.

    Returns:
        A tuple (prompt_filenames, clue_type_filenames, red_herring_filenames, solution_filenames, puzzle_folder, clue_type_folder, red_herring_folder, solution_folder), where:
            prompt_filenames: List of prompt file names.
            clue_type_filenames: List of clue type file names.
            red_herring_filenames: List of red herring file names.
            solution_filenames: List of solution file names.
            puzzle_folder: Path to the folder for prompt files.
            clue_type_folder: Path to the folder for clue type files.
            red_herring_folder: Path to the folder for red herring files.
            solution_folder: Path to the folder for solution files.
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

    # Convert data_folder from UNIX string to Path object. Both "/" and "\" should work on Windows"
    data_folder = Path(data_folder_str)

    # Define folders
    subfolder = (
        data_folder / theme / f"{n_objects}x{n_attributes}" / f"{n_red_herring_clues}rh"
    )
    puzzle_folder = subfolder / "puzzles"
    clue_type_folder = subfolder / "clue_types"
    red_herring_folder = subfolder / "red_herrings"
    solution_folder = subfolder / "solutions"

    # Clean folders
    clean_folder(folder_path=puzzle_folder, keep_files=prompt_filenames)
    clean_folder(folder_path=clue_type_folder, keep_files=clue_type_filenames)
    clean_folder(folder_path=red_herring_folder, keep_files=red_herring_filenames)
    clean_folder(folder_path=solution_folder, keep_files=solution_filenames)

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
    data_folder_str: str,
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[str], Path, str, Path]:
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
        data_folder_str: Path to the data folder as a string.

    Returns:
        A tuple (puzzle_paths, solution_paths, reduced_puzzle_paths, reduced_clue_type_paths, response_filenames, response_folder, score_filename, score_folder), where:
            puzzle_paths: Paths to the puzzles.
            solution_paths: Paths to the solutions.
            reduced_puzzle_paths: Paths to the puzzles after reducing the number of red herrings.
            reduced_clue_type_paths: Paths to the clue types after reducing the number of red herrings.
            response_filenames: Names of the response files.
            response_folder: Path to the folder to save the responses in.
            score_filename: Name of the score file.
            score_folder: Path to the folder to save the scores in.
    """
    # Convert data_folder from UNIX string to Path object. Both "/" and "\" should work on Windows"
    data_folder = Path(data_folder_str)

    # Define the subfolders for puzzles, solutions, responses, and evaluations
    puzzle_subfolder = Path(theme) / f"{n_objects}x{n_attributes}"

    # Get sorted names of all prompt files in the data folder
    puzzle_folder = (
        data_folder / puzzle_subfolder / f"{n_red_herring_clues}rh" / "puzzles"
    )

    # Load the puzzle paths
    puzzle_paths = sorted(list(puzzle_folder.glob("zebra_puzzle_*.txt")))

    # Sort the puzzle paths by the puzzle index (the number after zebra_puzzle_ and before .txt)
    puzzle_indices = [
        int(file_path.name.split("_")[2].split(".txt")[0]) for file_path in puzzle_paths
    ]
    puzzle_paths = [
        file_path for _, file_path in sorted(zip(puzzle_indices, puzzle_paths))
    ]

    solution_folder = puzzle_folder.parent / "solutions"

    solution_paths = [
        solution_folder / f"{puzzle_path.stem}_solution.json"
        for puzzle_path in puzzle_paths
    ]

    # Create reponse file names
    response_filenames = [
        f"{file_path.stem}_response.json" for file_path in puzzle_paths
    ]

    score_filename = f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"

    # Define evaluation folders
    response_folder = (
        data_folder
        / puzzle_subfolder
        / f"{n_red_herring_clues_evaluated}rh"
        / "responses"
        / model
    )
    score_folder = data_folder / "scores" / model / f"{n_red_herring_clues_evaluated}rh"
    reduced_puzzle_folder = (
        data_folder
        / puzzle_subfolder
        / f"{n_red_herring_clues_evaluated}rh"
        / "reduced_puzzles"
    )
    reduced_clue_type_folder = (
        data_folder
        / puzzle_subfolder
        / f"{n_red_herring_clues_evaluated}rh"
        / "reduced_clue_types"
    )

    # Get the paths of the reduced puzzles and clue types based on reduced_puzzle_folder
    reduced_puzzle_paths = [
        reduced_puzzle_folder.joinpath(f"{file_path.stem}_reduced.txt")
        for file_path in puzzle_paths
    ]
    reduced_clue_type_paths = [
        reduced_clue_type_folder.joinpath(f"{file_path.stem}_clue_types_reduced.txt")
        for file_path in puzzle_paths
    ]

    if generate_new_responses:
        reduced_puzzle_filenames = [path.stem + ".txt" for path in reduced_puzzle_paths]
        reduced_clue_type_filenames = [
            path.stem + ".txt" for path in reduced_clue_type_paths
        ]

        # Clean or create reponses folder
        clean_folder(folder_path=response_folder, keep_files=response_filenames)
        clean_folder(
            folder_path=reduced_puzzle_folder, keep_files=reduced_puzzle_filenames
        )
        clean_folder(
            folder_path=reduced_clue_type_folder, keep_files=reduced_clue_type_filenames
        )

    # Create the score folder if it does not exist
    os.makedirs(str(score_folder), exist_ok=True)

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


def get_evaluated_params(data_folder: Path) -> tuple[list[str], list[int]]:
    """Get the names and number of red herrings of the models used in the evaluation.

    Args:
        data_folder: Path to the data folder.

    Returns:
        A tuple (model_names, rh_values) where:
            model_names: List of model names.
            rh_values: List of the values of the set of red herring numbers evaluated.
    """
    # Define the path to the scores folder
    scores_path = data_folder / "scores"

    # Get the names of all models in the scores folder
    model_names = [model.name for model in scores_path.iterdir() if model.is_dir()]

    # Get the set of red herring numbers evaluated
    # Do this by getting the names of all folders in the scores folder and splitting the folder names by "rh"
    rh_values = set()
    for model in model_names:
        rh_values.update(
            [
                int(rh.name.split("rh")[0])
                for rh in (scores_path / model).iterdir()
                if rh.is_dir()
            ]
        )

    return model_names, list(rh_values)


def get_score_file_paths(
    data_folder: Path,
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
    scores_path = data_folder / "scores" / model / f"{n_red_herring_clues_evaluated}rh"

    # Get sorted names of all score files in the data folder that are evaluated on the correct number of puzzles
    score_file_paths = sorted(
        list(
            scores_path.glob(
                f"puzzle_scores_{model}_{theme}*_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"
            )
        )
    )

    return score_file_paths


def get_clue_type_file_paths(
    data_folder: Path,
    n_red_herring_clues_evaluated: int,
    theme: str,
    n_puzzles: int,
    reduced_flag: bool,
) -> dict[str, list[Path]]:
    """Get the paths of the clue type files.

    The clue type files are stored in the "clue_types" folder for the original puzzles and in the "reduced_clue_types" folder for the reduced puzzles.

    The clue type file paths are sorted by the puzzle indices.

    Args:
        data_folder: Path to the data folder.
        model: LLM model name.
        n_red_herring_clues_evaluated: Number of red herring clues evaluated.
        theme: Theme name.
        n_puzzles: Number of puzzles evaluated.
        reduced_flag: Whether the number of red herrings has been reduced.
            If True, the clue type files are in the "reduced_clue_types" folder.
            If False, the clue type files are in the "clue_types" folder.

    Returns:
        Dictionary of the clue type file paths for each puzzle size.
    """
    # Check all size puzzles in the data/theme folder

    clue_type_path = data_folder / theme

    # Get the puzzle size folders
    puzzle_size_folders = [
        folder for folder in clue_type_path.iterdir() if folder.is_dir()
    ]

    # Define the dictionary to store the clue type file paths
    clue_type_file_paths_all_sizes: dict[str, list[Path]] = {}

    for puzzle_size_folder in puzzle_size_folders:
        # Get sorted names of all clue type files in the data folder
        if reduced_flag:
            clue_type_file_paths = list(
                puzzle_size_folder.glob(
                    f"{n_red_herring_clues_evaluated}rh/reduced_clue_types/zebra_puzzle_*_clue_types_reduced.txt"
                )
            )
        else:
            clue_type_file_paths = list(
                puzzle_size_folder.glob(
                    f"{n_red_herring_clues_evaluated}rh/clue_types/zebra_puzzle_*_clue_types.txt"
                )
            )

        # Sort the clue type file paths by the puzzle index (the number after zebra_puzzle_)
        puzzle_indices = [
            int(file_path.name.split("_")[2]) for file_path in clue_type_file_paths
        ]
        clue_type_file_paths = [
            file_path
            for _, file_path in sorted(zip(puzzle_indices, clue_type_file_paths))
        ]

        # Check that the number of clue type files is equal to the number of puzzles
        if len(clue_type_file_paths) < n_puzzles:
            raise ValueError(
                f"Not enough clue type files found in {clue_type_path}. Found {len(clue_type_file_paths)}, expected {n_puzzles}."
            )
        if len(clue_type_file_paths) > n_puzzles:
            raise ValueError(
                f"Too many clue type files found in {clue_type_path}. Found {len(clue_type_file_paths)}, expected {n_puzzles}."
            )

        # Add the clue type file paths to the dictionary
        clue_type_file_paths_all_sizes[puzzle_size_folder.name] = clue_type_file_paths

    return clue_type_file_paths_all_sizes


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
