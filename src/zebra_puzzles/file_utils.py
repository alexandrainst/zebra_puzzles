"""Module for handling files and directories."""

import os
from pathlib import Path


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


def save_dataset(data: str, filename: str, folder: str = "data") -> None:
    """Save a file.

    Args:
        data: Data to save.
        filename: Name of the file.
        folder: Folder to save the file in.

    """
    with open(folder + "/" + filename, "w") as file:
        file.write(data)


def prepare_data_folders(
    n_puzzles: int,
    theme: str,
    n_objects: int,
    n_attributes: int,
    n_red_herring_clues: int,
) -> tuple[list[str], list[str], list[str], str, str, str]:
    """Prepare the data folders for the dataset.

    Args:
        n_puzzles: Number of puzzles to generate.
        theme: Theme of the puzzles.
        n_objects: Number of objects a the puzzle.
        n_attributes: Number of attributes of each object.
        n_red_herring_clues: Number of red herring clues to include in the puzzle as an integer.

    Returns:
        A tuple (prompt_filenames, solution_filenames, puzzle_folder, solution_folder), where:
            prompt_filenames: List of prompt file names.
            solution_filenames: List of solution file names.
            red_herring_filenames: List of red herring file names.
            puzzle_folder: Folder for the prompt files.
            solution_folder: Folder for the solution files
            red_herring_folder: Folder for the red herring files.

    """
    # Create data file names
    prompt_filenames = ["zebra_puzzle_{}.txt".format(i) for i in range(n_puzzles)]
    solution_filenames = [
        str(file.split(".")[0]) + "_solution.json" for file in prompt_filenames
    ]
    red_herring_filenames = [
        str(file.split(".")[0]) + "_red_herrings.txt" for file in prompt_filenames
    ]

    # Define folders
    subfolder = f"{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh"
    puzzle_folder = f"data/{subfolder}/puzzles"
    solution_folder = f"data/{subfolder}/solutions"
    red_herring_folder = f"data/{subfolder}/red_herrings"

    # Clean folders
    clean_folder(folder=puzzle_folder, keep_files=prompt_filenames)
    clean_folder(folder=solution_folder, keep_files=solution_filenames)

    return (
        prompt_filenames,
        solution_filenames,
        red_herring_filenames,
        puzzle_folder,
        solution_folder,
        red_herring_folder,
    )


def prepare_eval_folders(
    theme: str,
    n_objects: int,
    n_attributes: int,
    n_red_herring_clues: int,
    model: str,
    n_puzzles: int,
    generate_new_responses: bool,
) -> tuple[list[Path], list[Path], list[str], str, str, str]:
    """Prepare the folders for the evaluation.

    Args:
        theme: The theme of the puzzles.
        n_objects: Number of objects in each puzzle as an integer.
        n_attributes: Number of attributes of each object as an integer.
        n_red_herring_clues: Number of red herring clues included in the puzzles as an integer.
        model: The model to use for the evaluation as a string.
        n_puzzles: Number of puzzles to evaluate as an integer.
        generate_new_responses: Whether to generate new responses or use existing ones.

    Returns:
        A tuple (puzzle_paths, solution_paths, response_filenames, response_folder, score_filename, score_folder), where:
            puzzle_paths: Paths to the puzzles.
            solution_paths: Paths to the solutions.
            response_filenames: Names of the response files.
            response_folder: Folder to save the responses in.
            score_filename: Name of the score file.
            score_folder: Folder to save the scores in.
    """
    # Define the subfolders for puzzles, solutions, responses, and evaluations
    puzzle_subfolder = f"{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh"

    # Get sorted names of all prompt files in the data folder
    puzzle_paths = sorted(list(Path(f"data/{puzzle_subfolder}/puzzles").glob("*.txt")))

    solution_paths = [
        puzzle_path.parent.parent.joinpath("solutions") for puzzle_path in puzzle_paths
    ]

    # Create reponse file names
    response_filenames = [
        f"{file_path.stem}_response.json" for file_path in puzzle_paths
    ]

    score_filename = f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_red_herring_clues}_rh_{n_puzzles}_puzzles.txt"

    # Define evaluation folders
    response_folder = f"data/{puzzle_subfolder}/responses/{model}"
    score_folder = f"scores/{puzzle_subfolder}/{model}"

    if generate_new_responses:
        # Clean or create reponses folder
        clean_folder(folder=response_folder, keep_files=response_filenames)

    # Create the score folder if it does not exist
    os.makedirs(score_folder, exist_ok=True)

    return (
        puzzle_paths,
        solution_paths,
        response_filenames,
        response_folder,
        score_filename,
        score_folder,
    )
