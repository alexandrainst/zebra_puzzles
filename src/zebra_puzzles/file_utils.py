"""Module for handling files and directories."""

import os
from pathlib import Path

from zebra_puzzles.clue_removal import remove_red_herrings


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
            print("Old files were deleted.")
        else:
            print("Old files were not deleted.")


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
    puzzle_folder = data_folder / puzzle_subfolder / f"{n_red_herring_clues}rh/puzzles"

    puzzle_paths = sorted(list(puzzle_folder.glob("*.txt")))

    solution_paths = [
        puzzle_path.parent.parent.joinpath("solutions") for puzzle_path in puzzle_paths
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
