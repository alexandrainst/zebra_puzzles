"""Script to combine datasets or fix puzzle indices in file names.

Example usage:
        uv run src/scripts/fix_files.py --data_folder="data/da_huse/4x5/5rh" --number_to_add=900
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main script.

    Fixes dataset file names or copies files from one folder to another.

    This script is useful when combining datasets.

    Consider fixing puzzle indices on a copy of the dataset to avoid mixing up the original indices.

    Args:
        config: Config file.
    """
    # Choose whether to fix indices or copy files based on user input
    action = (
        input("Do you want to fix puzzle indices or copy files? (fix/copy): ")
        .strip()
        .lower()
    )
    if action not in ["fix", "copy"]:
        log.error("Invalid action. Please enter 'fix' or 'copy'.")
        return
    if action == "fix":
        log.info("You chose to fix puzzle indices.")
        # Call fix_filenames_pipeline function with user input
        fix_filenames_pipeline(
            data_folder=input("Enter the path to the dataset folder: "),
            number_to_add=int(input("Enter the number to add to the puzzle indices: ")),
            text_to_remove=input(
                "Enter the text to remove from the file names (leave empty if none): "
            ),
        )
    else:
        log.info("You chose to copy files.")
        # Call copy_files function with user input
        copy_files(
            source_folder=input("Enter the source folder to copy files from: "),
            destination_folder=input("Enter the destination folder to copy files to: "),
        )


def copy_files(source_folder: str, destination_folder: str) -> None:
    """Copies files from source_folder to destination_folder."""
    source_path = Path(source_folder)
    destination_path = Path(destination_folder)
    if not source_path.exists():
        log.error(f"Source folder {source_folder} does not exist.")
        return
    if not destination_path.exists():
        log.info(
            f"Destination folder {destination_folder} does not exist. Creating it."
        )
        destination_path.mkdir(parents=True, exist_ok=True)
    for file in source_path.glob("*"):
        if file.is_file():
            destination_file = destination_path / file.name
            # Ask about overwriting existing files
            if destination_file.exists():
                overwrite = (
                    input(f"{destination_file} already exists. Overwrite? (y/n): ")
                    .strip()
                    .lower()
                )
                if overwrite != "y":
                    log.info(
                        f"Skipping {file} as it already exists in the destination folder."
                    )
                    continue
            try:
                file.replace(destination_file)  # Use replace to move the file
            except Exception as e:
                log.error(f"Failed to copy {file} to {destination_file}: {e}")
        else:
            log.warning(f"{file} is not a file. Skipping.")


def fix_filenames_pipeline(
    data_folder: str, number_to_add: int = 0, text_to_remove=""
) -> None:
    """Pipeline to fix filenames in the dataset.

    This is useful when combining datasets generated with different numbers of puzzles.

    Runs checks before fixing puzzle indices in the files to avoid overwriting files or creating negative indices.

    Args:
        data_folder: Path to the dataset folder.
        number_to_add: Number to add to the puzzle indices in file names.
        text_to_remove: Text to remove from the file names.

    TODO: Handle folders for reduced puzzles and clue types.
    """
    # Get the paths to the folders to fix puzzle indices in
    list_of_folders = select_folders(data_folder=data_folder)

    # For each folder, rename the files by adding the number to the puzzle index
    for folder in list_of_folders:
        files = list(folder.glob("*"))

        # Sort the files descendingly by the number in their names if number_to_add is positive
        files, lowest_index = sort_files_by_index(
            files=files, number_to_add=number_to_add
        )

        # --- Run checks before fixing puzzle indices --- #
        # Check if a file contains '-' and warn if number_to_add is not zero
        if stop_due_to_current_negative_indices(
            folder=folder, number_to_add=number_to_add, files=files
        ):
            return

        # Check if indices would become negative and warn the user
        if stop_due_to_future_negative_indices(
            folder=folder, number_to_add=number_to_add, lowest_index=lowest_index
        ):
            return

        # --- Fix puzzle indices in the files --- #
        fix_filenames(
            folder=folder, number_to_add=number_to_add, text_to_remove=text_to_remove
        )
    # Log the changes
    log.info(
        f"Fixed filenames in {data_folder} by removing {text_to_remove} and adding {number_to_add} to the indices."
    )


def fix_filenames(
    folder: Path, number_to_add: int = 0, text_to_remove: str = ""
) -> None:
    """Fix puzzle indices in file names in the dataset.

    Adds a number to the puzzle indices in file names in the dataset and/or removes specified text from the file names.

    Args:
        folder: Path to the dataset folder.
        number_to_add: Number to add to the puzzle indices in file names.
        text_to_remove: Text to remove from the file names.
    """
    for file in folder.glob("*"):
        if not file.is_file():
            continue
        # If text_to_remove is specified, remove it from the file name
        if text_to_remove:
            new_file_name = file.name.replace(text_to_remove, "")
            new_file_path = file.parent / new_file_name
            # Rename the file
            # Check if the new file name already exists
            if new_file_path.exists():
                log.warning(f"{new_file_path} already exists. Skipping renaming.")
                continue
            file.rename(new_file_path)
            file = new_file_path

        # Extract the current index from the file name, which is the only number

        current_index = int("".join(filter(str.isdigit, file.name)))
        new_index = current_index + number_to_add
        # Create the new file name by replacing the old index with the new index
        new_file_name = file.name.replace(str(current_index), str(new_index))
        new_file_path = file.parent / new_file_name
        # Rename the file
        file.rename(new_file_path)


def stop_due_to_current_negative_indices(
    folder: Path, number_to_add: int, files: list[Path]
) -> bool:
    """Check if any files in the dataset have negative indices.

    Args:
        folder: Path to the dataset folder.
        number_to_add: Number to add to the puzzle indices in file names.
        files: List of files in the folder.

    Returns:
        True if any files have negative indices and the user does not want to continue, False otherwise.
    """
    if any("-" in file.name for file in files) and number_to_add != 0:
        if number_to_add < 0:
            raise ValueError(
                f"Files in {folder} contain '-' in their names. This may lead to unexpected results when subtracting {abs(number_to_add)} from the indices."
            )
        confirm = (
            input(
                f"Files in {folder} contain '-' in their names. This may lead to unexpected results when adding {number_to_add} to the indices. Do you want to continue? (y/n): "
            )
            .strip()
            .lower()
        )
        if confirm != "y":
            log.info("Aborting renaming of files.")
            # Output a sign to stop the script
            return True
        else:
            log.warning(
                f"Continuing with renaming files in {folder} even though it may lead to unexpected results."
            )
    return False


def stop_due_to_future_negative_indices(
    folder: Path, number_to_add: int, lowest_index: int
) -> bool:
    """Check if any files in the dataset would have negative indices after renaming.

    Args:
        folder: Path to the dataset folder.
        number_to_add: Number to add to the puzzle indices in file names.
        lowest_index: The lowest puzzle index in the files.

    Returns:
        True if adding number_to_add would result in negative indices and the user does not want to continue, False otherwise.
    """
    if lowest_index + number_to_add < 0:
        confirm = (
            input(
                f"The lowest index in {folder} is {lowest_index}. Adding {number_to_add} would result in negative indices. Do you want to continue? (y/n): "
            )
            .strip()
            .lower()
        )
        if confirm != "y":
            log.info("Aborting renaming of files.")
            # Output a sign to stop the script
            return True
        else:
            log.warning(
                f"Continuing with renaming files in {folder} even though it will result in negative indices."
            )
    return False


def select_folders(data_folder: str) -> list[Path]:
    """Select folders to fix puzzle indices in.

    Args:
        data_folder: Path to the dataset folder.

    Returns:
        A list of folders to fix puzzle indices in.

    """
    # Get the paths to the clue types, puzzles, red_herring clues, responses and solutions folders
    data_folder_path = Path(data_folder)
    clue_types_folder = data_folder_path / "clue_types"
    puzzles_folder = data_folder_path / "puzzles"
    red_herrings_folder = data_folder_path / "red_herrings"
    responses_folder = data_folder_path / "responses"
    solutions_folder = data_folder_path / "solutions"

    list_of_folders = [
        clue_types_folder,
        puzzles_folder,
        red_herrings_folder,
        responses_folder,
        solutions_folder,
    ]

    # If list_of_folders is empty, use the data_folder_path directly
    if not any(folder.exists() for folder in list_of_folders):
        log.info(
            f"No subfolders found in {data_folder}. Using the data folder directly."
        )
        list_of_folders = [data_folder_path]
    else:
        # Filter out non-existing folders
        list_of_folders = [folder for folder in list_of_folders if folder.exists()]

    # Add responses subfolders if they exist
    if (data_folder_path / "responses").exists():
        # Add each subfolder to list_of_folders
        for subfolder in (data_folder_path / "responses").iterdir():
            if subfolder.is_dir():
                list_of_folders.append(subfolder)
    return list_of_folders


def sort_files_by_index(
    files: list[Path], number_to_add: int = 0
) -> tuple[list[Path], int]:
    """Sort files to avoid overwriting files.

    Sort descendingly by the number in their names if number_to_add is positive,
    otherwise sort ascendingly.

    Args:
        files: List of files to sort.
        number_to_add: Number to add to the puzzle indices in file names.

    Returns:
        A tuple (files, lowest_index) where:
            files: Sorted list of files.
            lowest_index: The lowest index in the sorted files.
    """
    if number_to_add > 0:
        files.sort(
            key=lambda x: int("".join(filter(str.isdigit, x.name))), reverse=True
        )
        lowest_index = int("".join(filter(str.isdigit, files[-1].name)))
    else:
        files.sort(key=lambda x: int("".join(filter(str.isdigit, x.name))))
        lowest_index = int("".join(filter(str.isdigit, files[0].name)))
    return files, lowest_index


if __name__ == "__main__":
    main()
