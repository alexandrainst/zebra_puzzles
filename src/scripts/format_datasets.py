"""Script to format datasets and push them to Hugging Face.

This script should run after build_dataset.py.

Usage:
    uv run src/scripts/format_datasets.py <config_key>=<config_value> ...
"""

import json
import logging
from pathlib import Path

import hydra
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main script.

    Formats datasets using Hugging Face's Datasets library.

    Args:
        config: Config file.
    """
    data_folder_train = config.data_folder_train
    data_folder_test = config.data_folder_test

    theme = config.language.theme
    n_red_herring_clues = config.n_red_herring_clues
    n_attributes = config.n_attributes
    n_objects = config.n_objects

    # Set number of puzzles for training and testing datasets and the data folder to save the datasets in
    n_puzzles_train = 128
    n_puzzles_test = 1024
    data_folder_current = "data"

    format_datasets_pipeline(
        data_folder_current=data_folder_current,
        data_folder_train=data_folder_train,
        data_folder_test=data_folder_test,
        n_puzzles_train=n_puzzles_train,
        n_puzzles_test=n_puzzles_test,
        theme=theme,
        n_red_herring_clues=n_red_herring_clues,
        n_attributes=n_attributes,
        n_objects=n_objects,
    )


def format_datasets_pipeline(
    data_folder_current: str,
    data_folder_train: str,
    data_folder_test: str,
    n_puzzles_train: int,
    n_puzzles_test: int,
    theme: str,
    n_red_herring_clues: int,
    n_attributes: int,
    n_objects: int,
) -> None:
    """Formats datasets.

    Args:
        data_folder_current: Path to the current data folder.
        data_folder_train: Path to the training data folder.
        data_folder_test: Path to the testing data folder.
        n_puzzles_train: Number of puzzles in the training dataset.
        n_puzzles_test: Number of puzzles in the testing dataset.
        theme: Theme of the puzzles.
        n_red_herring_clues: Number of red herring clues in the puzzles.
        n_attributes: Number of attributes in the puzzles.
        n_objects: Number of objects in the puzzles.

    Returns:
        None
    """
    # Check if dataset already exists
    dataset_name = f"dataset_{theme}_{n_objects}x{n_attributes}_{n_red_herring_clues}rh"
    if (Path(data_folder_current) / dataset_name).exists():
        log.info(f"Dataset {dataset_name} already exists. Skipping formatting.")
        split_dataset = DatasetDict.load_from_disk(
            Path(data_folder_current) / dataset_name, keep_in_memory=True
        )
        log.info(f"Dataset {dataset_name} loaded from {data_folder_current}.")
    else:
        train_dataset = load_and_format_a_dataset(
            data_folder=data_folder_train,
            theme=theme,
            n_puzzles=n_puzzles_train,
            n_red_herring_clues=n_red_herring_clues,
            n_attributes=n_attributes,
            n_objects=n_objects,
        )
        test_dataset = load_and_format_a_dataset(
            data_folder=data_folder_test,
            theme=theme,
            n_puzzles=n_puzzles_test,
            n_red_herring_clues=n_red_herring_clues,
            n_attributes=n_attributes,
            n_objects=n_objects,
        )

        split_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

        # Save datasets
        split_dataset.save_to_disk(Path(data_folder_current) / dataset_name)
        log.info(
            f"Dataset {dataset_name} formatted and saved to {data_folder_current}/."
        )

    # Ask user if they want to push the dataset to Hugging Face Hub
    push_to_hub = (
        input("Do you want to push the dataset to Hugging Face Hub? (y/n): ")
        .strip()
        .lower()
    )
    if push_to_hub == "y":
        split_dataset.push_to_hub(
            "alexandrainst/zebra_puzzles",
            dataset_name,
            private=True,
            embed_external_files=True,
        )


def load_and_format_a_dataset(
    data_folder: str,
    theme: str,
    n_puzzles: int,
    n_red_herring_clues: int,
    n_attributes: int,
    n_objects: int,
) -> Dataset:
    """Format a dataset in a specific folder.

    Args:
        data_folder: Path to the folder containing the dataset files.
        theme: Theme of the puzzles.
        n_puzzles: Number of puzzles to load.
        n_red_herring_clues: Number of red herring clues in the puzzles.
        n_attributes: Number of attributes in the puzzles.
        n_objects: Number of objects in the puzzles.

    Returns:
        Dataset: Formatted dataset.
    """
    # Get the full path to the data folder
    full_data_path = (
        Path(data_folder)
        / theme
        / f"{n_objects}x{n_attributes}"
        / f"{n_red_herring_clues}rh"
    )

    # Load dataset
    puzzles, clue_files, red_herring_files, solution_files = load_dataset(
        full_data_path=full_data_path, n_puzzles=n_puzzles
    )

    # Format the dataset
    data_dict = format_a_dataset(
        puzzles=puzzles,
        clue_files=clue_files,
        red_herring_files=red_herring_files,
        solution_files=solution_files,
    )
    dataset = Dataset.from_dict(data_dict)
    return dataset


def load_dataset(
    full_data_path: Path, n_puzzles: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load dataset from a folder.

    Each dataset is loaded as a list of strings.

    Args:
        full_data_path: Path to the folder containing the dataset files.
        n_puzzles: Expected number of puzzles in the folder.

    Returns:
        A tuple (puzzles, clue_files, red_herring_files, solution_files), where:
            puzzles: List of puzzle strings.
            clue_files: List of clue type strings.
            red_herring_files: List of red herring indices strings.
            solution_files: List of solution strings.
    """
    puzzles_path = full_data_path / "puzzles"
    puzzles = load_files(data_path=puzzles_path, n_puzzles=n_puzzles)

    clue_types_path = full_data_path / "clue_types"
    clue_files = load_files(data_path=clue_types_path, n_puzzles=n_puzzles)

    red_herrings_path = full_data_path / "red_herrings"
    red_herring_files = load_files(data_path=red_herrings_path, n_puzzles=n_puzzles)

    solutions_path = full_data_path / "solutions"
    solution_files = load_files(data_path=solutions_path, n_puzzles=n_puzzles)

    return puzzles, clue_files, red_herring_files, solution_files


def format_a_dataset(
    puzzles: list[str],
    clue_files: list[str],
    red_herring_files: list[str],
    solution_files: list[str],
) -> dict[str, list]:
    """Format a dataset.

    Combines data from multiple files into a dictionary. The input lists of strings correspond to the following contents:
        * puzzles: list[str]
        * clue_types: list[list[str]]
        * red_herrings: list[list[int]]
        * solutions: list[dict[str,list[str]]]

    Args:
        puzzles: List of puzzle strings.
        clue_files: List of clue type strings.
        red_herring_files: List of red herring indices strings.
        solution_files: List of solution strings.

    Returns:
        Dictionary containing a formatted dataset.
    """
    # Format puzzles
    # Split them into introduction, clues and format_instructions
    introductions: list[str] = []
    clues: list[list[str]] = []
    questions: list[str] = []
    format_instructions: list[str] = []
    format_examples: list[str] = []

    for puzzle in puzzles:
        # The introduction is everything before the first clue
        introductions.append(puzzle.split("1.")[0])

        # Clues are all lines starting with a number
        clues.append(
            [
                line.strip()
                for line in puzzle.split("\n")
                if line.strip() and line[0].isdigit()
            ]
        )

        # Question is the line after the clues ending with a question mark
        text_after_clues = puzzle.split(clues[-1][-1])[-1].strip()
        questions.append(
            [
                line.strip()
                for line in text_after_clues.split("\n")
                if line.strip().endswith("?")
            ][0]
        )

        # Format instructions is the following line
        format_instructions.append(
            text_after_clues.split(questions[-1])[-1].strip().split("\n")[0]
        )

        # Format example is everything after the format instructions
        format_examples.append(
            text_after_clues.split(format_instructions[-1])[-1].strip()
        )

    # Format clue types
    clue_files_formatted: list[list[str]] = []
    for i, clue_str in enumerate(clue_files):
        clue_files_formatted.append([clue.strip() for clue in clue_str.split(",")])

    # Format red herrings
    red_herring_files_formatted: list[list[int]] = []
    for i, red_herring_str in enumerate(red_herring_files):
        red_herring_files_formatted.append(
            [int(idx.strip()) for idx in red_herring_str.split(",")]
        )

    # Format solutions
    solution_files_formatted: list[dict[str, list[str]]] = []
    for i, solution_str in enumerate(solution_files):
        solution_files_formatted.append(json.loads(solution_str))
        # TODO: Consider validating the solution format

    if (
        len(format_instructions) != len(clue_files_formatted)
        or len(questions) != len(solution_files_formatted)
        or len(format_examples) != len(introductions)
    ):
        print(
            "Mismatch in columns:"
            f"\n  Introductions: {introductions[0]}"
            f"\n  Clues: {clues[0]}"
            f"\n  Questions: {questions[0]}"
            f"\n  Format Instructions: {format_instructions[0]}"
            f"\n  Format Examples: {format_examples[0]}"
            f"\n  Solutions: {solution_files_formatted[0]}"
            f"\n  Clue Types: {clue_files_formatted[0]}"
        )
        raise ValueError("Mismatch in columns.")

    return {
        "introductions": introductions,
        "clues": clues,
        "questions": questions,
        "format_instructions": format_instructions,
        "format_examples": format_examples,
        "solutions": solution_files_formatted,
        "clue_types": clue_files_formatted,
        "red_herrings": red_herring_files_formatted,
    }


def load_files(data_path: Path, n_puzzles: int) -> list[str]:
    """Load n_puzzles files in a folder as strings.

    Args:
        data_path: Path to the folder containing files.
        n_puzzles: Expected number of puzzles in the folder.

    Returns:
        List of file contents as strings.
    """
    filenames = list(data_path.glob("*"))

    files_list = []
    for i, filename in enumerate(filenames):
        with filename.open() as file:
            file_content = file.read()
            files_list.append(file_content)
        if i == n_puzzles - 1:
            break

    # Check if the number of puzzles matches
    if len(files_list) != n_puzzles:
        raise ValueError(
            f"Expected {n_puzzles} puzzles, but only found {len(files_list)} in {data_path}."
        )
    if len(filenames) > n_puzzles:
        log.warning(
            f"Found more files than expected ({len(filenames)}), only the first {n_puzzles} will be used."
        )

    return files_list


if __name__ == "__main__":
    main()
