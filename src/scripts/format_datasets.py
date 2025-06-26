"""Script to format datasets.

This script should run after build_dataset.py.

Usage:
    uv run src/scripts/format_datasets.py <config_key>=<config_value> ...
"""

import logging
import json
from pathlib import Path
from datasets import Dataset, DatasetDict

import hydra
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
    n_puzzles = config.n_puzzles

    theme = config.language.theme
    n_red_herring_clues = config.n_red_herring_clues
    n_attributes = config.n_attributes
    n_objects = config.n_objects

    # For testing purposes, change to the desired number of puzzles. TODO: Remove this line.
    n_puzzles = 3 

    format_datasets_pipeline(data_folder_train=data_folder_train,
                            data_folder_test=data_folder_test,
                            n_puzzles=n_puzzles,
                            theme=theme,
                            n_red_herring_clues=n_red_herring_clues,
                            n_attributes=n_attributes,
                            n_objects=n_objects
                            )


def format_datasets_pipeline(data_folder_train:str,data_folder_test:str, n_puzzles: int, theme: str, n_red_herring_clues: int, n_attributes: int, n_objects: int
) -> None:
    """Formats datasets.

    Args:
        data_folder_train: Path to the training data folder.
        data_folder_test: Path to the testing data folder.
        n_puzzles: Number of puzzles to load.
        theme: Theme of the puzzles.
        n_red_herring_clues: Number of red herring clues in the puzzles.
        n_attributes: Number of attributes in the puzzles.
        n_objects: Number of objects in the puzzles.

    Returns:
        None
    """
    train_dataset =  format_a_dataset(
        data_folder=data_folder_train,
        theme=theme,
        n_puzzles=n_puzzles,
        n_red_herring_clues=n_red_herring_clues,
        n_attributes=n_attributes,
        n_objects=n_objects
    )
    test_dataset  =  format_a_dataset(
        data_folder=data_folder_test,
        theme=theme,
        n_puzzles=n_puzzles,
        n_red_herring_clues=n_red_herring_clues,
        n_attributes=n_attributes,
        n_objects=n_objects
    )

    split_dataset = DatasetDict({"train":train_dataset,"test":test_dataset})
    
    # Save datasets
    # TODO

def format_a_dataset(
    data_folder: str,
    theme: str,
    n_puzzles: int,
    n_red_herring_clues: int,
    n_attributes: int,
    n_objects: int
    ) -> Dataset:
    """ Format a dataset in a specific folder.
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
    full_data_path = Path(data_folder) / theme / f"{n_objects}x{n_attributes}" / f"{n_red_herring_clues}rh"

    # Load dataset
    data_dict = load_dataset(full_data_path=full_data_path,n_puzzles=n_puzzles)

    # Format the dataset
    dataset =  Dataset.from_dict(data_dict)
    return dataset


def load_dataset(full_data_path: Path, n_puzzles: int) -> dict[str, list]:
    """Load dataset from a folder.

    Combines data from multiple files into a dictionary with the following structure:
    * puzzles: list[str]
    * clue_types: list[list[str]]
    * red_herrings: list[list[int]]
    * solutions: list[dict[str,list[str]]]
    
    Args:
        full_data_path: Path to the folder containing the dataset files.
        n_puzzles: Expected number of puzzles in the folder.
    """

    puzzles_path = full_data_path / "puzzles"
    puzzles = load_files(data_path = puzzles_path, n_puzzles=n_puzzles)

    clue_types_path = full_data_path / "clue_types"
    clue_files = load_files(data_path = clue_types_path, n_puzzles=n_puzzles)

    red_herrings_path = full_data_path / "red_herrings"
    red_herring_files = load_files(data_path = red_herrings_path, n_puzzles=n_puzzles)

    solutions_path = full_data_path / "solutions"
    solution_files = load_files(data_path = solutions_path, n_puzzles=n_puzzles)

    # Format clue types
    clue_files_formatted: list[list[str]] = [None] * len(clue_files)
    for i, clue_str in enumerate(clue_files):
        clue_files_formatted[i] = [clue.strip() for clue in clue_str.split(",")]

    # Format red herrings
    red_herring_files_formatted: list[list[int]] = [None] * len(red_herring_files)
    for i, red_herring_str in enumerate(red_herring_files):
        red_herring_files_formatted[i] = [int(idx.strip()) for idx in red_herring_str.split(",")]

    # Format solutions
    solution_files_formatted: list[dict[str, list[str]]] = [None] * len(solution_files)
    for i, solution_str in enumerate(solution_files):
        solution_files_formatted[i]  = json.loads(solution_str)
        # TODO: Consider validating the solution format
    
    return {
        "clue_types": clue_files_formatted,
        "puzzles": puzzles,
        "red_herrings": red_herring_files_formatted,
        "solutions": solution_files_formatted,
    }

def load_files(data_path: Path, n_puzzles: int)-> list[str]:
    """Load n_puzzles files in a folder as strings.
    
    Args:
        data_path: Path to the folder containing files.
        n_puzzles: Expected number of puzzles in the folder.

    Returns:
        List of file contents as strings.
    """
    filenames = list(data_path.glob("*"))

    files_list = []
    for i,filename in enumerate(filenames):
        with filename.open() as file:
            file_content = file.read()
            files_list.append(file_content)
        if i == n_puzzles - 1:
            break

    # Check if the number of puzzles matches
    if len(files_list) != n_puzzles:
        raise ValueError(f"Expected {n_puzzles} puzzles, but only found {len(files_list)} in {data_path}.")
    if len(filenames) > n_puzzles:
        log.warning(f"Found more files than expected ({len(filenames)}), only the first {n_puzzles} will be used.")

    return files_list


if __name__ == "__main__":
    main()