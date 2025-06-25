"""Script to format datasets.

This script should run after build_dataset.py.

Usage:
    uv run src/scripts/format_datasets.py <config_key>=<config_value> ...
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main script.

    Formats datasets using Hugging Face's Datasets library.

    Args:
        config: Config file.
    """
    n_puzzles = config.n_puzzles
    theme = config.language.theme
    data_folder = config.data_folder
    n_red_herring_clues = config.n_red_herring_clues
    n_attributes = config.n_attributes
    n_objects = config.n_objects

    format_datasets_pipeline(data_folder=data_folder,
                            n_puzzles=n_puzzles,
                            theme=theme,
                            n_red_herring_clues=n_red_herring_clues,
                            n_attributes=n_attributes,
                            n_objects=n_objects
                            )


def format_datasets_pipeline(data_folder:str, n_puzzles: int, theme: str, n_red_herring_clues: int, n_attributes: int, n_objects: int
) -> None:
    """Formats datasets
    
    * (list[str]) puzzles
    * (list[list[str]]) clue_types
    * (list[list[int]]) red_herrings
    * (list[dict[str,list[str]]]) solutions

    Args:
        data_folder: Path to the folder containing the dataset files.

    Returns:
        None
    """
    data_folder = Path(data_folder)
    full_data_path = data_folder / theme / f"{n_objects}x{n_attributes}" / f"{n_red_herring_clues}rh"

    # Load dataset
    data_dict = load_dataset(full_data_path=full_data_path,n_puzzles=n_puzzles)

    """ TODO:
    DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "puzzles": [],
                    "clue_types": [],
                    "red_herrings": [],
                    "solutions": [],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "puzzles": [],
                    "clue_types": [],
                    "red_herrings": [],
                    "solutions": [],
                }
            ),
        }
    )"""

    return

def load_dataset(full_data_path: str, n_puzzles: int) -> dict[str, list]:
    """Load dataset from a folder.
    
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
        "clue_types": clue_files,
        "puzzles": puzzles,
        "red_herrings": red_herring_files,
        "solutions": solution_files,
    }

def load_files(data_path: Path, n_puzzles: int)-> list[str]:
    """Load files in a folder as strings.
    
    Args:
        data_path: Path to the folder containing files.
        n_puzzles: Expected number of puzzles in the folder.

    Returns:
        List of file contents as strings.
    """
    filenames = list(data_path.glob("*"))

    files_list = []
    for filename in filenames:
        with filename.open() as file:
            file_content = file.read()
            files_list.append(file_content)

    # Check if the number of puzzles matches
    if len(files_list) != n_puzzles:
        raise ValueError(f"Expected {n_puzzles} puzzles, but found {len(files_list)} in {data_path}.")
    return files_list


if __name__ == "__main__":
    main()