"""Define test configuations."""

from pathlib import Path
from shutil import rmtree
from typing import Generator

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from zebra_puzzles.pipeline import build_dataset

initialize(config_path="../config", version_base=None)


@pytest.fixture(scope="session", params=[1, 2])
def config(request) -> Generator[DictConfig, None, None]:
    """Hydra configuration.

    This fixture yields a Hydra configuration object for the tests. It uses params to create multiple configurations.
    #TODO: Test multiple values of n_red_herring_clues and n_red_herring_clues_evaluated
    """
    yield compose(
        config_name="config",
        overrides=[
            f"n_puzzles={request.param}",
            "n_objects=3",
            "n_attributes=3",
            "n_red_herring_clues=5",
            "n_red_herring_clues_evaluated=1",
            "data_folder=tests/test_data",
            "model=gpt-4o-mini",
        ],
    )


@pytest.fixture(scope="session")
def data_paths(config) -> Generator[tuple[Path, Path, Path], None, None]:
    """Fixture to generate a small dataset of zebra puzzles."""
    build_dataset(
        attributes=config.language.attributes,
        clues_dict=config.language.clues_dict,
        clue_weights=config.clue_weights,
        prompt_templates=config.language.prompt_templates,
        prompt_and=config.language.prompt_and,
        n_objects=config.n_objects,
        n_attributes=config.n_attributes,
        n_puzzles=config.n_puzzles,
        theme=config.language.theme,
        n_red_herring_clues=config.n_red_herring_clues,
        red_herring_clues_dict=config.language.red_herring_clues_dict,
        red_herring_attributes=config.language.red_herring_attributes,
        red_herring_facts=config.language.red_herring_facts,
        red_herring_clue_weights=config.red_herring_clue_weights,
        data_folder=config.data_folder,
    )

    # Load the generated puzzle
    data_folder = config.data_folder
    theme = config.language.theme
    n_objects = config.n_objects
    n_attributes = config.n_attributes
    n_red_herring_clues = config.n_red_herring_clues

    puzzle_path = Path(
        f"{data_folder}/{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh/puzzles/"
    )
    solution_path = Path(
        f"{data_folder}/{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh/solutions/"
    )
    red_herrings_path = Path(
        f"{data_folder}/{theme}/{n_objects}x{n_attributes}/{n_red_herring_clues}rh/red_herrings/"
    )

    yield puzzle_path, solution_path, red_herrings_path

    # Cleanup
    rmtree(puzzle_path.parent.parent, ignore_errors=True)
