"""Define test configurations."""

import random
from pathlib import Path
from shutil import rmtree
from typing import Generator

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from zebra_puzzles.evaluation.eval_pipeline import evaluate_all
from zebra_puzzles.performance.plot_pipeline import plot_results
from zebra_puzzles.puzzle_creation.build_pipeline import build_dataset

initialize(config_path="../config", version_base=None)


# Test configurations for pytest
# Each set of parameters represents a different configuration for the tests.
@pytest.fixture(
    scope="session", params=[(1, 2, 0), (2, 2, 0), (1, 2, 2), (2, 2, 2), (3, 0, 0)]
)
def config(request) -> Generator[DictConfig, None, None]:
    """Hydra configuration.

    This fixture yields a Hydra configuration object for the tests. It uses params to create multiple configurations.
    """
    # Set the seeds of random and numpy
    random.seed(42)
    np.random.seed(42)

    n_puzzles, n_red_herring_clues, n_red_herring_clues_evaluated = request.param

    yield compose(
        config_name="config",
        overrides=[
            f"n_puzzles={n_puzzles}",
            "n_objects=3",
            "n_attributes=3",
            f"n_red_herring_clues={n_red_herring_clues}",
            f"n_red_herring_clues_evaluated={n_red_herring_clues_evaluated}",
            "data_folder=tests/test_data",
            "model=gpt-4o-mini",
            "generate_new_responses=True",
        ],
    )


@pytest.fixture(scope="session")
def data_paths_fixture(config) -> Generator[tuple[Path, Path, Path], None, None]:
    """Fixture to generate a small dataset of zebra puzzles."""
    build_dataset(
        attributes=config.language.attributes,
        clues_dict=config.language.clues_dict,
        clue_weights=config.clue_weights,
        clue_cases_dict=config.language.clue_cases_dict,
        prompt_templates=config.language.prompt_templates,
        prompt_and=config.language.prompt_and,
        prompt_replacements=config.language.prompt_replacements,
        n_objects=config.n_objects,
        n_attributes=config.n_attributes,
        n_puzzles=config.n_puzzles,
        theme=config.language.theme,
        n_red_herring_clues=config.n_red_herring_clues,
        red_herring_clues_dict=config.language.red_herring_clues_dict,
        red_herring_attributes=config.language.red_herring_attributes,
        red_herring_facts=config.language.red_herring_facts,
        red_herring_clue_weights=config.red_herring_clue_weights,
        red_herring_cases_dict=config.language.red_herring_cases_dict,
        data_folder_str=config.data_folder,
    )

    # Load the generated puzzle
    data_folder_str = config.data_folder
    theme = config.language.theme
    n_objects = config.n_objects
    n_attributes = config.n_attributes
    n_red_herring_clues = config.n_red_herring_clues

    data_folder = Path(data_folder_str)

    puzzle_path = (
        data_folder
        / theme
        / f"{n_objects}x{n_attributes}"
        / f"{n_red_herring_clues}rh"
        / "puzzles"
    )
    solution_path = (
        data_folder
        / theme
        / f"{n_objects}x{n_attributes}"
        / f"{n_red_herring_clues}rh"
        / "solutions"
    )
    red_herrings_path = (
        data_folder
        / theme
        / f"{n_objects}x{n_attributes}"
        / f"{n_red_herring_clues}rh"
        / "red_herrings"
    )

    yield puzzle_path, solution_path, red_herrings_path

    # Cleanup
    rmtree(puzzle_path.parent.parent, ignore_errors=True)


@pytest.fixture(scope="session")
def eval_paths_fixture(
    data_paths_fixture, config
) -> Generator[tuple[Path, Path], None, None]:
    """Fixture to evaluate puzzles after generating them by the data_paths fixture."""
    # Evaluate the dataset
    evaluate_all(
        n_puzzles=config.n_puzzles,
        n_objects=config.n_objects,
        n_attributes=config.n_attributes,
        model=config.model,
        theme=config.language.theme,
        generate_new_responses=config.generate_new_responses,
        n_red_herring_clues=config.n_red_herring_clues,
        n_red_herring_clues_evaluated=config.n_red_herring_clues_evaluated,
        data_folder_str=config.data_folder,
    )

    # Load the response files
    data_folder_str = config.data_folder
    theme = config.language.theme
    n_objects = config.n_objects
    n_attributes = config.n_attributes
    n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated
    model = config.model

    data_folder = Path(data_folder_str)

    scores_path = data_folder / "scores" / model / f"{n_red_herring_clues_evaluated}rh"
    responses_path = (
        data_folder
        / theme
        / f"{n_objects}x{n_attributes}"
        / f"{n_red_herring_clues_evaluated}rh"
        / "responses"
        / model
    )

    yield scores_path, responses_path

    # Cleanup
    rmtree(scores_path, ignore_errors=True)
    rmtree(responses_path, ignore_errors=True)


@pytest.fixture(scope="session")
def plot_paths_fixture(
    eval_paths_fixture, config
) -> Generator[tuple[Path, list, Path, Path], None, None]:
    """Fixture to generate plots after evaluating puzzles by the eval_paths fixture."""
    # Run the plotting script
    n_puzzles = config.n_puzzles
    theme = config.language.theme
    data_folder = config.data_folder
    clue_types = list(config.clue_weights.keys())
    red_herring_clue_types = list(config.red_herring_clue_weights.keys())
    n_red_herring_clues = config.n_red_herring_clues
    model = config.model
    n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated

    plot_results(
        n_puzzles=n_puzzles,
        theme=theme,
        data_folder_str=data_folder,
        clue_types=clue_types,
        red_herring_clue_types=red_herring_clue_types,
        n_red_herring_clues_generated=n_red_herring_clues,
    )

    plots_path = Path(data_folder) / "plots" / theme

    # Get the folder names in the plots path (corresponding to the model names and comparisons)
    plots_model_paths = [
        p
        for p in plots_path.iterdir()
        if p.is_dir() and p.name != "clue_type_frequencies"
    ]

    model_folder = plots_path / model

    red_herring_folder = model_folder / f"{n_red_herring_clues_evaluated}rh"

    yield plots_path, plots_model_paths, model_folder, red_herring_folder

    # Cleanup
    rmtree(plots_path, ignore_errors=True)
