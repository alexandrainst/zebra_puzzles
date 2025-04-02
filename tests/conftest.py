"""Define test configuations."""

from typing import Generator

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

initialize(config_path="../config", version_base=None)


@pytest.fixture(scope="session")
def config() -> Generator[DictConfig, None, None]:
    """Hydra configuration."""
    yield compose(
        config_name="config",
        overrides=[
            "n_puzzles=1",
            "n_objects=3",
            "n_attributes=3",
            "data_folder=tests/test_data",
            "model=gpt-4o-mini",
        ],
    )
