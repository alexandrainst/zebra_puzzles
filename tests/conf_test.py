"""Define test configuations."""

from typing import Generator

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

initialize(config_path="../config", version_base=None)


@pytest.fixture(scope="session")
def config() -> Generator[DictConfig, None, None]:
    """Hydra configuration."""
    yield compose(config_name="config")
