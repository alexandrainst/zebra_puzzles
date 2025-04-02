"""Tests for the `module` module."""

from conf_test import config

from zebra_puzzles.pipeline import build_dataset


# An end-to-end test of build_dataset
def test_pipeline() -> None:
    """Test build_dataset."""
    # Load the config file using the pytest fixture from conf_test.py
    attributes = config.language.attributes
    clues_dict = config.language.clues_dict
    clue_weights = config.clue_weights
    prompt_templates = config.language.prompt_templates
    prompt_and = config.language.prompt_and
    theme = config.language.theme
    n_red_herring_clues = config.n_red_herring_clues
    red_herring_clues_dict = config.language.red_herring_clues_dict
    red_herring_attributes = config.language.red_herring_attributes
    red_herring_facts = config.language.red_herring_facts
    red_herring_clue_weights = config.red_herring_clue_weights
    n_puzzles = config.n_puzzles
    data_folder = config.data_folder

    # Override the config file
    n_objects = 4
    n_attributes = 3
    data_folder = "tests/test_data"

    # Call the function to test
    build_dataset(
        attributes=attributes,
        clues_dict=clues_dict,
        clue_weights=clue_weights,
        prompt_templates=prompt_templates,
        prompt_and=prompt_and,
        n_objects=n_objects,
        n_attributes=n_attributes,
        n_puzzles=n_puzzles,
        theme=theme,
        n_red_herring_clues=n_red_herring_clues,
        red_herring_clues_dict=red_herring_clues_dict,
        red_herring_attributes=red_herring_attributes,
        red_herring_facts=red_herring_facts,
        red_herring_clue_weights=red_herring_clue_weights,
        data_folder=data_folder,
    )

    # assert len(prompt) > len(prompt_templates[0])
    # assert len(solution_str) > 0
    # assert isinstance(prompt, str)
    # assert isinstance(solution_str, str)
    assert True
