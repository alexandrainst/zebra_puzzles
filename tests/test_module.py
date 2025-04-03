"""Tests for the `module` module.

Use 'pytest tests/test_module.py::test_name' to run a single test.

Use 'make test' to run all tests.
"""

import json
import os


def test_prompt(data_paths, config) -> None:
    """Test the prompt generated by build_dataset."""
    # Get the puzzle path
    puzzle_path = data_paths[0]

    # Load the first file in the puzzle directory
    puzzle_filename = os.listdir(puzzle_path)[0]
    puzzle_file_path_str = puzzle_path / puzzle_filename

    # Load a generated puzzle
    with open(puzzle_file_path_str, "r") as f:
        prompt = f.read()

    prompt_templates = config.language.prompt_templates

    # Check if the prompt is a string and has a length greater than the first template
    assert isinstance(prompt, str)
    assert len(prompt) > len(prompt_templates[0])


def test_solution(data_paths, config) -> None:
    """Test the solutions generated by build_dataset."""
    # Get the solution path
    solution_path = data_paths[1]

    # Load a generated solution
    solution_file_path = solution_path / "zebra_puzzle_0_solution.json"
    with open(solution_file_path, "r") as f:
        solution = f.read()

    # Check the dimensions of the solution
    solution_dict = json.loads(solution)
    assert isinstance(solution_dict, dict)
    assert len(solution_dict) == config.n_objects
    assert len(solution_dict["object_1"]) == config.n_attributes


def test_red_herring_files(data_paths, config) -> None:
    """Test the red herring files generated by build_dataset."""
    # Get the red herring path
    red_herring_path = data_paths[2]

    # Load a generated red herring file
    red_herring_file_path = red_herring_path / "zebra_puzzle_0_red_herrings.txt"
    with open(red_herring_file_path, "r") as f:
        red_herring_indices_str = f.read()

    # Check that the file is empty or contains comma-separated indices
    assert red_herring_indices_str == "" or isinstance(red_herring_indices_str, str)

    # If the file is not empty, check that it contains comma-separated indices
    # and that the number of indices matches n_red_herring_clues
    # and that the maximum index is greater than or equal to n_red_herring_clues
    if red_herring_indices_str != "":
        red_herring_indices_list = red_herring_indices_str.split(",")
        i_red_herrings = [int(i) for i in red_herring_indices_list]
        assert isinstance(i_red_herrings, list)
        assert len(i_red_herrings) == config.n_red_herring_clues
        assert max(i_red_herrings) >= config.n_red_herring_clues


def test_scores(eval_paths, config) -> None:
    """Test the evaluation scores of a dataset of zebra puzzles."""
    # Get the scores path
    scores_path = eval_paths[0]

    theme = config.language.theme
    n_objects = config.n_objects
    n_attributes = config.n_attributes
    n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated
    model = config.model
    n_puzzles = config.n_puzzles

    scores_file_path = (
        scores_path
        / f"puzzle_scores_{model}_{theme}_{n_objects}x{n_attributes}_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"
    )
    with open(scores_file_path, "r") as f:
        scores_str = f.read()

    # Get the number after "best permuted cell score:"
    best_permuted_cell_score_str = scores_str.split("best permuted cell score: ")[1]
    best_permuted_cell_score_str = best_permuted_cell_score_str.split("\n")[0]
    best_permuted_cell_score = float(best_permuted_cell_score_str)

    # Check that the score file is not empty and the best permuted cell score is greater than 0
    assert isinstance(scores_str, str)
    assert len(scores_str) > 0
    assert best_permuted_cell_score > 0


def test_responses(eval_paths, config) -> None:
    """Test the responses generated by the evaluation."""
    # Get the response path
    responses_path = eval_paths[1]
    responses_file_path = responses_path / "zebra_puzzle_0_response.json"

    with open(responses_file_path, "r") as f:
        response = f.read()

    # Check the dimensions of the solution
    response_dict = json.loads(response)

    assert isinstance(response_dict, dict)
    assert len(response_dict) == config.n_objects
    assert len(response_dict["object_1"]) == config.n_attributes
