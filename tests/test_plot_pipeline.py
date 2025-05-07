"""Tests for the `plot_pipeline` module.

Use 'pytest tests/test_plot_pipeline.py::test_name' to run a single test.

Use 'make test' to run all tests.

TODO: Do not check every parameter combination for all tests.
"""

import os


class TestHeatmapsForCurrentModel:
    """Tests related to heatmaps for the current model."""

    def test_model_plot_folder_exists(self, plot_paths_fixture, config) -> None:
        """Test that the model in the config has a folder in the plots folder."""
        # Get the path to the model currently in the config
        model_folder = plot_paths_fixture[2]

        # Check that the model folder exists
        assert model_folder.exists()
        assert model_folder.is_dir()

    def test_red_herring_plot_folder_exists(self, plot_paths_fixture, config) -> None:
        """Test that the model in the config has a folder for the chosen number of evaluated red herrings in the plots folder."""
        # Get the path to the model and number of evaluated red herring clues currently in the config
        red_herring_folder = plot_paths_fixture[3]

        # Check that the model folder contains a subfolder for the number of evaluated red herring clues
        assert red_herring_folder.exists()
        assert red_herring_folder.is_dir()

    def test_heatmaps_for_current_model(self, plot_paths_fixture, config) -> None:
        """Test that the cell score heatmap for the current evaluation is saved in the plots folder."""
        # Get the path to the model and number of evaluated red herring clues currently in the config
        red_herring_folder = plot_paths_fixture[3]

        # Get the path to the model currently in the config
        model = config.model
        n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated

        # Check that the folder contains the expected cell score file
        n_puzzles = config.n_puzzles
        cell_score_plot_filename = f"mean_cell_score_{model}_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.png"
        cell_score_plot_file_path = red_herring_folder / cell_score_plot_filename

        assert cell_score_plot_file_path.exists()
        assert cell_score_plot_file_path.is_file()
        assert os.path.getsize(cell_score_plot_file_path) > 0


def test_clue_type_difficulty_plot(plot_paths_fixture, config) -> None:
    """Test that the clue type difficulty plot file exists.

    Or check that it does not exist if n_puzzles is 1.

    """
    # Get the plotting path
    plots_path = plot_paths_fixture[0]
    model = config.model
    n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated
    n_puzzles = config.n_puzzles

    # Get the path to the model currently in the config
    model_folder = plots_path / model

    # Define the filename for the clue type difficulty plot
    clue_type_difficulty_plot_filename = f"clue_type_difficulties_{model}_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.png"
    clue_type_difficulty_plot_file_path = (
        model_folder / clue_type_difficulty_plot_filename
    )

    if n_puzzles == 1:
        # If n_puzzles is 1, the plot is not generated
        assert not clue_type_difficulty_plot_file_path.exists()
    else:
        # Check that the clue type difficulty plot file exists
        assert clue_type_difficulty_plot_file_path.exists()
        assert clue_type_difficulty_plot_file_path.is_file()
        assert os.path.getsize(clue_type_difficulty_plot_file_path) > 0


def test_clue_type_frequency_plots(plot_paths_fixture, config) -> None:
    """Test that the clue type frequency plot files exist."""
    # Get the plotting path
    plots_path = plot_paths_fixture[0]
    n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated
    n_puzzles = config.n_puzzles

    # Define the filename for the clue type frequency plot
    clue_type_frequency_plot_filename = f"clue_type_frequencies_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.png"
    clue_type_frequency_plot_file_path = (
        plots_path / "clue_type_frequencies" / clue_type_frequency_plot_filename
    )

    # Check that the clue type frequency plot file exists
    assert clue_type_frequency_plot_file_path.exists()
    assert clue_type_frequency_plot_file_path.is_file()
    assert os.path.getsize(clue_type_frequency_plot_file_path) > 0


def test_model_comparisons(plot_paths_fixture, config) -> None:
    """Test the comparisons between models in the plots folder.

    TODO: Make sure a comparison is actually done, by running two models.
    """
    # Get the list of paths to plots for each LLM model / comparison
    plots_model_paths = plot_paths_fixture[1]

    # Check that the model folders exist
    assert len(plots_model_paths) > 0

    # If multiple models are present, test the comparisons
    if len(plots_model_paths) > 1:
        # Select a folder with "vs" in the name
        comparison_folder = next((p for p in plots_model_paths if "vs" in p.name), None)

        # Check that the folder exists
        assert comparison_folder is not None
        assert comparison_folder.exists()
        assert comparison_folder.is_dir()

        # Check that the folder contains the expected cell score diff file
        n_red_herring_clues_evaluated = config.n_red_herring_clues_evaluated
        n_puzzles = config.n_puzzles

        cell_score_diff_plot_filename = f"mean_cell_score_*{comparison_folder.name}*_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.png"
        cell_score_diff_plot_file_path = (
            comparison_folder / cell_score_diff_plot_filename
        )

        # Check that the file exists and is not empty
        assert cell_score_diff_plot_file_path.exists()
        assert cell_score_diff_plot_file_path.is_file()
        assert os.path.getsize(cell_score_diff_plot_file_path) > 0

        # Check that the folder contains the expected comparison txt file
        comparison_txt_filename = next(
            (
                file
                for file in comparison_folder.iterdir()
                if file.name.startswith("comparison_")
                and file.name.endswith(
                    "_{n_red_herring_clues_evaluated}rh_{n_puzzles}_puzzles.txt"
                )
            ),
            None,
        )
        assert comparison_txt_filename is not None
        assert comparison_txt_filename.exists()
        assert comparison_txt_filename.is_file()
        assert os.path.getsize(comparison_txt_filename) > 0
