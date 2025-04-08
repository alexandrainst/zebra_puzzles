"""Module for plotting the results of LLM's trying to solve zebra puzzles."""


def plot_results(
    n_puzzles: int,
    n_objects: int,
    n_attributes: int,
    model: str,
    theme: str,
    n_red_herring_clues_evaluated: int,
    data_folder: str,
) -> None:
    """Plot the results of the LLM's trying to solve zebra puzzles.

    Args:
        n_puzzles: Number of puzzles.
        n_objects: Number of objects.
        n_attributes: Number of attributes.
        model: LLM model name.
        theme: Theme name.
        n_red_herring_clues_evaluated: Number of red herring clues evaluated.
        data_folder: Path to the data folder.

    TODO: Consider just plotting everything in a folder instead of specifying n_puzzles, model etc.
    """

    # Import results from score files

    # Format the data

    # Plot heatmap of the results

    # Prepare path for plots
