"""Module for creating plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zebra_puzzles.zebra_utils import round_using_std


def plot_heatmaps(
    scores_array: np.ndarray,
    score_types: list[str],
    plot_path: Path,
    n_red_herring_clues_evaluated_str: str,
    std_scores_array: np.ndarray,
    single_model: bool,
    model: str,
    n_puzzles: int,
) -> None:
    """Plot heatmaps of the mean scores.

    Args:
        scores_array: Array of mean scores.
        score_types: List of score types as strings.
        plot_path: Path to save the plots.
        n_red_herring_clues_evaluated_str: Number of red herring clues evaluated as a string.
        std_scores_array: Array of sample standard deviations of scores.
        single_model: Boolean indicating if the scores are from a single model.
        model: Name of the model or models as a string.
        n_puzzles: Number of puzzles evaluated.
        compare_mode: Mode of comparison, either "models" or "red_herrings".

    NOTE: Consider using subplots instead of saving separate figures for each score type.
    NOTE: Consider using i_not_evaluated_by_both instead of looking for -999 in the scores.
    """
    for score_type, score_type_array, std_score_type_array in zip(
        score_types, scores_array, std_scores_array
    ):
        # Set the figure size
        fig, ax = plt.subplots(figsize=(10, 8))

        # Fill untested cells with grey
        empty_cells = np.ones_like(score_type_array)
        empty_cells[score_type_array != -999] = 0
        ax.imshow(empty_cells, cmap="Greys", alpha=0.5)

        # Plot the mean scores
        score_type_array_not_empty = np.ma.masked_where(
            score_type_array == -999, score_type_array
        )
        image = ax.imshow(
            score_type_array_not_empty,
            cmap="Greens",
            aspect="equal",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        # Make a colorbar
        fig.colorbar(mappable=image, orientation="vertical", fraction=0.037, pad=0.04)

        # Set the title and labels
        title = choose_heatmap_title(
            single_model=single_model,
            score_type=score_type,
            n_red_herring_clues_evaluated_str=n_red_herring_clues_evaluated_str,
            model=model,
        )

        ax.set_title(title)
        ax.set_xlabel("# Attributes")
        ax.set_ylabel("# Objects")

        # Set the ticks and tick labels
        n_objects_max = score_type_array.shape[0]
        n_attributes_max = score_type_array.shape[1]
        ax.set_xticks(
            ticks=np.arange(n_attributes_max),
            labels=[str(i + 1) for i in range(n_attributes_max)],
        )

        ax.set_yticks(
            ticks=np.arange(n_objects_max),
            labels=[str(i + 2) for i in range(n_objects_max)],
        )

        # Annotate the cells with the mean scores (except for puzzle scores)
        ax = annotate_heatmap(
            ax=ax,
            data=score_type_array,
            std_data=std_score_type_array,
            single_model=single_model,
            score_type=score_type,
            n_x_max=n_attributes_max,
            n_y_max=n_objects_max,
        )

        # Adjust the layout
        fig.tight_layout()

        # Save the plot
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_filename = f"mean_{score_type}_{model}_{n_red_herring_clues_evaluated_str}rh_{n_puzzles}_puzzles.png"
        plot_filename = plot_filename.replace(" ", "_")
        plt.savefig(plot_path / plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def choose_heatmap_title(
    single_model: bool,
    score_type: str,
    n_red_herring_clues_evaluated_str: str,
    model: str,
) -> str:
    """Choose the title for the heatmap.

    Args:
        single_model: Boolean indicating if the scores are from a single model.
        score_type: Type of score as a string.
        n_red_herring_clues_evaluated_str: Number of red herring clues evaluated as a string.
        model: Name of the model or models as a string.

    Returns:
        title: Title for the heatmap.
    """
    if single_model:
        if not score_type == "puzzle score":
            title = f"{score_type.capitalize()}s with {n_red_herring_clues_evaluated_str} red herrings incl. sample std. dev. for model {model}"
        else:
            title = f"{score_type.capitalize()}s with {n_red_herring_clues_evaluated_str} red herrings for model {model}"
    else:
        title = f"Difference in mean {score_type} with {n_red_herring_clues_evaluated_str.replace('vs', '-')} red herrings for model {model.replace('vs', '-')} incl. std. error"
    return title


def annotate_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    std_data: np.ndarray,
    single_model: bool,
    score_type: str,
    n_x_max: int,
    n_y_max: int,
) -> plt.Axes:
    """Annotate the heatmap with the mean scores and standard deviations.

    Args:
        ax: Axes object to annotate.
        data: Array of mean scores.
        std_data: Array of sample standard deviations of scores.
        single_model: Boolean indicating if the scores are from a single model.
        score_type: Type of score as a string.
        n_x_max: Maximum number of attributes.
        n_y_max: Maximum number of objects.

    Returns:
        ax: Annotated Axes object.
    """
    for i in range(n_y_max):
        for j in range(n_x_max):
            if data[i, j] != -999:
                # If we are showing puzzle scores for a single evaluation, do not show the standard deviations, as the Bernoulli standard deviations can appear confusing
                if score_type == "puzzle score" and single_model:
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )
                else:
                    # Round to the correct number significant digits
                    score_rounded, std_rounded = round_using_std(
                        value=data[i, j], std=std_data[i, j]
                    )
                    ax.text(
                        j,
                        i,
                        f"{score_rounded} ± {std_rounded}",
                        ha="center",
                        va="center",
                        color="black",
                    )
    return ax


def plot_clue_type_frequencies(
    clue_type_frequencies_all_sizes: dict[str, dict[int, dict[str, int]]],
    clue_type_frequencies_all_sizes_normalised: dict[str, dict[int, dict[str, float]]],
    n_red_herring_clues_evaluated_str: str,
    data_folder: Path,
    theme: str,
    n_objects_max: int,
    n_attributes_max: int,
    n_puzzles: int,
) -> None:
    """Plot the frequencies of clue types.

    We plot the mean frequencies of clue types for each puzzle size after normalising the frequencies to sum to 1 in each puzzle.

    Args:
        clue_type_frequencies_all_sizes: Dictionary of dictionaries of dictionaries of clue type frequencies.
            The outer dictionary is for each puzzle size, the middle dictionary is for a puzzle index, and the inner dictionary is for each clue type.
        clue_type_frequencies_all_sizes_normalised: Dictionary of dictionaries of dictionaries of normalised clue type frequencies. The format matches clue_type_frequencies_all_sizes.
        n_red_herring_clues_evaluated_str: Number of red herring clues evaluated as a string.
        data_folder: Path to the data folder.
        theme: Theme name as a string.
        n_objects_max: Maximum number of objects in puzzles as an integer.
        n_attributes_max: Maximum number of attributes in puzzles as an integer.
        n_puzzles: The number of puzzles as an integer.
    """
    # Initialise the figure of n_objects_max_all_models x n_attributes_max_all_models subplots
    # TODO: Make plots wider
    fig, axs = plt.subplots(
        n_objects_max,
        n_attributes_max,
        figsize=(n_attributes_max * 3, n_objects_max * 3),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        f"Frequencies of clue types with {n_red_herring_clues_evaluated_str} red herrings for theme {theme}"
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # TODO: Sort n_objects in reverse and remove n_objects=1
    puzzle_sizes = clue_type_frequencies_all_sizes.keys()

    for puzzle_size in puzzle_sizes:
        # Get the number of objects and attributes from the puzzle size
        n_objects, n_attributes = map(int, puzzle_size.split("x"))

        # Get the subplot for this puzzle size
        ax = axs[n_objects - 1, n_attributes - 1]

        clue_type_frequencies_normalised_mean_one_size = get_clue_frequencies_per_puzzle_size(
            clue_type_frequencies_all_sizes_normalised=clue_type_frequencies_all_sizes_normalised,
            puzzle_size=puzzle_size,
            n_puzzles=n_puzzles,
        )

        # Create a bar plot for this puzzle size
        # TODO: Improve the plot layout and style
        ax.bar(
            clue_type_frequencies_normalised_mean_one_size.keys(),
            clue_type_frequencies_normalised_mean_one_size.values(),
        )
        ax.set_xlabel("Clue Type")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{n_objects}x{n_attributes}")
        ax.set_xticklabels(
            clue_type_frequencies_normalised_mean_one_size.keys(),
            rotation=45,
            ha="right",
        )
        ax.set_ylim(0, 0.4)
        ax.grid(axis="y")

    plt.tight_layout()

    # Save the plot
    plot_path = data_folder / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_filename = f"clue_type_frequencies_{n_red_herring_clues_evaluated_str}rh.png"
    plt.savefig(plot_path / plot_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_clue_frequencies_per_puzzle_size(
    clue_type_frequencies_all_sizes_normalised: dict[str, dict[int, dict[str, float]]],
    puzzle_size: str,
    n_puzzles: int,
) -> dict[str, float]:
    """Get the mean of the normalised frequencies of each clue type for puzzles of a specific size.

    Args:
        clue_type_frequencies_all_sizes_normalised: Dictionary of dictionaries of dictionaries of normalised clue type frequencies.
            The outer dictionary is for each puzzle size, the middle dictionary is for a puzzle index, and the inner dictionary is for each clue type.
        puzzle_size: String describing the puzzle size.
        n_puzzles: The number of puzzles for each puzzle size.

    Returns:
        A dictionary with clue types as keys and the mean of the normalised frequencies as the values.
    """
    # Take the mean of the clue type frequencies across all puzzles of this size
    clue_type_frequencies_normalised_mean_one_size: dict[str, float] = {}
    # TODO: Refactor a bit
    for (
        puzzle_index,
        clue_type_frequencies_normalised,
    ) in clue_type_frequencies_all_sizes_normalised[puzzle_size].items():
        # Sum the normalised frequencies of all puzzles of the chosen size
        for clue_type, freq_norm in clue_type_frequencies_normalised.items():
            if clue_type not in clue_type_frequencies_normalised_mean_one_size:
                clue_type_frequencies_normalised_mean_one_size[clue_type] = 0.0
            clue_type_frequencies_normalised_mean_one_size[clue_type] += freq_norm

    # Divide the sum for each clue type by the number of puzzles
    for (
        clue_type,
        freq_norm_sum,
    ) in clue_type_frequencies_normalised_mean_one_size.items():
        clue_type_frequencies_normalised_mean_one_size[clue_type] = (
            freq_norm_sum / float(n_puzzles)
        )

    sum_all_normalised_frequencies = sum(
        clue_type_frequencies_normalised_mean_one_size.values()
    )
    if abs(sum_all_normalised_frequencies - 1.0) > 0.0001:
        raise ValueError(
            f"The normalised frequencies do not sum to 1. They sum to {sum_all_normalised_frequencies}."
        )

    return clue_type_frequencies_normalised_mean_one_size
