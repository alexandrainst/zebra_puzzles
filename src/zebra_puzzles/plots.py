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
