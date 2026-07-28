"""Script for creating and uploading zebra puzzles to Hugging Face.

This script generates train (128), validation (128), and test (1024) puzzles
for puzzles of size 2x3 and 4x5, and then formats and uploads them to Hugging
Face, for one or more languages/themes.

Usage:
    uv run src/scripts/create_and_upload_dataset.py <language>/<theme> [<language>/<theme> ...]

Examples:
    uv run src/scripts/create_and_upload_dataset.py en/houses
    uv run src/scripts/create_and_upload_dataset.py en/houses da/smoerrebroed
"""

import subprocess
import sys

SIZES = [(2, 3), (4, 5)]
DATA_SPLITS = [("data_train", 128), ("data_val", 128), ("data_test", 1024)]


def main() -> None:
    """Generate, format and upload train/val/test datasets for one or more languages/themes."""
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: uv run src/scripts/create_and_upload_dataset.py "
            "<language>/<theme> [<language>/<theme> ...]"
        )
    language_themes = sys.argv[1:]

    auto_confirm = (
        input(
            "Automatically overwrite existing datasets and publish to Hugging"
            " Face Hub without asking? (y/n): "
        )
        .strip()
        .lower()
        == "y"
    )

    for language_theme in language_themes:
        for n_objects, n_attributes in SIZES:
            for data_folder, n_puzzles in DATA_SPLITS:
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "src/scripts/build_dataset.py",
                        f"language={language_theme}",
                        f"data_folder={data_folder}",
                        f"n_puzzles={n_puzzles}",
                        f"n_objects={n_objects}",
                        f"n_attributes={n_attributes}",
                        f"auto_confirm={auto_confirm}",
                    ],
                    check=True,
                )

            subprocess.run(
                [
                    "uv",
                    "run",
                    "src/scripts/format_datasets.py",
                    f"language={language_theme}",
                    f"n_objects={n_objects}",
                    f"n_attributes={n_attributes}",
                    f"auto_confirm={auto_confirm}",
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
