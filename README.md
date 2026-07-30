<a href="https://github.com/alexandrainst/zebra_puzzles">
<img
    src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
	width="239"
	height="175"
	align="right"
/>
</a>

# MultiZebraLogic

Generation and LLM evaluation of zebra puzzles in multiple languages and themes including red herrings.

Available languages and themes:

- Houses theme (44 languages - 39 European, 5 _non-European_):
    - Finished versions: Danish (da) 🇩🇰, Dutch (nl) 🇳🇱, English (en) 🇬🇧, Faroese (fo) 🇫🇴, German (de) 🇩🇪, Icelandic (is) 🇮🇸, Norwegian Bokmål (nb) 🇳🇴, Norwegian Nynorsk (nn) 🇳🇴 and Swedish (sv) 🇸🇪.
    - Preliminary versions: Albanian (sq) 🇦🇱, _Arabic (ar)_ 🇸🇦, Basque (eu) 🏴, Belarusian (be) 🇧🇾, Bosnian (bs) 🇧🇦, Bulgarian (bg) 🇧🇬, Catalan (ca) 🏴󠁥󠁳󠁣󠁴󠁿, _Chinese (zh)_ 🇨🇳, Croatian (hr) 🇭🇷, Czech (cs) 🇨🇿, Estonian (et) 🇪🇪, Finnish (fi) 🇫🇮, French (fr) 🇫🇷, Greek (el) 🇬🇷, _Hindi (hi)_ 🇮🇳, Hungarian (hu) 🇭🇺, Irish (ga) 🇮🇪, Italian (it) 🇮🇹, _Japanese (ja_) 🇯🇵, Latvian (lv) 🇱🇻, Lithuanian (lt) 🇱🇹, Luxembourgish (lb) 🇱🇺, Macedonian (mk) 🇲🇰, _Marathi (mr)_ 🇮🇳, Polish (pl) 🇵🇱, Portuguese (pt) 🇵🇹, Romanian (ro) 🇷🇴, Russian (ru) 🇷🇺, Scots (sco) 🏴󠁧󠁢󠁳󠁣󠁴󠁿, Serbian (sr) 🇷🇸, Slovak (sk) 🇸🇰, Slovenian (sl) 🇸🇮, Spanish (es) 🇪🇸, Ukrainian (uk) 🇺🇦 and West Frisian (fy) 🇳🇱.
- Smørrebrød theme:
    - Finished versions: Danish (da) 🇩🇰.

Dataset on the Hugging Face Hub: https://huggingface.co/datasets/alexandrainst/zebra_puzzles

Paper: [ArXiv preprint](https://arxiv.org/abs/2511.03553)

Contributions are welcome! Please read the [contributing guide](CONTRIBUTING.md) before submitting a PR. There are many ways to contribute, including opening issues for linguistic errors, adding new languages or themes, and adding new clue types.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-83%25-yellowgreen.svg)](https://github.com/alexandrainst/zebra_puzzles/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/zebra_puzzles)
[![License](https://img.shields.io/github/license/alexandrainst/zebra_puzzles)](https://github.com/alexandrainst/zebra_puzzles/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/zebra_puzzles)](https://github.com/alexandrainst/zebra_puzzles/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/zebra_puzzles/blob/main/CODE_OF_CONDUCT.md)

Developers:

- Sofie Helene Bruun (sofie.bruun@alexandra.dk)
- Dan Saattrup Smart (dan.smart@alexandra.dk)

## Usage

Run `uv run src/scripts/build_dataset.py` to generate puzzles.

Run `uv run src/scripts/evaluate.py` to evaluate puzzles.

Run `uv run src/scripts/plot_performance.py` to plot and compare puzzle evaluation performance.

Run `uv run src/scripts/fix_files.py` to combine datasets. Use the script to edit many filenames at once and/or move files to another folder.

Run `uv run src/scripts/format_datasets.py` to format and push a dataset to Hugging Face.

Run `uv run src/scripts/create_and_upload_dataset.py <language/theme>` to generate full train, validation and test datasets for a theme and optionally push to Hugging Face after user confirmation. This will generate 128 puzzles for training, 128 puzzles for validation, and 1024 puzzles for testing for sizes 2x3 and 4x5. It is possible to specify multiple languages and themes, e.g. `uv run src/scripts/create_and_upload_dataset.py en/houses da/smoerrebroed de/Hauser`.

Use the configuration in `config/config.yaml` to specify:
- language and theme of puzzles
- model for evaluation (e.g. gpt-4o-mini, gpt-4o, o3-mini, o3)
- whether to generate new LLM responses
- data folders
- number of puzzles to generate
- puzzle dimensions
- clue type weights
- number of red herrings to include

The chosen main data folder contains puzzles, their solutions, LLM responses, chosen clue types and the indices to red herring clues in each puzzle. LLM scores are saved in the 'scores' subfolder. Plots and cross-model comparisons are saved in the 'plots' subfolder.

Puzzles can be evaluated using fewer red herrings than they were generated with. This allows for measuring the impact of red herrings. If the number of red herrings is reduced, the new version of the puzzle is saved in a 'reduced_puzzles' folder, and the clue types are saved in a 'reduced_clue_types' folder.

## Example

The following is an example of a 2x3 puzzle with 5 red herrings. The theme is houses and the language is English.
```
A row of houses have numbers 1 to 2 from left to right.

In each house lives a person with a unique attribute in each of the following categories:

Jobs: baker and police officer.
Drinks: milk and tea.
Favourite book genres: fantasy and science fiction.

We also know the following:

1. The milk drinker lives in house no. 1.
2. The person who plays video games does not live in house no. 1.
3. The person with glasses lives in house no. 1.
4. The police officer lives to the left of the fantasy reader.
5. The tea drinker lives next to the person who plays the guitar.
6. The milk drinker is good friends with the person who watches ski jumping.
7. The person with a guinea pig lives in house no. 1.

Who has which attributes and lives in which house?

Please submit your answer as a JSON dictionary. Each key must be object_X where X is the house number. Each value must be a list of the attributes from the aforementioned categories that belong to the person in house no. X.

The following is an example of the answer format:

{
    "object_1": [
        "jobs_1",
        "drinks_1",
        "favourite_book_genres_1"
    ],
    "object_2": [
        "jobs_2",
        "drinks_2",
        "favourite_book_genres_2"
    ]
}
```

## Typical runtimes

Typical runtimes for generating a puzzle of size n_objects x n_attributes are (using all clue types):
- 2x3: 0.002 s
- 4x5: 0.01 s
- 4x6: 0.02 s
- 5x5: 0.5 s
- 6x6: 1.3 s

Increase n_expected_clues to reduce runtime if needed.

Typical times for evaluation of a puzzle without red herrings:

gpt-4o-mini:
- 3x3: 1.5 s
- 4x4: 2 s
- 4x5: 2 s

o3-mini:
- 2x2: 6 s
- 3x3: 25 s  (35 s with 5 red herrings)
- 4x4: 2 min
- 4x5: 8 min


## Adding a new language or theme

If you are using an agent, run `/add-language` in the project to use the guided skill, which handles config creation, validation, puzzle generation and review.

To add a new language or theme manually:

1. For a new language, create a folder in `config/language`.
2. Copy an existing config file such as `config/language/en/houses.yaml`.
3. Translate/replace words and phrases to fit your language/theme.
    - No attributes, categories or clues should have identical keys.
    - Please make sure the templates and clue types are unambiguous and that no attributes can be confused with the red herring attributes.
    - Attribute versions should be presented in the following order:
        1. Nominative
        2. Phrase connecting it to the subject
        3. Phrase disconnecting it from the subject
        4. Accusative
        5. Dative
        6. Genitive
        ...

        Only the first 3 are mandatory. See Icelandic (is) for an example of using all 6 versions.
        For red herring attributes, we skip the disconnecting phrase.

        Map the order of cases using `attribute_cases` and `red_herring_attribute_cases`.

    - For new themes, the number of attributes, red herring attributes and red herring facts can be changed without adapting other files.
    - For translations of existing themes, please prioritize keeping the meaning consistent unless this would sound unnatural or be difficult to implement.
    - If you do not wish to include all red herring or clue types, remember to change the settings in `config/config.yaml` during puzzle generation.
    - If a specific combination of words should be replaced, add it to prompt_replacements. E.g. `von dem: vom` in German.
    - If the language uses commas around relative clauses, remember to add `',.': .` to prompt_replacements.
4. All language- or theme-specific settings should be included in the config file, but if necessary, grammatical rules can be adapted in `src/zebra_puzzles/puzzle_creation/clue_selection.py` and `src/zebra_puzzles/puzzle_creation/red_herring_selection.py`. Please make sure the code will still run as expected for other languages and themes, and please try to make any new rules as general as possible.
5. Edit `README.md` to mention the new language/theme.
6. Generate some puzzles to test the language/theme. We recommend using large puzzles, so all clue types are applicable. Set the new theme with `language` in `config/config.yaml` and run e.g. `uv run src/scripts/build_dataset.py n_objects=4 n_attributes=5 n_red_herring_clues=5 n_puzzles=10`.

Read the skill at `.claude/skills/add-language/SKILL.md` for details on common translation issues.

## Setup

### Installation

1. Run `make install`, which sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. (Optional) Run `make install-pre-commit`, which installs pre-commit hooks for linting, formatting and type checking.


### Adding and Removing Packages

To install new PyPI packages, run:
```
uv add <package-name>
```

To remove them again, run:
```
uv remove <package-name>
```

To show all installed packages, run:
```
uv pip list
```


## All Built-in Commands

The project includes the following convenience commands:

- `make install`: Install the project and its dependencies in a virtual environment.
- `make install-pre-commit`: Install pre-commit hooks for linting, formatting and type checking.
- `make lint`: Lint the code using `ruff`.
- `make format`: Format the code using `ruff`.
- `make type-check`: Type check the code using `mypy`.
- `make test`: Run tests using `pytest` and update the coverage badge in the readme.
- `make docker`: Build a Docker image and run the Docker container.
- `make docs`: View documentation locally in a browser.
- `make publish-docs`: Publish documentation to GitHub Pages.
- `make tree`: Show the project structure as a tree.


## A Word on Modules and Scripts
In the `src` directory there are two subdirectories, `zebra_puzzles`
and `scripts`. This is a brief explanation of the differences between the two.

### Modules
All Python files in the `zebra_puzzles` directory are _modules_
internal to the project package. Examples here could be a general data loading script,
a definition of a model, or a training function. Think of modules as all the building
blocks of a project.

When a module is importing functions/classes from other modules we use the _relative
import_ notation - here's an example:

```
from .other_module import some_function
```

### Scripts
Python files in the `scripts` folder are scripts, which are short code snippets that
are _external_ to the project package, and which is meant to actually run the code. As
such, _only_ scripts will be called from the terminal. An analogy here is that the
internal `numpy` code are all modules, but the Python code you write where you import
some `numpy` functions and actually run them, that a script.

When importing module functions/classes when you're in a script, you do it like you
would normally import from any other package:

```
from zebra_puzzles import some_function
```

Note that this is also how we import functions/classes in tests, since each test Python
file is also a Python script, rather than a module.


## Features

### Docker Setup

A Dockerfile is included in the new repositories, which by default runs
`src/scripts/main.py`. You can build the Docker image and run the Docker container by
running `make docker`.

### Automatic Documentation

Run `make docs` to create the documentation in the `docs` folder, which is based on
your docstrings in your code. You can publish this documentation to Github Pages by
running `make publish-docs`. To add more manual documentation pages, simply add more
Markdown files to the `docs` directory; this will automatically be included in the
documentation.

### Automatic Test Coverage Calculation

Run `make test` to test your code, which also updates the "coverage badge" in the
README, showing you how much of your code base that is currently being tested.

### Continuous Integration

Github CI pipelines are included in the repo, running all the tests in the `tests`
directory, as well as building online documentation, if Github Pages has been enabled
for the repository (can be enabled on Github in the repository settings).

### Code Spaces

Code Spaces is a new feature on Github, that allows you to develop on a project
completely in the cloud, without having to do any local setup at all. This repo comes
included with a configuration file for running code spaces on Github. When hosted on
`alexandrainst/zebra_puzzles` then simply press the `<> Code` button
and add a code space to get started, which will open a VSCode window directly in your
browser.
