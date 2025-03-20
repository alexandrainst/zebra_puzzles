<a href="https://github.com/alexandrainst/zebra_puzzles">
<img
    src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
	width="239"
	height="175"
	align="right"
/>
</a>

# Zebra Puzzles

Generation and LLM evaluation of zebra puzzles in multiple languages and themes.

Run `uv run src/scripts/build_dataset.py` to generate puzzles.
Run `uv run src/scripts/evaluate.py` to evaluate puzzles.

Use the configuration in `config/config.yaml` to specify:
- number of puzzles to generate
- puzzle dimensions
- language and theme
- number of red herrings to include
- model for evaluation (e.g. gpt-4o-mini, gpt-4o, o3-mini, o3)

Puzzles and their solutions are saved in the data folder, LLM reponses are saved in the reponse folder and LLM scores are saved in the scores folder.

Typical runtimes for generating a puzzle of size n_objects x n_attributes are (using all clue types):
- 3x7: 0.7 s
- 4x4: 0.6 s
- 4x5: 13 s
- 4x6: 3 min
- 5x3: 3.8 s
- 5x6: >10 min
- 6x3: 4 min

Typical times for evaluation of a puzzle:

gpt-4o-mini:
-

o3:
- 4x5: 8 min

GitHub Copilot has been used for this project.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/zebra_puzzles/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/zebra_puzzles)
[![License](https://img.shields.io/github/license/alexandrainst/zebra_puzzles)](https://github.com/alexandrainst/zebra_puzzles/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/zebra_puzzles)](https://github.com/alexandrainst/zebra_puzzles/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/zebra_puzzles/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Sofie Helene Bruun (sofie.bruun@alexandra.dk)


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
