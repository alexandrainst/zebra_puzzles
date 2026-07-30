# Reviewing a Language Config

This guide is for a native speaker who wants to review and improve an existing
language config for the zebra puzzle generator — most usefully one of the
"Preliminary" languages listed in the [README](README.md), since they have not
been checked already.

Short version: Read and edit the config, generate some puzzles, check that the text is correct, and submit a PR with your changes.

## Priorities:
1) Correctness. Text must be linguistically acceptable.
2) Unambiguity. Clues must represent a unique solution.
3) Naturalness. Phrases should sound typical of the chosen language.
4) Ease of generation. Puzzle generation should be simple, but if code changes are relevant, please note them in the PR.
5) Consistency. Text should be consistent in meaning
and form across languages.


## 1. Find the config file

Each language/theme combination has one config file at:

```
config/language/<language code>/<theme name>.yaml
```

For example, French houses puzzles live in `config/language/fr/maisons.yaml`.

## 2. Understand the structure of the config file

The config file contains all the vocabulary, grammar, and sentence templates used to generate the puzzles. You will be checking that the config is correct, natural, and unambiguous in the target language. If you find any issues, please edit the config file directly and regenerate the puzzles to check your changes (see step 2 below).

Most of the config consists of attributes describing the objects of the puzzle theme (jobs, drinks, hobbies, etc.) and the sentence templates used to generate clues. The config also contains some metadata about the language (like its grammatical cases), the puzzle templates and a few global find-and-replace rules applied to the generated text.

In addition to the clue attributes, there are also "red herring" attributes — these are extra attributes that are not actually part of the puzzle solution, but are included to confuse the solver. They should never be ambiguous with any of the real attributes.

Any attribute can be combined with any other attribute in a clue, so this must be possible without
grammatical errors. This is why we try to avoid explicitly gendered forms when possible.

The dictionary keys need to be translated for clue attributes, but this is not necessary for red herring attributes.

Some configs have comments explaining the reasoning behind certain choices. Feel free to delete or ignore them if you think they are irrelevant.

Ask a maintainer if you have any questions about the config structure or how to edit it.

## 2. Generate puzzles to read

From the repository root, run:

```bash
uv run src/scripts/build_dataset.py \
  language=<language code>/<theme name> \
  n_objects=4 \
  n_attributes=5 \
  n_puzzles=3 \
  n_red_herring_clues=5
```

This writes three sample puzzles to:

```
data/<language code>_<theme name>/4x5/5rh/puzzles/zebra_puzzle_0.txt
data/<language code>_<theme name>/4x5/5rh/puzzles/zebra_puzzle_1.txt
data/<language code>_<theme name>/4x5/5rh/puzzles/zebra_puzzle_2.txt
```

We generate 4x5 puzzles with 5 red herring clues so there are enough clues to check the config and to represent clue types that are excluded from small puzzles.

Some "clues" are actually so-called red herrings, so you will see attributes mentioned that are not part of the solution. This is intended to confuse the solver which needs to filter out irrelevant information. The red herrings should never actually affect the solution or make it ambiguous according to the rules.

## 3. Edit the config file

Go through some puzzles and the config file itself to identify any issues and fix them.

If you find a problem, edit the config file directly and regenerate the puzzles to check your changes. Repeat until you're happy with the puzzles.

We try to keep language-specific issues in the config file itself, and most grammatical issues can be fixed by adding a new case, changing a template or adding a prompt_replacement. If a code change is needed or would improve naturalness, please note it in the PR.

You can compare with other language configs to see how they handle similar issues. The meaning should stay almost the same across languages, but the exact phrasing or structure can vary. README.md has a list of already reviewed languages.

### Common issues

- **Gender**: In the houses theme, the person in each house has no stated
  gender. If your language marks grammatical gender, check that they will combine correctly in sentences. A neutral form or a single
  gender used for everyone is fine - choose what would naturally describe an unknown person with any combination of the attributes.
- **Case forms**: For clues that use a preposition (like "next to", "just left
  of", "between"), check the noun/adjective after the preposition is in the
  grammatically correct case for languages that have case marking.
- **"Between N houses" clues**: When a clue says there are several houses
  between two people, check it's unambiguous *how many* — it should not be
  readable as a distance instead of a count.
- **Format description**: Would "JSON dictionary", "key" and "value" typically be translated or kept in English? Edit the prompt template accordingly.
- **Number agreement**: If your language changes a noun's form depending on
  the number in front of it (common in Slavic languages, for example), check
  that the form used in "between N houses" clues matches the specific number
  shown. We will typically use 4 houses, this should at least work for up to N=2.

### Naturalness

Avoid overly literal translations from English or stilted phrasing.

Try to use short phrases e.g. "The cat owner" instead of "The person who owns a cat" to avoid repeatedly using the same long phrase in every clue.

The order of categories in the config file determines the preferred order of categories in the generated clues. If a category generally uses longer, it is likely best to put it later in the list so that it is less likely to be used as the first attribute in a clue. For example, we prefer "The nurse drinks tea" over "The person who drinks tea is a nurse", so the job category should be listed before the drink category.

### Ambiguity

- Could any attribute be confused with one in a different category? Hobbies should be phrased as activities, not job titles to avoid confusion with the job category.
- Could any red herring accidentally be interpreted as affecting the solution due to confusion with a real attribute?
- Are the rules clear in the prompt template?

### Punctuation

Check spacing and punctuation look right — trailing commas before periods,
contracted articles, spacing around question marks or dashes, and so on.

## 5. Edit README.md

If you have reviewed the full config, you can now edit the README to mark the language as "Finished" instead of "Preliminary". 🎉

## 6. Submit your changes

Open a pull request with your config changes — see the
[contributing guide](CONTRIBUTING.md) for the general process. In the PR
description, mention that this is a native-speaker review and summarize what
you checked and changed, plus anything you're unsure about.

Thank you for improving the zebra puzzle generator!
