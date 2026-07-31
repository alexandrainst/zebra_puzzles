# Reviewing a Language Config

This guide is for a native or fluent speaker who wants to review and improve an existing
language config for the zebra puzzle generator — most usefully one of the
"Preliminary" languages listed in the [README](README.md), since they have not
been checked already.

**Short version**: Generate some puzzles, read and edit the config, check that the text is correct, and submit a PR with your changes and any notes on remaining issues.

### Priorities:
1) Correctness. Text must be linguistically acceptable.
2) Unambiguity. Clues must represent a unique solution.
3) Naturalness. Phrases should sound typical of the chosen language.
4) Ease of generation. Puzzle generation should be simple, but if code changes are relevant, please note them in the PR.
5) Consistency. Text should preferably be consistent in meaning and form across languages.

## 1. Generate puzzle examples

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

Some "clues" are actually so-called **red herrings**, so you will see attributes mentioned that are not part of the solution. This is intended to confuse the solver which needs to filter out irrelevant information. The red herrings should never actually affect the solution according to the presented rules.

## 2. Find the config file

Each language/theme combination has one config file at:

```
config/language/<language code>/<theme name>.yaml
```

For example, French houses puzzles live in `config/language/fr/maisons.yaml`.


## 3. Understand the structure of the config file

The config file contains the vocabulary, grammar, and sentence templates used to generate the puzzles: attributes describing the objects of the puzzle theme (jobs, drinks, hobbies, etc.) and the sentence templates used to generate clues, plus some metadata about the language (like relevant grammatical cases), the puzzle templates, and a few global find-and-replace rules applied to the generated text.

In addition to the clue attributes, there are also "red herring" attributes included to confuse the solver. They should look like real attributes, but never cause true ambiguity in the puzzle solution.

Any attribute can be combined with any other attribute (or red herring attribute) in a clue, so this must be possible without grammatical errors. This is why we try to avoid explicitly gendered forms when possible.

The dictionary keys need to be translated for clue attributes, but this is not necessary for red herring attributes.

Some configs have comments explaining the reasoning behind certain choices. Feel free to delete or ignore them if they are not helpful.

Ask a maintainer if you have any questions about the config structure.

## 4. Edit the config file

Go through some puzzles and the config file to find issues, then edit the config directly and regenerate the puzzles to check your changes. Repeat until you're happy with the puzzles.

You can compare with other language configs to see how they handle similar issues. The meaning should stay almost the same across languages, but the exact phrasing or structure can vary. README.md has a list of already reviewed languages.

We try to keep language-specific issues in the config file itself. Most grammatical issues can be fixed by adding a new case, changing a template, or adding a prompt_replacement. If a code change is needed or would improve naturalness, please note it in the PR.

### Common issues

Most important:
- **Gender** *(if your language marks grammatical gender or includes gendered attribute phrases)*: In the houses
  theme, the person in each house has no stated gender. Check that it will
  combine correctly in sentences. A neutral form or a single gender used for
  everyone is fine - choose what would naturally describe an unknown person
  with any combination of the attributes. The "nurse" attribute is typically
  the most difficult one to combine with other attributes.
- **Case forms** *(if your language has case marking)*: For clues that use a
  preposition (like "next to", "just left of", "between"), check the
  noun/adjective after the preposition is in the grammatically correct case.
- **"Between N houses" clues**: When a clue says there are several houses
  between two people, check it's unambiguous *how many* — it should not be
  readable as a distance instead of a count. E.g. we use "there are 2 houses between" instead of "lives 2 houses away from" to avoid ambiguity.

Other issues:
- **Puzzle enjoyment**: The red herring fact about enjoying puzzles can result in e.g. "X knows that it is fun to solve puzzles", where "X enjoys solving puzzles" would be more natural. Add this to prompt_replacements if needed. See English (en) for an example.
- **Format description**: Would "JSON dictionary", "key" and "value" typically be translated or kept in English? Edit the prompt template accordingly.
- **Number agreement** *(if your language inflects nouns by the preceding
  number, common in Slavic languages)*: Check that the form used in "between
  N houses" clues matches the specific number shown. We will typically use 4
  houses, this should at least work for up to N=2.
- **Multiple clue template versions** *(if needed to cover all the attributes)*: See
  e.g. Irish (ga) "same_object_templates", or Hindi (hi)
  "same_herring_templates". See Basque (eu) "attribute_subject_cases" if the
  case depends on the attribute category. Or just note the issue in the PR.
- **Word order of relative clauses** *(if your language orders relative
  clauses differently from main clauses)*: We need extra templates in
  red_herring_facts. See German (de) for an example.
- **Contractions** *(if your language has contractions, e.g. von dem -> vom)*:
  Add the relevant ones to prompt_replacements.

### Naturalness

- Avoid overly literal translations from English or stilted phrasing. If a native word exists, it is usually better than a loanword. It is ok to split, combine or change the order of sentences.
- Use short phrases, e.g. "The cat owner" instead of "The person who owns a cat", to avoid repeatedly using the same long phrase in every clue.
- Category order in the config sets the preferred order of categories in generated clues. Categories with longer or less natural nominative phrases are best placed later in the list, so they're less likely to open a clue — e.g. we prefer "The nurse loves oranges" over "The person who loves oranges is a nurse", so job comes before favourite fruit.

### Ambiguity

- Could any attribute be confused with one in a different category? Hobbies should be phrased as activities, not job titles to avoid confusion with the job category.
- Could any red herring accidentally be interpreted as affecting the solution due to confusion with a real attribute?
- Are the rules clear in the prompt template?

### Punctuation

- Check spacing and punctuation look right — commas, contracted articles, spacing around question marks or dashes, and so on. This can be corrected with the variables used in e.g. Japanese (ja) or with prompt_replacements.
- If commas are used around relative clauses, you can include both commas in the attributes and remove any commas placed before full stops by adding a prompt_replacement. See German (de) for an example.

## 5. Edit README.md

If the puzzles look correct, you can now edit the README to mark the language as "Finished" instead of "Preliminary". 🎉

## 6. Submit your changes

Open a pull request with your config changes — see the
[contributing guide](CONTRIBUTING.md) for the general process. In the PR
description, mention that this is a native-speaker review and summarize what
you checked and changed, plus anything you're unsure about.

Thank you for improving the zebra puzzle generator!
