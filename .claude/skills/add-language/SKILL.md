---
name: add-language
description: Add a new language and theme to the zebra puzzle generator. Creates the YAML config, updates README and config.yaml, validates the config, generates sample puzzles, and asks for a grammar review.
context:
  - config/language/en/houses.yaml
---

From the conversation, identify:
- `language_name`: human-readable name (e.g. "French")
- `lang_code`: ISO 639-1 two-letter code (e.g. "fr")
- `theme_name`: the theme filename without extension (e.g. "maisons")
- `config_path`: `config/language/<lang_code>/<theme_name>.yaml`

## Steps

### 1. Choose a template config
`config/language/en/houses.yaml` is already loaded as a structural reference showing all required keys. For languages with no grammatical case inflection (e.g. French, Italian, Spanish), it is also the right content template.

For languages that inflect attributes by case, additionally read the matching template:

| Case system | Template |
|---|---|
| Dative only (e.g. German) | `config/language/de/Hauser.yaml` |
| Accusative + dative (e.g. Faroese) | `config/language/fo/hus.yaml` |
| Accusative + dative + genitive (e.g. Icelandic) | `config/language/is/husum.yaml` |
| Genitive only (e.g. Finnish) | `config/language/fi/talot.yaml` |

Before writing the config, briefly scan whether the language has any feature that clearly cannot be expressed by translating the existing clue templates and attribute forms — for example, a radically different word order, a grammatical category with no analog in existing configs, or a number agreement rule that affects clue templates. Note anything flagged here and address it at step 4.

### 2. Create the config file
Create `config/language/<lang_code>/` directory if needed, then write `<theme_name>.yaml`.

**Before writing any attribute forms, make three decisions:**

1. **Gender default**: If the language marks grammatical gender, decide the default now. Write it as a YAML comment at the top of the file. Apply it to ALL nom/is/is_not forms for both regular attributes AND red herring attributes — they can appear in the same puzzle describing the same person of unknown gender. Semantic gender neutrality is preferred. In the houses theme, attributes can describe both men and women, but their genders are never mentioned.

2. **Case requirements**: Look at the clue templates you will translate and identify which prepositions or postpositions are used for positional clues (`next_to`, `just_left_of`, `between`). The case these prepositions govern determines whether you need extra cases beyond `[nom, is, is_not]`. If all positional prepositions take nominative (as in Hungarian), no extra cases are needed.

3. **Prompt replacements**: Think ahead — will relative clauses leave trailing commas before periods? Will articles contract? Note these now and add to `prompt_replacements` as you encounter them while writing the config. If you are considering adding a lot of replacements, it may be better to edit the code to handle the grammar.
   - Romance languages typically contract `de + article`: French `de le → du`, Portuguese `de o → do`, Spanish `de el → del`, Italian `di il → del`. Since nom forms follow positional prepositions in clue templates (e.g. `à esquerda de {attribute_desc}`), these contractions arise naturally. Add the relevant `"de o ": "do "` style rule upfront.

**Key clue template grammar** (refer to this when writing `is` forms and clue templates):
- `same_object`: `{attribute_desc_1} {attribute_desc_2}.` — desc_1 is the subject (nom), desc_2 is a bare predicate (the `is` form used directly). For languages that omit the copula (e.g. Hungarian 3rd-person present), this works as-is.
- `friends`: `{attribute_desc} {attribute_desc_herring}.` (or your translation) — BOTH descriptions must function as co-subjects. Use nominative for both. If the language would require a different case for the second subject, restructure the template to use "X and Y are friends" with explicit conjunction instead.
- All positional clues (`found_at`, `next_to`, `just_left_of`, `between`): all desc forms are subjects (nom).

Required top-level keys (in this order):
1. Header comment: `# Config file for generating zebra puzzles in <LanguageName> with the <theme> theme.`
2. `theme: <lang_code>_<theme_name>`
3. `attribute_cases: [nom, is, is_not, ...]` — list all grammatical case forms used, in the order they appear in each attribute's description list
4. `red_herring_attribute_cases: [nom, is, ...]` — same but for red herring attributes (no `is_not` form)
5. `attributes:` — nested dict: category → value → list of description strings
6. `red_herring_attributes:` — dict: key → list of description strings
7. `red_herring_facts:` — dict: key → list of 1–2 description strings
8. `clues_dict:` — clue type → template string (must include all clue types from the template)
9. `clue_cases_dict:` — clue type → list of case names (must only use cases from `attribute_cases` plus `none`)
10. `red_herring_clues_dict:` — red herring clue type → template string
11. `red_herring_cases_dict:` — red herring clue type → list of case names (must only use cases from `red_herring_attribute_cases` plus `none`)
12. `prompt_templates:` — list of prompt section strings
13. `prompt_and:` — word for "and" in lists (e.g. "and", "und", "et")
14. `prompt_replacements:` — dict of literal string substitutions applied to the entire generated prompt after all templates are filled in. Use it to fix awkward phrasings that arise when attribute descriptions combine with clue or fact templates in unexpected ways. Example in English: `knows that it is fun to solve: enjoys solving`. Can be empty (`{}`) if no fixups are needed. If the language uses commas after relative clauses, you can add a comma at the end of some attribute descriptions and then remove it at the end of sentences with a replacement rule, e.g. `',.': .`. You can also use this for contractions.

The meaning should be consistent across languages, unless this would compromise grammar, unambiguity or make puzzles too complicated to generate.

### 3. Validate the config before generating puzzles

Check every item below and fix any problems found:

**Config validation**
Run the validation tool — it checks list lengths, unknown case references, and required `is`/`is_not` entries:
```bash
uv run .claude/skills/add-language/validate_config.py language=<lang_code>/<theme_name>
```

**"is" form of red herring attributes**
The `same_herring` and `double_herring` templates use the red herring's `is` form as a direct predicate after a nominative subject:
```
{attribute_desc} {attribute_desc_herring}.
```
So `is` must be a full predicate phrase that makes sense after a nominative subject — NOT just a bare noun.

❌ Wrong: `'on polkupyörä'` → gives "X on polkupyörä" = "X is a bicycle"
✓ Right: `'omistaa polkupyörän'` → gives "X omistaa polkupyörän" = "X owns a bicycle"
✓ Right: `'porte des lunettes'` → gives "X porte des lunettes" = "X wears glasses"

Check every red herring attribute's `is` form. If the form is just a noun or a "has/is + noun" copular phrase that doesn't work standalone, fix it to a verb phrase.

Note: the `cactus` red herring intentionally uses a *negative* `is` form (e.g. "does not own a cactus") as a deliberate puzzle trick. This is the only exception — do not replicate it for other attributes.

**`multiple_between` vs `one_between` ambiguity**
For `multiple_between` the puzzle description says there are N houses between two people (N > 1). The template must be unambiguous — the solver must be able to tell how many houses are between them, not just that it's more than one.

Bad (ambiguous): "X ja Y asuvat N talon päässä toisistaan" (could mean distance N, i.e. N−1 houses between)
Good (unambiguous): "X ja Y välissä on N taloa" (explicitly N houses between)

**`none` in cases dicts**
`none` is a sentinel: in a positive clue it resolves to the `is` form; in a negative clue it resolves to the `is_not` form. Use `none` in `clue_cases_dict` wherever the clue template uses the predicate form of an attribute.

**Unambiguous templates**
- `prompt_templates` must be unambiguous. It must be clear that each object has exactly one value from each category and that each value is assigned to exactly one object. E.g. "In each house lives a person with a unique attribute in each of the following categories:".

The technical terms "JSON dictionary", "key" and "value" should not be translated in most languages, but they can be formatted differently if that would look natural in the language when discussing code.

**Unambiguous attributes**
Each attribute must be unambiguous. It should not be easy to confuse one attribute with another or a red herring attribute. In particular, check red herring attributes against occupation names in the target language — a word that means both a red herring concept and an occupation (e.g. Hungarian "nővér" = both "sister" and "nurse") should be replaced. Warn the user if avoiding ambiguity requires changing the meaning.

### 4. Check if code changes are needed

Code changes are needed when:
- Number agreement rules require special handling beyond what templates support (e.g. Finnish partitive after numbers 2+, which was solved via the template directly)
- Major changes are required to clue templates to make them unambiguous or grammatically correct in the new language. For example if word order in the new language is very different and cannot be handled by existing templates or a few prompt_replacements.

If changes are needed, explain what they are before implementing.

Run tests with `make test` to verify that any code changes are correct and don't break existing languages.

### 5. Run make check

Check formatting by running `make check`. Fix any issues found.

### 6. Generate puzzles
Run the following to generate 3 puzzles with 4 objects, 5 attributes and 5 red herrings:

```bash
uv run src/scripts/build_dataset.py \
  language=<lang_code>/<theme_name> \
  n_objects=4 \
  n_attributes=5 \
  n_puzzles=3 \
  n_red_herring_clues=5
```

If the build fails with a `ValueError`, read the message — it will point to the exact config key and entry that is wrong (wrong list length, unknown case name, etc.).

### 7. Show and review the puzzles
Read and display all three generated puzzles:
- `data/<lang_code>_<theme_name>/4x5/5rh/puzzles/zebra_puzzle_0.txt`
- `data/<lang_code>_<theme_name>/4x5/5rh/puzzles/zebra_puzzle_1.txt`
- `data/<lang_code>_<theme_name>/4x5/5rh/puzzles/zebra_puzzle_2.txt`

Checking multiple puzzles increases the chance of seeing rare clue types (such as `multiple_between`) that may not appear in puzzle 0.

First self-review:
1. Does the grammar look correct?
2. Does it sound natural?
3. Is the puzzle unambiguous and looks solvable?
4. Is the punctuation correct?

Make any corrections to the config, delete the generated puzzles for this language, and re-run puzzle generation to verify the fixes.

### 8. Update README.md
Add the new language to the language/theme list under the relevant theme. Use the same format as existing entries:
```
- <Theme> theme:
    - Preliminary versions: ... and <LanguageName> 🏳️.
```

### 9. Update config/config.yaml
Add `<lang_code>/<theme_name>` to the comment block listing all valid language/theme combinations near the top of the file.

### 10. Consider improving this skill
If you found any part of this process confusing or error-prone, suggest improvements to this skill. Did you have to make decisions or look up information that could be included in the skill? Did you have to read the code to understand how it works? If so, consider adding that information to this skill.

### 11. User-review

Show the first puzzle to the user and ask if the puzzle looks correct.
