---
name: add-language
description: Add a new language and theme to the zebra puzzle generator. Creates the YAML config, updates README and config.yaml, validates the config, generates sample puzzles, and asks for a grammar review.
---

From the conversation, identify:
- `language_name`: human-readable name (e.g. "French")
- `lang_code`: ISO 639-1 two-letter code (e.g. "fr")
- `theme_name`: the theme filename without extension (e.g. "maisons")
- `config_path`: `config/language/<lang_code>/<theme_name>.yaml`

## Steps

### 1. Choose a template config
Pick the existing config that best matches the target language's case system:

| Case system | Template |
|---|---|
| No inflection (e.g. French, Italian, Spanish) | `config/language/en/houses.yaml` |
| Dative only (e.g. German) | `config/language/de/Hauser.yaml` |
| Accusative + dative (e.g. Faroese) | `config/language/fo/hus.yaml` |
| Accusative + dative + genitive (e.g. Icelandic) | `config/language/is/husum.yaml` |
| Genitive only (e.g. Finnish) | `config/language/fi/talot.yaml` |

Read the chosen template in full so you know the complete required structure.

### 2. Create the config file
Create `config/language/<lang_code>/` directory if needed, then write `<theme_name>.yaml`.

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
14. `prompt_replacements:` — dict of string substitutions applied to the final prompt

The meaning should be consistent across languages, unless this would compromise grammar, unambiguity or make puzzles too complicated to generate.

### 3. Validate the config before generating puzzles

Check every item below and fix any problems found:

**Config validation**
- You can use `tests/validate_config.py` to check automatically.

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

**`multiple_between` vs `one_between` ambiguity**
For `multiple_between` the puzzle description says there are N houses between two people (N > 1). The template must be unambiguous — the solver must be able to tell how many houses are between them, not just that it's more than one.

Bad (ambiguous): "X ja Y asuvat N talon päässä toisistaan" (could mean distance N, i.e. N−1 houses between)
Good (unambiguous): "X ja Y välissä on N taloa" (explicitly N houses between)

**`none` in cases dicts**
`none` is a sentinel: in a positive clue it resolves to the `is` form; in a negative clue it resolves to the `is_not` form. Use `none` in `clue_cases_dict` wherever the clue template uses the predicate form of an attribute.

**Unambiguous templates**
- `prompt_templates` must be unambiguous. It must be clear that each object has exactly one value from each category.

### 4. Update README.md
Add the new language to the language/theme list under the relevant theme. Use the same format as existing entries:
```
- <Theme> theme:
    - Preliminary versions: ... and <LanguageName> 🏳️.
```

### 5. Update config/config.yaml
Add `<lang_code>/<theme_name>` to the comment block listing all valid language/theme combinations near the top of the file.

### 6. Check if code changes are needed

Code changes are needed when:
- A new grammatical feature requires a new clue type (rare)
- Number agreement rules require special handling beyond what templates support (e.g. Finnish partitive after numbers 2+, which was solved via the template directly)
- Major changes are required to clue templates to make them unambiguous or grammatically correctin the new language

If changes are needed, explain what they are before implementing.

### 7. Generate puzzles
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

### 8. Show and review the puzzles
Read and display `data/<lang_code>_<theme_name>/4x5/5rh/puzzles/zebra_puzzle_0.txt`.

Ask yourself:
1. Does the grammar look correct?
2. Are all clue templates natural in the language?
3. Are the `multiple_between` / `one_between` clues unambiguous?

Make any corrections to the config before the language is considered complete. Then re-run the puzzle generation to verify the fixes.

### 9. Consider improving this skill
If you found any part of this process confusing or error-prone, please suggest improvements to this skill.
