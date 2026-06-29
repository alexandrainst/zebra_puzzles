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
| Genitive + instrumental (e.g. Ukrainian) | `config/language/uk/budynky.yaml` |
| Masculine definite article split (e.g. Bulgarian) | `config/language/bg/kashti.yaml` |

**Human-validated configs:** da, de, en, fo, is, nb, nl, nn, sv. These are the most reliable translation references.

Before writing the config, briefly scan whether the language has any feature that clearly cannot be expressed by translating the existing clue templates and attribute forms — for example, a radically different word order, a grammatical category with no analog in existing configs, or a number agreement rule that affects clue templates. Note anything flagged here and address it at step 4.

### 2. Create the config file
Create `config/language/<lang_code>/` directory if needed, then write `<theme_name>.yaml`.

**Before writing any attribute forms, make three decisions:**

1. **Gender default**: If the language marks grammatical gender, decide the default now. Write it as a YAML comment at the top of the file. Apply it to all nom/is/is_not forms for both regular attributes AND red herring attributes — they can appear in the same puzzle describing the same person of unknown gender. But if a neutral form is available and natural, use that instead. In the houses theme, attributes can describe both men and women, but their genders are never mentioned. The same attribute string must work for any person regardless of their actual gender — a grammatically gendered form is fine as long as it is used consistently for everyone (e.g. "the cat owner" in a grammatically masculine form is acceptable; using masculine for some persons and feminine for others is not). Prefer short, natural forms over verbose periphrastic constructions.

2. **Case requirements**: Look at the clue templates you will translate and identify which prepositions or postpositions are used for positional clues (`next_to`, `just_left_of`, `between`). The case these prepositions govern determines whether you need extra cases beyond `[nom, is, is_not]`. If all positional prepositions take nominative (as in Hungarian), no extra cases are needed.

   A few patterns that come up across many European languages:

   - **Locative/prepositional case** (Polish, Czech, Serbian, Croatian, Lithuanian, Slovenian): many languages have a locative case used after prepositions of location. Check whether the prepositions for `next_to` and `between` govern locative — in most Slavic languages they govern instrumental or genitive instead, so locative is typically not needed for attribute forms.

   - **Number agreement in `multiple_between`** (most Slavic languages): many Slavic languages require different noun forms after different numerals, e.g. Bosnian/Croatian/Serbian: 2–4 → genitive singular ("kuće"), 5+ → genitive plural ("kuća"). Since `multiple_between` only arises for n≥2 and typical puzzle sizes keep n≤4, using the 2–4 form throughout is acceptable for a preliminary config. Note the limitation in a comment in the config file.

   - **Animate/inanimate accusative split** (most Slavic languages): animate masculine nouns use the genitive form in accusative position, while inanimates use a distinct form. This does not require any special case name — since every attribute stores its own form list, just supply the correct surface form for each attribute's accusative slot.

   - **Case syncretism** (when two grammatical cases always share the same surface form): list them as one entry in `attribute_cases` with a descriptive combined name, e.g. `gen_dat` for Romanian where genitive = dative. Use that single name in `clue_cases_dict` wherever either case is needed.

   - **Postpositive (suffixed) definite articles** (Romanian, Bulgarian, Albanian, Macedonian): definiteness is marked by a suffix on the noun rather than a separate word. Store the fully inflected form including the article suffix directly in the YAML — no special handling is needed. Use `prompt_replacements` for any preposition–article interactions. **Bulgarian exception**: masculine definite nouns have two forms — a full form (`-ят/-ят`, used in subject/nominative position) and a short form (`-а/-я`, used after prepositions). This split requires `attribute_cases: [nom, is, is_not, prep]` where `prep` is the short masculine form. Feminine and neuter nouns have only one definite form and use the same string for both `nom` and `prep` slots.

3. **Prompt replacements**: Think ahead — will relative clauses leave trailing commas before periods? Will articles contract? Note these now and add to `prompt_replacements` as you encounter them while writing the config. If you are considering adding a lot of replacements, it may be better to edit the code to handle the grammar.
   - Romance languages typically contract `de + article`: French `de le → du`, Portuguese `de o → do`, Spanish `de el → del`, Italian `di il → del`. Since nom forms follow positional prepositions in clue templates (e.g. `à esquerda de {attribute_desc}`), these contractions arise naturally. Add the relevant `"de o ": "do "` style rule upfront.

**Key clue template grammar** (refer to this when writing `is` forms and clue templates):
- `same_object`: `{attribute_desc_1} {attribute_desc_2}.` — desc_1 is the subject (nom), desc_2 is a bare predicate (the `is` form used directly). For languages that omit the copula (e.g. Hungarian 3rd-person present), this works as-is.
- `friends`: `{attribute_desc} {attribute_desc_herring}.` (or your translation) — BOTH descriptions must function as co-subjects. Use nominative for both. If the language would require a different case for the second subject, restructure the template to use "X and Y are friends" with explicit conjunction instead.
- In `found_at`/`not_at`, all desc forms are subjects (nom). In `next_to`, `just_left_of`, `between`, etc., the *first* desc is the grammatical subject (nom), but the *second* desc (and further) appear after prepositions and may require a different case. When `clue_cases_dict['next_to']` uses a non-nominative case for the second argument (e.g. `['nom', 'prep']`), `red_herring_cases_dict['next_to_herring']` must mirror it, and that case must be added to `red_herring_attribute_cases` as a third entry.


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

**Known content-sensitive attributes** — these have been mistranslated in several past configs. Verify each one before finishing step 2:

- **Wild strawberry** (`ahomansikka` / `fraise des bois` / `szamóca`): This is the small woodland fruit (*Fragaria vesca*), not a regular strawberry and **not a raspberry**. Use the local term for wild/woodland strawberry.
- **Mango (red herring)**: The person likes mango but considers some other fruit better — mango is their runner-up. Avoid phrasings like "second favourite fruit" that imply mango is one of their top picks. Frame it as a ranking where something else comes first, e.g. "thinks the second-best fruit is mango".
- **Soda**: This is a generic carbonated soft drink (cola, fizzy drink).
- **Bouldering**: This is rope-free climbing on low boulders/walls, **not** general rock-climbing or mountaineering.
- **Cocoa**: This is the hot chocolate drink made from cocoa powder, not the cocoa bean, powder, or tree.
- **Stick insect**: This is the insect that looks like a stick, not a general term for any insect or a wooden stick.
- **Football**: This describes someone who plays football (soccer) as a hobby, not someone who watches it, is a fan or a professional player.

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
Each attribute must be unambiguous. It should not be easy to confuse an attribute in one category with an attribute in another category. Check every category pair where confusion is plausible:

- **Hobbies vs. occupations**: Hobby attributes must use activity phrases, not occupation nouns. A solver reading "the sailor" cannot tell whether it describes a job or a pastime. Check any hobby that has a corresponding profession in the language (painter, footballer, tennis player, etc.).
- **Red herring attributes vs. occupations**: A word that means both a red herring concept and an occupation (e.g. Hungarian "nővér" = both "sister" and "nurse") must be replaced. Warn the user if avoiding ambiguity requires changing the meaning.

### 4. Check if code changes are needed

Code changes are needed when:
- Number agreement rules require special handling beyond what templates support (e.g. Finnish partitive after numbers 2+, which was solved via the template directly)
- Major changes are required to clue templates to make them unambiguous or grammatically correct in the new language. For example if word order in the new language is very different and cannot be handled by existing templates or a few prompt_replacements.

If changes are needed, explain what they are before implementing.

Run tests with `make test` to verify that any code changes are correct and don't break existing languages.

### 5. Run make check

Check formatting by running `make check`. Fix any issues found.

### 6. Check for semantic drift
Compare the meaning of the new config with the English template. If any attribute or clue meaning has changed, note it here and explain why it was necessary or align the config with the English meaning.

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

### 9. Update README.md
Add the new language to the language/theme list under the relevant theme. Use the same format as existing entries:
```
- <Theme> theme:
    - Preliminary versions: ... and <LanguageName> 🏳️.
```

### 10. Update config/config.yaml
Add `<lang_code>/<theme_name>` to the comment block listing all valid language/theme combinations near the top of the file.

### 11. Consider improving this skill
If you found any part of this process confusing or error-prone, suggest improvements to this skill. Did you have to make decisions or look up information that could be included in the skill? Did you have to read the code to understand how it works? If so, consider adding that information to this skill.

### 12. User-review

Show the first puzzle to the user and ask if the puzzle looks correct.
