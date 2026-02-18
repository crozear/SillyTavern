# Word Replacement Config Expansion — Design

**Date:** 2026-02-18
**File:** `src/endpoints/backends/chat-completions.js` → `WORD_REPLACEMENT_CONFIG`

## Goal

Add ~22 new source/replacement group pairs to fill gaps in anatomical coverage, add male/female pronoun-bound euphemism groups, expand action verbs, and handle arousal state phrases with proper tense splitting.

## Strategy

- **Bare matching** for inherently anatomical terms with no plausible non-sexual homograph (vagina, erection, scrotum, pre-ejaculate)
- **Pronoun-bound matching** for ambiguous words that need context (core, folds, member, breasts) — mirrors existing `herSens`/`hisSens` pattern
- **Tense-split groups** for arousal states to prevent verb-tense grammar errors

## New Groups

### Bare anatomical

| Group | Source | Replacements |
|---|---|---|
| `vagina` | `vagina` | `pussy`, `cunt` |
| `member` | `member`, `manhood` | `cock`, `dick` |
| `erection` | `erection` | `boner`, `hard-on`, `stiffy` |
| `scrotum` | `scrotum` | `sack`, `nutsack`, `ball sack` |
| `preCum` | `pre-ejaculate`, `pre-seminal fluid` | `pre-cum`, `cock drool` |

### Pronoun-bound: female genitalia euphemisms

Covers LLM soft terms: sex, core, center, flower, folds, depths, womanhood, entrance

| Group | Pronoun | Replacements |
|---|---|---|
| `herSexNoun` | `her` | `her pussy`, `her cunt`, `her slit` |
| `yourSexNoun` | `your` | `your pussy`, `your cunt`, `your slit` |
| `mySexNoun` | `my` | `my pussy`, `my cunt`, `my slit` |
| `theirSexNoun` | `their` | `their pussy`, `their cunt`, `their slit` |

### Pronoun-bound: male genital euphemisms

Covers: member, manhood, hardness, length (where not already caught by `yourLength`/`phallus`)

| Group | Pronoun | Replacements |
|---|---|---|
| `hisMember` | `his` | `his cock`, `his dick` |
| `myMember` | `my` | `my cock`, `my dick` |
| `yourMemberMore` | `your` | `your cock`, `your dick` |
| `theirMember` | `their` | `their cock`, `their dick` |

### Pronoun-bound: breasts

Covers: breasts, breast, bosom, bust (not `chest` — already caught by `chestFemale`)

| Group | Pronoun | Replacements |
|---|---|---|
| `herBreasts` | `her` | `her tits`, `her titties`, `her boobs` |
| `yourBreasts` | `your` | `your tits`, `your titties`, `your boobs` |
| `myBreasts` | `my` | `my tits`, `my titties`, `my boobs` |

### Arousal state (tense-split)

| Group | Source | Replacements |
|---|---|---|
| `becameAroused` | `became aroused`, `grew aroused`, `felt aroused` | `got horny`, `got turned on`, `got hot` |
| `gettingAroused` | `becoming aroused`, `growing aroused`, `getting aroused`, `feeling aroused` | `getting horny`, `getting turned on`, `getting hot` |

### Actions

| Group | Source | Replacements |
|---|---|---|
| `actionsForeplay` | `fondle`, `caress` | `grope`, `knead`, `manhandle`, `paw at` |

### Sensations source expansion

Extend existing `sensations` source from `["ache"]` to `["ache", "tingle", "flutter"]`. Replacements unchanged.

## Conflict Analysis

- `herSexNoun` includes `her entrance` — longer phrase takes precedence over bare `entrance` group (no double-replacement since `pussy`/`cunt`/`slit` are not source words)
- `hisMember` includes `his hardness` — longer phrase takes precedence over bare `hardness` in `phallus` source
- `yourMemberMore` excludes `your manhood` (same as `your member` but added) and `your length`/`your hardness` (already covered by `yourLength` and `phallus`)
- `member` bare group: "member of a group" is a theoretical false positive, accepted as extremely rare in erotica context
