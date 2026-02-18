# Word Replacement Config Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ~22 new source/replacement group pairs to `WORD_REPLACEMENT_CONFIG` in `chat-completions.js` to cover anatomical gaps, pronoun-bound euphemisms, arousal state phrases, and new action verbs.

**Architecture:** All changes are pure config additions to two objects (`sourceGroups` and `replacementGroups`) inside `WORD_REPLACEMENT_CONFIG` in one file. No logic changes required — the existing `buildReplacementRules`, `applyWordReplacements`, and streaming infrastructure handles everything automatically once new groups are added.

**Tech Stack:** Node.js, plain JavaScript. No build step — server restart picks up changes.

---

### Task 1: Add bare anatomical groups

**Files:**
- Modify: `src/endpoints/backends/chat-completions.js:156-246` (sourceGroups) and `:247-295` (replacementGroups)

These terms are inherently anatomical with no plausible non-sexual homograph in erotica output.

**Step 1: Add to `sourceGroups` block**

In `chat-completions.js`, locate the `sourceGroups` object (starts around line 156). Add the following entries after the existing `testicles` line (near line 170):

```js
vagina: ['vagina'],
member: ['member', 'manhood'],
erection: ['erection'],
scrotum: ['scrotum'],
preCum: ['pre-ejaculate', 'pre-seminal fluid'],
```

**Step 2: Add matching entries to `replacementGroups` block**

In the `replacementGroups` object (starts around line 247), add after the existing `testicles` line (near line 261):

```js
vagina: ['pussy', 'cunt'],
member: ['cock', 'dick'],
erection: ['boner', 'hard-on', 'stiffy'],
scrotum: ['sack', 'nutsack', 'ball sack'],
preCum: ['pre-cum', 'cock drool'],
```

**Step 3: Manual smoke test**

Start the server (`npm start`). Send a test message through any provider with word replacement enabled that includes "his erection", "her vagina", "his scrotum", "pre-ejaculate". Verify they are replaced in the response.

**Step 4: Commit**

```bash
git add src/endpoints/backends/chat-completions.js
git commit -m "feat: add bare anatomical replacement groups (vagina, member, erection, scrotum, preCum)"
```

---

### Task 2: Add pronoun-bound female genitalia euphemism groups

**Files:**
- Modify: `src/endpoints/backends/chat-completions.js` — `sourceGroups` and `replacementGroups`

These mirror the existing `herSens`/`hisSens` pattern. The pronoun prefix prevents false positives on words like "core" or "folds" in non-sexual sentences.

**Step 1: Add to `sourceGroups`**

Add after the existing `chestFemale` entry (around line 240):

```js
herSexNoun: ['her sex', 'her core', 'her center', 'her flower', 'her folds', 'her depths', 'her womanhood', 'her entrance'],
yourSexNoun: ['your sex', 'your core', 'your center', 'your flower', 'your folds', 'your depths', 'your womanhood', 'your entrance'],
mySexNoun: ['my sex', 'my core', 'my center', 'my flower', 'my folds', 'my depths', 'my womanhood', 'my entrance'],
theirSexNoun: ['their sex', 'their core', 'their center', 'their flower', 'their folds', 'their depths', 'their womanhood', 'their entrance'],
```

**Step 2: Add to `replacementGroups`**

Add after the existing `chestFemale` entry (around line 289):

```js
herSexNoun: ['her pussy', 'her cunt', 'her slit'],
yourSexNoun: ['your pussy', 'your cunt', 'your slit'],
mySexNoun: ['my pussy', 'my cunt', 'my slit'],
theirSexNoun: ['their pussy', 'their cunt', 'their slit'],
```

**Step 3: Commit**

```bash
git add src/endpoints/backends/chat-completions.js
git commit -m "feat: add pronoun-bound female genitalia euphemism groups (herSexNoun etc.)"
```

---

### Task 3: Add pronoun-bound male genital euphemism groups

**Files:**
- Modify: `src/endpoints/backends/chat-completions.js` — `sourceGroups` and `replacementGroups`

Note: `your length` is already caught by the existing `yourLength` group. `your hardness` / bare `hardness` is already caught by the existing `phallus` group. The new `yourMemberMore` group only adds `your member` and `your manhood` to avoid redundancy.

**Step 1: Add to `sourceGroups`**

Add after the `theirSexNoun` entries from Task 2:

```js
hisMember: ['his member', 'his manhood', 'his hardness', 'his length'],
myMember: ['my member', 'my manhood', 'my hardness', 'my length'],
yourMemberMore: ['your member', 'your manhood'],
theirMember: ['their member', 'their manhood', 'their length', 'their hardness'],
```

**Step 2: Add to `replacementGroups`**

```js
hisMember: ['his cock', 'his dick'],
myMember: ['my cock', 'my dick'],
yourMemberMore: ['your cock', 'your dick'],
theirMember: ['their cock', 'their dick'],
```

**Step 3: Commit**

```bash
git add src/endpoints/backends/chat-completions.js
git commit -m "feat: add pronoun-bound male genital euphemism groups (hisMember etc.)"
```

---

### Task 4: Add pronoun-bound breast groups

**Files:**
- Modify: `src/endpoints/backends/chat-completions.js` — `sourceGroups` and `replacementGroups`

Intentionally excludes `her chest` / `her bare chest` — already caught by the existing `chestFemale` group. No `hisMember` parallel here since "his breasts" is not a common phrase.

**Step 1: Add to `sourceGroups`**

```js
herBreasts: ['her breasts', 'her breast', 'her bosom', 'her bust'],
yourBreasts: ['your breasts', 'your breast', 'your bosom', 'your bust'],
myBreasts: ['my breasts', 'my breast', 'my bosom', 'my bust'],
```

**Step 2: Add to `replacementGroups`**

```js
herBreasts: ['her tits', 'her titties', 'her boobs'],
yourBreasts: ['your tits', 'your titties', 'your boobs'],
myBreasts: ['my tits', 'my titties', 'my boobs'],
```

**Step 3: Commit**

```bash
git add src/endpoints/backends/chat-completions.js
git commit -m "feat: add pronoun-bound breast groups (herBreasts, yourBreasts, myBreasts)"
```

---

### Task 5: Add tense-split arousal state groups and foreplay action group

**Files:**
- Modify: `src/endpoints/backends/chat-completions.js` — `sourceGroups` and `replacementGroups`

Two arousal groups are tense-split to keep grammar clean (past→past, present-participle→present-participle). No conflict with existing `arousalNoun` (bare "arousal") or `herSens`/`hisSens` ("her/his arousal").

**Step 1: Add to `sourceGroups`**

Add after the existing `actionsStimulation` entry (around line 245):

```js
becameAroused: ['became aroused', 'grew aroused', 'felt aroused'],
gettingAroused: ['becoming aroused', 'growing aroused', 'getting aroused', 'feeling aroused'],
actionsForeplay: ['fondle', 'caress'],
```

**Step 2: Add to `replacementGroups`**

Add after the existing `actionsStimulation` entry (around line 294):

```js
becameAroused: ['got horny', 'got turned on', 'got hot'],
gettingAroused: ['getting horny', 'getting turned on', 'getting hot'],
actionsForeplay: ['grope', 'knead', 'manhandle', 'paw at'],
```

**Step 3: Commit**

```bash
git add src/endpoints/backends/chat-completions.js
git commit -m "feat: add tense-split arousal state groups and actionsForeplay"
```

---

### Task 6: Expand existing sensations source list

**Files:**
- Modify: `src/endpoints/backends/chat-completions.js:243` (sensations source line)

Add `tingle` and `flutter` to the existing `sensations` source group. Neither word appears in the replacement list so no cycle risk.

**Step 1: Find the sensations source line**

It currently reads:
```js
sensations: ["ache"],
```

**Step 2: Expand it**

Change to:
```js
sensations: ['ache', 'tingle', 'flutter'],
```

**Step 3: Commit**

```bash
git add src/endpoints/backends/chat-completions.js
git commit -m "feat: expand sensations source group with tingle and flutter"
```

---

### Task 7: Reset replacement cycle cache

After all config changes are saved, the `cachedReplacementRules` cache must be cleared so the server picks up all new groups. The cache is reset automatically on server restart, but if the server is running hot-reload style, you can confirm by restarting.

**Step 1: Restart the server**

```bash
npm start
```

**Step 2: Verify end-to-end**

Send a message through any LLM provider with word replacement enabled. Include phrases from each new group (e.g. "her core", "his manhood", "her bosom", "became aroused", "she caressed") and confirm they are replaced correctly in the response.

**Step 3: Done**
