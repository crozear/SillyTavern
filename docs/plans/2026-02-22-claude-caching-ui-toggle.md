# Claude Caching UI Toggle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Enable prompt caching" checkbox under "Use system prompt" that sets `cachingAtDepth` to `0` (on) or `-1` (off) per request, overriding `config.yaml`.

**Architecture:** The checkbox is a standard SillyTavern setting — HTML element → `settingsToUpdate` binding → `generate_data` pass-through → backend override in `sendClaudeRequest`. No new endpoints. The backend reads `request.body.claude_enable_caching_at_depth` and applies it after resolving the config value.

**Tech Stack:** Vanilla JS + jQuery (frontend), Node.js + Express (backend)

---

### Task 1: Add checkbox to index.html

**Files:**
- Modify: `public/index.html` (~line 1984, after the `use_sysprompt` block)

**Step 1: Locate insertion point**

Find this closing tag in `public/index.html`:
```html
                                    </div>
                                    <div class="range-block" data-source="makersuite,vertexai,aimlapi,openrouter,claude,xai,electronhub,chutes,nanogpt">
```
It appears right after the `use_sysprompt` block ends (~line 1984). Insert the new block between those two.

**Step 2: Insert the checkbox block**

Add this between the closing `</div>` of the `use_sysprompt` block and the opening of the web search block:

```html
                                    <div class="range-block" data-source="claude">
                                        <label for="claude_enable_caching_at_depth" class="checkbox_label widthFreeExpand">
                                            <input id="claude_enable_caching_at_depth" type="checkbox" />
                                            <span>
                                                <span data-i18n="Enable prompt caching">Enable prompt caching</span>
                                            </span>
                                        </label>
                                        <div class="toggle-description justifyLeft marginBot5">
                                            <span data-i18n="Cache the prompt at depth 0 for Claude models. Reduces cost and latency for repeated context.">
                                                Cache the prompt at depth 0 for Claude models. Reduces cost and latency for repeated context.
                                            </span>
                                        </div>
                                    </div>
```

**Step 3: Verify visually**

Start the server (`npm start`) and open the Claude settings panel. Confirm the checkbox appears below "Use system prompt" and above "Enable web search", and only shows when Claude source is selected.

**Step 4: Commit**

```bash
git add public/index.html
git commit -m "feat: add Claude prompt caching checkbox to settings panel"
```

---

### Task 2: Wire up the setting in openai.js

**Files:**
- Modify: `public/scripts/openai.js`

There are four places to touch. Search for `use_sysprompt` in the file — each instance will have a parallel addition nearby.

**Step 1: Add to `settingsToUpdate`**

Find:
```js
    use_sysprompt: ['#use_sysprompt', 'use_sysprompt', true, false],
```
Add immediately after:
```js
    claude_enable_caching_at_depth: ['#claude_enable_caching_at_depth', 'claude_enable_caching_at_depth', true, false],
```

**Step 2: Add to `default_settings`**

Find:
```js
    use_sysprompt: false,
```
Add immediately after:
```js
    claude_enable_caching_at_depth: false,
```

**Step 3: Add to Claude generate data**

Find:
```js
        generate_data.use_sysprompt = settings.use_sysprompt;
```
(There are two occurrences — one in the Claude block ~line 2691, one in the MakerSuite block ~line 2717. Add only to the Claude block, not the MakerSuite one.)

Add immediately after the Claude occurrence:
```js
        generate_data.claude_enable_caching_at_depth = settings.claude_enable_caching_at_depth;
```

**Step 4: Add the change event handler**

Find:
```js
    $('#use_sysprompt').on('change', function () {
        oai_settings.use_sysprompt = !!$('#use_sysprompt').prop('checked');
        saveSettingsDebounced();
    });
```
Add immediately after:
```js
    $('#claude_enable_caching_at_depth').on('change', function () {
        oai_settings.claude_enable_caching_at_depth = !!$('#claude_enable_caching_at_depth').prop('checked');
        saveSettingsDebounced();
    });
```

**Step 5: Verify in browser**

Open the settings panel, check the box, reload the page — it should remain checked. Open browser devtools → Application → Local Storage and confirm `claude_enable_caching_at_depth: true` is persisted in settings.

**Step 6: Commit**

```bash
git add public/scripts/openai.js
git commit -m "feat: wire claude_enable_caching_at_depth setting in openai.js"
```

---

### Task 3: Apply override in sendClaudeRequest

**Files:**
- Modify: `src/endpoints/backends/chat-completions.js` (~line 1272)

**Step 1: Locate the cachingAtDepth resolution block**

Find this block in `sendClaudeRequest`:
```js
    let cachingAtDepth = getConfigValue('claude.cachingAtDepth', -1, 'number');
    // Disabled if not an integer or negative
    if (!Number.isInteger(cachingAtDepth) || cachingAtDepth < 0) {
        cachingAtDepth = -1;
    }
```

**Step 2: Add the UI override immediately after**

```js
    // UI toggle: override cachingAtDepth from request body
    if (request.body.claude_enable_caching_at_depth === false) {
        cachingAtDepth = -1;
    } else if (request.body.claude_enable_caching_at_depth === true && cachingAtDepth < 0) {
        cachingAtDepth = 0;
    }
```

Logic:
- `false` → always force off (UI says disabled)
- `true` + config already has depth ≥ 0 → keep config value (respect finer-grained config)
- `true` + config is -1 → use 0 (UI says enable, config has nothing set)
- `undefined` (old clients) → no change, existing behavior preserved

**Step 3: Verify the beta header is emitted**

When `cachingAtDepth === 0`, the existing code path at ~line 1363 will push `'prompt-caching-2024-07-31'` to `betaHeaders`. Confirm that block reads:
```js
    if (enableSystemPromptCache || cachingAtDepth !== -1) {
        betaHeaders.push('prompt-caching-2024-07-31');
        betaHeaders.push('extended-cache-ttl-2025-04-11');
    }
```
No change needed there — it already handles `cachingAtDepth !== -1`.

**Step 4: Manual smoke test**

- Enable the checkbox, send a message to Claude, check server console for `Claude request:` log — confirm `anthropic-beta` header includes `prompt-caching-2024-07-31`.
- Disable the checkbox, send again — confirm `prompt-caching-2024-07-31` is absent (unless `enableSystemPromptCache` is also on in config).

**Step 5: Commit**

```bash
git add src/endpoints/backends/chat-completions.js
git commit -m "feat: apply claude_enable_caching_at_depth UI override in sendClaudeRequest"
```

---

### Task 4: Final verification

**Step 1: Run lint**

```bash
npm run lint
```
Expected: no errors. Fix any reported issues before proceeding.

**Step 2: Run unit tests**

```bash
cd tests && npm run test:unit
```
Expected: all pass (no tests directly cover this path, but ensures no regressions).

**Step 3: Commit if lint required fixes**

Only if lint auto-fix changed anything:
```bash
git add -p
git commit -m "style: lint fixes for caching toggle"
```
