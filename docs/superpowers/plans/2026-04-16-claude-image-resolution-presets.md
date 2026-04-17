# Claude Image Resolution Presets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Claude-specific image resolution preset selector (min/low/medium/high) that downscales images before sending to the Anthropic API, plus a static token-cost estimate label in the settings panel.

**Architecture:** Extend the existing `compressImage()` method on the `Message` class in `public/scripts/openai.js` with a Claude-specific branch that reads a new `claude_image_resolution` setting and calls the already-present `createThumbnail()` helper to downscale. A new dropdown + hint label in `public/index.html` (scoped to `data-source="claude"`) surfaces the setting. No backend changes needed — resizing happens before base64 is sent.

**Tech Stack:** Vanilla JS, jQuery, Canvas API (via existing `createThumbnail`), HTML

---

## File Map

| File | Change |
|---|---|
| `public/scripts/openai.js` | Add default setting, `settingsToUpdate` entry, `updateClaudeResolutionHint()`, event listener, Claude branch in `compressImage()`, call hint init in `loadOpenAISettings()` |
| `public/index.html` | Add resolution dropdown + token hint label inside existing Claude inline-media block |

---

### Task 1: Add the `claude_image_resolution` setting

**Files:**
- Modify: `public/scripts/openai.js:379` (settingsToUpdate)
- Modify: `public/scripts/openai.js:487` (default_settings)

- [ ] **Step 1: Add to `settingsToUpdate`**

In `public/scripts/openai.js`, find line 379:
```javascript
    request_image_resolution: ['#request_image_resolution', 'request_image_resolution', false, false],
```
Add immediately after it:
```javascript
    claude_image_resolution: ['#claude_image_resolution', 'claude_image_resolution', false, false],
```

- [ ] **Step 2: Add to `default_settings`**

Find line 487:
```javascript
    request_image_resolution: '',
```
Add immediately after it:
```javascript
    claude_image_resolution: 'medium',
```

- [ ] **Step 3: Verify the app still loads**

Run: `npm start` — open the browser. No console errors. No crash.

- [ ] **Step 4: Commit**

```bash
git add public/scripts/openai.js
git commit -m "feat: add claude_image_resolution setting (default medium)"
```

---

### Task 2: Add the HTML dropdown and hint label

**Files:**
- Modify: `public/index.html` — after the `openai_inline_image_quality` block (line ~2075)

The new block goes immediately after the closing `</div>` of the `data-source="openai,custom,..."` quality div (around line 2075), still inside the parent `range-block` div. Scope it to `data-source="claude"` so it only appears for the Claude provider.

- [ ] **Step 1: Insert the new HTML block**

In `public/index.html`, find the exact string:
```html
                                        </div>
                                    </div>
                                    <div id="request_images_block" class="range-block" data-source="makersuite,vertexai">
```
Insert the new block between the two closing `</div>` tags and the `request_images_block` div:
```html
                                        </div>
                                        <div class="flex-container flexFlowColumn wide100p textAlignCenter marginTop10" data-source="claude">
                                            <div class="flex-container oneline-dropdown">
                                                <label for="claude_image_resolution" data-i18n="Claude Image Resolution">
                                                    Claude Image Resolution
                                                </label>
                                                <select id="claude_image_resolution">
                                                    <option value="min">Min (322 px)</option>
                                                    <option value="low">Low (644 px)</option>
                                                    <option value="medium">Medium (1,288 px)</option>
                                                    <option value="high">High (2,576 px)</option>
                                                </select>
                                            </div>
                                            <small id="claude_image_resolution_hint" class="opacity50p marginTop5"></small>
                                        </div>
                                    </div>
                                    <div id="request_images_block" class="range-block" data-source="makersuite,vertexai">
```

- [ ] **Step 2: Verify the dropdown appears for Claude**

Start the server, switch chat completion source to Claude — the "Claude Image Resolution" dropdown should appear below the media inlining section. Switch to OpenAI — it should be hidden.

- [ ] **Step 3: Commit**

```bash
git add public/index.html
git commit -m "feat: add Claude image resolution dropdown to settings UI"
```

---

### Task 3: Wire up the hint label and event listener

**Files:**
- Modify: `public/scripts/openai.js` — add `updateClaudeResolutionHint()` function, event listener near line 6913, and call from `loadOpenAISettings()`

- [ ] **Step 1: Add `updateClaudeResolutionHint()` function**

Find the line in `public/scripts/openai.js`:
```javascript
    $('#openai_inline_image_quality').on('input', function () {
        oai_settings.inline_image_quality = String($(this).val());
        saveSettingsDebounced();
    });
```
Add a new function definition **before** that block (or anywhere nearby in the event-listener section is fine — just keep it grouped with the image quality code):

```javascript
    /**
     * Update the Claude image resolution hint label with a worst-case token estimate.
     */
    function updateClaudeResolutionHint() {
        const resolutionPresets = { min: 256, low: 512, medium: 1024, high: 1568, opus: 2576 };
        const preset = oai_settings.claude_image_resolution || default_settings.claude_image_resolution;
        const maxEdge = resolutionPresets[preset] ?? resolutionPresets.medium;
        const paddedEdge = Math.ceil(maxEdge / 28) * 28;
        const tokens = Math.round(paddedEdge * paddedEdge / 750);
        $('#claude_image_resolution_hint').text(`Max ~${tokens.toLocaleString()} tokens (${maxEdge.toLocaleString()} px long edge)`);
    }
```

- [ ] **Step 2: Add the event listener**

Immediately after the `#openai_inline_image_quality` handler (after `saveSettingsDebounced();` and its closing `});`), add:

```javascript
    $('#claude_image_resolution').on('input', function () {
        oai_settings.claude_image_resolution = String($(this).val());
        updateClaudeResolutionHint();
        saveSettingsDebounced();
    });
```

- [ ] **Step 3: Call hint init in `loadOpenAISettings()`**

In `loadOpenAISettings()`, find the line:
```javascript
    updateVertexAIServiceAccountStatus();
```
Add immediately after it:
```javascript
    updateClaudeResolutionHint();
```

- [ ] **Step 4: Verify the hint text appears**

Start the server, switch to Claude. The hint below the dropdown should read e.g. `Max ~2,212 tokens (1,288 px long edge)` for Medium. Change the dropdown to High — it should update to `Max ~8,848 tokens (2,576 px long edge)`. Reload the page — hint should restore from saved setting.

- [ ] **Step 5: Commit**

```bash
git add public/scripts/openai.js
git commit -m "feat: add Claude resolution hint label with token estimate"
```

---

### Task 4: Implement Claude-specific downscaling in `compressImage()`

**Files:**
- Modify: `public/scripts/openai.js:3627-3645` (`compressImage()` method)

- [ ] **Step 1: Read the current `compressImage()` method**

Current code at `public/scripts/openai.js:3627`:
```javascript
    async compressImage(image) {
        const compressImageSources = [
            chat_completion_sources.OPENROUTER,
            chat_completion_sources.MAKERSUITE,
            chat_completion_sources.MISTRALAI,
            chat_completion_sources.VERTEXAI,
        ];
        const sizeThreshold = 2 * 1024 * 1024;
        const dataSize = image.length * 0.75;
        const safeMimeTypes = ['image/jpeg', 'image/png', 'image/webp'];
        const mimeType = image?.split(';')?.[0]?.split(':')?.[1];
        if (compressImageSources.includes(oai_settings.chat_completion_source) && dataSize > sizeThreshold) {
            const maxSide = 2048;
            image = await createThumbnail(image, maxSide, maxSide);
        } else if (!safeMimeTypes.includes(mimeType)) {
            image = await createThumbnail(image, null, null);
        }
        return image;
    }
```

- [ ] **Step 2: Replace `compressImage()` with the updated version**

Replace the entire method body. The new version adds a Claude branch between the existing two `if` clauses:

```javascript
    async compressImage(image) {
        const compressImageSources = [
            chat_completion_sources.OPENROUTER,
            chat_completion_sources.MAKERSUITE,
            chat_completion_sources.MISTRALAI,
            chat_completion_sources.VERTEXAI,
        ];
        const sizeThreshold = 2 * 1024 * 1024;
        const dataSize = image.length * 0.75;
        const safeMimeTypes = ['image/jpeg', 'image/png', 'image/webp'];
        const mimeType = image?.split(';')?.[0]?.split(':')?.[1];
        if (compressImageSources.includes(oai_settings.chat_completion_source) && dataSize > sizeThreshold) {
            const maxSide = 2048;
            image = await createThumbnail(image, maxSide, maxSide);
        } else if (oai_settings.chat_completion_source === chat_completion_sources.CLAUDE) {
            const resolutionPresets = { min: 256, low: 512, medium: 1024, high: 1568, opus: 2576 };
            const preset = oai_settings.claude_image_resolution || default_settings.claude_image_resolution;
            const maxEdge = resolutionPresets[preset] ?? resolutionPresets.medium;
            const size = await getImageSizeFromDataURL(image);
            const needsResize = size.width > maxEdge || size.height > maxEdge;
            const needsMimeConvert = !safeMimeTypes.includes(mimeType);
            if (needsResize || needsMimeConvert) {
                image = await createThumbnail(image, needsResize ? maxEdge : null, needsResize ? maxEdge : null);
            }
        } else if (!safeMimeTypes.includes(mimeType)) {
            image = await createThumbnail(image, null, null);
        }
        return image;
    }
```

- [ ] **Step 3: Verify downscaling works**

Start the server. In Claude chat, attach a large image (e.g. a 3000×2000 photo). Open DevTools Network tab. Send a message. Inspect the request payload — the `image_url.url` data URL should decode to an image with a long edge ≤ 1288 px (Medium preset). Switch to Min (322 px), attach the same image, send — the decoded image should be ≤ 322 px on the long edge.

- [ ] **Step 4: Verify small images are NOT upscaled**

Attach a 100×80 image on Medium preset. The data URL in the outgoing request should remain at 100×80 (no upscaling).

- [ ] **Step 5: Verify non-Claude sources are unaffected**

Switch to OpenAI or OpenRouter. Attach an image. The behavior should be identical to before this change.

- [ ] **Step 6: Commit**

```bash
git add public/scripts/openai.js
git commit -m "feat: downscale Claude images per resolution preset before sending"
```

---

## Verification Checklist

- [ ] Claude dropdown appears when Claude source is selected, hidden for all others
- [ ] Hint label shows correct token estimate for each preset:
  - Min → `Max ~151 tokens (322 px long edge)`
  - Low → `Max ~553 tokens (644 px long edge)`
  - Medium → `Max ~2,212 tokens (1,288 px long edge)`
  - High → `Max ~8,848 tokens (2,576 px long edge)`
- [ ] Hint updates immediately on dropdown change (no page reload needed)
- [ ] Hint is correct after page reload (restored from saved setting)
- [ ] Large images (>preset edge) are downscaled before send
- [ ] Small images (<preset edge) are NOT upscaled
- [ ] Unsafe MIME types (e.g. GIF, AVIF) are still converted to PNG for Claude
- [ ] No regression on OpenRouter/MakerSuite/Mistral/VertexAI compress path
- [ ] No regression on OpenAI (unaffected path)
