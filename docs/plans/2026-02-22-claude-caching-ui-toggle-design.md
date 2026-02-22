# Design: Claude Caching At Depth UI Toggle

**Date:** 2026-02-22

## Problem

`cachingAtDepth` in `sendClaudeRequest` is controlled only by `claude.cachingAtDepth` in `config.yaml`. There is no per-session UI toggle — users must edit the server config to enable/disable caching at depth.

## Goal

Add a checkbox in the chat completion settings panel, directly under "Use system prompt", that lets users enable or disable Claude's prompt caching at depth per session.

- **Unchecked:** `cachingAtDepth = -1` (disabled)
- **Checked:** `cachingAtDepth = 0` (cache from depth 0)

System prompt cache (`enableSystemPromptCache`) is unaffected and remains server-config-only.

## Approach Selected

**Option A — `cachingAtDepth` only (minimal override)**

The UI sends a `claude_enable_caching_at_depth` boolean with each generate request. In `sendClaudeRequest`, this overrides the server config value:
- `false` → force `cachingAtDepth = -1`
- `true` and config has no depth set → use `cachingAtDepth = 0`
- `true` and config already has a depth ≥ 0 → keep config value

## Changes

### `public/index.html`
- Add `div.range-block[data-source="claude"]` after the `use_sysprompt` block (~line 1984)
- Checkbox id: `claude_enable_caching_at_depth`
- Label: "Enable prompt caching"
- Description: "Cache the prompt at depth 0 for supported Claude models. Reduces cost and latency for repeated context."

### `public/scripts/openai.js`
- Add to `settingsToUpdate`: `claude_enable_caching_at_depth: ['#claude_enable_caching_at_depth', 'claude_enable_caching_at_depth', true, false]`
- Add to `default_settings`: `claude_enable_caching_at_depth: false`
- Add `change` event handler that calls `saveSettingsDebounced()`
- Add to Claude generate data block: `generate_data.claude_enable_caching_at_depth = settings.claude_enable_caching_at_depth`

### `src/endpoints/backends/chat-completions.js`
- In `sendClaudeRequest`, after resolving `cachingAtDepth` from config, apply the UI override:
  ```js
  if (request.body.claude_enable_caching_at_depth === false) {
      cachingAtDepth = -1;
  } else if (request.body.claude_enable_caching_at_depth === true && cachingAtDepth < 0) {
      cachingAtDepth = 0;
  }
  ```

## Scope

- 3 files modified
- ~15 lines of new code
- No new endpoints, no schema changes, no migrations needed
