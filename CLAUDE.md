# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# SillyTavern - Custom Fork

This is a customized fork of [SillyTavern](https://github.com/SillyTavern/SillyTavern), a frontend for LLM APIs.
The upstream remote is `upstream` (official SillyTavern), and `origin` is the fork (crozear/SillyTavern).
The working branch is `release`.

## Tech Stack

- **Backend:** Node.js + Express (plain JavaScript, no TypeScript)
- **Frontend:** Vanilla JS + jQuery (no React/Vue/Angular)
- **No ORM** — file-based storage, no database
- **Package manager:** npm

## Project Structure

```
public/                    # Frontend (served statically)
  index.html               # Main UI — dropdowns, settings panels, model selectors
  script.js                # Core frontend logic (Generate function, streaming, etc.)
  scripts/
    openai.js              # Chat completion settings, message preparation, streaming reply parsing
    reasoning.js           # Extracts reasoning/thinking content from API responses
    sse-stream.js          # SSE stream event parsing (OpenAI, Claude, Responses API)
    extensions/            # Plugin system (TTS, image gen, vectors, captions, etc.)
src/                       # Backend (Express server)
  endpoints/
    backends/
      chat-completions.js  # THE main file — all LLM API request handlers (Claude, OpenAI, Google, etc.)
  prompt-converters.js     # Converts between message formats (OpenAI ↔ Claude ↔ Google etc.)
  constants.js             # Model lists, API URLs, reasoning effort maps
  util.js                  # Shared utilities
default/
  config.yaml              # Server configuration defaults
  content/settings.json    # Default frontend settings
```

## Custom Modifications (vs upstream)

### 1. Word Replacement System (largest custom feature)
Server-side regex system in `chat-completions.js` that swaps clinical/euphemistic language with more explicit alternatives in LLM responses. Includes:
- `WORD_REPLACEMENT_CONFIG` — source/replacement word group mappings
- `createWordReplacementStream()` — Transform stream for SSE streaming responses
- `enforceWordReplacementsOnResponse()` — for non-streaming responses
- `forwardFetchResponseWithWordReplacements()` — replaces upstream's `forwardFetchResponse()`
- Toggle via UI checkbox (`word_replacement_enabled` in openai.js) and config.yaml
- Dedicated endpoint: `POST /api/chat-completions/word-replacements`
- Applied across ALL providers (Claude, OpenAI, Google, DeepSeek, xAI, etc.)

### 2. New Model Support
- **Claude Sonnet 4.6** — added to model selectors, regex patterns, caption settings
- **Claude Opus 4.6 adaptive thinking** — `thinking: { type: 'adaptive' }` instead of budget_tokens
- `getClaudeAdaptiveEffort()` in prompt-converters.js maps reasoning effort to API effort levels
- **Reasoning effort "none"** option added (disables thinking entirely)
- Default OpenAI model changed from gpt-4-turbo → gpt-5.1
- GPT 5.2 compatibility (delete top_p)

### 3. OpenAI Responses API Support
- Auto-detects reasoning models and routes to `/v1/responses` instead of `/chat/completions`
- `convertToResponsesApiRequest()` transforms request format
- Frontend handles `response.*` SSE events in sse-stream.js and openai.js
- `X-Response-Format: responses` header signals frontend to use Responses API parsing
- Reasoning summary extraction from Responses API output items

### 4. Service Tier / Proxy Enhancements
- **Service tier selector** (flex/default/priority) in UI and request params
- **Reverse proxy improvements:** skips upstream status check, passes `instructions` and `verbosity`
- Jailbreak instruction extraction (`getLastJailbreakInstructions()`) and forwarding to proxy
- SillyTavern variables (`getPromptVariablesForProxy()`) passed to reverse proxies

### 5. Claude API Tweaks
- Allows `top_p` when < 0.95 (instead of always deleting it)
- Deletes `temperature` when set to 1 (API default)
- Removed verbosity beta header for Claude (moved to adaptive effort)
- Google API: added `presencePenalty` and `frequencyPenalty` pass-through

### 6. Other Changes
- `bias/` directory — logit bias presets (Default.json, Main.json) with a Python runner
- `st-agent-proxy/` — separate proxy server project (its own package.json)
- Pollinations API: set `private: true`, `referrer: 'sillytavern'`, removed auth header
- Group chats: modified `isValidImageUrl` null guard
- Various `// @ts-ignore` additions for toastr calls

## Commands

```bash
# Run the server
npm install
npm start              # or Start.bat on Windows
npm run debug          # with Node.js inspector

# Lint (from repo root)
npm run lint           # check src/**/*.js, public/**/*.js
npm run lint:fix       # auto-fix

# Tests (separate package in tests/ subdirectory)
cd tests && npm install
npm test               # unit (Jest) + e2e (Playwright)
npm run test:unit      # Jest only
npm run test:e2e       # Playwright only

# Run a single Jest test file
node --experimental-vm-modules node_modules/jest/bin/jest.js --config jest.config.json util.test.js
```

## Code Style

Enforced by ESLint (`.eslintrc.cjs`). Key rules:
- Single quotes, semicolons required, 4-space indentation
- Trailing commas on multiline (`comma-dangle: always-multiline`)
- `no-unused-vars` is an error (args exempted)
- Browser globals: `toastr`, `SillyTavern`, `ePub`, `pdfjsLib` — don't import these
- `// @ts-ignore` is acceptable for external library calls (toastr pattern used throughout)

## Common Tasks

When adding support for a **new model**:
1. Add to model selector in `public/index.html` (search for existing model options)
2. Update regex patterns in `chat-completions.js` (search for model name patterns like `/^claude-/`)
3. Update `src/constants.js` if the model needs reasoning effort or special handling
4. Update `src/prompt-converters.js` if message format differs
5. Add to caption model list in `public/scripts/extensions/caption/settings.html` if multimodal

When modifying **API request handling**:
- All provider-specific logic is in `src/endpoints/backends/chat-completions.js`
- Each provider has its own `send*Request()` function (e.g., `sendClaudeRequest`, `sendMakerSuiteRequest`)
- The generic OpenAI-compatible path is in `router.post('/generate', ...)`
- Word replacements must be applied to both streaming (`forwardFetchResponseWithWordReplacements`) and non-streaming (`sendWithWordReplacements`) responses

When modifying **frontend settings**:
- UI controls go in `public/index.html` with `data-source` attributes controlling which providers show them
- Setting binding is in `public/scripts/openai.js` in the `settingsToUpdate` object
- Default values go in `default_settings` in the same file

## Skills (Slash Commands)

These skills are available via the `Skill` tool. Check before any response — if there's even a 1% chance a skill applies, invoke it first.

### Workflow / Process Skills (invoke these first — they determine HOW to work)

| Trigger | Skill |
|---------|-------|
| Starting any conversation | `superpowers:using-superpowers` |
| Any bug, test failure, or unexpected behavior | `superpowers:systematic-debugging` |
| Before implementing any feature or fix | `superpowers:test-driven-development` |
| Building something new or adding functionality | `superpowers:brainstorming` (before any code) |
| Have a spec/requirements for a multi-step task | `superpowers:writing-plans` |
| Executing a written plan in the current session | `superpowers:subagent-driven-development` |
| Executing a written plan in a new/separate session | `superpowers:executing-plans` |
| 2+ independent tasks that can run in parallel | `superpowers:dispatching-parallel-agents` |
| About to claim work is complete/fixed/passing | `superpowers:verification-before-completion` |
| Implementation complete, deciding how to integrate | `superpowers:finishing-a-development-branch` |
| Starting feature work needing workspace isolation | `superpowers:using-git-worktrees` |

### Code Review Skills

| Trigger | Skill |
|---------|-------|
| Completed a task or feature, before merging | `superpowers:requesting-code-review` |
| Received code review feedback to implement | `superpowers:receiving-code-review` |
| Reviewing a pull request | `code-review:code-review` |

### Implementation Skills (invoke after brainstorming/planning)

| Trigger | Skill |
|---------|-------|
| Building UI components, pages, or web interfaces | `frontend-design:frontend-design` |
| Guided feature development with codebase analysis | `feature-dev:feature-dev` |
| Customizing keyboard shortcuts or keybindings | `keybindings-help` |
| Creating or editing skills themselves | `superpowers:writing-skills` |

### Skill Priority

1. **Process skills first** — `systematic-debugging`, `brainstorming`, `writing-plans`
2. **Implementation skills second** — `frontend-design`, `feature-dev`

`brainstorming` always leads to `writing-plans`, which leads to `subagent-driven-development` or `executing-plans`.

## Merging Upstream

The fork regularly merges from `upstream/release`. Custom changes are concentrated in a small number of files, making conflicts manageable. The biggest conflict risk is always `chat-completions.js` due to the extensive word replacement system woven through every provider handler.
