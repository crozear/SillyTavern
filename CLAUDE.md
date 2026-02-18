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

## Running

```bash
npm install
npm start        # or use Start.bat
```

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

## Merging Upstream

The fork regularly merges from `upstream/release`. Custom changes are concentrated in a small number of files, making conflicts manageable. The biggest conflict risk is always `chat-completions.js` due to the extensive word replacement system woven through every provider handler.
