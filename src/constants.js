export const PUBLIC_DIRECTORIES = {
    images: 'public/img/',
    backups: 'backups/',
    sounds: 'public/sounds',
    extensions: 'public/scripts/extensions',
    globalExtensions: 'public/scripts/extensions/third-party',
};

export const SETTINGS_FILE = 'settings.json';

/**
 * @type {import('./users.js').UserDirectoryList}
 * @readonly
 * @enum {string}
 */
export const USER_DIRECTORY_TEMPLATE = Object.freeze({
    root: '',
    thumbnails: 'thumbnails',
    thumbnailsBg: 'thumbnails/bg',
    thumbnailsAvatar: 'thumbnails/avatar',
    thumbnailsPersona: 'thumbnails/persona',
    worlds: 'worlds',
    user: 'user',
    avatars: 'User Avatars',
    userImages: 'user/images',
    groups: 'groups',
    groupChats: 'group chats',
    chats: 'chats',
    characters: 'characters',
    backgrounds: 'backgrounds',
    novelAI_Settings: 'NovelAI Settings',
    koboldAI_Settings: 'KoboldAI Settings',
    openAI_Settings: 'OpenAI Settings',
    textGen_Settings: 'TextGen Settings',
    themes: 'themes',
    movingUI: 'movingUI',
    extensions: 'extensions',
    instruct: 'instruct',
    context: 'context',
    quickreplies: 'QuickReplies',
    assets: 'assets',
    comfyWorkflows: 'user/workflows',
    files: 'user/files',
    vectors: 'vectors',
    backups: 'backups',
    sysprompt: 'sysprompt',
    reasoning: 'reasoning',
});

/**
 * @type {import('./users.js').User}
 * @readonly
 */
export const DEFAULT_USER = Object.freeze({
    handle: 'default-user',
    name: 'User',
    created: Date.now(),
    password: '',
    admin: true,
    enabled: true,
    salt: '',
});

export const UNSAFE_EXTENSIONS = [
    '.php',
    '.exe',
    '.com',
    '.dll',
    '.pif',
    '.application',
    '.gadget',
    '.msi',
    '.jar',
    '.cmd',
    '.bat',
    '.reg',
    '.sh',
    '.py',
    '.js',
    '.jse',
    '.jsp',
    '.pdf',
    '.html',
    '.htm',
    '.hta',
    '.vb',
    '.vbs',
    '.vbe',
    '.cpl',
    '.msc',
    '.scr',
    '.sql',
    '.iso',
    '.img',
    '.dmg',
    '.ps1',
    '.ps1xml',
    '.ps2',
    '.ps2xml',
    '.psc1',
    '.psc2',
    '.msh',
    '.msh1',
    '.msh2',
    '.mshxml',
    '.msh1xml',
    '.msh2xml',
    '.scf',
    '.lnk',
    '.inf',
    '.reg',
    '.doc',
    '.docm',
    '.docx',
    '.dot',
    '.dotm',
    '.dotx',
    '.xls',
    '.xlsm',
    '.xlsx',
    '.xlt',
    '.xltm',
    '.xltx',
    '.xlam',
    '.ppt',
    '.pptm',
    '.pptx',
    '.pot',
    '.potm',
    '.potx',
    '.ppam',
    '.ppsx',
    '.ppsm',
    '.pps',
    '.ppam',
    '.sldx',
    '.sldm',
    '.ws',
];

export const GEMINI_SAFETY = [
    {
        category: 'HARM_CATEGORY_HARASSMENT',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_HATE_SPEECH',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_CIVIC_INTEGRITY',
        threshold: 'OFF',
    },
];

export const VERTEX_SAFETY = [
    {
        category: 'HARM_CATEGORY_IMAGE_HATE',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_IMAGE_HARASSMENT',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT',
        threshold: 'OFF',
    },
    {
        category: 'HARM_CATEGORY_JAILBREAK',
        threshold: 'OFF',
    },
];

export const CHAT_COMPLETION_SOURCES = {
    OPENAI: 'openai',
    CLAUDE: 'claude',
    OPENROUTER: 'openrouter',
    AI21: 'ai21',
    MAKERSUITE: 'makersuite',
    VERTEXAI: 'vertexai',
    MISTRALAI: 'mistralai',
    CUSTOM: 'custom',
    COHERE: 'cohere',
    PERPLEXITY: 'perplexity',
    GROQ: 'groq',
    CHUTES: 'chutes',
    ELECTRONHUB: 'electronhub',
    NANOGPT: 'nanogpt',
    DEEPSEEK: 'deepseek',
    AIMLAPI: 'aimlapi',
    XAI: 'xai',
    POLLINATIONS: 'pollinations',
    MOONSHOT: 'moonshot',
    FIREWORKS: 'fireworks',
    COMETAPI: 'cometapi',
    AZURE_OPENAI: 'azure_openai',
    ZAI: 'zai',
    SILICONFLOW: 'siliconflow',
    MINIMAX: 'minimax',
    WORKERS_AI: 'workers_ai',
};

/**
 * Path to multer file uploads under the data root.
 */
export const UPLOADS_DIRECTORY = '_uploads';

// TODO: this is copied from the client code; there should be a way to de-duplicate it eventually
export const TEXTGEN_TYPES = {
    OOBA: 'ooba',
    MANCER: 'mancer',
    VLLM: 'vllm',
    APHRODITE: 'aphrodite',
    TABBY: 'tabby',
    KOBOLDCPP: 'koboldcpp',
    TOGETHERAI: 'togetherai',
    LLAMACPP: 'llamacpp',
    OLLAMA: 'ollama',
    INFERMATICAI: 'infermaticai',
    DREAMGEN: 'dreamgen',
    OPENROUTER: 'openrouter',
    FEATHERLESS: 'featherless',
    HUGGINGFACE: 'huggingface',
    GENERIC: 'generic',
};

export const INFERMATICAI_KEYS = [
    'model',
    'prompt',
    'max_tokens',
    'temperature',
    'top_p',
    'top_k',
    'repetition_penalty',
    'stream',
    'stop',
    'presence_penalty',
    'frequency_penalty',
    'min_p',
    'seed',
    'ignore_eos',
    'n',
    'best_of',
    'min_tokens',
    'spaces_between_special_tokens',
    'skip_special_tokens',
    'logprobs',
];

export const FEATHERLESS_KEYS = [
    'model',
    'prompt',
    'best_of',
    'echo',
    'frequency_penalty',
    'logit_bias',
    'logprobs',
    'max_tokens',
    'n',
    'presence_penalty',
    'seed',
    'stop',
    'stream',
    'suffix',
    'temperature',
    'top_p',
    'user',

    'use_beam_search',
    'top_k',
    'min_p',
    'repetition_penalty',
    'length_penalty',
    'early_stopping',
    'stop_token_ids',
    'ignore_eos',
    'min_tokens',
    'skip_special_tokens',
    'spaces_between_special_tokens',
    'truncate_prompt_tokens',

    'include_stop_str_in_output',
    'response_format',
    'guided_json',
    'guided_regex',
    'guided_choice',
    'guided_grammar',
    'guided_decoding_backend',
    'guided_whitespace_pattern',
];

// https://docs.together.ai/reference/completions
export const TOGETHERAI_KEYS = [
    'model',
    'prompt',
    'max_tokens',
    'temperature',
    'top_p',
    'top_k',
    'repetition_penalty',
    'min_p',
    'presence_penalty',
    'frequency_penalty',
    'stream',
    'stop',
];

// https://github.com/ollama/ollama/blob/main/docs/api.md#request-8
export const OLLAMA_KEYS = [
    'num_predict',
    'num_ctx',
    'num_batch',
    'stop',
    'temperature',
    'repeat_penalty',
    'presence_penalty',
    'frequency_penalty',
    'top_k',
    'top_p',
    'tfs_z',
    'typical_p',
    'seed',
    'repeat_last_n',
    'min_p',
];

// https://platform.openai.com/docs/api-reference/completions
export const OPENAI_KEYS = [
    'model',
    'prompt',
    'stream',
    'temperature',
    'top_p',
    'frequency_penalty',
    'presence_penalty',
    'stop',
    'seed',
    'logit_bias',
    'logprobs',
    'max_tokens',
    'n',
    'best_of',
];

export const AVATAR_WIDTH = 512;
export const AVATAR_HEIGHT = 768;
export const DEFAULT_AVATAR_PATH = './public/img/ai4.png';

export const OPENROUTER_HEADERS = {
    'HTTP-Referer': 'https://sillytavern.app',
    'X-Title': 'SillyTavern',
};

export const AIMLAPI_HEADERS = {
    'HTTP-Referer': 'https://sillytavern.app',
    'X-Title': 'SillyTavern',
};

export const FEATHERLESS_HEADERS = {
    'HTTP-Referer': 'https://sillytavern.app',
    'X-Title': 'SillyTavern',
};

export const OPENROUTER_KEYS = [
    'max_tokens',
    'temperature',
    'top_k',
    'top_p',
    'presence_penalty',
    'frequency_penalty',
    'repetition_penalty',
    'min_p',
    'top_a',
    'seed',
    'logit_bias',
    'model',
    'stream',
    'prompt',
    'stop',
    'provider',
    'include_reasoning',
];

// https://github.com/vllm-project/vllm/blob/0f8a91401c89ac0a8018def3756829611b57727f/vllm/entrypoints/openai/protocol.py#L220
export const VLLM_KEYS = [
    'model',
    'prompt',
    'best_of',
    'echo',
    'frequency_penalty',
    'logit_bias',
    'logprobs',
    'max_tokens',
    'n',
    'presence_penalty',
    'seed',
    'stop',
    'stream',
    'suffix',
    'temperature',
    'top_p',
    'user',

    'use_beam_search',
    'top_k',
    'min_p',
    'repetition_penalty',
    'length_penalty',
    'early_stopping',
    'stop_token_ids',
    'ignore_eos',
    'min_tokens',
    'skip_special_tokens',
    'spaces_between_special_tokens',
    'truncate_prompt_tokens',

    'include_stop_str_in_output',
    'response_format',
    'guided_json',
    'guided_regex',
    'guided_choice',
    'guided_grammar',
    'guided_decoding_backend',
    'guided_whitespace_pattern',
];

export const AZURE_OPENAI_KEYS = [
    'messages',
    'temperature',
    'frequency_penalty',
    'presence_penalty',
    'top_p',
    'max_tokens',
    'max_completion_tokens',
    'stream',
    'logit_bias',
    'stop',
    'n',
    'logprobs',
    'seed',
    'tools',
    'tool_choice',
    'reasoning_effort',
];

export const OPENAI_VERBOSITY_MODELS = /^gpt-5[^chat]*/;

export const OPENAI_REASONING_EFFORT_MODELS = [
    'o1',
    'o3-mini',
    'o3-mini-2025-01-31',
    'o4-mini',
    'o4-mini-2025-04-16',
    'o3',
    'o3-2025-04-16',
    'gpt-5',
    'gpt-5-2025-08-07',
    'gpt-5-mini',
    'gpt-5-mini-2025-08-07',
    'gpt-5-nano',
    'gpt-5-nano-2025-08-07',
    'gpt-5.1',
    'gpt-5.1-2025-11-13',
    'gpt-5.1-chat-latest',
    'gpt-5.2',
    'gpt-5.2-2025-12-11',
    'gpt-5.2-chat-latest',
    'gpt-5.3-chat-latest',
    'gpt-5.4',
    'gpt-5.4-2026-03-05',
    'gpt-5.4-mini',
    'gpt-5.4-mini-2026-03-17',
    'gpt-5.4-nano',
    'gpt-5.4-nano-2026-03-17',
    'gpt-5.5',
    'gpt-5.5-2026-04-23',
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'gpt-6-astra',
];

export const OPENAI_REASONING_EFFORT_MAP = {
    none: 'none',
    auto: 'auto',
    min: 'minimal',
    low: 'low',
    medium: 'medium',
    high: 'high',
    xhigh: 'xhigh',
    max: 'xhigh',
};

/**
 * Models that only accept a single fixed reasoning effort value.
 * @type {Record<string, string>}
 */
export const OPENAI_FIXED_REASONING_EFFORT = {
    'gpt-5.3-chat-latest': 'medium',
};

/**
 * Models that support the Responses API `reasoning.mode` parameter (standard/pro).
 */
export const OPENAI_PRO_REASONING_MODELS = /^gpt-5\.6/;

/**
 * Models that support Responses API prompt cache controls (`prompt_cache_options`).
 * GPT-5.6 and later model families.
 */
export const OPENAI_PROMPT_CACHE_OPTIONS_MODELS = /^gpt-(?:5\.(?:[6-9]|\d{2,})|[6-9]|\d{2,})|gpt-6-astra/;

/**
 * Models that should use the OpenAI Responses API (/v1/responses) instead of Chat Completions.
 * These are reasoning models that benefit from the richer output format.
 */
export const OPENAI_RESPONSES_API_MODELS = [OPENAI_REASONING_EFFORT_MODELS, /gpt-5[^chat]*/];

export const NANOGPT_REASONING_EFFORT_MAP = {
    min: 'none',
    low: 'minimal',
    medium: 'low',
    high: 'medium',
    max: 'high',
};

/**
 * Ordered effort tiers as the UI presents them. Index position is what matters:
 * a model's `effortLevels` list is indexed by the tier the user picked, clamped
 * to the list length. That way Opus 4.6 (whose top tier is named "max", not
 * "xhigh") still gets its top tier when the user asks for Extra High.
 * @type {string[]}
 */
export const CLAUDE_EFFORT_TIERS = ['low', 'medium', 'high', 'xhigh', 'max'];

/**
 * Conservative baseline for any Claude model we don't recognise, including the
 * arbitrary names a reverse proxy can pass through. Assumes the OLDER, more
 * permissive request shape on purpose: sending `budget_tokens` to a model that
 * wanted adaptive thinking is a loud, recoverable 400, whereas assuming the
 * newer shape would silently strip the user's temperature on every custom model.
 */
const CLAUDE_DEFAULT_CAPABILITIES = {
    /** @type {'none'|'manual'|'adaptive'|'both'} Which `thinking` field shapes the API accepts. `both` = adaptive and the deprecated manual budget (4.6 gen). */
    thinkingMode: 'manual',
    /** Whether omitting `thinking` entirely still produces thinking (5-series). */
    thinkingDefaultOn: false,
    /** @type {'never'|'always'|'effort-capped'} */
    canDisableThinking: 'always',
    /** Highest effort that may accompany `thinking: {type:'disabled'}`, when effort-capped. */
    disableEffortCap: 'high',
    supportsPrefill: true,
    /** @type {'full'|'limited'|'none'} `limited` means temperature XOR top_p, never both. */
    samplingMode: 'limited',
    /** @type {string[]} API-accepted effort names, ascending. Empty = model has no effort param. */
    effortLevels: [],
    /** Whether `thinking.display` is accepted (and therefore required to see any thinking text). */
    thinkingDisplay: false,
    supportsTaskBudget: false,
    supportsWebSearch: false,
    contextWindow: 200000,
    maxOutput: 8192,
    /** 2576px long edge instead of 1568px. */
    highResImages: false,
};

/**
 * Per-model Claude API capabilities, matched most-specific-first.
 *
 * Encoded from Anthropic's model migration guide. Every Claude code path should
 * read this table rather than growing another inline regex — adding the next
 * model should be one entry here, not a dozen scattered edits.
 *
 * @type {{ pattern: RegExp, caps: Partial<typeof CLAUDE_DEFAULT_CAPABILITIES> }[]}
 */
export const CLAUDE_MODEL_CAPABILITIES = [
    {
        // Fable 5: adaptive thinking is unconditional. Both
        // `thinking: {type:'disabled'}` and manual budget_tokens return 400.
        pattern: /^claude-(fable)-5/,
        caps: {
            thinkingMode: 'adaptive',
            thinkingDefaultOn: true,
            canDisableThinking: 'never',
            supportsPrefill: false,
            samplingMode: 'none',
            effortLevels: ['low', 'medium', 'high', 'xhigh', 'max'],
            thinkingDisplay: true,
            supportsTaskBudget: true,
            supportsWebSearch: true,
            contextWindow: 1000000,
            maxOutput: 128000,
            highResImages: true,
        },
    },
    {
        // Opus 5: thinking may be disabled, but only at effort `high` or below.
        // Pairing `disabled` with xhigh/max is a 400, checked per request.
        pattern: /^claude-opus-5/,
        caps: {
            thinkingMode: 'adaptive',
            thinkingDefaultOn: true,
            canDisableThinking: 'effort-capped',
            disableEffortCap: 'high',
            supportsPrefill: false,
            samplingMode: 'none',
            effortLevels: ['low', 'medium', 'high', 'xhigh', 'max'],
            thinkingDisplay: true,
            supportsTaskBudget: true,
            supportsWebSearch: true,
            contextWindow: 1000000,
            maxOutput: 128000,
            highResImages: true,
        },
    },
    {
        // Sonnet 5: same family, but `disabled` is accepted at any effort level.
        pattern: /^claude-sonnet-5/,
        caps: {
            thinkingMode: 'adaptive',
            thinkingDefaultOn: true,
            canDisableThinking: 'always',
            supportsPrefill: false,
            samplingMode: 'none',
            effortLevels: ['low', 'medium', 'high', 'xhigh', 'max'],
            thinkingDisplay: true,
            supportsWebSearch: true,
            contextWindow: 1000000,
            maxOutput: 128000,
            highResImages: true,
        },
    },
    {
        // Opus 4.7 / 4.8: adaptive thinking, but OFF unless asked for. First
        // models to reject any non-default temperature/top_p/top_k outright.
        pattern: /^claude-opus-4-(7|8)/,
        caps: {
            thinkingMode: 'adaptive',
            thinkingDefaultOn: false,
            canDisableThinking: 'always',
            supportsPrefill: false,
            samplingMode: 'none',
            effortLevels: ['low', 'medium', 'high', 'xhigh', 'max'],
            thinkingDisplay: true,
            supportsTaskBudget: true,
            supportsWebSearch: true,
            contextWindow: 1000000,
            maxOutput: 128000,
            highResImages: true,
        },
    },
    {
        // 4.6: first adaptive generation, but the only one that still accepts
        // manual `enabled` + budget_tokens (deprecated, not rejected). Top effort
        // tier is named `max` here; `xhigh` did not exist yet. Still honours one
        // of temperature/top_p.
        pattern: /^claude-(opus|sonnet)-4-6/,
        caps: {
            thinkingMode: 'both',
            thinkingDefaultOn: false,
            canDisableThinking: 'always',
            supportsPrefill: false,
            samplingMode: 'limited',
            effortLevels: ['low', 'medium', 'high', 'max'],
            thinkingDisplay: true,
            supportsWebSearch: true,
            contextWindow: 1000000,
            maxOutput: 128000,
        },
    },
    {
        pattern: /^claude-sonnet-4-5/,
        caps: { supportsWebSearch: true, contextWindow: 1000000, maxOutput: 64000 },
    },
    {
        // Opus 4.5: manual thinking, but the only pre-4.6 model with the effort
        // parameter (sent alongside budget_tokens). No xhigh/max tiers.
        pattern: /^claude-opus-4-5/,
        caps: { effortLevels: ['low', 'medium', 'high'], supportsWebSearch: true, maxOutput: 64000 },
    },
    {
        pattern: /^claude-haiku-4-5/,
        caps: { supportsWebSearch: true, maxOutput: 64000 },
    },
    {
        pattern: /^claude-opus-4-1/,
        caps: { supportsWebSearch: true, maxOutput: 32000 },
    },
    {
        pattern: /^claude-opus-4/,
        caps: { supportsWebSearch: true, maxOutput: 32000 },
    },
    {
        pattern: /^claude-sonnet-4/,
        caps: { supportsWebSearch: true, maxOutput: 64000 },
    },
    {
        pattern: /^claude-3-7/,
        caps: { supportsWebSearch: true, maxOutput: 64000 },
    },
    {
        // Claude 3.x predates extended thinking entirely.
        pattern: /^claude-3-5/,
        caps: { thinkingMode: 'none', samplingMode: 'full', supportsWebSearch: true, maxOutput: 8192 },
    },
    {
        pattern: /^claude-3/,
        caps: { thinkingMode: 'none', samplingMode: 'full', maxOutput: 4096 },
    },
];

/**
 * Resolve the API capabilities of a Claude model name.
 * Unknown names (custom deployments, reverse proxy passthrough) fall back to the
 * conservative baseline rather than assuming the newest request shape.
 * @param {string} model Model identifier
 * @returns {typeof CLAUDE_DEFAULT_CAPABILITIES} Resolved capabilities
 */
export function getClaudeCapabilities(model) {
    const name = String(model ?? '');
    const match = CLAUDE_MODEL_CAPABILITIES.find(entry => entry.pattern.test(name));
    return { ...CLAUDE_DEFAULT_CAPABILITIES, ...(match?.caps ?? {}) };
}

/**
 * Map a UI effort tier onto the effort name a given model actually accepts.
 * Resolves positionally and clamps to the model's top tier, so a request for
 * Extra High on a model that stops at `max` still gets that model's ceiling.
 * @param {string} tier UI tier: low/medium/high/xhigh/max (or `min`, treated as low)
 * @param {typeof CLAUDE_DEFAULT_CAPABILITIES} caps Resolved model capabilities
 * @returns {string|null} Effort value for `output_config.effort`, or null for the API default
 */
export function resolveClaudeEffort(tier, caps) {
    const levels = caps?.effortLevels ?? [];
    if (!levels.length) {
        return null;
    }

    // `min` is a SillyTavern-only tier that predates the effort parameter.
    const index = CLAUDE_EFFORT_TIERS.indexOf(tier === 'min' || tier === 'minimal' ? 'low' : tier);
    if (index === -1) {
        return null;
    }

    return levels[Math.min(index, levels.length - 1)];
}

/**
 * Claude token prices in USD per million tokens, most-specific-first.
 * Base (non-batch) rates; the Message Batches API bills at 50% of these.
 * @type {{ pattern: RegExp, input: number, output: number, until?: string, then?: { input: number, output: number } }[]}
 */
export const CLAUDE_MODEL_PRICING = [
    { pattern: /^claude-(fable)-5/, input: 10, output: 50 },
    { pattern: /^claude-opus-5/, input: 5, output: 25 },
    // Sonnet 5 launched on introductory pricing that reverts on 2026-09-01.
    { pattern: /^claude-sonnet-5/, input: 2, output: 10, until: '2026-09-01', then: { input: 3, output: 15 } },
    { pattern: /^claude-opus-4-(5|6|7|8)/, input: 5, output: 25 },
    { pattern: /^claude-opus-4/, input: 15, output: 75 },
    { pattern: /^claude-sonnet-4/, input: 3, output: 15 },
    { pattern: /^claude-haiku-4-5/, input: 1, output: 5 },
    { pattern: /^claude-3-7/, input: 3, output: 15 },
    { pattern: /^claude-3-5-haiku/, input: 0.8, output: 4 },
    { pattern: /^claude-3-5/, input: 3, output: 15 },
    { pattern: /^claude-3-opus/, input: 15, output: 75 },
    { pattern: /^claude-3-haiku/, input: 0.25, output: 1.25 },
];

/**
 * Look up per-million token prices for a Claude model.
 * @param {string} model Model identifier
 * @param {Date} [now] Clock, for the introductory-pricing cutover
 * @returns {{ input: number, output: number }|null} Prices in USD per million tokens, or null if unknown
 */
export function getClaudePricing(model, now = new Date()) {
    const entry = CLAUDE_MODEL_PRICING.find(e => e.pattern.test(String(model ?? '')));
    if (!entry) {
        return null;
    }

    if (entry.until && entry.then && now >= new Date(entry.until)) {
        return { input: entry.then.input, output: entry.then.output };
    }

    return { input: entry.input, output: entry.output };
}

export const LOG_LEVELS = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
};

/**
 * An array of supported media file extensions.
 * This is used to validate file uploads and ensure that only supported media types are processed.
 */
export const MEDIA_EXTENSIONS = [
    'bmp',
    'png',
    'jpg',
    'webp',
    'jpeg',
    'jfif',
    'gif',
    'mp4',
    'avi',
    'mov',
    'wmv',
    'flv',
    'webm',
    '3gp',
    'mkv',
    'mpg',
    'mp3',
    'wav',
    'ogg',
    'flac',
    'aac',
    'm4a',
    'aiff',
];

/**
 * Bitwise flag-style media request types.
 */
export const MEDIA_REQUEST_TYPE = {
    IMAGE: 0b001,
    VIDEO: 0b010,
    AUDIO: 0b100,
};


export const ZAI_ENDPOINT = {
    COMMON: 'common',
    CODING: 'coding',
};

export const SILICONFLOW_ENDPOINT = {
    GLOBAL: 'global',
    CN: 'cn',
};

export const MINIMAX_ENDPOINT = {
    GLOBAL: 'global',
    CN: 'cn',
};
