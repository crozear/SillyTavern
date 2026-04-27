/* eslint-disable dot-notation */
import process from 'node:process';
import util from 'node:util';
import { Transform } from 'node:stream';
import express from 'express';
import fetch from 'node-fetch';
import urlJoin from 'url-join';

import {
    AIMLAPI_HEADERS,
    AZURE_OPENAI_KEYS,
    CHAT_COMPLETION_SOURCES,
    GEMINI_SAFETY,
    NANOGPT_REASONING_EFFORT_MAP,
    OPENAI_FIXED_REASONING_EFFORT,
    OPENAI_REASONING_EFFORT_MAP,
    OPENAI_REASONING_EFFORT_MODELS,
    OPENAI_RESPONSES_API_MODELS,
    OPENAI_VERBOSITY_MODELS,
    OPENROUTER_HEADERS,
    VERTEX_SAFETY,
    SILICONFLOW_ENDPOINT,
    ZAI_ENDPOINT,
} from '../../constants.js';
import {
    getConfigValue,
    tryParse,
    uuidv4,
    mergeObjectWithYaml,
    excludeKeysByYaml,
    color,
    trimTrailingSlash,
    flattenSchema,
} from '../../util.js';
import {
    convertClaudeMessages,
    convertGooglePrompt,
    convertTextCompletionPrompt,
    convertCohereMessages,
    convertMistralMessages,
    convertAI21Messages,
    convertXAIMessages,
    cachingAtDepthForOpenRouterClaude,
    cachingAtDepthForClaude,
    getPromptNames,
    calculateClaudeBudgetTokens,
    getClaudeAdaptiveEffort,
    calculateGoogleBudgetTokens,
    postProcessPrompt,
    PROMPT_PROCESSING_TYPE,
    addAssistantPrefix,
    embedOpenRouterMedia,
    addReasoningContentToToolCalls,
    cachingSystemPromptForOpenRouter,
    addOpenRouterSignatures,
} from '../../prompt-converters.js';

import { readSecret, SECRET_KEYS } from '../secrets.js';
import {
    getTokenizerModel,
    getSentencepiceTokenizer,
    getTiktokenTokenizer,
    sentencepieceTokenizers,
    TEXT_COMPLETION_MODELS,
    webTokenizers,
    getWebTokenizer,
} from '../tokenizers.js';
import { getVertexAIAuth, getProjectIdFromServiceAccount } from '../google.js';

const API_OPENAI = 'https://api.openai.com/v1';
const API_CLAUDE = 'https://api.anthropic.com/v1';
const API_MISTRAL = 'https://api.mistral.ai/v1';
const API_COHERE_V1 = 'https://api.cohere.ai/v1';
const API_COHERE_V2 = 'https://api.cohere.ai/v2';
const API_PERPLEXITY = 'https://api.perplexity.ai';
const API_GROQ = 'https://api.groq.com/openai/v1';
const API_MAKERSUITE = 'https://generativelanguage.googleapis.com';
const API_VERTEX_AI = 'https://us-central1-aiplatform.googleapis.com';
const API_AI21 = 'https://api.ai21.com/studio/v1';
const API_CHUTES = 'https://llm.chutes.ai/v1';
const API_ELECTRONHUB = 'https://api.electronhub.ai/v1';
const API_NANOGPT = 'https://nano-gpt.com/api/v1';
const API_DEEPSEEK = 'https://api.deepseek.com/beta';
const API_XAI = 'https://api.x.ai/v1';
const API_AIMLAPI = 'https://api.aimlapi.com/v1';
const API_POLLINATIONS = 'https://gen.pollinations.ai/v1';
const API_MOONSHOT = 'https://api.moonshot.ai/v1';
const API_FIREWORKS = 'https://api.fireworks.ai/inference/v1';
const API_COMETAPI = 'https://api.cometapi.com/v1';
const API_ZAI_COMMON = 'https://api.z.ai/api/paas/v4';
const API_ZAI_CODING = 'https://api.z.ai/api/coding/paas/v4';
const API_SILICONFLOW = 'https://api.siliconflow.com/v1';
const API_SILICONFLOW_CN = 'https://api.siliconflow.cn/v1';
const API_OPENROUTER = 'https://openrouter.ai/api/v1';

/**
 * Module-scoped Claude caching configuration values.
 */
const configExtendedTTL = getConfigValue('claude.extendedTTL', false, 'boolean');
function getCacheTTL(request) {
    const extendedTTL = request?.body?.claude_extendedTTL || configExtendedTTL;
    return extendedTTL ? '1h' : '5m';
}
const enableSystemPromptCache = getConfigValue('claude.enableSystemPromptCache', false, 'boolean');
const cachingAtDepth = (() => {
    const value = getConfigValue('claude.cachingAtDepth', -1, 'number');
    return Number.isInteger(value) && value >= 0 ? value : -1;
})();
const enableAdaptiveThinking = getConfigValue('claude.enableAdaptiveThinking', true, 'boolean');

/**
 * Cache for cacheable (writing) OpenRouter model IDs.
 * @type {string[]}
 */
const openRouterCacheableModels = [];

/**
 * Checks if an OpenRouter model supports prompt cache writing.
 * Uses a cache to avoid repeated API calls.
 * @param {string} modelId - The OpenRouter model ID
 * @returns {Promise<boolean>} `true` if the model supports writing cache
 */
async function isOpenRouterModelCacheable(modelId) {
    if (openRouterCacheableModels.includes(modelId)) {
        return true;
    }

    try {
        const response = await fetch(`${API_OPENROUTER}/models`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
            signal: AbortSignal.timeout(5000),
        });

        if (!response.ok) {
            console.warn(`OpenRouter models API returned ${response.status}: ${response.statusText}`);
            return false;
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            console.warn('OpenRouter API response format unexpected');
            return false;
        }

        const model = data.data.find(m => m.id === modelId);
        const supportsCache = model?.pricing?.input_cache_write != null;

        if (supportsCache) {
            openRouterCacheableModels.push(modelId);
        }

        return supportsCache;
    } catch (error) {
        console.warn(`Failed to check OpenRouter cache support for ${modelId}:`, error.message);
        return false;
    }
}

const WORD_REPLACEMENT_CONFIG = {
    sourceGroups: {

        // state-of-arousal words bound to pronouns — only clearly emotional/abstract terms
        // (heat/warmth/arousal removed: too physical, "her warmth give way" ≠ "her neediness give way")
        yourSens: [
            'your desire',
            'your need',
            'your want',
            'your excitement',
        ],
        mySens: [
            'my desire',
            'my need',
            'my want',
            'my excitement',
        ],
        herSens: [
            'her desire',
            'her need',
            'her want',
            'her excitement',
        ],
        hisSens: [
            'his desire',
            'his need',
            'his want',
            'his excitement',
        ],
        theirSens: [
            'their desire',
            'their need',
            'their want',
            'their excitement',
        ],

        withSens: [
            'with desire',
            'with need',
            'with want',
            'with excitement',
        ],

        // “chest (if female)” – only in clearly gendered phrasing so you don’t
        // turn “knife in his chest” into “knife in his tits”
        chestFemale: ['her chest', 'her bare chest'],
        herSexNoun: ['her sex', 'her opening', 'her core', 'her center', 'her flower', 'her folds', 'her passage', 'her depths', 'her womanhood', 'her entrance', 'her arousal', 'her femininity'],
        // your/my/their: only unambiguously female terms (sex/core/center removed — gender-ambiguous false positives)
        yourSexNoun: ['your flower', 'your folds', 'your depths', 'your womanhood', 'your entrance', 'your femininity'],
        mySexNoun: ['my flower', 'my folds', 'my depths', 'my womanhood', 'my entrance', 'my femininity'],
        theirSexNoun: ['their flower', 'their folds', 'their depths', 'their womanhood', 'their entrance', 'their femininity'],

        hisMember: ['his member', 'his manhood', 'his hardness', 'his length', 'his arousal', 'his girth'],
        myMember: ['my member', 'my manhood', 'my hardness', 'my length', 'my girth'],
        yourMember: ['your member', 'your manhood', 'your hardness', 'your length', 'your girth'],
        theirMember: ['their member', 'their manhood', 'their length', 'their hardness', 'their girth'],
        loveMuscle: ['love muscle', 'tumescence'],

        herBreasts: ['her breasts', 'her breast', 'her bosom', 'her bust'],
        yourBreasts: ['your breasts', 'your breast', 'your bosom', 'your bust'],
        myBreasts: ['my breasts', 'my breast', 'my bosom', 'my bust'],

        hisPeaks: ['his peaks'],
        herPeaks: ['her peaks'],
        myPeaks: ['my peaks'],
        theirPeaks: ['their peaks'],
        yourPeaks: ['your peaks'],

        hisPeak: ['his peak'],
        herPeak: ['her peak'],
        myPeak: ['my peak'],
        theirPeak: ['their peak'],
        yourPeak: ['your peak'],

        hisRelease: ['his release'],
        herRelease: ['her release'],
        myRelease: ['my release'],
        theirRelease: ['their release'],
        yourRelease: ['your release'],

        hisMound: ['his mound'],
        herMound: ['her mound'],
        myMound: ['my mound'],
        theirMound: ['their mound'],
        yourMound: ['your mound'],

        yourLength: ['your length'],
        sensations:    ['tingle', 'flutter', 'pulse'],
        sensationsS:   ['tingles', 'flutters', 'pulses'],
        sensationsEd:  ['tingled', 'fluttered', 'pulsed'],
        sensationsIng: ['tingling', 'fluttering', 'pulsing'],
        // "ache" only in sexual-context phrases to avoid false positives ("my head aches")
        acheFor:    ['ache for'],
        achesFor:   ['aches for'],
        achedFor:   ['ached for'],
        achingFor:  ['aching for'],
        actionsPenetration:    ['insert', 'penetrate'],
        actionsPenetrationS:   ['inserts', 'penetrates'],
        actionsPenetrationEd:  ['inserted', 'penetrated'],
        actionsPenetrationIng: ['inserting', 'penetrating'],
        actionsStimulation:    ['stimulate', 'arouse'],
        actionsStimulationS:   ['stimulates', 'arouses'],
        actionsStimulationEd:  ['stimulated', 'aroused'],
        actionsStimulationIng: ['stimulating', 'arousing'],
        becameAroused: ['became aroused', 'grew aroused', 'felt aroused'],
        gettingAroused: ['becoming aroused', 'growing aroused', 'getting aroused', 'feeling aroused'],

        phallus: ['phallus', 'shaft', 'penis', 'hardness'],
        phallusTip: ['glans'],
        urethra: ['urethra'],
        clitoris: ['clitoris', 'nub', 'bundle of nerves', 'bud'],
        inner: ['labia minora','secret place', 'most intimate place'],
        outer: ['labia majora', 'labia', 'petals'],
        innerEntrance: ['vaginal walls', 'vaginal canal', 'inner walls'],
        entrance: ['entrance'],
        pubicHair: ['pubic hair'],
        pubicArea: ['pubic', 'crotch', 'pelvis', 'perineum'],
        butt: ['buttocks', 'butt', 'hindquarters', 'rear'],
        buttHole: ['butthole'],
        testicles: ['testicles'],
        vagina: ['vagina'],
        member: ['member', 'manhood'],
        erection: ['erection', 'stiffness'],
        scrotum: ['scrotum'],
        preCum: ['pre-ejaculate', 'pre-seminal fluid'],
        cleavage: ['cleavage'],
        wetness: ['slicked'],
        slicked: ['wetness', 'slickness'],
        slickWith: ['slick with'],
        slickIn: ['slick in', 'slick inside'],
        slicking: ['slicking'],
        ofSlick: ['of slick'],
        slick: ['slick'],
        femCum: ['vaginal lubrication'],
        cum: ['semen', 'seed'],
        fluidVague: ['discharge', 'secretions', 'nectar'],
        iOrgasm: ['I orgasm', 'I climax', 'I ejaculate'],
        orgasmIng: ['ejaculating', 'climaxing', 'orgasming'],
        orgasmEd: ['orgasmed', 'climaxed', 'ejaculated'],
        // now also catch “orgasm” as a noun directly
        orgasm: ['climax', 'ejaculate', 'orgasm', 'orgasms'],

    },
    replacementGroups: {

        // pronoun-specific arousal replacements so grammar stays clean
        yourSens: ['your horniness', 'your lewdness'],
        mySens: ['my horniness', 'my lewdness'],
        herSens: ['her horniness', 'her lewdness'],
        hisSens: ['his horniness', 'his lewdness'],
        theirSens: ['their horniness', 'their lewdness'],
        withSens: ['with horniness', 'with lewdness', 'with lust'],

        // female chest → tits/breasts/nipples, keeping the pronoun
        chestFemale: ['her tits', 'her breasts', 'her nipples', 'her nips'],

        herSexNoun: ['her pussy', 'her cunt', 'her slit'],
        yourSexNoun: ['your pussy', 'your cunt', 'your slit'],
        mySexNoun: ['my pussy', 'my cunt', 'my slit'],
        theirSexNoun: ['their pussy', 'their cunt', 'their slit'],

        hisMember: ['his cock', 'his dick'],
        myMember: ['my cock', 'my dick'],
        yourMember: ['your cock', 'your dick'],
        theirMember: ['their cock', 'their dick'],
        loveMuscle: ['cock', 'dick'],

        herBreasts: ['her tits', 'her titties', 'her boobs'],
        yourBreasts: ['your tits', 'your titties', 'your boobs'],
        myBreasts: ['my tits', 'my titties', 'my boobs'],

        hisPeak: ['his limit and cum'],
        herPeak: ['her limit and squirt', 'her limit and cum'],
        myPeak: ['my limit and cum'],
        theirPeak: ['their limit and cum'],
        yourPeak: ['your limit and cum'],

        hisRelease: ['him cumming'],
        herRelease: ['her squirting'],
        myRelease: ['me cumming'],
        theirRelease: ['them cumming'],
        yourRelease: ['you cumming'],

        hisMound: ['around his cock', 'around his dick'],
        herMound: ['around her pussy', 'around her cunt'],
        myMound: ['around my pussy', 'around my cunt'],
        theirMound: ['around their pussy', 'around their cunt'],
        yourMound: ['around your pussy', 'around your cunt'],

        hisPeaks: ['his nipples', 'his nips'],
        herPeaks: ['her nipples', 'her nips'],
        myPeaks: ['my nipples', 'my nips'],
        theirPeaks: ['their nipples', 'their nips'],
        yourPeaks: ['your nipples', 'your nips'],

        yourLength: ['your cock', 'your dick'],
        sensations:    ['throb', 'quiver', 'swell', 'clench', 'burn', 'sting'],
        sensationsS:   ['throbs', 'quivers', 'swells', 'clenches', 'burns', 'stings'],
        sensationsEd:  ['throbbed', 'quivered', 'swelled', 'clenched', 'burned', 'stung'],
        sensationsIng: ['throbbing', 'quivering', 'swelling', 'clenching', 'burning', 'stinging'],
        acheFor:    ['throb for', 'burn for', 'clench for'],
        achesFor:   ['throbs for', 'burns for', 'clenches for'],
        achedFor:   ['throbbed for', 'burned for', 'clenched for'],
        achingFor:  ['throbbing for', 'burning for', 'clenching for'],
        actionsPenetration:    ['fuck', 'hammer', 'pound', 'pump', 'thrust', 'slam', 'ram', 'drive', 'bury', 'hilt'],
        actionsPenetrationS:   ['fucks', 'hammers', 'pounds', 'pumps', 'thrusts', 'slams', 'rams', 'drives', 'buries', 'hilts'],
        actionsPenetrationEd:  ['fucked', 'hammered', 'pounded', 'pumped', 'thrust', 'slammed', 'rammed', 'drove', 'buried', 'hilted'],
        actionsPenetrationIng: ['fucking', 'hammering', 'pounding', 'pumping', 'thrusting', 'slamming', 'ramming', 'driving', 'burying', 'hilting'],
        actionsStimulation:    ['grind', 'suck', 'ravage', 'rub', 'stroke', 'tug', 'squeeze', 'lap', 'lick', 'swirl', 'milk'],
        actionsStimulationS:   ['grinds', 'sucks', 'ravages', 'rubs', 'strokes', 'tugs', 'squeezes', 'laps', 'licks', 'swirls', 'milks'],
        actionsStimulationEd:  ['ground', 'sucked', 'ravaged', 'rubbed', 'stroked', 'tugged', 'squeezed', 'lapped', 'licked', 'swirled', 'milked'],
        actionsStimulationIng: ['grinding', 'sucking', 'ravaging', 'rubbing', 'stroking', 'tugging', 'squeezing', 'lapping', 'licking', 'swirling', 'milking'],
        becameAroused: ['got horny', 'got turned on', 'got hot'],
        gettingAroused: ['getting horny', 'getting turned on', 'getting hot'],

        phallus: ['cock', 'dick', 'shaft'],
        phallusTip: ['cock tip', 'dick tip', 'tip', 'cockhead', 'head', 'dick ridge'],
        urethra: ['piss slit', 'piss hole'],
        clitoris: ['clit'],
        inner: ['pussy slit', 'cunt slit', 'pussy', 'cunt', 'inner pussy lips'],
        outer: ['outer pussy lips', 'pussy lips', 'cunt lips', 'pussy', 'cunt'],
        innerEntrance: ['pussy walls', 'pussy tunnel', 'cunt walls', 'cunt tunnels', 'inner pussy'],
        entrance: ['hole', 'slit', 'opening'],
        pubicHair: ['pubes'],
        pubicArea: ['sexy goods'],
        butt: ['ass', 'cheeks', 'ass cheeks'],
        buttHole: ['asshole', 'rim', 'ring'],
        testicles: ['balls', 'nuts'],
        vagina: ['pussy', 'cunt'],
        member: ['cock', 'dick'],
        erection: ['boner', 'hard-on', 'stiffy'],
        scrotum: ['sack', 'nutsack', 'ball sack'],
        preCum: ['pre-cum', 'cock drool'],
        cleavage: ['tits', 'breasts', 'titties', 'boobs'],
        wetness: ['soaked', 'dripping', 'gushing', 'coated'],
        slicked: ['juiciness', 'slickness', 'slipperiness'],
        slickWith: ['drenched with', 'soaked with', 'dripping with'],
        slickIn: ['coated inside', 'soaked inside', 'covered inside'],
        slickIng: ['coating', 'covering', 'lubing', 'greasing', 'sliming'],
        ofSlick: ['of slimy juices', 'of musky fluids', 'of shiny juice'],
        slick: ['slimy', 'slippery', 'glossy', 'shiny', 'glistening', 'wet'],
        femCum: ['femcum'],
        cum: ['cum'],
        fluidVague: ['juices'],
        iOrgasm: ['I cum'],
        orgasmIng: ['cumming'],
        orgasmEd: ['came'],
        orgasm: ['cum'],
    },
};

function areWordReplacementsEnabled(requestValue) {
    const configEnabled = getConfigValue('wordReplacement.enabled', true, 'boolean');
    return typeof requestValue === 'boolean' ? requestValue : configEnabled;
}

function resolveWordReplacementEnabledOverride(request) {
    return request?.body?.word_replacement_enabled;
}

function getWordReplacementEnabled(request) {
    return areWordReplacementsEnabled(resolveWordReplacementEnabledOverride(request));
}

let cachedReplacementRules = null;
let longestReplacementSourceLength = 0;
const replacementCycles = new Map();

function applyWordReplacements(text, enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    if (!text || typeof text !== 'string') {
        return text;
    }

    if (!isEnabled) {
        return text;
    }

    if (!cachedReplacementRules) {
        cachedReplacementRules = buildReplacementRules(WORD_REPLACEMENT_CONFIG);
        longestReplacementSourceLength = cachedReplacementRules.reduce((max, rule) => Math.max(max, rule.maxLength), 0);
    }

    return cachedReplacementRules.reduce((current, rule) => {
        return current.replace(rule.pattern, (match) => {
            const nextValue = pickReplacement(rule.group, WORD_REPLACEMENT_CONFIG);
            if (!nextValue) return match;
            return applyCaseToReplacement(match, nextValue);
        });
    }, text);
}

function buildReplacementRules(config) {
    const rules = [];
    Object.entries(config.sourceGroups || {}).forEach(([group, words]) => {
        const replacements = getReplacementPool(group, config);
        if (!Array.isArray(words) || !words.length || !Array.isArray(replacements) || !replacements.length) {
            return;
        }

        const terms = words
            .flatMap(word => String(word || '').split('/'))
            .map(word => word.trim())
            .filter(Boolean)
            .sort((a, b) => b.length - a.length)
            .map(escapeForRegex);

        if (!terms.length) return;

        rules.push({
            group,
            maxLength: terms[0]?.length || 0,
            pattern: new RegExp(`\\b(?:${terms.join('|')})\\b`, 'gi'),
        });
    });

    return rules.sort((a, b) => b.maxLength - a.maxLength);
}

function pickReplacement(group, config) {
    const pool = getReplacementPool(group, config);
    if (!Array.isArray(pool) || !pool.length) {
        return null;
    }

    if (!replacementCycles.has(group)) {
        replacementCycles.set(group, { pool: [...pool], remaining: [...pool] });
    }

    const cycle = replacementCycles.get(group);
    if (!cycle.remaining.length) {
        cycle.remaining = [...cycle.pool];
    }

    const index = Math.floor(Math.random() * cycle.remaining.length);
    const [choice] = cycle.remaining.splice(index, 1);
    return choice;
}

function escapeForRegex(term) {
    return term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+');
}

function applyCaseToReplacement(source, replacement) {
    if (!replacement) return source;

    const isAllCaps = source.toUpperCase() === source;
    const isCapitalized = source[0] === source[0]?.toUpperCase();

    if (isAllCaps) return replacement.toUpperCase();
    if (isCapitalized) return replacement[0].toUpperCase() + replacement.slice(1);
    return replacement;
}

function getReplacementPool(group, config) {
    return Array.isArray(config.replacementGroups?.[group])
        ? config.replacementGroups[group]
        : [];
}

function enforceWordReplacementsOnResponse(response, enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    if (!isEnabled) {
        return response;
    }

    if (!response || typeof response !== 'object') {
        return applyWordReplacements(response, isEnabled);
    }

    if (Array.isArray(response.choices)) {
        response.choices.forEach((choice) => {
            if (typeof choice?.text === 'string') {
                choice.text = applyWordReplacements(choice.text, isEnabled);
            }
            enforceWordReplacementsOnMessage(choice?.message, isEnabled);
            enforceWordReplacementsOnMessage(choice?.delta, isEnabled);
        });
    }

    if (Array.isArray(response.output_text)) {
        response.output_text = response.output_text.map(segment =>
            typeof segment === 'string' ? applyWordReplacements(segment, isEnabled) : segment,
        );
    }

    if (Array.isArray(response.output)) {
        response.output.forEach((item) => {
            if (Array.isArray(item?.content)) {
                item.content = item.content.map(block => enforceWordReplacementsOnContentBlock(block, isEnabled));
            } else if (typeof item?.content === 'string') {
                item.content = applyWordReplacements(item.content, isEnabled);
            }
            // Handle reasoning summary text in Responses API output
            if (Array.isArray(item?.summary)) {
                item.summary.forEach((summaryBlock) => {
                    if (typeof summaryBlock?.text === 'string') {
                        summaryBlock.text = applyWordReplacements(summaryBlock.text, isEnabled);
                    }
                });
            }
        });
    }

    if (response?.response_metadata?.raw_response) {
        response.response_metadata.raw_response = enforceWordReplacementsOnResponse(
            response.response_metadata.raw_response,
            isEnabled,
        );
    }

    return applyWordReplacementsDeep(response, new WeakSet(), isEnabled);
}

function enforceWordReplacementsOnMessage(message, enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    if (!isEnabled) {
        return;
    }

    if (!message || typeof message !== 'object') {
        return;
    }

    if (typeof message.content === 'string') {
        message.content = applyWordReplacements(message.content, isEnabled);
    } else if (Array.isArray(message.content)) {
        message.content = message.content.map(block => enforceWordReplacementsOnContentBlock(block, isEnabled));
    }

    if (typeof message.reasoning_content === 'string') {
        message.reasoning_content = applyWordReplacements(message.reasoning_content, isEnabled);
    } else if (Array.isArray(message.reasoning_content)) {
        message.reasoning_content = message.reasoning_content.map(block => enforceWordReplacementsOnContentBlock(block, isEnabled));
    }
}

function enforceWordReplacementsOnContentBlock(block, enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    if (!isEnabled) {
        return block;
    }

    if (block === null || block === undefined) {
        return block;
    }

    if (typeof block === 'string') {
        return applyWordReplacements(block, isEnabled);
    }

    if (typeof block !== 'object') {
        return block;
    }

    if (typeof block.text === 'string') {
        block.text = applyWordReplacements(block.text, isEnabled);
    }

    if (typeof block.refusal === 'string') {
        block.refusal = applyWordReplacements(block.refusal, isEnabled);
    }

    if (Array.isArray(block.content)) {
        block.content = block.content.map(nextBlock => enforceWordReplacementsOnContentBlock(nextBlock, isEnabled));
    }

    return block;
}

function applyWordReplacementsDeep(value, seen = new WeakSet(), enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    if (!isEnabled) {
        return value;
    }

    if (typeof value === 'string') {
        return applyWordReplacements(value, isEnabled);
    }

    if (value === null || typeof value !== 'object') {
        return value;
    }

    if (seen.has(value)) {
        return value;
    }
    seen.add(value);

    if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i += 1) {
            value[i] = applyWordReplacementsDeep(value[i], seen, isEnabled);
        }
        return value;
    }

    Object.keys(value).forEach((key) => {
        value[key] = applyWordReplacementsDeep(value[key], seen, isEnabled);
    });

    return value;
}

function createWordReplacementStream(enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    if (!isEnabled) {
        return new Transform({
            transform(chunk, _encoding, callback) {
                callback(null, chunk);
            },
        });
    }

    // Ensure replacement rules are initialized so longestReplacementSourceLength is valid
    if (!cachedReplacementRules) {
        cachedReplacementRules = buildReplacementRules(WORD_REPLACEMENT_CONFIG);
        longestReplacementSourceLength = cachedReplacementRules.reduce(
            (max, rule) => Math.max(max, rule.maxLength),
            0,
        );
    }

    const carryLimit = Math.max((longestReplacementSourceLength || 0) - 1, 64);
    const decoder = new TextDecoder();

    // Track per-choice, per-field buffers so phrases split across events still match
    const fieldBuffers = new Map();
    let pending = '';

    function bufferKey(choiceIndex, field) {
        return `${choiceIndex}:${field}`;
    }

    function isWordChar(ch) {
        return /\w/.test(ch); // same notion of "word" as \b in your regexes
    }

    // Cache multi-word source phrases so we don't cut through them
    let multiWordPhrases = null;
    let maxPhraseLength = 0;

    function initMultiWordPhrases() {
        if (multiWordPhrases) return;

        multiWordPhrases = [];
        maxPhraseLength = 0;

        const groups = WORD_REPLACEMENT_CONFIG?.sourceGroups || {};
        Object.values(groups).forEach((words) => {
            if (!Array.isArray(words)) return;
            words.forEach((w) => {
                const term = String(w || '').trim();
                // Only care about phrases with whitespace (multi-word)
                if (term && /\s/.test(term)) {
                    multiWordPhrases.push(term);
                    if (term.length > maxPhraseLength) {
                        maxPhraseLength = term.length;
                    }
                }
            });
        });
    }

    // Returns true if the cut index is *inside* any multi-word source phrase
    function inMiddleOfSourcePhrase(str, cutIndex) {
        initMultiWordPhrases();
        if (!multiWordPhrases.length) return false;

        const start = Math.max(0, cutIndex - maxPhraseLength);
        const end = Math.min(str.length, cutIndex + maxPhraseLength);
        const region = str.slice(start, end);

        for (const phrase of multiWordPhrases) {
            const rel = region.toLowerCase().indexOf(phrase);
            if (rel === -1) continue;

            const phraseStart = start + rel;
            const phraseEnd = phraseStart + phrase.length;

            // cutIndex is between characters, so forbid cuts strictly inside the phrase
            if (phraseStart < cutIndex && cutIndex < phraseEnd) {
                return true;
            }
        }

        return false;
    }

    function appendAndExtract(choiceIndex, field, text = '', flush = false) {
        const key = bufferKey(choiceIndex, field);
        const current = fieldBuffers.get(key) || '';
        let combined = current + (text || '');

        if (flush) {
            fieldBuffers.delete(key);
            return combined ? applyWordReplacements(combined, isEnabled) : '';
        }

        if (!combined) {
            return '';
        }

        // If we haven't exceeded the carry limit yet, just buffer everything.
        if (combined.length <= carryLimit) {
            fieldBuffers.set(key, combined);
            return '';
        }

        // Start by cutting so that we keep the last `carryLimit` chars.
        let cut = combined.length - carryLimit;

        // Move `cut` left until it's at a safe boundary:
        // - not in the middle of a word
        // - not in the middle of a multi-word replacement phrase
        while (
            cut > 0 &&
            (
                (isWordChar(combined[cut - 1]) && isWordChar(combined[cut])) ||
                inMiddleOfSourcePhrase(combined, cut)
            )
        ) {
            cut--;
        }

        // If we couldn't find a safe cut position, keep everything for next time.
        if (cut <= 0) {
            fieldBuffers.set(key, combined);
            return '';
        }

        const head = combined.slice(0, cut);
        const tail = combined.slice(cut);

        fieldBuffers.set(key, tail);

        return applyWordReplacements(head, isEnabled);
    }

    function flushChoice(choiceIndex) {
        return {
            content: appendAndExtract(choiceIndex, 'content', '', true),
            reasoning_content: appendAndExtract(choiceIndex, 'reasoning_content', '', true),
            text: appendAndExtract(choiceIndex, 'text', '', true),
        };
    }

    function flushAll(push) {
        const keys = Array.from(fieldBuffers.keys());
        const byChoice = new Map();

        keys.forEach((key) => {
            const [choiceIndex, field] = key.split(':');
            const idx = Number(choiceIndex) || 0;
            const flushed = appendAndExtract(idx, field, '', true);
            if (!flushed) return;
            if (!byChoice.has(idx)) {
                byChoice.set(idx, {});
            }
            byChoice.get(idx)[field] = flushed;
        });

        byChoice.forEach((fields, idx) => {
            const choice = { index: idx, delta: {} };
            if (fields.content) choice.delta.content = fields.content;
            if (fields.reasoning_content) choice.delta.reasoning_content = fields.reasoning_content;
            if (fields.text) choice.text = fields.text;
            push(`data: ${JSON.stringify({ choices: [choice] })}\n\n`);
        });
    }

    function processChoice(choice) {
        const idx = Number.isInteger(choice?.index) ? choice.index : 0;
        choice.delta = choice.delta || {};

        if (typeof choice.text === 'string') {
            const emitted = appendAndExtract(idx, 'text', choice.text, false);
            if (emitted) {
                choice.text = emitted;
            } else {
                delete choice.text;
            }
        }

        if (typeof choice.delta.content === 'string') {
            const emitted = appendAndExtract(idx, 'content', choice.delta.content, false);
            if (emitted) {
                choice.delta.content = emitted;
            } else {
                delete choice.delta.content;
            }
        } else if (Array.isArray(choice.delta.content)) {
            const updated = [];
            choice.delta.content.forEach((block) => {
                if (typeof block === 'string') {
                    const emitted = appendAndExtract(idx, 'content', block, false);
                    if (emitted) updated.push(emitted);
                    return;
                }
                if (block?.type === 'text' && typeof block.text === 'string') {
                    const emitted = appendAndExtract(idx, 'content', block.text, false);
                    if (emitted) {
                        updated.push({ ...block, text: emitted });
                    }
                    return;
                }
                updated.push(block);
            });

            if (updated.length) {
                choice.delta.content = updated;
            } else {
                delete choice.delta.content;
            }
        }

        if (typeof choice.delta.reasoning_content === 'string') {
            const emitted = appendAndExtract(idx, 'reasoning_content', choice.delta.reasoning_content, false);
            if (emitted) {
                choice.delta.reasoning_content = emitted;
            } else {
                delete choice.delta.reasoning_content;
            }
        } else if (Array.isArray(choice.delta.reasoning_content)) {
            const updated = [];
            choice.delta.reasoning_content.forEach((block) => {
                if (typeof block === 'string') {
                    const emitted = appendAndExtract(idx, 'reasoning_content', block, false);
                    if (emitted) updated.push(emitted);
                    return;
                }
                if (block?.type === 'text' && typeof block.text === 'string') {
                    const emitted = appendAndExtract(idx, 'reasoning_content', block.text, false);
                    if (emitted) {
                        updated.push({ ...block, text: emitted });
                    }
                    return;
                }
                updated.push(block);
            });

            if (updated.length) {
                choice.delta.reasoning_content = updated;
            } else {
                delete choice.delta.reasoning_content;
            }
        }

        // When the stream signals completion, flush any buffered tail for this choice
        if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
            const extras = flushChoice(idx);
            if (extras.content) {
                if (typeof choice.delta.content === 'string' || choice.delta.content === undefined) {
                    choice.delta.content = (choice.delta.content || '') + extras.content;
                } else if (Array.isArray(choice.delta.content)) {
                    choice.delta.content.push(extras.content);
                }
            }
            if (extras.reasoning_content) {
                if (typeof choice.delta.reasoning_content === 'string' || choice.delta.reasoning_content === undefined) {
                    choice.delta.reasoning_content = (choice.delta.reasoning_content || '') + extras.reasoning_content;
                } else if (Array.isArray(choice.delta.reasoning_content)) {
                    choice.delta.reasoning_content.push(extras.reasoning_content);
                }
            }
            if (extras.text) {
                choice.text = (choice.text || '') + extras.text;
            }
        }
    }

    const transformStream = new Transform({
        transform(chunk, _encoding, callback) {
            try {
                pending += decoder.decode(chunk, { stream: true });
                let eventEnd;
                while ((eventEnd = pending.indexOf('\n\n')) !== -1) {
                    const eventChunk = pending.slice(0, eventEnd);
                    pending = pending.slice(eventEnd + 2);
                    processEvent.call(this, eventChunk);
                }
                callback();
            } catch (error) {
                callback(error);
            }
        },
        flush(callback) {
            try {
                const remaining = pending + decoder.decode();
                if (remaining) {
                    processEvent.call(this, remaining);
                }
                flushAll((event) => this.push(event));
                callback();
            } catch (error) {
                callback(error);
            }
        },
    });

    function processEvent(eventChunk) {
        const trimmed = eventChunk.trimEnd();
        const lines = trimmed.split('\n');
        const dataLineIndex = lines.findIndex(line => line.startsWith('data:'));

        if (dataLineIndex === -1) {
            this.push(`${eventChunk}\n\n`);
            return;
        }

        const payload = lines[dataLineIndex].replace(/^data:\s*/, '').trim();

        if (payload === '[DONE]') {
            flushAll((event) => this.push(event));
            this.push('data: [DONE]\n\n');
            return;
        }

        let data;
        try {
            data = JSON.parse(payload);
        } catch (error) {
            this.push(`${eventChunk}\n\n`);
            return;
        }

        if (Array.isArray(data?.choices)) {
            data.choices.forEach(processChoice);
        }

        // Handle OpenAI Responses API streaming events
        if (typeof data?.type === 'string' && data.type.startsWith('response.')) {
            processResponsesApiEvent(data);

            // Flush buffered content before forwarding terminal events
            if (data.type === 'response.completed' || data.type === 'response.incomplete' || data.type === 'response.failed') {
                const extras = flushChoice(0);
                if (extras.reasoning_content) {
                    this.push(`data: ${JSON.stringify({
                        type: 'response.reasoning_summary_text.delta',
                        delta: extras.reasoning_content,
                    })}\n\n`);
                }
                if (extras.content) {
                    this.push(`data: ${JSON.stringify({
                        type: 'response.output_text.delta',
                        delta: extras.content,
                    })}\n\n`);
                }
            }
        }

        // Handle Claude native streaming events
        if (typeof data?.type === 'string') {
            processClaudeNativeEvent.call(this, data);
        }

        lines[dataLineIndex] = `data: ${JSON.stringify(data)}`;
        this.push(`${lines.join('\n')}\n\n`);
    }

    /**
     * Apply word replacements to OpenAI Responses API streaming events.
     * @param {object} data The parsed SSE event data
     */
    function processResponsesApiEvent(data) {
        if (typeof data.delta !== 'string') return;

        if (data.type === 'response.output_text.delta') {
            const emitted = appendAndExtract(0, 'content', data.delta, false);
            if (emitted) {
                data.delta = emitted;
            } else {
                data.delta = '';
            }
        } else if (data.type === 'response.reasoning_summary_text.delta' || data.type === 'response.reasoning_content_text.delta') {
            const emitted = appendAndExtract(0, 'reasoning_content', data.delta, false);
            if (emitted) {
                data.delta = emitted;
            } else {
                data.delta = '';
            }
        }
    }

    /**
     * Apply word replacements to Claude native streaming events.
     * Handles content_block_delta (text_delta, thinking_delta) and flushes on message_stop.
     * @param {object} data The parsed SSE event data
     */
    function processClaudeNativeEvent(data) {
        if (data.type === 'content_block_delta' && data.delta) {
            if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
                const emitted = appendAndExtract(0, 'content', data.delta.text, false);
                if (emitted) {
                    data.delta.text = emitted;
                } else {
                    data.delta.text = '';
                }
            } else if (data.delta.type === 'thinking_delta' && typeof data.delta.thinking === 'string') {
                const emitted = appendAndExtract(0, 'reasoning_content', data.delta.thinking, false);
                if (emitted) {
                    data.delta.thinking = emitted;
                } else {
                    data.delta.thinking = '';
                }
            }
        } else if (data.type === 'message_stop' || data.type === 'message_delta') {
            const extras = flushChoice(0);
            if (extras.reasoning_content) {
                this.push(`data: ${JSON.stringify({
                    type: 'content_block_delta',
                    index: 0,
                    delta: { type: 'thinking_delta', thinking: extras.reasoning_content },
                })}\n\n`);
            }
            if (extras.content) {
                this.push(`data: ${JSON.stringify({
                    type: 'content_block_delta',
                    index: 0,
                    delta: { type: 'text_delta', text: extras.content },
                })}\n\n`);
            }
        }
    }

    return transformStream;
}

/**
 * Parses a raw SSE response body and returns a human-readable summary.
 * Extracts thinking deltas and text deltas, ignoring protocol noise.
 * @param {string} raw - The full SSE response string
 * @returns {string}
 */
function formatStreamingResponse(raw) {
    let thinking = '';
    let text = '';
    let metaBase = null;
    let finalOutputTokens = null;

    for (const line of raw.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr || jsonStr === '[DONE]') continue;
        let parsed;
        try { parsed = JSON.parse(jsonStr); } catch { continue; }

        if (parsed?.type === 'message_start' && parsed.message) {
            const msg = parsed.message;
            const u = msg.usage ?? {};
            metaBase = { model: msg.model, inputTokens: u.input_tokens ?? '?', cacheRead: u.cache_read_input_tokens ?? 0, cacheCreated: u.cache_creation_input_tokens ?? 0 };
        }

        if (parsed?.type === 'message_delta' && parsed.usage?.output_tokens != null) {
            finalOutputTokens = parsed.usage.output_tokens;
        }

        const delta = parsed?.delta ?? parsed?.choices?.[0]?.delta;
        if (!delta) continue;

        if (delta.type === 'thinking_delta' && delta.thinking) thinking += delta.thinking;
        else if (delta.type === 'text_delta' && delta.text) text += delta.text;
        else if (typeof delta.content === 'string') text += delta.content;
    }

    const parts = [];
    if (thinking) parts.push(`[Thinking]\n${thinking}`);
    if (text) parts.push(`[Response]\n${text}`);
    if (metaBase) {
        const out = finalOutputTokens ?? '?';
        parts.push(`model: ${metaBase.model} | in: ${metaBase.inputTokens} | out: ${out} | cache_read: ${metaBase.cacheRead} | cache_created: ${metaBase.cacheCreated}`);
    }
    return parts.length ? parts.join('\n\n') : '(no text content)';
}

function forwardFetchResponseWithWordReplacements(from, to, enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    let statusCode = from.status;
    let statusText = from.statusText;

    if (!from.ok) {
        console.warn(`Streaming request failed with status ${statusCode} ${statusText}`);
    }

    if (statusCode === 401) {
        statusCode = 400;
    }

    to.statusCode = statusCode;
    to.statusMessage = statusText;

    if (from.body && to.socket) {
        const endResponse = () => {
            if (!to.writableEnded) {
                to.end();
            }
        };

        const destroySource = () => {
            if (typeof from.body.destroy === 'function') {
                from.body.destroy();
            }
        };

        if (!isEnabled) {
            const chunks = [];
            from.body.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
            from.body.pipe(to);

            to.socket.on('close', function () {
                destroySource();
                endResponse();
            });

            from.body.on('end', function () {
                console.info('Streaming request finished.\n' + formatStreamingResponse(Buffer.concat(chunks).toString('utf8')));
                endResponse();
            });

            from.body.on('error', function (error) {
                console.error('Streaming request error:', error);
                endResponse();
            });
        } else {
            const chunks = [];
            from.body.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
            const transformStream = createWordReplacementStream(isEnabled);
            from.body.pipe(transformStream).pipe(to);

            to.socket.on('close', function () {
                destroySource();
                transformStream.end();
                endResponse();
            });

            transformStream.on('end', function () {
                console.info('Streaming request finished.\n' + formatStreamingResponse(Buffer.concat(chunks).toString('utf8')));
                endResponse();
            });

            transformStream.on('error', function (error) {
                console.error('Word replacement streaming error:', error);
                endResponse();
            });
        }
    } else {
        to.end();
    }
}

function sendWithWordReplacements(res, payload, enabled) {
    const isEnabled = areWordReplacementsEnabled(enabled);
    if (!isEnabled) {
        return res.send(payload);
    }

    return res.send(enforceWordReplacementsOnResponse(payload, isEnabled));
}

/**
 * Gets OpenRouter transforms based on the request.
 * @param {import('express').Request} request Express request
 * @returns {string[] | undefined} OpenRouter transforms
 */
function getOpenRouterTransforms(request) {
    switch (request.body.middleout) {
        case 'on':
            return ['middle-out'];
        case 'off':
            return [];
        case 'auto':
            return undefined;
    }
}

/**
 * Gets OpenRouter plugins based on the request.
 * @param {import('express').Request} request
 * @returns {any[]} OpenRouter plugins
 */
function getOpenRouterPlugins(request) {
    const plugins = [];

    if (request.body.enable_web_search) {
        plugins.push({ 'id': 'web' });
    }

    return plugins;
}

/**
 * Hacky way to use JSON schema only if json_object format is supported.
 * @param {object} bodyParams Additional body parameters
 * @param {object[]} messages Array of messages
 * @param {object} jsonSchema JSON schema object
 */
function setJsonObjectFormat(bodyParams, messages, jsonSchema) {
    bodyParams['response_format'] = {
        type: 'json_object',
    };
    const message = {
        role: 'user',
        content: `JSON schema for the response:\n${JSON.stringify(jsonSchema.value, null, 4)}`,
    };
    messages.push(message);
}

/**
 * Sends a request to Claude API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendClaudeRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_CLAUDE).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.CLAUDE);
    const divider = '-'.repeat(process.stdout.columns);
    const enableSystemPromptCache = Boolean(request.body.claude_enable_caching);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);
    let cachingAtDepth = getConfigValue('claude.cachingAtDepth', -1, 'number');
    // Disabled if not an integer or negative
    if (!Number.isInteger(cachingAtDepth) || cachingAtDepth < 0) {
        cachingAtDepth = -1;
    }
    // UI toggle: override cachingAtDepth from request body
    if (request.body.claude_enable_caching_at_depth === false) {
        cachingAtDepth = -1;
    } else if (request.body.claude_enable_caching_at_depth === true && cachingAtDepth < 0) {
        cachingAtDepth = 0;
    }

    if (!apiKey) {
        console.warn(color.red(`Claude API key is missing.\n${divider}`));
        return response.status(400).send({ error: true });
    }

    try {
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });
        const additionalHeaders = {};
        const betaHeaders = ['output-300k-2026-03-24'];
        const useTools = Array.isArray(request.body.tools) && request.body.tools.length > 0;
        const useSystemPrompt = Boolean(request.body.use_sysprompt);
        const convertedPrompt = convertClaudeMessages(request.body.messages, request.body.assistant_prefill, useSystemPrompt, useTools, getPromptNames(request));
        const useThinking = /^claude-(3-7|opus-4|sonnet-4|haiku-4-5|opus-4-5|opus-4-6|sonnet-4-6|opus-4-7)/.test(request.body.model);
        const isAdaptiveThinking = /^claude-(opus-4-6|sonnet-4-6|opus-4-7)/.test(request.body.model) && request.body.claude_use_adaptive_thinking !== false || /^claude-opus-4-7/.test(request.body.model);
        const useWebSearch = /^claude-(3-5|3-7|opus-4|sonnet-4|haiku-4-5|opus-4-5|opus-4-6|sonnet-4-6|opus-4-7)/.test(request.body.model) && Boolean(request.body.enable_web_search);
        const isLimitedSampling = /^claude-(opus-4-1|sonnet-4-5|haiku-4-5|opus-4-5|opus-4-6|sonnet-4-6)/.test(request.body.model);
        const noPrefillModel = /^claude-(opus-4-6|sonnet-4-6|opus-4-7)/.test(request.body.model);
        let fixThinkingPrefill = false;
        // Add custom stop sequences
        const stopSequences = [];
        if (Array.isArray(request.body.stop)) {
            stopSequences.push(...request.body.stop);
        }

        const requestBody = {
            /** @type {any} */ system: [],
            messages: convertedPrompt.messages,
            model: request.body.model,
            max_tokens: request.body.max_tokens,
            stop_sequences: stopSequences,
            temperature: request.body.temperature,
            top_p: request.body.top_p,
            top_k: request.body.top_k,
            stream: request.body.stream,
        };
        if (useSystemPrompt) {
            if (enableSystemPromptCache && Array.isArray(convertedPrompt.systemPrompt) && convertedPrompt.systemPrompt.length) {
                convertedPrompt.systemPrompt[convertedPrompt.systemPrompt.length - 1].cache_control = { type: 'ephemeral', ttl: getCacheTTL(request) };
            }

            requestBody.system = convertedPrompt.systemPrompt;
        } else {
            delete requestBody.system;
        }
        if (useTools) {
            betaHeaders.push('tools-2024-05-16');
            requestBody.tool_choice = { type: request.body.tool_choice };
            requestBody.tools = request.body.tools
                .filter(tool => tool.type === 'function')
                .map(tool => tool.function)
                .map(fn => ({ name: fn.name, description: fn.description, input_schema: flattenSchema(fn.parameters, request.body.chat_completion_source) }));

            if (enableSystemPromptCache && requestBody.tools.length) {
                requestBody.tools[requestBody.tools.length - 1].cache_control = { type: 'ephemeral', ttl: getCacheTTL(request) };
            }
        }
        if (/^claude-opus-4-7/.test(request.body.model)) {
                delete requestBody.top_k;
                delete requestBody.temperature;
                delete requestBody.top_p;
        }
        // Structured output is a forced tool
        if (request.body.json_schema) {
            const jsonTool = {
                name: request.body.json_schema.name,
                description: request.body.json_schema.description || 'Well-formed JSON object',
                input_schema: request.body.json_schema.value,
            };
            requestBody.tools = [...(requestBody.tools || []), jsonTool];
            requestBody.tool_choice = { type: 'tool', name: request.body.json_schema.name };
        }

        if (useWebSearch) {
            const webSearchTool = [{
                'type': 'web_search_20250305',
                'name': 'web_search',
            }];
            requestBody.tools = [...webSearchTool, ...(requestBody.tools || [])];
        }

        if (cachingAtDepth !== -1) {
            cachingAtDepthForClaude(convertedPrompt.messages, cachingAtDepth, getCacheTTL(request));
        }

        if (enableSystemPromptCache || cachingAtDepth !== -1) {
            betaHeaders.push('prompt-caching-2024-07-31');
            betaHeaders.push('extended-cache-ttl-2025-04-11');
        }

        if (isLimitedSampling) {
            if (requestBody.temperature < 1) {
                delete requestBody.top_p;
            } else {
                delete requestBody.temperature;
            }
        }

        const reasoningEffort = request.body.reasoning_effort;
        const isThinkingDisabled = !reasoningEffort || reasoningEffort === 'none';

        if (useThinking && !isThinkingDisabled) {
            // No prefill when thinking
            fixThinkingPrefill = true;
            const minThinkTokens = 1024;
            if (requestBody.max_tokens <= minThinkTokens) {
                const newValue = requestBody.max_tokens + minThinkTokens;
                console.warn(color.yellow(`Claude thinking requires a minimum of ${minThinkTokens} response tokens.`));
                console.info(color.blue(`Increasing response length to ${newValue}.`));
                requestBody.max_tokens = newValue;
            }

            if (isAdaptiveThinking) {
                // Opus 4.6-4.7 / Sonnet 4.6: use adaptive thinking
                requestBody.thinking = { type: 'adaptive' };

                const effort = getClaudeAdaptiveEffort(reasoningEffort, request.body.model);
                if (effort) {
                    requestBody.output_config ??= {};
                    requestBody.output_config.effort = effort;
                }
            } else {
                // Older models: use enabled thinking with budget_tokens
                const budgetTokens = calculateClaudeBudgetTokens(requestBody.max_tokens, reasoningEffort, requestBody.stream);
                if (Number.isInteger(budgetTokens)) {
                    requestBody.thinking = {
                        type: 'enabled',
                        budget_tokens: budgetTokens,
                    };
                }
            }

            // NO I CAN'T SILENTLY IGNORE THE TEMPERATURE.
            delete requestBody.temperature;
            delete requestBody.top_k;

            if (requestBody.top_p < 0.95) {
                delete requestBody.top_p;
            }
        }

        if ((fixThinkingPrefill || noPrefillModel) && convertedPrompt.messages.length && convertedPrompt.messages[convertedPrompt.messages.length - 1].role === 'assistant') {
            convertedPrompt.messages[convertedPrompt.messages.length - 1].role = 'user';
        }


        if (betaHeaders.length) {
            additionalHeaders['anthropic-beta'] = betaHeaders.join(',');
        }

        if (request.body.top_p === 1)
            delete requestBody.top_p;

        if (request.body.temperature === 1)
            delete requestBody.temperature;

        console.debug('Claude request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/messages', {
            method: 'POST',
            signal: controller.signal,
            body: JSON.stringify(requestBody),
            headers: {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                'x-api-key': apiKey,
                ...additionalHeaders,
            },
        });

        if (request.body.stream) {
            // Pipe remote SSE stream to Express response
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const generateResponseText = await generateResponse.text();
                console.warn(color.red(`Claude API returned error: ${generateResponse.status} ${generateResponse.statusText}\n${generateResponseText}\n${divider}`));
                return response.status(500).send({ error: true });
            }

            /** @type {any} */
            const generateResponseJson = await generateResponse.json();
            const responseText = generateResponseJson?.content?.[0]?.text || '';
            console.debug('Claude response:', generateResponseJson);

            // Wrap it back to OAI format + save the original content
            const reply = { choices: [{ 'message': { 'content': responseText } }], content: generateResponseJson.content };
            return sendWithWordReplacements(response, reply, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error(color.red(`Error communicating with Claude: ${error}\n${divider}`));
        if (!response.headersSent) {
            return response.status(500).send({ error: true });
        }
    }
}

/**
 * Sends a request to Google AI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMakerSuiteRequest(request, response) {
    const useVertexAi = request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.VERTEXAI;
    const apiName = useVertexAi ? 'Google Vertex AI' : 'Google AI Studio';
    let apiUrl;
    let apiKey;
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    let authHeader;
    let authType;

    if (useVertexAi) {
        apiUrl = new URL(request.body.reverse_proxy || API_VERTEX_AI);

        try {
            const auth = await getVertexAIAuth(request);
            authHeader = auth.authHeader;
            authType = auth.authType;
            console.debug(`Using Vertex AI authentication type: ${authType}`);
        } catch (error) {
            console.warn(`${apiName} authentication failed: ${error.message}`);
            return response.status(400).send({ error: true, message: error.message });
        }
    } else {
        apiUrl = new URL(request.body.reverse_proxy || API_MAKERSUITE);
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE);

        if (!request.body.reverse_proxy && !apiKey) {
            console.warn(`${apiName} API key is missing.`);
            return response.status(400).send({ error: true });
        }

        authHeader = `Bearer ${apiKey}`;
        authType = 'api_key';
    }

    const model = String(request.body.model);
    const stream = Boolean(request.body.stream);
    const enableWebSearch = Boolean(request.body.enable_web_search);
    const requestImages = Boolean(request.body.request_images);
    const reasoningEffort = String(request.body.reasoning_effort);
    const includeReasoning = Boolean(request.body.include_reasoning);
    const aspectRatio = String(request.body.request_image_aspect_ratio);
    const imageSize = String(request.body.request_image_resolution);
    const isGemma = model.includes('gemma');
    const isLearnLM = model.includes('learnlm');

    const responseMimeType = request.body.responseMimeType ?? (request.body.json_schema ? 'application/json' : undefined);
    const responseSchema = request.body.responseSchema ?? (request.body.json_schema ? request.body.json_schema.value : undefined);

    const generationConfig = {
        stopSequences: request.body.stop,
        candidateCount: 1,
        maxOutputTokens: request.body.max_tokens,
        temperature: request.body.temperature,
        topP: request.body.top_p,
        topK: request.body.top_k || undefined,
        presencePenalty: request.body.presence_penalty,
        frequencyPenalty: request.body.frequency_penalty,
        responseMimeType: responseMimeType,
        responseSchema: responseSchema,
        seed: request.body.seed,
    };

    function getGeminiBody() {
        // #region UGLY MODEL LISTS AREA
        const imageGenerationModels = [
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash-exp-image-generation',
            'gemini-2.0-flash-preview-image-generation',
            'gemini-2.5-flash-image-preview',
            'gemini-2.5-flash-image',
            'gemini-3-pro-image-preview',
            'gemini-3.1-flash-image-preview',
        ];

        const isThinkingConfigModel = m => (/^gemini-2.5-(flash|pro)/.test(m) && !/-image(-preview)?$/.test(m)) || (/^gemini-3[.\d]*-(flash|pro)/.test(m));
        const isImageSizeModel = m => /^gemini-3/.test(m);

        const noSearchModels = [
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash-lite-001',
            'gemini-2.0-flash-lite-preview-02-05',
            'gemini-robotics-er-1.5-preview',
        ];
        // #endregion

        if (!Array.isArray(generationConfig.stopSequences) || !generationConfig.stopSequences.length) {
            delete generationConfig.stopSequences;
        }

        const enableImageModality = requestImages && imageGenerationModels.includes(model);
        const enableImageConfig = enableImageModality && (aspectRatio || imageSize);
        if (enableImageModality) {
            generationConfig.responseModalities = ['text', 'image'];
            if (enableImageConfig) {
                generationConfig.imageConfig = {};
                if (imageSize && isImageSizeModel(model)) {
                    generationConfig.imageConfig.imageSize = imageSize;
                }
                if (aspectRatio) {
                    generationConfig.imageConfig.aspectRatio = aspectRatio;
                }
            }
        }

        const useSystemPrompt = !enableImageModality && !isGemma && request.body.use_sysprompt;

        const tools = [];
        const prompt = convertGooglePrompt(request.body.messages, model, useSystemPrompt, getPromptNames(request));
        const safetySettings = [...GEMINI_SAFETY, ...(useVertexAi ? VERTEX_SAFETY : [])];

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0 && !enableImageModality && !isGemma) {
            const functionDeclarations = [];
            const customTools = [];
            for (const tool of request.body.tools) {
                if (tool.type === 'function') {
                    if (tool.function.parameters?.$schema) {
                        delete tool.function.parameters.$schema;
                    }
                    if (tool.function.parameters?.properties && Object.keys(tool.function.parameters.properties).length === 0) {
                        delete tool.function.parameters;
                    }
                    functionDeclarations.push(tool.function);
                } else if (tool[tool.type]) {
                    customTools.push({ [tool.type]: tool[tool.type] });
                }
            }
            if (functionDeclarations.length > 0) {
                tools.push({ function_declarations: functionDeclarations });
            }
            // Custom tools are only supported when no function calling is present
            if (functionDeclarations.length === 0 && customTools.length > 0) {
                tools.push(...customTools);
            }
        }

        if (enableWebSearch && !enableImageModality && !isGemma && !isLearnLM && !noSearchModels.includes(model)) {
            // Tool use with function calling is unsupported
            if (!tools.some(t => t.function_declarations)) {
                tools.push({ google_search: {} });
            }
        }

        if (isThinkingConfigModel(model)) {
            const thinkingConfig = { includeThoughts: includeReasoning };

            const thinkingBudget = calculateGoogleBudgetTokens(generationConfig.maxOutputTokens, reasoningEffort, model);
            if (typeof thinkingBudget === 'number' && Number.isInteger(thinkingBudget)) {
                thinkingConfig.thinkingBudget = thinkingBudget;
            }

            if (typeof thinkingBudget === 'string' && thinkingBudget.length > 0) {
                thinkingConfig.thinkingLevel = thinkingBudget;
            }

            // Vertex doesn't allow mixing disabled thinking with includeThoughts
            if (useVertexAi && thinkingBudget === 0 && thinkingConfig.includeThoughts) {
                console.info('Thinking budget is 0, but includeThoughts is true. Thoughts will not be included in the response.');
                thinkingConfig.includeThoughts = false;
            }

            generationConfig.thinkingConfig = thinkingConfig;
        }

        let body = {
            contents: prompt.contents,
            safetySettings: safetySettings,
            generationConfig: generationConfig,
        };

        if (useSystemPrompt && Array.isArray(prompt.system_instruction.parts) && prompt.system_instruction.parts.length) {
            body.systemInstruction = prompt.system_instruction;
        }

        if (tools.length) {
            body.tools = tools;

            const toolChoice = request.body.tool_choice;
            let functionCallingConfig;

            // Translate OpenAI's `tool_choice` to Gemini's `functionCallingConfig`
            if (typeof toolChoice === 'string') {
                switch (toolChoice) {
                    case 'none':
                        functionCallingConfig = { mode: 'NONE' };
                        break;
                    case 'required':
                        functionCallingConfig = { mode: 'ANY' };
                        break;
                    case 'auto':
                        functionCallingConfig = { mode: 'AUTO' };
                        break;
                }
            } else if (typeof toolChoice === 'object' && toolChoice?.function?.name) {
                // Force a specific function call
                functionCallingConfig = {
                    mode: 'ANY',
                    allowedFunctionNames: [toolChoice.function.name],
                };
            }

            if (functionCallingConfig) {
                body.toolConfig = { functionCallingConfig };
            }
        }

        return body;
    }

    const body = getGeminiBody();
    console.debug(`${apiName} request:`, body);

    try {
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        const apiVersion = getConfigValue('gemini.apiVersion', 'v1beta');
        const responseType = (stream ? 'streamGenerateContent' : 'generateContent');

        let url;
        let headers = {
            'Content-Type': 'application/json',
        };

        if (useVertexAi) {
            if (authType === 'express') {
                // For Express mode (API key authentication), use the key parameter
                const keyParam = authHeader.replace('Bearer ', '');
                const region = request.body.vertexai_region || 'us-central1';
                const projectId = request.body.vertexai_express_project_id;
                const baseUrl = region === 'global'
                    ? 'https://aiplatform.googleapis.com'
                    : `https://${region}-aiplatform.googleapis.com`;
                url = projectId
                    ? `https://aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}?key=${keyParam}${stream ? '&alt=sse' : ''}`
                    : `${baseUrl}/v1/publishers/google/models/${model}:${responseType}?key=${keyParam}${stream ? '&alt=sse' : ''}`;
            } else if (authType === 'full') {
                // For Full mode (service account authentication), use project-specific URL
                // Get project ID from Service Account JSON
                const serviceAccountJson = readSecret(request.user.directories, SECRET_KEYS.VERTEXAI_SERVICE_ACCOUNT);
                if (!serviceAccountJson) {
                    console.warn('Vertex AI Service Account JSON is missing.');
                    return response.status(400).send({ error: true });
                }

                let projectId;
                try {
                    const serviceAccount = JSON.parse(serviceAccountJson);
                    projectId = getProjectIdFromServiceAccount(serviceAccount);
                } catch (error) {
                    console.error('Failed to extract project ID from Service Account JSON:', error);
                    return response.status(400).send({ error: true });
                }
                const region = request.body.vertexai_region || 'us-central1';
                // Handle global region differently - no region prefix in hostname
                if (region === 'global') {
                    url = `https://aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                } else {
                    url = `https://${region}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                }
                headers['Authorization'] = authHeader;
            } else {
                // For proxy mode, use the original URL with Authorization header
                url = `${apiUrl.toString().replace(/\/$/, '')}/v1/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                headers['Authorization'] = authHeader;
            }
        } else {
            url = `${apiUrl.toString().replace(/\/$/, '')}/${apiVersion}/models/${model}:${responseType}?key=${apiKey}${stream ? '&alt=sse' : ''}`;
        }

        const generateResponse = await fetch(url, {
            body: JSON.stringify(body),
            method: 'POST',
            headers: headers,
            signal: controller.signal,
        });

        if (stream) {
            try {
                // Pipe remote SSE stream to Express response
                forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
            } catch (error) {
                console.error('Error forwarding streaming response:', error);
                if (!response.headersSent) {
                    return response.status(500).send({ error: true });
                }
            }
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`${apiName} API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }

            /** @type {any} */
            const generateResponseJson = await generateResponse.json();

            const candidates = generateResponseJson?.candidates;
            if (!candidates || candidates.length === 0) {
                let message = `${apiName} API returned no candidate`;
                console.warn(message, generateResponseJson);
                if (generateResponseJson?.promptFeedback?.blockReason) {
                    message += `\nPrompt was blocked due to : ${generateResponseJson.promptFeedback.blockReason}`;
                }
                return response.send({ error: { message } });
            }

            const responseContent = candidates[0].content ?? candidates[0].output;
            const functionCall = (candidates?.[0]?.content?.parts ?? []).some(part => part.functionCall);
            const inlineData = (candidates?.[0]?.content?.parts ?? []).some(part => part.inlineData);
            console.debug(`${apiName} response:`, util.inspect(generateResponseJson, { depth: 5, colors: true }));

            const responseText = typeof responseContent === 'string' ? responseContent : responseContent?.parts?.filter(part => !part.thought)?.map(part => part.text)?.join('\n\n');
            if (!responseText && !functionCall && !inlineData) {
                let message = `${apiName} Candidate text empty`;
                console.warn(message, generateResponseJson);
                return response.send({ error: { message } });
            }

            // Wrap it back to OAI format (responseContent includes thought signatures in parts array)
            const reply = { choices: [{ 'message': { 'content': responseText } }], responseContent };
            return sendWithWordReplacements(response, reply, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error(`Error communicating with ${apiName} API:`, error);
        if (!response.headersSent) {
            return response.status(500).send({ error: true });
        }
    }
}

/**
 * Sends a request to AI21 API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendAI21Request(request, response) {
    if (!request.body) return response.sendStatus(400);

    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AI21);
    if (!apiKey) {
        console.warn('AI21 API key is missing.');
        return response.status(400).send({ error: true });
    }
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    const bodyParams = {};
    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });
    // Hack to support JSON schema
    if (request.body.json_schema) {
        bodyParams.response_format = {
            type: 'json_object',
        };
        const message = {
            role: 'user',
            content: `JSON schema for the response:\n${JSON.stringify(request.body.json_schema.value, null, 4)}`,
        };
        request.body.messages.push(message);
    }
    const convertedPrompt = convertAI21Messages(request.body.messages, getPromptNames(request));
    const body = {
        messages: convertedPrompt,
        model: request.body.model,
        max_tokens: request.body.max_tokens,
        temperature: request.body.temperature,
        top_p: request.body.top_p,
        stop: request.body.stop,
        stream: request.body.stream,
        tools: request.body.tools,
        ...bodyParams,
    };
    const options = {
        method: 'POST',
        headers: {
            accept: 'application/json',
            'content-type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
    };

    console.debug('AI21 request:', body);

    try {
        const generateResponse = await fetch(API_AI21 + '/chat/completions', options);
        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`AI21 API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('AI21 response:', generateResponseJson);
            return sendWithWordReplacements(response, generateResponseJson, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error('Error communicating with AI21 API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to MistralAI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMistralAIRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_MISTRAL).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    if (!apiKey) {
        console.warn('MistralAI API key is missing.');
        return response.status(400).send({ error: true });
    }

    try {
        const messages = convertMistralMessages(request.body.messages, getPromptNames(request));
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        const requestBody = {
            'model': request.body.model,
            'messages': messages,
            'temperature': request.body.temperature,
            'top_p': request.body.top_p,
            'frequency_penalty': request.body.frequency_penalty,
            'presence_penalty': request.body.presence_penalty,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'safe_prompt': request.body.safe_prompt,
            'random_seed': request.body.seed === -1 ? undefined : request.body.seed,
            'stop': Array.isArray(request.body.stop) && request.body.stop.length > 0 ? request.body.stop : undefined,
        };

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            requestBody['tools'] = request.body.tools;
            requestBody['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.json_schema) {
            requestBody['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
            timeout: 0,
        };

        console.debug('MisralAI request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);
        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`MistralAI API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('MistralAI response:', generateResponseJson);
            return sendWithWordReplacements(response, generateResponseJson, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error('Error communicating with MistralAI API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to Cohere API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendCohereRequest(request, response) {
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);
    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    if (!apiKey) {
        console.warn('Cohere API key is missing.');
        return response.status(400).send({ error: true });
    }

    try {
        const convertedHistory = convertCohereMessages(request.body.messages, getPromptNames(request));
        const tools = [];

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            tools.push(...request.body.tools);
            tools.forEach(tool => {
                if (tool?.function?.parameters?.$schema) {
                    delete tool.function.parameters.$schema;
                }
            });
        }

        // https://docs.cohere.com/reference/chat
        const requestBody = {
            stream: Boolean(request.body.stream),
            model: request.body.model,
            messages: convertedHistory.chatHistory,
            temperature: request.body.temperature,
            max_tokens: request.body.max_tokens,
            k: request.body.top_k,
            p: request.body.top_p,
            seed: request.body.seed,
            stop_sequences: request.body.stop,
            frequency_penalty: request.body.frequency_penalty,
            presence_penalty: request.body.presence_penalty,
            documents: [],
            tools: tools,
        };

        const canDoSafetyMode = String(request.body.model).endsWith('08-2024');
        if (canDoSafetyMode) {
            requestBody.safety_mode = 'OFF';
        }

        if (request.body.json_schema) {
            requestBody.response_format = {
                type: 'json_schema',
                schema: request.body.json_schema.value,
            };
        }

        console.debug('Cohere request:', requestBody);

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
            timeout: 0,
        };

        const apiUrl = API_COHERE_V2 + '/chat';

        if (request.body.stream) {
            const stream = await fetch(apiUrl, config);
            forwardFetchResponseWithWordReplacements(stream, response, wordReplacementsEnabled);
        } else {
            const generateResponse = await fetch(apiUrl, config);
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`Cohere API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('Cohere response:', generateResponseJson);
            return sendWithWordReplacements(response, generateResponseJson, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error('Error communicating with Cohere API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to DeepSeek API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendDeepSeekRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_DEEPSEEK).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    if (!apiKey && !request.body.reverse_proxy) {
        console.warn('DeepSeek API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;

            // DeepSeek doesn't permit empty required arrays
            bodyParams.tools.forEach(tool => {
                const required = tool?.function?.parameters?.required;
                if (Array.isArray(required) && required.length === 0) {
                    delete tool.function.parameters.required;
                }
            });
        }

        // Hack to support JSON schema
        if (request.body.json_schema) {
            bodyParams.response_format = {
                type: 'json_object',
            };
            const message = {
                role: 'user',
                content: `JSON schema for the response:\n${JSON.stringify(request.body.json_schema.value, null, 4)}`,
            };
            request.body.messages.push(message);
        }

        const processedMessages = addAssistantPrefix(postProcessPrompt(request.body.messages, PROMPT_PROCESSING_TYPE.SEMI_TOOLS, getPromptNames(request)), bodyParams.tools, 'prefix');

        if (/-reasoner/.test(request.body.model)) {
            addReasoningContentToToolCalls(processedMessages);
        }

        const requestBody = {
            'messages': processedMessages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'stop': request.body.stop,
            'seed': request.body.seed,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('DeepSeek request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`DeepSeek API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('DeepSeek response:', generateResponseJson);
            return sendWithWordReplacements(response, generateResponseJson, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error('Error communicating with DeepSeek API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to XAI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendXaiRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_XAI).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    if (!apiKey && !request.body.reverse_proxy) {
        console.warn('xAI API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        if (request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort === 'high' ? 'high' : 'low';
        }

        if (request.body.enable_web_search) {
            bodyParams['search_parameters'] = {
                mode: 'on',
                sources: [
                    { type: 'web', safe_search: false },
                    { type: 'news', safe_search: false },
                    { type: 'x' },
                ],
            };
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    strict: request.body.json_schema.strict ?? true,
                    schema: request.body.json_schema.value,
                },
            };
        }

        const processedMessages = request.body.messages = convertXAIMessages(request.body.messages, getPromptNames(request));

        const requestBody = {
            'messages': processedMessages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'max_completion_tokens': request.body.max_completion_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'seed': request.body.seed,
            'n': request.body.n,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('xAI request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`xAI API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('xAI response:', generateResponseJson);
            return sendWithWordReplacements(response, generateResponseJson, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error('Error communicating with xAI API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to AI/ML API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendAimlapiRequest(request, response) {
    const apiUrl = API_AIMLAPI;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    if (!apiKey) {
        console.warn('AI/ML API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        if (request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const requestBody = {
            'messages': request.body.messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'seed': request.body.seed,
            'n': request.body.n,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
                ...AIMLAPI_HEADERS,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('AI/ML API request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`AI/ML API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('AI/ML API response:', generateResponseJson);
            return sendWithWordReplacements(response, generateResponseJson, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error('Error communicating with AI/ML API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to Electron Hub.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendElectronHubRequest(request, response) {
    const apiUrl = API_ELECTRONHUB;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.ELECTRONHUB);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    if (!apiKey) {
        console.warn('Electron Hub key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.enable_web_search) {
            bodyParams['web_search'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const isClaude = /^claude-/.test(request.body.model);

        if (Array.isArray(request.body.messages) && isClaude) {
            if (enableSystemPromptCache) {
                cachingSystemPromptForOpenRouter(request.body.messages, '5m');
            }

            if (cachingAtDepth !== -1) {
                cachingAtDepthForOpenRouterClaude(request.body.messages, cachingAtDepth, '5m');
            }
        }

        const requestBody = {
            'messages': request.body.messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'top_k': request.body.top_k,
            'logit_bias': request.body.logit_bias,
            'seed': request.body.seed,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('Electron Hub request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn('Electron Hub returned error: ', errorText);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('Electron Hub response:', generateResponseJson);
            return sendWithWordReplacements(response, generateResponseJson, wordReplacementsEnabled);
        }
    } catch (error) {
        console.error('Error communicating with Electron Hub: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to Chutes.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendChutesRequest(request, response) {
    const apiUrl = API_CHUTES;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.CHUTES);

    if (!apiKey) {
        console.warn('Chutes key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const requestBody = {
            'messages': request.body.messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'repetition_penalty': request.body.repetition_penalty,
            'min_p': request.body.min_p,
            'top_p': request.body.top_p,
            'top_k': request.body.top_k,
            'seed': request.body.seed,
            'stop': request.body.stop,
            'reasoning_effort': request.body.reasoning_effort,
            'logit_bias': request.body.logit_bias,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('Chutes request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn('Chutes returned error: ', errorText);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('Chutes response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with Chutes: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a chat completion request to Azure OpenAI.
 * @param {express.Request} request Express request object (contains request.body with all generate_data)
 * @param {express.Response} response Express response object
 */
async function sendAzureOpenAIRequest(request, response) {
    // 1. GATHER & VALIDATE SETTINGS
    const { azure_base_url, azure_deployment_name, azure_api_version } = request.body;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AZURE_OPENAI);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);
    if (!azure_base_url || !azure_deployment_name || !azure_api_version || !apiKey) {
        return response.status(400).send({
            error: {
                message: 'Azure OpenAI configuration is incomplete. Please provide Base URL, Deployment Name, API Version, and API Key in the connection settings.',
            },
        });
    }

    // 2. PREPARE THE REQUEST
    const url = new URL(`/openai/deployments/${azure_deployment_name}/chat/completions`, azure_base_url);
    url.searchParams.set('api-version', azure_api_version);
    const endpointUrl = url.toString();

    // Create the base payload with all standard parameters
    const apiRequestBody = /** @type {any} */ ({});
    for (const key of AZURE_OPENAI_KEYS) {
        if (Object.hasOwn(request.body, key)) {
            apiRequestBody[key] = request.body[key];
        }
    }

    // Handle Structured Output (JSON Mode) by translating the custom `json_schema` object.
    if (request.body.json_schema) {
        apiRequestBody['response_format'] = {
            type: 'json_schema',
            json_schema: {
                name: request.body.json_schema.name,
                strict: request.body.json_schema.strict ?? true,
                schema: request.body.json_schema.value,
            },
        };
    }

    // Adjust logprobs for Azure OpenAI, which follows the OpenAI Chat Completions API spec.
    if (typeof apiRequestBody.logprobs === 'number' && apiRequestBody.logprobs > 0) {
        apiRequestBody.top_logprobs = apiRequestBody.logprobs;
        apiRequestBody.logprobs = true;
    }

    // Do not send reasoning effort to models which do not support it
    apiRequestBody['reasoning_effort'] = OPENAI_REASONING_EFFORT_MODELS.includes(request.body.model)
        ? OPENAI_FIXED_REASONING_EFFORT[request.body.model] ?? OPENAI_REASONING_EFFORT_MAP[request.body.reasoning_effort] ?? request.body.reasoning_effort
        : undefined;

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', () => controller.abort());

    const config = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'api-key': apiKey,
        },
        body: JSON.stringify(apiRequestBody),
        signal: controller.signal,
    };

    console.info(`Sending request to Azure OpenAI: ${endpointUrl}`);
    console.debug('Azure OpenAI Request Body:', apiRequestBody);
    try {
        const fetchResponse = await fetch(endpointUrl, config);

        if (request.body.stream) {
            return forwardFetchResponseWithWordReplacements(fetchResponse, response, wordReplacementsEnabled);
        }

        if (fetchResponse.ok) {
            /** @type {any} */
            const json = await fetchResponse.json();
            console.debug('Azure OpenAI response:', json);
            return sendWithWordReplacements(response, json, wordReplacementsEnabled);
        }

        const text = await fetchResponse.text();
        const data = tryParse(text) || { error: { message: fetchResponse.statusText || 'Unknown error occurred' } };
        return response.status(500).send(data);
    } catch (error) {
        const message = error.name === 'AbortError'
            ? 'Request was aborted by the client.'
            : (error.message || 'An unknown network error occurred.');
        return response.status(500).send({ error: { message, ...error } });
    }
}

export const router = express.Router();

router.post('/status', async function (request, statusResponse) {
    try {
        if (!request.body) return statusResponse.sendStatus(400);

        let apiUrl = '';
        let apiKey = '';
        let headers = {};
        let queryParams = {};

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_OPENAI).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER) {
            apiUrl = 'https://openrouter.ai/api/v1';
            apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER);
            // OpenRouter needs to pass the Referer and X-Title: https://openrouter.ai/docs#requests
            headers = { ...OPENROUTER_HEADERS };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MISTRALAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_MISTRAL).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            apiUrl = request.body.custom_url;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM);
            headers = {};
            mergeObjectWithYaml(headers, request.body.custom_include_headers);
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COHERE) {
            apiUrl = API_COHERE_V1;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CHUTES) {
            apiUrl = API_CHUTES;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CHUTES);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ELECTRONHUB) {
            apiUrl = API_ELECTRONHUB;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.ELECTRONHUB);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NANOGPT) {
            apiUrl = API_NANOGPT;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT);
            headers = {};
            queryParams = { detailed: true };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.DEEPSEEK) {
            apiUrl = new URL(request.body.reverse_proxy || API_DEEPSEEK.replace('/beta', '')).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.XAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_XAI).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AIMLAPI) {
            apiUrl = API_AIMLAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI);
            headers = { ...AIMLAPI_HEADERS };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
            apiUrl = 'https://gen.pollinations.ai/text';
            apiKey = readSecret(request.user.directories, SECRET_KEYS.POLLINATIONS);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.GROQ) {
            apiUrl = API_GROQ;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COMETAPI) {
            apiUrl = API_COMETAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI);
            headers = {};
            throw new Error('This provider is temporarily disabled.');
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT) {
            apiUrl = new URL(request.body.reverse_proxy || API_MOONSHOT).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MOONSHOT);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
            apiUrl = API_FIREWORKS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MAKERSUITE) {
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE);
            apiUrl = trimTrailingSlash(request.body.reverse_proxy || API_MAKERSUITE);
            const apiVersion = getConfigValue('gemini.apiVersion', 'v1beta');
            const modelsUrl = !apiKey && request.body.reverse_proxy
                ? `${apiUrl}/${apiVersion}/models`
                : `${apiUrl}/${apiVersion}/models?key=${apiKey}`;

            if (!apiKey && !request.body.reverse_proxy) {
                console.warn('Google AI Studio API key is missing.');
                return statusResponse.status(400).send({ error: true });
            }

            try {
                const response = await fetch(modelsUrl);

                if (response.ok) {
                    /** @type {any} */
                    const data = await response.json();
                    // Transform Google AI Studio models to OpenAI format
                    const models = data.models
                        ?.filter(model => model.supportedGenerationMethods?.includes('generateContent'))
                        ?.map(model => ({
                            id: model.name.replace('models/', ''),
                        })) || [];

                    console.info('Available Google AI Studio models:', models.map(m => m.id));
                    return statusResponse.send({ data: models });
                } else {
                    console.warn('Google AI Studio models endpoint failed:', response.status, response.statusText);
                    return statusResponse.send({ error: true, bypass: true, data: { data: [] } });
                }
            } catch (error) {
                console.error('Error fetching Google AI Studio models:', error);
                return statusResponse.send({ error: true, bypass: true, data: { data: [] } });
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CLAUDE) {
            const apiKey = readSecret(request.user.directories, SECRET_KEYS.CLAUDE);

            if (!apiKey) {
                console.warn('Claude API key is missing.');
                return statusResponse.status(400).send({ error: true });
            }

            try {
                const modelsUrl = new URL(urlJoin(API_CLAUDE, '/models'));
                modelsUrl.searchParams.set('limit', '100');
                const response = await fetch(modelsUrl, {
                    method: 'GET',
                    headers: {
                        'x-api-key': apiKey,
                        'anthropic-version': '2023-06-01',
                    },
                });

                if (response.ok) {
                    /** @type {any} */
                    const data = await response.json();
                    const models = (data.data || []).map(model => ({ id: model.id }));
                    console.info('Available Claude models:', models.map(m => m.id));
                    return statusResponse.send({ data: models });
                } else {
                    console.warn('Claude models endpoint failed:', response.status, response.statusText);
                    return statusResponse.send({ error: true, data: [] });
                }
            } catch (error) {
                console.error('Error fetching Claude models:', error);
                return statusResponse.send({ error: true, data: [] });
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AZURE_OPENAI) {
            const { azure_base_url, azure_deployment_name, azure_api_version } = request.body;
            const apiKey = readSecret(request.user.directories, SECRET_KEYS.AZURE_OPENAI);

            // 1) Validate configuration from the frontend
            if (!apiKey || !azure_base_url || !azure_deployment_name || !azure_api_version) {
                console.warn('Azure OpenAI status check failed: missing config from frontend.');
                return statusResponse.status(400).send({ error: true, message: 'Azure configuration is incomplete.' });
            }
            // 2) Build URLs using the URL API for consistency and robustness.
            const modelsUrl = new URL('/openai/models', azure_base_url);
            modelsUrl.searchParams.set('api-version', azure_api_version);

            const chatUrl = new URL(`/openai/deployments/${azure_deployment_name}/chat/completions`, azure_base_url);
            chatUrl.searchParams.set('api-version', azure_api_version);

            // Map common status codes to user-friendly error messages
            const azureStatusErrorMap = {
                400: 'API version may be invalid for this resource.',
                401: 'Invalid API key or insufficient permissions.',
                403: 'Invalid API key or insufficient permissions.',
                404: 'Endpoint URL appears incorrect (404).',
            };

            try {
                // ---- A) GET /models: fast sanity check for endpoint + api key + api version ----
                const apiConfigTest = await fetch(modelsUrl, {
                    method: 'GET',
                    headers: { 'api-key': apiKey, 'Accept': 'application/json' },
                });

                if (!apiConfigTest.ok) {
                    let errText = '';
                    try { errText = await apiConfigTest.text(); } catch { /* response body may be empty */ }

                    console.warn('Azure OpenAI GET /models failed:', apiConfigTest.status, apiConfigTest.statusText, errText || '');

                    const defaultMessage = `Azure Models endpoint error: ${apiConfigTest.statusText}`;
                    const message = azureStatusErrorMap[apiConfigTest.status] ?? defaultMessage;
                    return statusResponse.status(apiConfigTest.status).send({ error: true, message });
                }

                // ---- B) POST /chat/completions: verify deployment + read underlying model ID ----
                // Small, deterministic probe to minimize cost/latency
                const modelPayload = {
                    messages: [{ role: 'user', content: 'Say word Hi' }],
                    stream: false,
                    max_completion_tokens: 5,
                };

                const modelRequest = await fetch(chatUrl, {
                    method: 'POST',
                    headers: { 'api-key': apiKey, 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify(modelPayload),
                });

                let modelResponse;
                try {
                    modelResponse = await modelRequest.json();
                } catch {
                    modelResponse = { raw: 'Failed to parse JSON response from chat completions probe.' };
                }

                const modelId = /** @type {any} */ (modelResponse)?.model;
                if (!modelId) {
                    console.warn('Azure status check succeeded but could not find a model ID in the response.');
                    console.debug('Azure Response Body:', modelResponse);
                    // Keep a benign success to avoid UX disruption in the UI
                    return statusResponse.send({ data: [] });
                }

                console.info(color.green('Azure OpenAI connection successful. Detected model:'), modelId);
                // Consistent response format: always an array of { id }
                return statusResponse.send({ data: [{ id: modelId }] });
            } catch (error) {
                console.error('Azure OpenAI status check connection error:', error);
                return statusResponse.status(500).send({ error: true, message: 'Failed to connect to the Azure endpoint.' });
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.SILICONFLOW) {
            const defaultApiUrl = request.body.siliconflow_endpoint === SILICONFLOW_ENDPOINT.CN
                ? API_SILICONFLOW_CN : API_SILICONFLOW;
            apiUrl = defaultApiUrl;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.SILICONFLOW);
            headers = {};
            queryParams = { type: 'text', sub_type: 'chat' };
        } else {
            console.warn('This chat completion source is not supported yet.');
            return statusResponse.status(400).send({ error: true });
        }

        if (!apiKey && !request.body.reverse_proxy && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.CUSTOM) {
            console.warn('Chat Completion API key is missing.');
            return statusResponse.status(400).send({ error: true });
        }

        if (request.body.reverse_proxy) {
            console.info('Reverse proxy detected; skipping upstream status check.');
            return statusResponse.send({ data: [] });
        }

        const modelsUrl = new URL(urlJoin(apiUrl, '/models'));
        Object.keys(queryParams).forEach(key => {
            modelsUrl.searchParams.append(key, queryParams[key]);
        });
        const response = await fetch(modelsUrl, {
            method: 'GET',
            headers: {
                'Authorization': 'Bearer ' + apiKey,
                ...headers,
            },
        });

        if (response.ok) {
            /** @type {any} */
            let data = await response.json();

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS && Array.isArray(data)) {
                data = { data: data.map(model => ({ id: model.name, ...model })) };
            }

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CHUTES && Array.isArray(data?.data)) {
                data.data = data.data
                    .filter(model => model?.id)
                    .map(model => {
                        if (model.pricing?.prompt !== undefined && model.pricing?.completion !== undefined) {
                            return {
                                ...model,
                                pricing: {
                                    ...model.pricing,
                                    input: model.pricing.prompt,
                                    output: model.pricing.completion,
                                },
                            };
                        }
                        return model;
                    });
            }

            statusResponse.send(data);

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COHERE && Array.isArray(data?.models)) {
                data.data = data.models.map(model => ({ id: model.name, ...model }));
            }

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER && Array.isArray(data?.data)) {
                let models = [];

                data.data.forEach(model => {
                    const context_length = model.context_length;
                    const tokens_dollar = Number(1 / (1000 * model.pricing?.prompt));
                    const tokens_rounded = (Math.round(tokens_dollar * 1000) / 1000).toFixed(0);
                    models[model.id] = {
                        tokens_per_dollar: tokens_rounded + 'k',
                        context_length: context_length,
                    };
                });

                console.info('Available OpenRouter models:', models);
            } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MISTRALAI) {
                const models = data?.data;
                console.info(models);
            } else {
                const models = data?.data;

                if (Array.isArray(models)) {
                    const modelIds = models.filter(x => x && typeof x === 'object').map(x => x.id).sort();
                    console.info('Available models:', modelIds);
                } else {
                    console.warn('Chat Completion endpoint did not return a list of models.');
                }
            }
        } else {
            console.error('Chat Completion status check failed. Either Access Token is incorrect or API endpoint is down.');
            statusResponse.send({ error: true, data: { data: [] } });
        }
    } catch (e) {
        console.error(e);

        if (!statusResponse.headersSent) {
            statusResponse.send({ error: true });
        } else {
            statusResponse.end();
        }
    }
});

router.post('/word-replacements', function (request, response) {
    const enabled = getWordReplacementEnabled(request);
    const value = request.body?.value;
    const processed = enforceWordReplacementsOnResponse(value, enabled);
    return response.send({ value: processed });
});

router.post('/bias', async function (request, response) {
    if (!request.body || !Array.isArray(request.body))
        return response.sendStatus(400);

    try {
        const result = {};
        const model = getTokenizerModel(String(request.query.model || ''));

        // no bias for claude
        if (model == 'claude') {
            return response.send(result);
        }

        let encodeFunction;

        if (sentencepieceTokenizers.includes(model)) {
            const tokenizer = getSentencepiceTokenizer(model);
            const instance = await tokenizer?.get();
            if (!instance) {
                console.error('Tokenizer not initialized:', model);
                return response.send({});
            }
            encodeFunction = (text) => new Uint32Array(instance.encodeIds(text));
        } else if (webTokenizers.includes(model)) {
            const tokenizer = getWebTokenizer(model);
            const instance = await tokenizer?.get();
            if (!instance) {
                console.warn('Tokenizer not initialized:', model);
                return response.send({});
            }
            encodeFunction = (text) => new Uint32Array(instance.encode(text));
        } else {
            const tokenizer = getTiktokenTokenizer(model);
            encodeFunction = (tokenizer.encode.bind(tokenizer));
        }

        for (const entry of request.body) {
            if (!entry || !entry.text) {
                continue;
            }

            try {
                const tokens = getEntryTokens(entry.text, encodeFunction);

                for (const token of tokens) {
                    result[token] = entry.value;
                }
            } catch {
                console.warn('Tokenizer failed to encode:', entry.text);
            }
        }

        // not needed for cached tokenizers
        //tokenizer.free();
        return response.send(result);

        /**
         * Gets tokenids for a given entry
         * @param {string} text Entry text
         * @param {(string) => Uint32Array} encode Function to encode text to token ids
         * @returns {Uint32Array} Array of token ids
         */
        function getEntryTokens(text, encode) {
            // Get raw token ids from JSON array
            if (text.trim().startsWith('[') && text.trim().endsWith(']')) {
                try {
                    const json = JSON.parse(text);
                    if (Array.isArray(json) && json.every(x => typeof x === 'number')) {
                        return new Uint32Array(json);
                    }
                } catch {
                    // ignore
                }
            }

            // Otherwise, get token ids from tokenizer
            return encode(text);
        }
    } catch (error) {
        console.error(error);
        return response.send({});
    }
});

/**
 * Converts a chat completions message content array into Responses API content parts.
 * Mirrors response_input_*_param.py shapes (input_text / input_image / input_file).
 * @param {any} content
 * @returns {any}
 */
function convertResponsesContent(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return content;
    return content.map(part => {
        if (!part || typeof part !== 'object') return part;
        if (part.type === 'text') return { type: 'input_text', text: part.text ?? '' };
        if (part.type === 'image_url') {
            const url = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
            const detail = part.image_url?.detail;
            return { type: 'input_image', image_url: url, ...(detail ? { detail } : {}) };
        }
        if (part.type === 'input_audio' || part.type === 'audio') {
            return { type: 'input_audio', input_audio: part.input_audio || part.audio };
        }
        if (part.type === 'file') {
            const f = part.file || {};
            /** @type {any} */
            const out = { type: 'input_file' };
            if (f.file_id) out.file_id = f.file_id;
            if (f.file_data) out.file_data = f.file_data;
            if (f.file_url) out.file_url = f.file_url;
            if (f.filename) out.filename = f.filename;
            return out;
        }
        // Already a Responses-API-shaped part (input_text/input_image/input_file/output_text)
        return part;
    });
}

/**
 * Converts a chat completions request body in-place to the OpenAI Responses API format.
 * @param {any} requestBody The request body to transform
 */
function convertToResponsesApiRequest(requestBody) {
    // messages → input; first developer message → instructions param, rest → system role
    if (requestBody.messages) {
        let firstSystemUsed = false;
        const input = [];
        for (const msg of requestBody.messages) {
            const cleanMsg = { role: msg.role, content: convertResponsesContent(msg.content) };
            if (msg.name) cleanMsg.name = msg.name;

            if (cleanMsg.role === 'developer' && !firstSystemUsed) {
                requestBody.instructions = typeof cleanMsg.content === 'string'
                    ? cleanMsg.content
                    : (Array.isArray(cleanMsg.content) ? cleanMsg.content.map(p => p.text ?? '').join('') : '');
                firstSystemUsed = true;
            } else if (cleanMsg.role === 'developer') {
                input.push({ ...cleanMsg, role: 'system' });
            } else {
                input.push(cleanMsg);
            }
        }
        requestBody.input = input;
        delete requestBody.messages;
    }

    // max_tokens / max_completion_tokens → max_output_tokens
    if (requestBody.max_completion_tokens) {
        requestBody.max_output_tokens = requestBody.max_completion_tokens;
        delete requestBody.max_completion_tokens;
    } else if (requestBody.max_tokens) {
        requestBody.max_output_tokens = requestBody.max_tokens;
    }
    delete requestBody.max_tokens;

    // reasoning_effort → reasoning.effort, always request summaries
    requestBody.reasoning = {
        ...(requestBody.reasoning_effort ? { effort: requestBody.reasoning_effort } : {}),
        summary: 'detailed',
    };
    delete requestBody.reasoning_effort;

    // response_format → text.format, verbosity → text.verbosity
    if (requestBody.response_format || requestBody.verbosity) {
        requestBody.text = {};
        if (requestBody.response_format) {
            requestBody.text.format = requestBody.response_format;
            delete requestBody.response_format;
        }
        if (requestBody.verbosity) {
            requestBody.text.verbosity = requestBody.verbosity;
            delete requestBody.verbosity;
        }
    }

    // Don't store conversations on OpenAI's servers by default; allow opt-in via flag
    if (typeof requestBody.responses_store === 'boolean') {
        requestBody.store = requestBody.responses_store;
    } else {
        requestBody.store = true;
    }
    delete requestBody.responses_store;

    // Remove unsupported parameters
    if (!requestBody.reasoning.effort || requestBody.reasoning.effort === 'none') {
        delete requestBody.reasoning;
    } else {
        delete requestBody.temperature;
        delete requestBody.top_p;
    }
    delete requestBody.top_k;
    delete requestBody.n;
    delete requestBody.logit_bias;
    delete requestBody.logprobs;
    delete requestBody.top_logprobs;
    delete requestBody.prompt;
    delete requestBody.frequency_penalty;
    delete requestBody.presence_penalty;
    delete requestBody.stop;
    delete requestBody.seed;
}

/**
 * Builds the `tools` and `include` arrays for an outgoing Responses API request based on
 * frontend-supplied flags. Mirrors *_param.py TypedDicts in the OpenAI SDK.
 * Mutates `requestBody` in place.
 * @param {any} originalBody The original request.body from the client (for flags)
 * @param {any} requestBody The outgoing Responses API request body to mutate
 */
function buildResponsesTools(originalBody, requestBody) {
    /** @type {any[]} */
    const tools = Array.isArray(requestBody.tools) ? requestBody.tools : [];
    /** @type {string[]} */
    const include = Array.isArray(requestBody.include) ? requestBody.include : [];

    const addInclude = (key) => { if (!include.includes(key)) include.push(key); };

    // file_search
    if (originalBody.enable_file_search && Array.isArray(originalBody.openai_vector_store_ids) && originalBody.openai_vector_store_ids.length) {
        /** @type {any} */
        const tool = {
            type: 'file_search',
            vector_store_ids: originalBody.openai_vector_store_ids,
        };
        if (Number.isFinite(originalBody.file_search_max_num_results)) {
            tool.max_num_results = Math.max(1, Math.min(50, Number(originalBody.file_search_max_num_results)));
        }
        /** @type {any} */
        const ranking = {};
        if (originalBody.file_search_ranker) ranking.ranker = originalBody.file_search_ranker;
        if (Number.isFinite(originalBody.file_search_score_threshold)) {
            ranking.score_threshold = Math.max(0, Math.min(1, Number(originalBody.file_search_score_threshold)));
        }
        if (Object.keys(ranking).length) tool.ranking_options = ranking;
        if (!tools.some(t => t?.type === 'file_search')) tools.push(tool);
        addInclude('file_search_call.results');
    }

    // web_search (existing behavior preserved)
    if (originalBody.enable_web_search) {
        if (!tools.some(t => t?.type === 'web_search' || t?.type === 'web_search_preview')) {
            tools.push({ type: 'web_search' });
        }
    }

    // code_interpreter
    if (originalBody.enable_code_interpreter) {
        if (!tools.some(t => t?.type === 'code_interpreter')) {
            tools.push({ type: 'code_interpreter', container: { type: 'auto' } });
        }
        addInclude('code_interpreter_call.outputs');
    }

    // user-defined custom function tools (Responses API shape: flat, not { function: { ... } })
    if (Array.isArray(originalBody.custom_functions)) {
        for (const fn of originalBody.custom_functions) {
            if (!fn?.name) continue;
            tools.push({
                type: 'function',
                name: fn.name,
                description: fn.description ?? null,
                parameters: fn.parameters ?? null,
                strict: fn.strict ?? true,
            });
        }
    }

    // skills (inline base64 zip OR skill_reference)
    if (Array.isArray(originalBody.skills)) {
        for (const sk of originalBody.skills) {
            if (!sk?.type) continue;
            if (sk.type === 'skill_reference' && sk.skill_id) {
                const entry = { type: 'skill_reference', skill_id: sk.skill_id };
                if (sk.version) entry.version = sk.version;
                tools.push(entry);
            } else if (sk.type === 'inline' && sk.name && sk.data_b64) {
                tools.push({
                    type: 'inline',
                    name: sk.name,
                    description: sk.description ?? '',
                    source: { type: 'base64', media_type: 'application/zip', data: sk.data_b64 },
                });
            }
        }
    }

    if (tools.length) requestBody.tools = tools;
    if (include.length) requestBody.include = include;

    // tool_choice passthrough (string or object)
    if (originalBody.tool_choice !== undefined && requestBody.tool_choice === undefined) {
        requestBody.tool_choice = originalBody.tool_choice;
    }
    if (typeof originalBody.parallel_tool_calls === 'boolean') {
        requestBody.parallel_tool_calls = originalBody.parallel_tool_calls;
    }
}

router.post('/generate', async function (request, response) {
    try {
        if (!request.body) return response.status(400).send({ error: true });
        const wordReplacementsEnabled = getWordReplacementEnabled(request);

        const postProcessingType = request.body.custom_prompt_post_processing;
        if (Array.isArray(request.body.messages) && postProcessingType) {
            console.info('Applying custom prompt post-processing of type', postProcessingType);
            request.body.messages = postProcessPrompt(
                request.body.messages,
                postProcessingType,
                getPromptNames(request));
        }

        if (request.body.json_schema?.value) {
            request.body.json_schema.value = flattenSchema(request.body.json_schema.value, request.body.chat_completion_source);
        }

        switch (request.body.chat_completion_source) {
            case CHAT_COMPLETION_SOURCES.CLAUDE: return await sendClaudeRequest(request, response);
            case CHAT_COMPLETION_SOURCES.AI21: return await sendAI21Request(request, response);
            case CHAT_COMPLETION_SOURCES.MAKERSUITE: return await sendMakerSuiteRequest(request, response);
            case CHAT_COMPLETION_SOURCES.VERTEXAI: return await sendMakerSuiteRequest(request, response);
            case CHAT_COMPLETION_SOURCES.MISTRALAI: return await sendMistralAIRequest(request, response);
            case CHAT_COMPLETION_SOURCES.COHERE: return await sendCohereRequest(request, response);
            case CHAT_COMPLETION_SOURCES.DEEPSEEK: return await sendDeepSeekRequest(request, response);
            case CHAT_COMPLETION_SOURCES.AIMLAPI: return await sendAimlapiRequest(request, response);
            case CHAT_COMPLETION_SOURCES.XAI: return await sendXaiRequest(request, response);
            case CHAT_COMPLETION_SOURCES.CHUTES: return await sendChutesRequest(request, response);
            case CHAT_COMPLETION_SOURCES.ELECTRONHUB: return await sendElectronHubRequest(request, response);
            case CHAT_COMPLETION_SOURCES.AZURE_OPENAI: return await sendAzureOpenAIRequest(request, response);
        }

        let apiUrl;
        let apiKey;
        let headers;
        let bodyParams;
        const isTextCompletion = Boolean(request.body.model && TEXT_COMPLETION_MODELS.includes(request.body.model)) || typeof request.body.messages === 'string';

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_OPENAI).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI);
            headers = {};
            bodyParams = {
                logprobs: request.body.logprobs,
                top_logprobs: undefined,
            };

            // Adjust logprobs params for Chat Completions API, which expects { top_logprobs: number; logprobs: boolean; }
            if (!isTextCompletion && bodyParams.logprobs > 0) {
                bodyParams.top_logprobs = bodyParams.logprobs;
                bodyParams.logprobs = true;
            }

            if (getConfigValue('openai.randomizeUserId', false, 'boolean')) {
                bodyParams['user'] = uuidv4();
            }

            embedOpenRouterMedia(request.body.messages, { audio: true, video: false });
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER) {
            apiUrl = 'https://openrouter.ai/api/v1';
            apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER);
            // OpenRouter needs to pass the Referer and X-Title: https://openrouter.ai/docs#requests
            headers = { ...OPENROUTER_HEADERS };
            const includeReasoning = Boolean(request.body.include_reasoning);
            bodyParams = {
                transforms: getOpenRouterTransforms(request),
                plugins: getOpenRouterPlugins(request),
                reasoning: {
                    exclude: !includeReasoning,
                },
            };

            if (request.body.min_p !== undefined) {
                bodyParams['min_p'] = request.body.min_p;
            }

            if (request.body.top_a !== undefined) {
                bodyParams['top_a'] = request.body.top_a;
            }

            if (request.body.repetition_penalty !== undefined) {
                bodyParams['repetition_penalty'] = request.body.repetition_penalty;
            }

            if (Array.isArray(request.body.provider) && request.body.provider.length > 0) {
                bodyParams['provider'] = {
                    allow_fallbacks: request.body.allow_fallbacks ?? true,
                    order: request.body.provider ?? [],
                };
            }

            if (Array.isArray(request.body.quantizations) && request.body.quantizations.length > 0) {
                bodyParams['provider'] ??= {};
                bodyParams['provider']['quantizations'] = request.body.quantizations;
            }

            if (request.body.use_fallback) {
                bodyParams['route'] = 'fallback';
            }

            if (request.body.reasoning_effort) {
                bodyParams['reasoning']['effort'] = request.body.reasoning_effort;
            }

            if (request.body.verbosity) {
                bodyParams['verbosity'] = request.body.verbosity;
            }

            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        name: request.body.json_schema.name,
                        strict: request.body.json_schema.strict ?? true,
                        schema: request.body.json_schema.value,
                    },
                };
            }

            const isClaude = /^anthropic\/claude/.test(request.body.model);
            const isGemini = /google\/gemini/.test(request.body.model);
            const isCacheableGemini = isGemini && await isOpenRouterModelCacheable(request.body.model);
            const enableGeminiSystemPromptCache = getConfigValue('gemini.enableSystemPromptCache', false, 'boolean');

            if (Array.isArray(request.body.messages)) {
                embedOpenRouterMedia(request.body.messages, { audio: true, video: true });
                addOpenRouterSignatures(request.body.messages, request.body.model);

                if (isClaude) {
                    if (enableSystemPromptCache) {
                        cachingSystemPromptForOpenRouter(request.body.messages, getCacheTTL(request));
                    }

                    if (cachingAtDepth !== -1) {
                        cachingAtDepthForOpenRouterClaude(request.body.messages, cachingAtDepth, getCacheTTL(request));
                    }
                }

                if (isCacheableGemini && enableGeminiSystemPromptCache) {
                    cachingSystemPromptForOpenRouter(request.body.messages);
                }
            }

            if (isGemini) {
                bodyParams['safety_settings'] = GEMINI_SAFETY;
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            apiUrl = request.body.custom_url;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM);
            headers = {};
            bodyParams = {
                logprobs: request.body.logprobs,
                top_logprobs: undefined,
            };

            // Adjust logprobs params for Chat Completions API, which expects { top_logprobs: number; logprobs: boolean; }
            if (!isTextCompletion && bodyParams.logprobs > 0) {
                bodyParams.top_logprobs = bodyParams.logprobs;
                bodyParams.logprobs = true;
            }

            mergeObjectWithYaml(bodyParams, request.body.custom_include_body);
            mergeObjectWithYaml(headers, request.body.custom_include_headers);
            embedOpenRouterMedia(request.body.messages, { audio: true, video: false });
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.PERPLEXITY) {
            apiUrl = API_PERPLEXITY;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.PERPLEXITY);
            headers = {};
            bodyParams = {
                reasoning_effort: request.body.reasoning_effort,
            };
            request.body.messages = postProcessPrompt(request.body.messages, PROMPT_PROCESSING_TYPE.STRICT, getPromptNames(request));
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        schema: request.body.json_schema.value,
                    },
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.GROQ) {
            apiUrl = API_GROQ;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ);
            headers = {};
            bodyParams = {};
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        name: request.body.json_schema.name,
                        description: request.body.json_schema.description,
                        schema: request.body.json_schema.value,
                        strict: request.body.json_schema.strict ?? true,
                    },
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
            apiUrl = API_FIREWORKS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS);
            headers = {};
            bodyParams = {};
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        name: request.body.json_schema.name,
                        description: request.body.json_schema.description,
                        schema: request.body.json_schema.value,
                        strict: request.body.json_schema.strict ?? true,
                    },
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NANOGPT) {
            apiUrl = API_NANOGPT;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT);
            headers = {};
            bodyParams = {};
            if (request.body.enable_web_search && !/:online$/.test(request.body.model)) {
                request.body.model = `${request.body.model}:online`;
            }
            if (request.body.min_p !== undefined) {
                bodyParams['min_p'] = request.body.min_p;
            }
            if (request.body.top_a !== undefined) {
                bodyParams['top_a'] = request.body.top_a;
            }
            if (request.body.repetition_penalty !== undefined) {
                bodyParams['repetition_penalty'] = request.body.repetition_penalty;
            }
            if (request.body.reasoning_effort) {
                const effort = NANOGPT_REASONING_EFFORT_MAP[request.body.reasoning_effort];
                bodyParams['reasoning'] = { effort: effort };
            }

            const enableSystemPromptCache = getConfigValue('claude.enableSystemPromptCache', false, 'boolean');
            const isClaude = /(?:^|\/)claude[-_]/.test(request.body.model);
            if (enableSystemPromptCache && isClaude) {
                bodyParams['cache_control'] = {
                    'enabled': true,
                    'ttl': getCacheTTL(request),
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
            apiUrl = API_POLLINATIONS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.POLLINATIONS);
            headers = {};
            bodyParams = {
                reasoning_effort: request.body.reasoning_effort,
                private: true,
                referrer: 'sillytavern',
                seed: request.body.seed ?? Math.floor(Math.random() * 99999999),
            };
            if (request.body.json_schema) {
                setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT) {
            apiUrl = new URL(request.body.reverse_proxy || API_MOONSHOT).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MOONSHOT);
            headers = {};
            bodyParams = {
                thinking: {
                    type: request.body.include_reasoning ? 'enabled' : 'disabled',
                },
            };
            request.body.json_schema
                ? setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema)
                : addAssistantPrefix(request.body.messages, [], 'partial');
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COMETAPI) {
            apiUrl = API_COMETAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI);
            headers = {};
            bodyParams = {
                reasoning_effort: request.body.reasoning_effort,
            };
            throw new Error('This provider is temporarily disabled.');
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ZAI) {
            const defaultApiUrl = request.body.zai_endpoint === ZAI_ENDPOINT.CODING ? API_ZAI_CODING : API_ZAI_COMMON;
            apiUrl = new URL(request.body.reverse_proxy || defaultApiUrl).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.ZAI);
            headers = {
                'Accept-Language': 'en-US,en',
            };
            bodyParams = {
                thinking: {
                    type: request.body.include_reasoning ? 'enabled' : 'disabled',
                },
            };
            if (request.body.json_schema) {
                setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.SILICONFLOW) {
            const defaultApiUrl = request.body.siliconflow_endpoint === SILICONFLOW_ENDPOINT.CN
                ? API_SILICONFLOW_CN : API_SILICONFLOW;
            apiUrl = defaultApiUrl;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.SILICONFLOW);
            headers = {};
            bodyParams = {};
            if (request.body.json_schema) {
                setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
            }
        } else {
            console.warn('This chat completion source is not supported yet.');
            return response.status(400).send({ error: true });
        }

        // A few of OpenAIs reasoning models support reasoning effort
        if (request.body.reasoning_effort && [CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI].includes(request.body.chat_completion_source)) {
            if (OPENAI_REASONING_EFFORT_MODELS.includes(request.body.model)) {
                bodyParams['reasoning_effort'] = OPENAI_FIXED_REASONING_EFFORT[request.body.model] ?? OPENAI_REASONING_EFFORT_MAP[request.body.reasoning_effort] ?? request.body.reasoning_effort;
            }
        }

        if (request.body.verbosity && [CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI].includes(request.body.chat_completion_source)) {
            if (OPENAI_VERBOSITY_MODELS.test(request.body.model)) {
                bodyParams['verbosity'] = request.body.verbosity;
            }
        }

        if ([CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI].includes(request.body.chat_completion_source)) {
            bodyParams['service_tier'] = request.body.service_tier;
        }

        if (!apiKey && !request.body.reverse_proxy && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.CUSTOM) {
            console.warn('OpenAI API key is missing.');
            return response.status(400).send({ error: true });
        }

        // Add custom stop sequences
        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        // Determine if we should use the OpenAI Responses API for this model
        const useResponsesApi = !isTextCompletion
            && request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI
            && OPENAI_RESPONSES_API_MODELS.flat().some(m => {
                const model = request.body.model;
                if (!model) return false;
                if (m instanceof RegExp) return m.test(model);
                return model.startsWith(m);
            });

        const textPrompt = isTextCompletion ? convertTextCompletionPrompt(request.body.messages) : '';
        const endpointUrl = useResponsesApi
            ? `${apiUrl}/responses`
            : isTextCompletion && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.OPENROUTER
                ? `${apiUrl}/completions`
                : `${apiUrl}/chat/completions`;

        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        if (!isTextCompletion && Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.json_schema && !bodyParams['response_format']) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    strict: request.body.json_schema.strict ?? true,
                    schema: request.body.json_schema.value,
                },
            };
        }

        const promptReference = isTextCompletion === true
            ? textPrompt
            : (request.body.reverse_proxy && request.body.prompt && typeof request.body.prompt === 'object' && !Array.isArray(request.body.prompt)
                ? request.body.prompt
                : undefined);

        // Strip non-standard fields from messages (extensions may add metadata like 'source', 'swipe_info', etc.)
        if (Array.isArray(request.body.messages)) {
            request.body.messages = request.body.messages.map(msg => {
                const clean = { role: msg.role, content: msg.content };
                if (msg.name) clean.name = msg.name;
                if (msg.tool_calls) clean.tool_calls = msg.tool_calls;
                if (msg.tool_call_id) clean.tool_call_id = msg.tool_call_id;
                return clean;
            });
        }

        const requestBody = {
            'messages': isTextCompletion === false ? request.body.messages : undefined,
            'prompt': promptReference,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'max_completion_tokens': request.body.max_completion_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'top_k': request.body.top_k,
            'stop': isTextCompletion === false ? request.body.stop : undefined,
            'logit_bias': request.body.logit_bias,
            'seed': request.body.seed,
            'n': request.body.n,
            ...bodyParams,
        };

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            excludeKeysByYaml(requestBody, request.body.custom_exclude_body);
        }

        // Transform request body for the OpenAI Responses API
        if (useResponsesApi) {
            convertToResponsesApiRequest(requestBody);
            buildResponsesTools(request.body, requestBody);
        }

        // Chat Completions web_search_options for non-Responses OpenAI / CUSTOM
        if (!useResponsesApi && request.body.enable_web_search && !isTextCompletion
            && [CHAT_COMPLETION_SOURCES.OPENAI, CHAT_COMPLETION_SOURCES.CUSTOM].includes(request.body.chat_completion_source)) {
            /** @type {any} */
            const rb = requestBody;
            rb.web_search_options = rb.web_search_options || {};
        }

        if (request.body.model?.startsWith('gpt') && requestBody.top_k !== undefined) {
            delete requestBody.top_k;
        }

        /** @type {import('node-fetch').RequestInit} */
        const config = {
            method: 'post',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
                ...headers,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug(useResponsesApi ? 'Responses API request:' : 'Chat Completion request:', requestBody);

        const fetchResponse = await fetch(endpointUrl, config);

        if (request.body.stream) {
            console.info('Streaming request in progress');
            if (useResponsesApi) {
                response.setHeader('X-Response-Format', 'responses');
            }
            forwardFetchResponseWithWordReplacements(fetchResponse, response, wordReplacementsEnabled);
            return;
        }

        if (fetchResponse.ok) {
            /** @type {any} */
            const json = await fetchResponse.json();
            if (useResponsesApi) {
                response.setHeader('X-Response-Format', 'responses');
            }
            sendWithWordReplacements(response, json, wordReplacementsEnabled);
            console.debug(useResponsesApi ? 'Responses API response:' : 'Chat Completion response:', json);
            return; // Response already sent by sendWithWordReplacements
        } else {
            const responseText = await fetchResponse.text();
            const errorData = tryParse(responseText);

            const message = fetchResponse.statusText || 'Unknown error occurred';
            const quota_error = fetchResponse.status === 429 && errorData?.error?.type === 'insufficient_quota';
            console.error('Chat completion request error: ', message, responseText);

            if (!response.headersSent) {
                response.send({ error: { message }, quota_error: quota_error });
            } else if (!response.writableEnded) {
                response.write(responseText);
            } else {
                response.end();
            }
        }
    } catch (error) {
        console.error('Generation failed', error);
        const message = error.code === 'ECONNREFUSED'
            ? `Connection refused: ${error.message}`
            : error.message || 'Unknown error occurred';

        if (!response.headersSent) {
            response.status(502).send({ error: { message, ...error } });
        } else {
            response.end();
        }
    }
});

const multimodalModels = express.Router();

multimodalModels.post('/pollinations', async (_req, res) => {
    try {
        const response = await fetch('https://gen.pollinations.ai/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data)) {
            return res.json([]);
        }

        const multimodalModels = data
            .filter(m => Array.isArray(m?.input_modalities))
            .filter(m => m.input_modalities.includes('image'))
            .map(m => m.name);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/aimlapi', async (_req, res) => {
    try {
        const response = await fetch('https://api.aimlapi.com/v1/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            return res.json([]);
        }

        const multimodalModels = data.data.filter(m => m?.features?.includes('openai/chat-completion.vision')).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/nanogpt', async (_req, res) => {
    try {
        const response = await fetch('https://nano-gpt.com/api/v1/models?detailed=true');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            return res.json([]);
        }

        const multimodalModels = data.data.filter(m => m?.capabilities?.vision).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/electronhub', async (_req, res) => {
    try {
        const response = await fetch('https://api.electronhub.ai/v1/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const multimodalModels = data.data.filter(m => m.metadata?.vision).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/chutes', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.CHUTES);

        if (!key) {
            return res.json([]);
        }

        const response = await fetch('https://llm.chutes.ai/v1/models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        const data = await response.json();

        const modelsData = /** @type {{object: string, data: Array<{id: string, input_modalities?: string[]}>}} */ (data);
        const multimodalModels = modelsData.data
            .filter(m => m.input_modalities?.includes('image'))
            .map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/mistral', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.MISTRALAI);

        if (!key) {
            return res.json([]);
        }

        const response = await fetch('https://api.mistral.ai/v1/models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const multimodalModels = data.data.filter(m => m.capabilities?.vision).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/xai', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.XAI);

        if (!key) {
            return res.json([]);
        }

        // xAI's /models endpoint doesn't return modality info, so we must use /language-models instead
        const response = await fetch('https://api.x.ai/v1/language-models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const multimodalModels = data.models.filter(m => m.input_modalities?.includes('image')).map(m => m.id);
        if (!multimodalModels.includes('grok-4-0709')) {
            // The endpoint says it doesn't support images, but it does
            multimodalModels.push('grok-4-0709');
        }
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/moonshot', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.MOONSHOT);

        if (!key) {
            return res.json([]);
        }

        const response = await fetch('https://api.moonshot.ai/v1/models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        const multimodalModels = data.data.filter(m => m.supports_image_in).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.use('/multimodal-models', multimodalModels);

router.post('/process', async function (request, response) {
    try {
        if (!Array.isArray(request.body.messages)) {
            return response.status(400).send({ error: 'Invalid messages format' });
        }

        if (!Object.values(PROMPT_PROCESSING_TYPE).includes(request.body.type)) {
            return response.status(400).send({ error: 'Unknown processing type' });
        }

        const messages = postProcessPrompt(request.body.messages, request.body.type, getPromptNames(request));
        return response.send({ messages });
    } catch (error) {
        console.error(error);
        return response.sendStatus(500);
    }
});
