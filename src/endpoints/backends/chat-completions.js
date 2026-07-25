/* eslint-disable dot-notation */
import process from 'node:process';
import util from 'node:util';
import fs from 'node:fs';
import path from 'node:path';
import { Transform } from 'node:stream';
import express from 'express';
import fetch from 'node-fetch';
import urlJoin from 'url-join';
import { sync as writeFileAtomicSync } from 'write-file-atomic';

import {
    AIMLAPI_HEADERS,
    AZURE_OPENAI_KEYS,
    CHAT_COMPLETION_SOURCES,
    CLAUDE_MODEL_CAPABILITIES,
    GEMINI_SAFETY,
    getClaudeCapabilities,
    getClaudePricing,
    NANOGPT_REASONING_EFFORT_MAP,
    OPENAI_FIXED_REASONING_EFFORT,
    OPENAI_PRO_REASONING_MODELS,
    OPENAI_PROMPT_CACHE_OPTIONS_MODELS,
    OPENAI_REASONING_EFFORT_MAP,
    OPENAI_REASONING_EFFORT_MODELS,
    OPENAI_RESPONSES_API_MODELS,
    OPENAI_VERBOSITY_MODELS,
    OPENROUTER_HEADERS,
    VERTEX_SAFETY,
    SILICONFLOW_ENDPOINT,
    MINIMAX_ENDPOINT,
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
const API_MINIMAX = 'https://api.minimax.io/v1';
const API_MINIMAX_CN = 'https://api.minimaxi.com/v1';
const API_WORKERS_AI = 'https://api.cloudflare.com/client/v4/accounts';

/**
 * Module-scoped Claude caching defaults from config.yaml.
 */
const configExtendedTTL = getConfigValue('claude.extendedTTL', false, 'boolean');
const configEnableSystemPromptCache = getConfigValue('claude.enableSystemPromptCache', false, 'boolean');
const configCachingAtDepth = (() => {
    const value = getConfigValue('claude.cachingAtDepth', -1, 'number');
    return Number.isInteger(value) && value >= 0 ? value : -1;
})();

/**
 * Resolves Claude caching configuration by merging config.yaml defaults
 * with request.body overrides from the frontend.
 * @param {import('express').Request} request
 * @returns {{ enableSystemPromptCache: boolean, cachingAtDepth: number, ttl: string }}
 */
function resolveClaudeCachingConfig(request) {
    const extendedTTL = request?.body?.claude_extendedTTL ?? configExtendedTTL;
    const ttl = extendedTTL ? '1h' : '5m';

    const enableSystemPromptCache = (request?.body?.claude_enable_caching !== undefined)
        ? Boolean(request.body.claude_enable_caching)
        : configEnableSystemPromptCache;

    let cachingAtDepth = configCachingAtDepth;
    if (request?.body?.claude_enable_caching_at_depth === false) {
        cachingAtDepth = -1;
    } else if (request?.body?.claude_enable_caching_at_depth === true && cachingAtDepth < 0) {
        cachingAtDepth = 0;
    }
    if (request?.body?.claude_enable_caching === false) {
        cachingAtDepth = -1;
    }

    return { enableSystemPromptCache, cachingAtDepth, ttl };
}

/**
 * Cache for cacheable (writing) OpenRouter model IDs.
 * @type {string[]}
 */

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
function formatStreamingResponse(raw, responseHeaders = null) {
    let thinking = '';
    let text = '';
    let streamedModel = null;
    /** @type {any} */
    let streamedUsage = null;

    for (const line of raw.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr || jsonStr === '[DONE]') continue;
        let parsed;
        try { parsed = JSON.parse(jsonStr); } catch { continue; }

        if (parsed?.type === 'message_start' && parsed.message) {
            streamedModel = parsed.message.model;
            streamedUsage = { ...(parsed.message.usage ?? {}) };
        }

        // Only the final message_delta carries the real output token count.
        if (parsed?.type === 'message_delta' && parsed.usage?.output_tokens != null && streamedUsage) {
            streamedUsage.output_tokens = parsed.usage.output_tokens;
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
    if (streamedUsage) {
        // Anthropic responses carry the org that served the request; the prompt
        // cache is scoped per org, so a flip here explains any cache_read: 0
        const canReadHeaders = responseHeaders && typeof responseHeaders.get === 'function';
        parts.push(formatClaudeUsageMeta(streamedModel, streamedUsage, {
            org: canReadHeaders ? responseHeaders.get('anthropic-organization-id') : '',
            req: canReadHeaders ? responseHeaders.get('request-id') : '',
        }));
    }
    return parts.length ? parts.join('\n\n') : '(no text content)';
}

/**
 * Stop reasons that end a generation successfully (HTTP 200) but produce no
 * usable output. Without explicit handling they surface as an empty message.
 */
const CLAUDE_ABNORMAL_STOP_REASONS = new Set(['refusal', 'model_context_window_exceeded']);

/**
 * Describe an abnormal Claude stop reason in user-facing terms.
 * @param {string} stopReason Value of `stop_reason`
 * @param {string} [category] Value of `stop_details.category` ("cyber", "bio", …)
 * @returns {string|null} Message to surface, or null when the generation ended normally
 */
function describeClaudeStop(stopReason, category) {
    if (!stopReason || !CLAUDE_ABNORMAL_STOP_REASONS.has(stopReason)) {
        return null;
    }

    if (stopReason === 'refusal') {
        // Safety classifiers (Fable 5, Sonnet 5, Opus 5) decline as a 200 response.
        return `Claude declined this request (stop_reason: refusal${category ? `, category: ${category}` : ''}). Any partial output was discarded.`;
    }

    return 'Claude stopped because the model context window was exceeded, not max_tokens. Reduce the context size and try again.';
}

/**
 * Watch a forwarded SSE stream for an abnormal Claude stop reason.
 * A refusal arrives as an ordinary stream that simply stops producing text, so
 * this writes an error frame the frontend already knows how to toast.
 * @param {import('express').Response} to Destination response
 * @returns {(text: string) => void} Chunk inspector, safe to call on every chunk
 */
function makeClaudeStopWatcher(to) {
    let notified = false;
    return (text) => {
        if (notified || !text.includes('stop_reason')) {
            return;
        }

        for (const line of text.split('\n')) {
            if (!line.startsWith('data:')) {
                continue;
            }

            // Chunk boundaries can split a frame; a partial parse just yields null.
            const parsed = tryParse(line.slice(5).trim());
            const payload = parsed?.delta ?? parsed?.message;
            const notice = describeClaudeStop(payload?.stop_reason, payload?.stop_details?.category);
            if (!notice) {
                continue;
            }

            notified = true;
            console.warn(color.red(notice));
            if (to && !to.writableEnded) {
                to.write(`data: ${JSON.stringify({ error: { message: notice } })}\n\n`);
            }
            return;
        }
    };
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

        const watchForStop = makeClaudeStopWatcher(to);

        if (!isEnabled) {
            const chunks = [];
            from.body.on('data', (chunk) => {
                chunks.push(Buffer.from(chunk));
                watchForStop(chunk.toString('utf8'));
            });
            from.body.pipe(to);

            to.socket.on('close', function () {
                destroySource();
                endResponse();
            });

            from.body.on('end', function () {
                console.info('Streaming request finished.\n' + formatStreamingResponse(Buffer.concat(chunks).toString('utf8'), from.headers));
                endResponse();
            });

            from.body.on('error', function (error) {
                console.error('Streaming request error:', error);
                endResponse();
            });
        } else {
            const chunks = [];
            from.body.on('data', (chunk) => {
                chunks.push(Buffer.from(chunk));
                watchForStop(chunk.toString('utf8'));
            });
            const transformStream = createWordReplacementStream(isEnabled);
            from.body.pipe(transformStream).pipe(to);

            to.socket.on('close', function () {
                destroySource();
                transformStream.end();
                endResponse();
            });

            transformStream.on('end', function () {
                console.info('Streaming request finished.\n' + formatStreamingResponse(Buffer.concat(chunks).toString('utf8'), from.headers));
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
 * Runs a non-streaming Claude agentic loop: makes API calls, handles tool_use responses,
 * and returns when Claude is done or a custom tool is encountered.
 * Built-in tools (web_search) are resolved by the Claude API internally and appear as
 * server_tool_use/server_tool_result blocks — they don't produce stop_reason "tool_use".
 * When stop_reason is "tool_use", the tool_use blocks are custom tools that need
 * frontend execution, so we break out and let the frontend handle them.
 * The loop continues only when stop_reason is "tool_use" but all tool_use blocks
 * can be handled (currently none can be — future server-side tool support goes here).
 * @param {object} requestBody The initial request body for Claude API
 * @param {string} apiUrl Claude API URL
 * @param {object} headers Request headers
 * @param {AbortController} controller Abort controller
 * @param {number} maxIterations Maximum loop iterations
 * @returns {Promise<{allText: string, lastResponse: object}>}
 */
async function runClaudeAgenticLoop(requestBody, apiUrl, headers, controller, maxIterations) {
    const messages = [...requestBody.messages];
    let allText = '';
    let lastResponse = null;

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const iterBody = { ...requestBody, messages, stream: false };
        const fetchResponse = await fetch(apiUrl + '/messages', {
            method: 'POST',
            signal: controller.signal,
            body: JSON.stringify(iterBody),
            headers,
        });

        if (!fetchResponse.ok) {
            const errorText = await fetchResponse.text();
            throw new Error(`Claude API error (iteration ${iteration}): ${fetchResponse.status} ${fetchResponse.statusText}\n${errorText}`);
        }

        lastResponse = await fetchResponse.json();
        console.debug(`Agentic loop iteration ${iteration}: stop_reason=${lastResponse.stop_reason}`);

        for (const block of lastResponse.content || []) {
            if (block.type === 'text') allText += block.text;
        }

        if (lastResponse.stop_reason !== 'tool_use') break;

        // stop_reason "tool_use" means Claude wants us to execute custom tools.
        // Break out so the frontend ToolManager can handle them via recursive Generate().
        break;
    }

    return { allText, lastResponse };
}

/**
 * Runs a streaming Claude agentic loop: streams text/thinking deltas to the client
 * while handling tool_use loops server-side.
 * When stop_reason is "tool_use", deferred message events are forwarded so the
 * frontend can detect tool calls, then the loop breaks for frontend handling.
 * @param {object} requestBody The initial request body for Claude API
 * @param {string} apiUrl Claude API URL
 * @param {object} headers Request headers
 * @param {AbortController} controller Abort controller
 * @param {express.Response} expressResponse Express response to stream to
 * @param {object|boolean} wordReplacementsEnabled Word replacement config
 * @param {number} maxIterations Maximum loop iterations
 */
async function runClaudeAgenticLoopStreaming(requestBody, apiUrl, headers, controller, expressResponse, wordReplacementsEnabled, maxIterations) {
    const messages = [...requestBody.messages];
    const isWrEnabled = areWordReplacementsEnabled(wordReplacementsEnabled);
    let isFirstIteration = true;

    expressResponse.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    });

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const iterBody = { ...requestBody, messages, stream: true };
        const fetchResponse = await fetch(apiUrl + '/messages', {
            method: 'POST',
            signal: controller.signal,
            body: JSON.stringify(iterBody),
            headers,
        });

        if (!fetchResponse.ok) {
            const errorText = await fetchResponse.text();
            console.warn(`Agentic loop streaming error (iteration ${iteration}): ${fetchResponse.status} ${errorText}`);
            if (isFirstIteration) {
                expressResponse.write(`data: ${JSON.stringify({ error: true, status: fetchResponse.status, message: errorText })}\n\n`);
            }
            break;
        }

        const reconstructed = await consumeAndForwardClaudeStream(fetchResponse, expressResponse, isWrEnabled, isFirstIteration);
        isFirstIteration = false;

        console.debug(`Agentic loop streaming iteration ${iteration}: stop_reason=${reconstructed.stop_reason}`);

        if (reconstructed.stop_reason !== 'tool_use') break;

        // Forward deferred message_delta/message_stop so the frontend detects tool_use
        for (const evt of reconstructed.deferredEvents) {
            if (!expressResponse.writableEnded) expressResponse.write(evt);
        }
        break;
    }
}

/**
 * Reads Claude SSE events, forwards text/thinking deltas to the Express response,
 * and reconstructs the full response for loop logic.
 * Suppresses message_delta/message_stop when stop_reason is tool_use (loop continues).
 * Forwards ALL content block events including tool_use so the frontend can detect them
 * if the loop breaks for custom tools.
 * @param {import('node-fetch').Response} fetchResponse Fetch response from Claude API
 * @param {express.Response} expressResponse Express response to stream to
 * @param {boolean} wordReplacementsEnabled Whether word replacements are active
 * @param {boolean} forwardMetaEvents Whether to forward message_start events (first iteration only)
 * @returns {Promise<{content: any[], stop_reason: string, usage: object, deferredEvents: string[]}>}
 */
async function consumeAndForwardClaudeStream(fetchResponse, expressResponse, wordReplacementsEnabled, forwardMetaEvents) {
    const content = [];
    let stopReason = '';
    let usage = {};
    const transformStream = wordReplacementsEnabled ? createWordReplacementStream(wordReplacementsEnabled) : null;
    const deferredEvents = [];

    const writeToClient = (data) => {
        if (!expressResponse.writableEnded) {
            expressResponse.write(data);
        }
    };

    const forward = (sseEvent) => {
        if (transformStream) transformStream.write(sseEvent);
        else writeToClient(sseEvent);
    };

    if (transformStream) {
        transformStream.on('data', (chunk) => writeToClient(chunk));
    }

    return new Promise((resolve, reject) => {
        let buffer = '';
        fetchResponse.body.on('data', (chunk) => {
            buffer += chunk.toString();
            const parts = buffer.split('\n\n');
            buffer = parts.pop();
            for (const part of parts) {
                const dataLine = part.split('\n').find(l => l.startsWith('data:'));
                if (!dataLine) continue;
                const payload = dataLine.replace(/^data:\s*/, '').trim();
                if (!payload || payload === '[DONE]') continue;
                let parsed;
                try { parsed = JSON.parse(payload); } catch { continue; }

                if (parsed.type === 'message_start' && parsed.message) {
                    usage = { ...usage, ...parsed.message.usage };
                    if (forwardMetaEvents) {
                        forward(`data: ${JSON.stringify(parsed)}\n\n`);
                    }
                }

                if (parsed.type === 'content_block_start') {
                    content[parsed.index] = { ...parsed.content_block };
                    if (parsed.content_block.type === 'text') content[parsed.index].text = '';
                    if (parsed.content_block.type === 'thinking') content[parsed.index].thinking = '';
                    forward(`data: ${JSON.stringify(parsed)}\n\n`);
                }

                if (parsed.type === 'content_block_delta' && parsed.delta) {
                    const block = content[parsed.index];
                    if (block) {
                        if (parsed.delta.type === 'text_delta') block.text = (block.text || '') + parsed.delta.text;
                        else if (parsed.delta.type === 'thinking_delta') block.thinking = (block.thinking || '') + parsed.delta.thinking;
                        else if (parsed.delta.type === 'input_json_delta') block._inputJson = (block._inputJson || '') + parsed.delta.partial_json;
                    }
                    forward(`data: ${JSON.stringify(parsed)}\n\n`);
                }

                if (parsed.type === 'content_block_stop') {
                    const block = content[parsed.index];
                    if (block && block._inputJson) {
                        try { block.input = JSON.parse(block._inputJson); } catch { block.input = {}; }
                        delete block._inputJson;
                    }
                    forward(`data: ${JSON.stringify(parsed)}\n\n`);
                }

                if (parsed.type === 'message_delta') {
                    if (parsed.delta?.stop_reason) stopReason = parsed.delta.stop_reason;
                    if (parsed.usage) usage = { ...usage, ...parsed.usage };
                    deferredEvents.push(`data: ${JSON.stringify(parsed)}\n\n`);
                }

                if (parsed.type === 'message_stop') {
                    deferredEvents.push(`data: ${JSON.stringify(parsed)}\n\n`);
                }
            }
        });

        fetchResponse.body.on('end', () => {
            const flush = () => {
                if (stopReason !== 'tool_use') {
                    for (const evt of deferredEvents) forward(evt);
                }
                resolve({ content, stop_reason: stopReason, usage, deferredEvents });
            };
            if (transformStream) {
                transformStream.end();
                transformStream.on('finish', flush);
            } else {
                flush();
            }
        });
        fetchResponse.body.on('error', reject);
    });
}

/**
 * Sends a request to Claude API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
/**
 * Builds the Claude Messages API request body and fetch headers from an incoming
 * request. Shared by the synchronous sendClaudeRequest path and the async batch
 * submit endpoint so both paths produce byte-identical request shapes.
 * @param {import('express').Request} request Express request
 * @param {string} apiKey Resolved API key (or proxy password) for the x-api-key header
 * @returns {{ requestBody: any, fetchHeaders: Record<string, string>, useTools: boolean, useWebSearch: boolean, caps: any, taskBudgetEnabled: boolean }}
 */
function buildClaudeRequestBody(request, apiKey) {
    const { enableSystemPromptCache, cachingAtDepth, ttl: cacheTTL } = resolveClaudeCachingConfig(request);
    const additionalHeaders = {};
    const betaHeaders = [];
    const model = String(request.body.model ?? '');
    // Every per-model decision below reads this table instead of an inline regex.
    const caps = getClaudeCapabilities(model);
    const isKnownClaudeModel = CLAUDE_MODEL_CAPABILITIES.some(entry => entry.pattern.test(model));
    const useTools = Array.isArray(request.body.tools) && request.body.tools.length > 0;
    const useSystemPrompt = Boolean(request.body.use_sysprompt);
    const reasoningEffort = request.body.reasoning_effort;
    const wantsThinkingOff = !reasoningEffort || reasoningEffort === 'none';
    // A manual thinking block also conflicts with a prefill, so decide this before
    // the converter runs: it has to drop the assistant turn BEFORE merging same-role
    // runs, or flipping it back to `user` afterwards doubles the final user message.
    const manualThinkingActive = caps.thinkingMode === 'manual' && !wantsThinkingOff && reasoningEffort !== 'auto';
    const allowPrefill = caps.supportsPrefill && !manualThinkingActive;
    const convertedPrompt = convertClaudeMessages(request.body.messages, request.body.assistant_prefill, useSystemPrompt, useTools, getPromptNames(request), allowPrefill);
    const useWebSearch = caps.supportsWebSearch && Boolean(request.body.enable_web_search);
    const taskBudgetEnabled = caps.supportsTaskBudget && Boolean(request.body.claude_task_budget_enabled);
    // openai.js already sends this on every request; it mirrors the "Show Thoughts" setting.
    const showThoughts = Boolean(request.body.include_reasoning);

    // The 300k output beta only means something on models that can exceed the
    // default cap. Unknown names (reverse proxy passthrough) keep the old
    // unconditional behaviour rather than silently losing headroom.
    if (!isKnownClaudeModel || caps.maxOutput >= 128000) {
        betaHeaders.push('output-300k-2026-03-24');
    }

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
            convertedPrompt.systemPrompt[convertedPrompt.systemPrompt.length - 1].cache_control = { type: 'ephemeral', ttl: cacheTTL };
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

        if (enableSystemPromptCache && requestBody.tools.length && cachingAtDepth !== -1) {
            // Must match the system/messages TTL: longer-TTL breakpoints have to
            // precede shorter ones, and tools come first in the cache hierarchy
            requestBody.tools[requestBody.tools.length - 1].cache_control = { type: 'ephemeral', ttl: cacheTTL };
        }
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
        cachingAtDepthForClaude(convertedPrompt.messages, cachingAtDepth, cacheTTL);
    }

    if (enableSystemPromptCache || cachingAtDepth !== -1) {
        betaHeaders.push('prompt-caching-2024-07-31');
        betaHeaders.push('extended-cache-ttl-2025-04-11');
    }

    if (caps.thinkingMode !== 'none') {
        // Fable 5 / Mythos 5 think unconditionally: both `disabled` and manual
        // budget_tokens are a 400, so "None" simply has nothing to send.
        const forcedOn = wantsThinkingOff && caps.canDisableThinking === 'never';
        if (forcedOn) {
            console.info(color.blue(`Thinking cannot be disabled on ${model}; the "None" reasoning effort has no effect.`));
        }

        if (wantsThinkingOff && !forcedOn) {
            requestBody.thinking = { type: 'disabled' };
        } else {
            const minThinkTokens = 1024;
            if (requestBody.max_tokens <= minThinkTokens) {
                const newValue = requestBody.max_tokens + minThinkTokens;
                console.warn(color.yellow(`Claude thinking requires a minimum of ${minThinkTokens} response tokens.`));
                console.info(color.blue(`Increasing response length to ${newValue}.`));
                requestBody.max_tokens = newValue;
            }

            if (caps.thinkingMode === 'adaptive') {
                requestBody.thinking = { type: 'adaptive' };
            } else {
                // Pre-4.6: manual extended thinking with an explicit budget.
                const budgetTokens = calculateClaudeBudgetTokens(requestBody.max_tokens, reasoningEffort, requestBody.stream, false);
                if (Number.isInteger(budgetTokens)) {
                    requestBody.thinking = {
                        type: 'enabled',
                        budget_tokens: budgetTokens,
                    };
                }
            }

            // Effort rides along wherever the model supports it — that's every
            // adaptive model, plus Opus 4.5 next to its manual budget.
            const effort = getClaudeAdaptiveEffort(reasoningEffort, caps);
            if (effort) {
                requestBody.output_config ??= {};
                requestBody.output_config.effort = effort;
            }
        }
    }

    const thinkingActive = Boolean(requestBody.thinking) && requestBody.thinking.type !== 'disabled';

    // Opus 5 rejects `thinking: {type:'disabled'}` above `high` effort, per request.
    if (requestBody.thinking?.type === 'disabled' && caps.canDisableThinking === 'effort-capped' && requestBody.output_config?.effort) {
        const capIndex = caps.effortLevels.indexOf(caps.disableEffortCap);
        const currentIndex = caps.effortLevels.indexOf(requestBody.output_config.effort);
        if (capIndex !== -1 && currentIndex > capIndex) {
            console.info(color.blue(`${model} caps effort at ${caps.disableEffortCap} while thinking is disabled; lowering from ${requestBody.output_config.effort}.`));
            requestBody.output_config.effort = caps.disableEffortCap;
        }
    }

    // Thinking text is omitted by default from Opus 4.7 onward, so the "Show
    // Thoughts" setting has to be asked for explicitly or nothing ever renders.
    if (thinkingActive && caps.thinkingDisplay) {
        requestBody.thinking.display = showThoughts ? 'summarized' : 'omitted';
    }

    // A task budget is advisory and orthogonal to thinking, so it is not gated on it.
    if (taskBudgetEnabled) {
        const total = Math.max(20000, Number(request.body.claude_task_budget_total) || 64000);
        requestBody.output_config ??= {};
        requestBody.output_config.task_budget = { type: 'tokens', total };
        betaHeaders.push('task-budgets-2026-03-13');
    }

    // Sampling, resolved in one place. Comparisons read the ORIGINAL request
    // values so the rules stay order-independent.
    if (caps.samplingMode === 'none') {
        delete requestBody.temperature;
        delete requestBody.top_p;
        delete requestBody.top_k;
    } else {
        if (caps.samplingMode === 'limited') {
            if (Number(request.body.temperature) < 1) {
                delete requestBody.top_p;
            } else {
                delete requestBody.temperature;
            }
        }

        if (thinkingActive) {
            // NO I CAN'T SILENTLY IGNORE THE TEMPERATURE.
            delete requestBody.temperature;
            delete requestBody.top_k;

            if (Number(request.body.top_p) < 0.95) {
                delete requestBody.top_p;
            }
        }

        // Sending an API default is pointless noise.
        if (request.body.top_p === 1) {
            delete requestBody.top_p;
        }

        if (request.body.temperature === 1) {
            delete requestBody.temperature;
        }
    }

    if (betaHeaders.length) {
        additionalHeaders['anthropic-beta'] = betaHeaders.join(',');
    }

    console.debug('Claude request:', requestBody);

    const fetchHeaders = {
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01',
        'x-api-key': apiKey,
        ...additionalHeaders,
    };

    return { requestBody, fetchHeaders, useTools, useWebSearch, caps, taskBudgetEnabled };
}

async function sendClaudeRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_CLAUDE).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.CLAUDE, request.body.secret_id);
    const divider = '-'.repeat(process.stdout.columns);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

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

        const { requestBody, fetchHeaders, useTools, useWebSearch, taskBudgetEnabled } = buildClaudeRequestBody(request, apiKey);

        const useAgenticLoop = taskBudgetEnabled && (useTools || useWebSearch);
        const maxIterations = Math.min(Math.max(1, Number(request.body.claude_task_budget_max_iterations) || 25), 50);

        if (useAgenticLoop && request.body.stream) {
            console.info(color.blue(`Starting Claude agentic streaming loop (max ${maxIterations} iterations)`));
            await runClaudeAgenticLoopStreaming(
                requestBody, apiUrl, fetchHeaders, controller, response, wordReplacementsEnabled, maxIterations,
            );

            if (!response.writableEnded) {
                response.write('data: [DONE]\n\n');
                response.end();
            }
        } else if (useAgenticLoop && !request.body.stream) {
            console.info(color.blue(`Starting Claude agentic loop (max ${maxIterations} iterations)`));
            const { allText, lastResponse } = await runClaudeAgenticLoop(
                requestBody, apiUrl, fetchHeaders, controller, maxIterations,
            );

            if (!lastResponse) {
                console.warn(color.red(`Claude agentic loop returned no response.\n${divider}`));
                return response.status(500).send({ error: true });
            }

            console.debug('Claude agentic response:', lastResponse);
            const reply = {
                choices: [{ message: { content: allText } }],
                content: lastResponse.content,
            };
            return sendWithWordReplacements(response, reply, wordReplacementsEnabled);
        } else if (request.body.stream) {
            const generateResponse = await fetch(apiUrl + '/messages', {
                method: 'POST',
                signal: controller.signal,
                body: JSON.stringify(requestBody),
                headers: fetchHeaders,
            });
            // Pipe remote SSE stream to Express response
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            const generateResponse = await fetch(apiUrl + '/messages', {
                method: 'POST',
                signal: controller.signal,
                body: JSON.stringify(requestBody),
                headers: fetchHeaders,
            });

            if (!generateResponse.ok) {
                const generateResponseText = await generateResponse.text();
                console.warn(color.red(`Claude API returned error: ${generateResponse.status} ${generateResponse.statusText}\n${generateResponseText}\n${divider}`));
                return response.status(500).send({ error: true });
            }

            /** @type {any} */
            const generateResponseJson = await generateResponse.json();
            const responseText = generateResponseJson?.content?.[0]?.text || '';
            console.debug('Claude response:', generateResponseJson);

            const usage = generateResponseJson?.usage ?? {};
            console.info(formatClaudeUsageMeta(generateResponseJson?.model, usage, {
                org: generateResponse.headers.get('anthropic-organization-id'),
                req: generateResponse.headers.get('request-id'),
            }));

            // A refusal or a blown context window is a successful 200 with no usable
            // text; report it instead of handing back an empty message.
            const stopNotice = describeClaudeStop(generateResponseJson?.stop_reason, generateResponseJson?.stop_details?.category);
            if (stopNotice) {
                console.warn(color.red(`${stopNotice}\n${divider}`));
                return response.send({ error: { message: stopNotice } });
            }

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
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE, request.body.secret_id);

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
    const isGemma3 = /gemma-3/.test(model);
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

        const useSystemPrompt = !enableImageModality && !isGemma3 && request.body.use_sysprompt;

        const tools = [];
        const prompt = convertGooglePrompt(request.body.messages, model, useSystemPrompt, getPromptNames(request));
        const safetySettings = [...GEMINI_SAFETY, ...(useVertexAi ? VERTEX_SAFETY : [])];

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0 && !enableImageModality && !isGemma3) {
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

        if (enableWebSearch && !enableImageModality && !isGemma3 && !isLearnLM && !noSearchModels.includes(model)) {
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
                const serviceAccountJson = readSecret(request.user.directories, SECRET_KEYS.VERTEXAI_SERVICE_ACCOUNT, request.body.secret_id);
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

    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AI21, request.body.secret_id);
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
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI, request.body.secret_id);
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
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE, request.body.secret_id);
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
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK, request.body.secret_id);
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
        addReasoningContentToToolCalls(processedMessages);

        if (request.body.include_reasoning && request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
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
            'thinking': { type: request.body.include_reasoning ? 'enabled' : 'disabled' },
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
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI, request.body.secret_id);
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
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI, request.body.secret_id);
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
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.ELECTRONHUB, request.body.secret_id);
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
            const { enableSystemPromptCache, cachingAtDepth, ttl } = resolveClaudeCachingConfig(request);
            if (enableSystemPromptCache) {
                cachingSystemPromptForOpenRouter(request.body.messages, ttl);
            }

            if (cachingAtDepth !== -1) {
                cachingAtDepthForOpenRouterClaude(request.body.messages, cachingAtDepth, ttl);
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
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.CHUTES, request.body.secret_id);

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
 * Sends a request to MiniMax.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMinimaxRequest(request, response) {
    const apiUrl = request.body.minimax_endpoint === MINIMAX_ENDPOINT.CN
        ? API_MINIMAX_CN : API_MINIMAX;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.MINIMAX, request.body.secret_id);
    const wordReplacementsEnabled = getWordReplacementEnabled(request);

    if (!apiKey) {
        console.warn('MiniMax key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        // MiniMax does not allow consecutive messages with the same role.
        // Merge them into a single message to avoid "invalid chat setting (2013)".
        const messages = postProcessPrompt(request.body.messages, PROMPT_PROCESSING_TYPE.MERGE_TOOLS, getPromptNames(request));

        let bodyParams = {};

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        const requestBody = {
            'messages': messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.model === 'M2-her' ? Math.min(request.body.max_tokens, 2048) : request.body.max_tokens,
            'stream': request.body.stream,
            'top_p': request.body.top_p,
            'stop': request.body.stop,
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

        console.debug('MiniMax request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponseWithWordReplacements(generateResponse, response, wordReplacementsEnabled);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn('MiniMax returned error: ', errorText);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('MiniMax response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with MiniMax: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * @param {express.Request} request Express request object (contains request.body with all generate_data)
 * @param {express.Response} response Express response object
 */
async function sendAzureOpenAIRequest(request, response) {
    // 1. GATHER & VALIDATE SETTINGS
    const { azure_base_url, azure_deployment_name, azure_api_version } = request.body;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AZURE_OPENAI, request.body.secret_id);
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
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER) {
            apiUrl = 'https://openrouter.ai/api/v1';
            apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER, request.body.secret_id);
            // OpenRouter needs to pass the Referer and X-Title: https://openrouter.ai/docs#requests
            headers = { ...OPENROUTER_HEADERS };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MISTRALAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_MISTRAL).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            apiUrl = request.body.custom_url;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM, request.body.secret_id);
            headers = {};
            mergeObjectWithYaml(headers, request.body.custom_include_headers);
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COHERE) {
            apiUrl = API_COHERE_V1;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CHUTES) {
            apiUrl = API_CHUTES;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CHUTES, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ELECTRONHUB) {
            apiUrl = API_ELECTRONHUB;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.ELECTRONHUB, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NANOGPT) {
            apiUrl = API_NANOGPT;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT, request.body.secret_id);
            headers = {};
            queryParams = { detailed: true };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.DEEPSEEK) {
            apiUrl = new URL(request.body.reverse_proxy || API_DEEPSEEK.replace('/beta', '')).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.XAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_XAI).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AIMLAPI) {
            apiUrl = API_AIMLAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI, request.body.secret_id);
            headers = { ...AIMLAPI_HEADERS };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
            apiUrl = 'https://gen.pollinations.ai/text';
            apiKey = readSecret(request.user.directories, SECRET_KEYS.POLLINATIONS, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.GROQ) {
            apiUrl = API_GROQ;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COMETAPI) {
            apiUrl = API_COMETAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI, request.body.secret_id);
            headers = {};
            throw new Error('This provider is temporarily disabled.');
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT) {
            apiUrl = new URL(request.body.reverse_proxy || API_MOONSHOT).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MOONSHOT, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
            apiUrl = API_FIREWORKS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MAKERSUITE) {
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE, request.body.secret_id);
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
                            ...model,
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
            const apiKey = readSecret(request.user.directories, SECRET_KEYS.AZURE_OPENAI, request.body.secret_id);

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
            apiKey = readSecret(request.user.directories, SECRET_KEYS.SILICONFLOW, request.body.secret_id);
            headers = {};
            queryParams = { type: 'text', sub_type: 'chat' };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.WORKERS_AI) {
            apiKey = readSecret(request.user.directories, SECRET_KEYS.WORKERS_AI, request.body.secret_id);

            if (!apiKey) {
                console.warn('Cloudflare Workers AI API key is missing.');
                return statusResponse.status(400).send({ error: true });
            }

            try {
                const accountId = String(request.body.workers_ai_account_id || '').trim();
                if (!accountId) {
                    console.warn('Cloudflare Workers AI Account ID is missing.');
                    return statusResponse.status(400).send({ error: true });
                }

                const modelsUrl = new URL(`${API_WORKERS_AI}/${encodeURIComponent(accountId)}/ai/models/search`);
                modelsUrl.searchParams.set('task', 'Text Generation');
                modelsUrl.searchParams.set('per_page', '1000');

                const response = await fetch(modelsUrl, {
                    method: 'GET',
                    headers: {
                        'Authorization': 'Bearer ' + apiKey,
                    },
                });

                if (response.ok) {
                    /** @type {any} */
                    const data = await response.json();
                    const models = Array.isArray(data?.result)
                        ? data.result.map(model => ({ ...model, id: model.name }))
                        : [];

                    console.debug('Available Cloudflare Workers AI models:', models.map(m => m.id));
                    return statusResponse.send({ data: models });
                } else {
                    console.warn('Cloudflare Workers AI models endpoint failed:', response.status, response.statusText);
                    return statusResponse.status(response.status).send({ error: true });
                }
            } catch (error) {
                console.error('Error fetching Cloudflare Workers AI models:', error);
                return statusResponse.status(500).send({ error: true });
            }
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

// ---------------------------------------------------------------------------
// Claude Message Batches ("Flex" tier) — asynchronous, 50%-cost generation.
// A single message generation is submitted as a batch of one, polled by the
// frontend, and its reply is delivered back into the origin chat when ready.
// Jobs are persisted per-user so a page reload / server restart can resume.
// ---------------------------------------------------------------------------

// Keep in sync with CLAUDE_BATCH_ONLY_MODELS in public/scripts/openai.js.
const CLAUDE_BATCH_MODELS = /^claude-fable-5/;

function getClaudeBatchStorePath(directories) {
    return path.join(directories.root, 'claude-batches.json');
}

function readClaudeBatchJobs(directories) {
    try {
        const storePath = getClaudeBatchStorePath(directories);
        if (!fs.existsSync(storePath)) return [];
        const parsed = JSON.parse(fs.readFileSync(storePath, 'utf8'));
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.error('Failed to read claude-batches.json:', error);
        return [];
    }
}

function writeClaudeBatchJobs(directories, jobs) {
    try {
        writeFileAtomicSync(getClaudeBatchStorePath(directories), JSON.stringify(jobs, null, 2), 'utf8');
    } catch (error) {
        console.error('Failed to write claude-batches.json:', error);
    }
}

function addClaudeBatchJob(directories, job) {
    const jobs = readClaudeBatchJobs(directories);
    jobs.push(job);
    writeClaudeBatchJobs(directories, jobs);
}

function findClaudeBatchJob(directories, jobId) {
    return readClaudeBatchJobs(directories).find(job => job.jobId === jobId) ?? null;
}

function updateClaudeBatchJob(directories, jobId, patch) {
    const jobs = readClaudeBatchJobs(directories);
    const index = jobs.findIndex(job => job.jobId === jobId);
    if (index >= 0) {
        jobs[index] = { ...jobs[index], ...patch };
        writeClaudeBatchJobs(directories, jobs);
    }
}

function removeClaudeBatchJob(directories, jobId) {
    const jobs = readClaudeBatchJobs(directories).filter(job => job.jobId !== jobId);
    writeClaudeBatchJobs(directories, jobs);
}

// Resolve the Claude key the same way sendClaudeRequest does: proxy password
// when a reverse proxy is set, otherwise the stored secret.
function resolveClaudeBatchKey(request) {
    return request.body.reverse_proxy
        ? request.body.proxy_password
        : readSecret(request.user.directories, SECRET_KEYS.CLAUDE, request.body.secret_id);
}

function claudeBatchHeaders(apiKey) {
    return {
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01',
        'x-api-key': apiKey,
    };
}

function claudeBatchBaseUrl(request) {
    return request.body.reverse_proxy || API_CLAUDE;
}

// "1m 20s" / "45s" — batches run for minutes, so wall time is the useful number.
function formatClaudeBatchElapsed(since) {
    if (!Number.isFinite(since)) return '';
    const seconds = Math.max(0, Math.round((Date.now() - since) / 1000));
    const minutes = Math.floor(seconds / 60);
    return minutes ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

// "0.018$", "0.01$" — trailing zeros trimmed so the line stays scannable.
function formatClaudeCost(amount) {
    return `${Number(amount.toFixed(6))}$`;
}

/**
 * Price a Claude generation from its usage block.
 * The Message Batches API bills at 50% of the listed rates.
 * @param {string} model Model that served the request
 * @param {object} usage Anthropic `usage` object
 * @param {boolean} batch Whether this was a batch generation
 * @returns {{ input: number, output: number, total: number }|null} USD amounts, or null for an unpriced model
 */
function computeClaudeCost(model, usage, batch) {
    const pricing = getClaudePricing(model);
    if (!pricing) {
        return null;
    }

    const discount = batch ? 0.5 : 1;
    const inputRate = (pricing.input / 1e6) * discount;
    const outputRate = (pricing.output / 1e6) * discount;

    const u = usage ?? {};
    const input = (Number(u.input_tokens) || 0) * inputRate;
    const output = (Number(u.output_tokens) || 0) * outputRate;

    // Anthropic reports input_tokens EXCLUSIVE of the cache counters, so these are
    // additional spend: reads bill at 0.1x the input rate, writes at 1.25x (2x for
    // the 1h extended TTL this fork can turn on).
    const cacheRead = (Number(u.cache_read_input_tokens) || 0) * inputRate * 0.1;
    const created = u.cache_creation ?? {};
    const wrote5m = Number(created.ephemeral_5m_input_tokens);
    const wrote1h = Number(created.ephemeral_1h_input_tokens);
    const cacheWrite = Number.isFinite(wrote5m) || Number.isFinite(wrote1h)
        ? ((wrote5m || 0) * 1.25 + (wrote1h || 0) * 2) * inputRate
        : (Number(u.cache_creation_input_tokens) || 0) * inputRate * 1.25;

    return { input, output, total: input + output + cacheRead + cacheWrite };
}

/**
 * Same one-line usage summary the synchronous Claude path prints, so batched and
 * blocking generations are comparable at a glance in the console.
 * @param {string} model Model that served the request
 * @param {object} usage Anthropic `usage` object
 * @param {Record<string, any>} [extras] Trailing `key: value` fields, skipped when falsy
 * @param {{ batch?: boolean }} [options] `batch` halves the per-token rates
 * @returns {string} Formatted line
 */
function formatClaudeUsageMeta(model, usage, extras = {}, { batch = false } = {}) {
    const u = usage ?? {};
    // Unknown model: print the token counts and skip the cost rather than "NaN$".
    const cost = computeClaudeCost(model, u, batch);
    const inCost = cost ? ` (${formatClaudeCost(cost.input)})` : '';
    const outCost = cost ? ` (${formatClaudeCost(cost.output)})` : '';
    const totalCost = cost ? ` | total cost: ${formatClaudeCost(cost.total)}` : '';
    let meta = `model: ${model} | in: ${u.input_tokens ?? '?'}${inCost} | out: ${u.output_tokens ?? '?'}${outCost}${totalCost} | cache_read: ${u.cache_read_input_tokens ?? 0} | cache_created: ${u.cache_creation_input_tokens ?? 0}`;
    for (const [key, value] of Object.entries(extras)) {
        if (value) meta += ` | ${key}: ${value}`;
    }
    return meta;
}

// Frontend batch settings, so the master switch, poll interval and give-up window
// all live in config.yaml. `enabled` is what the frontend uses to decide whether a
// batch-only model goes through the batch API at all — it's the only off switch.
function getClaudeBatchClientSettings() {
    return {
        batchEnabled: getConfigValue('claude.batchFlex.enabled', false, 'boolean'),
        pollIntervalMs: Math.max(5000, getConfigValue('claude.batchFlex.pollIntervalMs', 20000, 'number')),
        maxWaitMinutes: Math.max(1, getConfigValue('claude.batchFlex.maxWaitMinutes', 90, 'number')),
    };
}

// Submit a single generation as a batch of one.
router.post('/claude-batch/submit', async function (request, response) {
    try {
        if (!getConfigValue('claude.batchFlex.enabled', false, 'boolean')) {
            return response.status(409).send({ ineligible: true, reason: 'Batch Processing is disabled in config.yaml (claude.batchFlex.enabled).' });
        }

        // The 50% discount only applies to a real API key. Refuse rather than fall
        // back, so an unbatchable request can never be billed at full price silently.
        const apiKey = resolveClaudeBatchKey(request);
        if (!apiKey || !apiKey.includes('sk-ant')) {
            return response.status(409).send({ ineligible: true, reason: 'Batch Processing needs an sk-ant API key as the proxy password. Nothing was sent.' });
        }

        // Mirrors isClaudeBatchModeOn on the frontend: every other Claude model is
        // cheaper on subscription usage than at half the API price, so batching it
        // would be a net loss. Re-checked here so resumed/stale callers can't slip past.
        if (!CLAUDE_BATCH_MODELS.test(String(request.body.model ?? ''))) {
            return response.status(409).send({ ineligible: true, reason: `Batch mode is not enabled for ${request.body.model}.` });
        }

        // The batch body is built from the same payload a synchronous /generate
        // takes, so a caller that skipped assembling it can't produce a valid batch.
        if (!Array.isArray(request.body.messages)) {
            return response.status(409).send({ ineligible: true, reason: 'Request is missing a messages array.' });
        }

        const { requestBody } = buildClaudeRequestBody(request, apiKey);
        delete requestBody.stream; // streaming is not allowed inside a batch

        const customId = ('st-' + uuidv4().replace(/-/g, '')).slice(0, 64);
        const batchPayload = { requests: [{ custom_id: customId, params: requestBody }] };

        const createUrl = urlJoin(claudeBatchBaseUrl(request), 'messages/batches');
        const proxyResponse = await fetch(createUrl, {
            method: 'POST',
            headers: claudeBatchHeaders(apiKey),
            body: JSON.stringify(batchPayload),
        });

        const data = await proxyResponse.json().catch(() => null);
        if (!proxyResponse.ok || !data?.id) {
            console.warn(color.red(`Claude batch submit failed: ${proxyResponse.status} ${JSON.stringify(data)}`));
            return response.status(502).send({ error: true, detail: data });
        }

        const job = {
            jobId: uuidv4(),
            batchId: data.id,
            customId,
            chatId: request.body.batch_chat_id ?? null,
            characterName: request.body.batch_character_name ?? null,
            model: request.body.model,
            createdAt: Date.now(),
            status: 'in_progress',
        };
        addClaudeBatchJob(request.user.directories, job);

        console.info(color.blue(`Claude batch queued: ${data.id} | model: ${job.model} | status: ${data.processing_status} | job: ${job.jobId}`));
        return response.send({
            jobId: job.jobId,
            batchId: job.batchId,
            customId,
            processing_status: data.processing_status,
            ...getClaudeBatchClientSettings(),
        });
    } catch (error) {
        console.error(color.red(`Claude batch submit error: ${error?.stack || error}`));
        return response.status(500).send({ error: true });
    }
});

// Poll a batch's processing status.
router.post('/claude-batch/status', async function (request, response) {
    try {
        const apiKey = resolveClaudeBatchKey(request);
        const statusUrl = urlJoin(claudeBatchBaseUrl(request), 'messages/batches', String(request.body.batchId));
        const proxyResponse = await fetch(statusUrl, { headers: claudeBatchHeaders(apiKey) });
        const data = await proxyResponse.json().catch(() => null);

        if (proxyResponse.ok && data?.processing_status && request.body.jobId) {
            // Polls run every ~20s, so only a state change is worth a console line.
            const previous = findClaudeBatchJob(request.user.directories, request.body.jobId);
            updateClaudeBatchJob(request.user.directories, request.body.jobId, { status: data.processing_status });

            if (previous && previous.status !== data.processing_status) {
                const counts = data.request_counts ?? {};
                const summary = Object.entries(counts).filter(([, n]) => n).map(([k, n]) => `${k}: ${n}`).join(', ');
                console.info(color.blue(`Claude batch ${request.body.batchId}: ${previous.status} → ${data.processing_status} after ${formatClaudeBatchElapsed(previous.createdAt)}${summary ? ` (${summary})` : ''}`));
            }
        }
        return response.status(proxyResponse.status).send(data ?? { error: true });
    } catch (error) {
        console.error(color.red(`Claude batch status error: ${error}`));
        return response.status(500).send({ error: true });
    }
});

// Retrieve and reshape a completed batch's single result (word replacements applied).
router.post('/claude-batch/result', async function (request, response) {
    try {
        const apiKey = resolveClaudeBatchKey(request);
        const resultsUrl = urlJoin(claudeBatchBaseUrl(request), 'messages/batches', String(request.body.batchId), 'results');
        const proxyResponse = await fetch(resultsUrl, { headers: claudeBatchHeaders(apiKey) });
        const text = await proxyResponse.text();

        if (!proxyResponse.ok) {
            return response.status(proxyResponse.status).send({ error: true, detail: tryParse(text) ?? text });
        }

        // Results are JSONL. For a batch of one, pick the line for our custom_id.
        const lines = text.split('\n').map(line => line.trim()).filter(Boolean);
        const line = lines.find(l => !request.body.customId || l.includes(request.body.customId)) ?? lines[0];
        if (!line) {
            return response.status(502).send({ error: true, detail: 'Empty batch results' });
        }

        const entry = JSON.parse(line);
        const result = entry?.result;

        const job = request.body.jobId ? findClaudeBatchJob(request.user.directories, request.body.jobId) : null;

        if (result?.type !== 'succeeded') {
            console.warn(color.red(`Claude batch ${request.body.batchId} ${result?.type ?? 'errored'}: ${JSON.stringify(result?.error ?? null)}`));
            if (request.body.jobId) {
                updateClaudeBatchJob(request.user.directories, request.body.jobId, { status: 'ended', resultType: result?.type ?? 'errored' });
            }
            return response.send({ resultType: result?.type ?? 'errored', error: result?.error ?? null });
        }

        const message = result.message ?? {};
        const responseText = message?.content?.find(part => part.type === 'text')?.text || message?.content?.[0]?.text || '';
        // Same hybrid shape the synchronous Claude path returns: OpenAI choices[] +
        // Claude-native content[] (which extractMessageFromData reads first).
        const reply = {
            choices: [{ message: { content: responseText } }],
            content: message.content,
            usage: message.usage,
        };

        console.debug('Claude batch response:', message);
        console.info(formatClaudeUsageMeta(message?.model, message?.usage, {
            org: proxyResponse.headers.get('anthropic-organization-id'),
            msg: message?.id,
            batch: request.body.batchId,
            waited: job ? formatClaudeBatchElapsed(job.createdAt) : '',
        }, { batch: true }));

        // Batches can come back refused (Fable 5 runs safety classifiers) or truncated
        // by the context window. Both are "succeeded" results with nothing to deliver.
        const stopNotice = describeClaudeStop(message?.stop_reason, message?.stop_details?.category);
        if (stopNotice) {
            console.warn(color.red(`Claude batch ${request.body.batchId}: ${stopNotice}`));
            if (request.body.jobId) {
                updateClaudeBatchJob(request.user.directories, request.body.jobId, { status: 'ended', resultType: 'refused' });
            }
            return response.send({ resultType: 'refused', error: { message: stopNotice } });
        }

        const processed = enforceWordReplacementsOnResponse(reply, getWordReplacementEnabled(request));

        if (request.body.jobId) {
            updateClaudeBatchJob(request.user.directories, request.body.jobId, { status: 'ready', resultType: 'succeeded', resultReply: processed });
        }
        return response.send({ resultType: 'succeeded', reply: processed });
    } catch (error) {
        console.error(color.red(`Claude batch result error: ${error}`));
        return response.status(500).send({ error: true });
    }
});

// Record how a completed reply should be written back into the chat. Sent right
// after submit, once the frontend has parked its placeholder and knows the target.
router.post('/claude-batch/annotate', function (request, response) {
    if (request.body.jobId) {
        updateClaudeBatchJob(request.user.directories, request.body.jobId, {
            mode: request.body.mode ?? 'normal',
            swipeId: request.body.swipeId,
            originalMes: request.body.originalMes,
        });
    }
    return response.send({ ok: true });
});

// Acknowledge delivery — drop the job from the persisted store.
router.post('/claude-batch/ack', function (request, response) {
    if (request.body.jobId) {
        removeClaudeBatchJob(request.user.directories, request.body.jobId);
    }
    return response.send({ ok: true });
});

// Cancel an in-progress batch.
router.post('/claude-batch/cancel', async function (request, response) {
    try {
        const apiKey = resolveClaudeBatchKey(request);
        const cancelUrl = urlJoin(claudeBatchBaseUrl(request), 'messages/batches', String(request.body.batchId), 'cancel');
        const proxyResponse = await fetch(cancelUrl, { method: 'POST', headers: claudeBatchHeaders(apiKey) });
        const data = await proxyResponse.json().catch(() => ({}));
        if (request.body.jobId) {
            removeClaudeBatchJob(request.user.directories, request.body.jobId);
        }
        return response.status(proxyResponse.status).send(data);
    } catch (error) {
        console.error(color.red(`Claude batch cancel error: ${error}`));
        return response.status(500).send({ error: true });
    }
});

// List this user's persisted batch jobs (for resume on reload / restart).
router.get('/claude-batch/list', function (request, response) {
    return response.send({ jobs: readClaudeBatchJobs(request.user.directories), ...getClaudeBatchClientSettings() });
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
 * @param {any} originalBody The original request.body from the client (for flags)
 */
function convertToResponsesApiRequest(requestBody, originalBody = {}) {
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
    const proMode = Boolean(originalBody.pro_reasoning_mode)
        && OPENAI_PRO_REASONING_MODELS.test(requestBody.model ?? '');
    requestBody.reasoning = {
        ...(requestBody.reasoning_effort ? { effort: requestBody.reasoning_effort } : {}),
        ...(proMode ? { mode: 'pro' } : {}),
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

    // Prompt caching (GPT-5.6+): implicit mode is the API default and bills cache writes
    // at 1.25x input. When the toggle is off, explicit mode with no breakpoints in the
    // prompt means the request neither reads nor writes the cache.
    if (originalBody.openai_enable_caching === false
        && OPENAI_PROMPT_CACHE_OPTIONS_MODELS.test(requestBody.model ?? '')) {
        requestBody.prompt_cache_options = { mode: 'explicit' };
    }

    // Don't store conversations on OpenAI's servers by default; allow opt-in via flag
    if (typeof requestBody.responses_store === 'boolean') {
        requestBody.store = requestBody.responses_store;
    } else {
        requestBody.store = true;
    }
    delete requestBody.responses_store;

    // Remove unsupported parameters
    if (requestBody.reasoning.effort === 'none'
        || (!requestBody.reasoning.effort && !requestBody.reasoning.mode)) {
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
            case CHAT_COMPLETION_SOURCES.MINIMAX: return await sendMinimaxRequest(request, response);
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
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI, request.body.secret_id);
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
            apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER, request.body.secret_id);
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

            if (Array.isArray(request.body.messages)) {
                embedOpenRouterMedia(request.body.messages, { audio: true, video: true });
                addOpenRouterSignatures(request.body.messages, request.body.model);

                if (isClaude) {
                    const { enableSystemPromptCache, cachingAtDepth, ttl } = resolveClaudeCachingConfig(request);
                    if (enableSystemPromptCache) {
                        cachingSystemPromptForOpenRouter(request.body.messages, ttl);
                    }

                    if (cachingAtDepth !== -1) {
                        cachingAtDepthForOpenRouterClaude(request.body.messages, cachingAtDepth, ttl);
                    }
                }

                // Gemini on OpenRouter has no top-level system_instruction field; the system prompt
                // must stay in the messages array. Caching uses Anthropic-style cache_control
                // breakpoints inside the system message, and OpenRouter manages the cache TTL itself.
                if (isGemini) {
                    const { enableSystemPromptCache } = resolveClaudeCachingConfig(request);
                    if (enableSystemPromptCache) {
                        cachingSystemPromptForOpenRouter(request.body.messages);
                    }
                }
            }

            if (isGemini) {
                bodyParams['safety_settings'] = GEMINI_SAFETY;
                bodyParams['service_tier'] = request.body.service_tier;
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            apiUrl = request.body.custom_url;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM, request.body.secret_id);
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
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.PERPLEXITY) {
            apiUrl = API_PERPLEXITY;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.PERPLEXITY, request.body.secret_id);
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
            apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ, request.body.secret_id);
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
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS, request.body.secret_id);
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
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT, request.body.secret_id);
            headers = {};
            bodyParams = {};
            if (request.body.nanogpt_provider) {
                headers['X-Provider'] = request.body.nanogpt_provider;
            }
            if (request.body.nanogpt_payg_override) {
                headers['X-Billing-Mode'] = 'paygo';
                bodyParams['billing_mode'] = 'paygo';
            }
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

            const isClaude = /(?:^|\/)claude[-_]/.test(request.body.model);
            if (isClaude) {
                const { enableSystemPromptCache, ttl } = resolveClaudeCachingConfig(request);
                if (enableSystemPromptCache) {
                    bodyParams['cache_control'] = {
                        'enabled': true,
                        'ttl': ttl,
                    };
                }
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
            apiUrl = API_POLLINATIONS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.POLLINATIONS, request.body.secret_id);
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
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MOONSHOT, request.body.secret_id);
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
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI, request.body.secret_id);
            headers = {};
            bodyParams = {
                reasoning_effort: request.body.reasoning_effort,
            };
            throw new Error('This provider is temporarily disabled.');
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ZAI) {
            const defaultApiUrl = request.body.zai_endpoint === ZAI_ENDPOINT.CODING ? API_ZAI_CODING : API_ZAI_COMMON;
            apiUrl = new URL(request.body.reverse_proxy || defaultApiUrl).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.ZAI, request.body.secret_id);
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
            apiKey = readSecret(request.user.directories, SECRET_KEYS.SILICONFLOW, request.body.secret_id);
            headers = {};
            bodyParams = {};
            if (request.body.json_schema) {
                setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.WORKERS_AI) {
            apiKey = readSecret(request.user.directories, SECRET_KEYS.WORKERS_AI, request.body.secret_id);
            const accountId = String(request.body.workers_ai_account_id || '').trim();
            if (!accountId) {
                console.warn('Cloudflare Workers AI Account ID is missing.');
                return response.status(400).send({ error: true });
            }
            apiUrl = `${API_WORKERS_AI}/${encodeURIComponent(accountId)}/ai/v1`;
            headers = {};
            bodyParams = {
                repetition_penalty: request.body.repetition_penalty,
            };
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: request.body.json_schema.value,
                };
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
            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM && /^koboldcpp\/(.+)$/.test(request.body.model)) {
                bodyParams['reasoning_effort'] = request.body.reasoning_effort;
            }
        }

        if (request.body.verbosity && [CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI].includes(request.body.chat_completion_source)) {
            if (OPENAI_VERBOSITY_MODELS.test(request.body.model)) {
                bodyParams['verbosity'] = request.body.verbosity;
            }
        }

        if ([CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI].includes(request.body.chat_completion_source)) {
            if (/^gpt-5/.test(request.body.model)) {
            bodyParams['service_tier'] = request.body.service_tier;
            } else {
                delete bodyParams['service_tier'];
            }
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
            convertToResponsesApiRequest(requestBody, request.body);
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

multimodalModels.post('/workers_ai', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.WORKERS_AI);
        const accountId = String(req.body.workers_ai_account_id || '').trim();

        if (!key || !accountId) {
            return res.json([]);
        }

        const apiUrl = `https://api.cloudflare.com/client/v4/accounts/${encodeURIComponent(accountId)}/ai/models/search?task=Text+Generation&per_page=1000`;
        const response = await fetch(apiUrl, {
            method: 'GET',
            headers: { 'Authorization': 'Bearer ' + key },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const models = Array.isArray(data?.result)
            ? data.result
                .filter(m => Array.isArray(m.properties) && m.properties.some(p => p.property_id === 'vision' && p.value === 'true'))
                .map(m => m.name)
            : [];
        return res.json(models);
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
