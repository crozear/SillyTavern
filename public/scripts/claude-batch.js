import {
    chat,
    cleanUpMessage,
    event_types,
    eventSource,
    extractMessageFromData,
    getCurrentChatId,
    getRequestHeaders,
    name2,
    saveChatConditional,
    saveReply,
    updateMessageBlock,
} from '../script.js';
import { extractReasoningFromData } from './reasoning.js';
import { createGenerationParameters, getChatCompletionModel, getClaudeBatchBlocker, getClaudeBatchRequestExtras, isClaudeBatchModeOn, oai_settings, setClaudeBatchServerEnabled } from './openai.js';
import { getRegexedString, regex_placement } from './extensions/regex/engine.js';
import { power_user } from './power-user.js';
import { t } from './i18n.js';

const API_BASE = '/api/backends/chat-completions/claude-batch';

/** Text shown in the chat while the batch is cooking. */
const PLACEHOLDER = '*⏳ Waiting for a batched reply…*';

/** `quiet` is extension-internal (summarize, vectors, …): the caller awaits a
 * returned string, which a detached job can't provide. It's never triggered by
 * hand, so it runs synchronously even with Batch Processing on — the one case
 * where a sync fallback can't be a cost surprise. */
const ALWAYS_SYNC_TYPES = new Set(['quiet']);

/** Active jobs, keyed by jobId. @type {Map<string, BatchJob>} */
const activeJobs = new Map();

let pollIntervalMs = 20000;
let maxWaitMinutes = 90;

/**
 * @typedef {object} BatchJob
 * @property {string} jobId Server-side job identifier
 * @property {string} batchId Anthropic batch identifier
 * @property {string} customId custom_id of the single request in the batch
 * @property {string|null} chatId Chat the reply belongs to
 * @property {string|null} characterName Character name for toasts
 * @property {number} createdAt Timestamp of submission
 * @property {'normal'|'swipe'|'continue'} mode How the reply gets written back
 * @property {number} [swipeId] Swipe slot to fill (mode 'swipe')
 * @property {string} [originalMes] Message text before the continuation (mode 'continue')
 * @property {number} [timer] setInterval handle
 * @property {object} [reply] Completed reply payload, awaiting delivery
 * @property {boolean} [delivering] Guard against re-entrant delivery
 * @property {boolean} [polling] Guard against overlapping polls
 * @property {boolean} [notified] Whether the "ready elsewhere" toast has been shown
 */

/**
 * Credentials/settings the backend needs on every batch call. `generate_data` is
 * long gone by the time a resumed poller runs, so these are re-read from settings.
 * @returns {object}
 */
function batchExtras() {
    return getClaudeBatchRequestExtras();
}

/**
 * @param {string} path Endpoint path under the batch API base
 * @param {object} body Request payload
 * @returns {Promise<Response>}
 */
function postBatch(path, body) {
    return fetch(`${API_BASE}/${path}`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ ...batchExtras(), ...body }),
    });
}

/**
 * Submits a generation as a Claude batch and returns immediately, leaving a
 * placeholder message in the chat that gets filled in when the batch finishes.
 * @param {string} type Generation type
 * @param {object} generateData Generation payload from Generate() — carries the prompt, not the API body
 * @param {import('../script.js').AdditionalRequestOptions} [options] Additional request options
 * @param {AbortSignal} [signal] Generation abort signal
 * @returns {Promise<'queued'|'sync'|'refused'>} 'sync' = run the normal request; 'refused' = abort, don't bill
 */
export async function startClaudeBatch(type, generateData, options = {}, signal = null) {
    if (!isClaudeBatchModeOn() || ALWAYS_SYNC_TYPES.has(type)) {
        return 'sync';
    }

    // A batch detaches the instant it's submitted, so an abort raised while the prompt
    // was still being assembled — the stop button, or Prompt Inspector's "Cancel
    // generation" — has to be caught here. The synchronous path gets this for free by
    // handing the signal to fetch; there's no in-flight request for it to cancel here.
    if (signal?.aborted) {
        return 'refused';
    }

    // Defensive only: Generate() already refuses these up front, with the toast, before
    // anything is committed to the chat. Never silently downgrade to a paid sync call.
    if (getClaudeBatchBlocker(type)) {
        return 'refused';
    }

    const chatId = getCurrentChatId();
    const mode = type === 'swipe' ? 'swipe' : (type === 'continue' ? 'continue' : 'normal');
    let response;

    // Generate() only hands over `{ prompt, … }`; the actual API body is assembled
    // by sendOpenAIRequest. Build it the same way here (including the settings-ready
    // event) so a batched request is byte-identical to a synchronous one.
    let generate_data;
    try {
        const model = getChatCompletionModel(oai_settings);
        ({ generate_data } = await createGenerationParameters(oai_settings, model, type, generateData?.prompt, options));
        await eventSource.emit(event_types.CHAT_COMPLETION_SETTINGS_READY, generate_data);
    } catch (error) {
        console.error('Claude batch parameters could not be built.', error);
        toastr.error(t`Couldn't build the batch request. Nothing was sent.`, t`Batch Processing`, { timeOut: 15000 });
        return 'refused';
    }

    // Building the parameters awaits CHAT_COMPLETION_SETTINGS_READY listeners, so the
    // user has had another window in which to cancel. Last check before it's billable.
    if (signal?.aborted) {
        return 'refused';
    }

    try {
        response = await fetch(`${API_BASE}/submit`, {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                ...generate_data,
                stream: false,
                batch_chat_id: chatId ?? null,
                batch_character_name: name2,
            }),
        });
    } catch (error) {
        console.error('Claude batch submit failed.', error);
        toastr.error(t`Couldn't reach the batch endpoint. Nothing was sent.`, t`Batch Processing`, { timeOut: 15000 });
        return 'refused';
    }

    if (!response.ok) {
        const detail = await response.json().catch(() => null);
        console.error('Claude batch submit error.', response.status, detail);
        toastr.error(
            detail?.reason || t`Batch submission failed. Nothing was sent — switch to another Claude model to generate normally.`,
            t`Batch Processing`,
            { timeOut: 15000, extendedTimeOut: 25000 },
        );
        return 'refused';
    }

    const data = await response.json();
    applyServerSettings(data);

    const job = {
        jobId: data.jobId,
        batchId: data.batchId,
        customId: data.customId,
        chatId: chatId ?? null,
        characterName: name2,
        createdAt: Date.now(),
        mode,
    };

    await placeholderFor(job);
    // Delivery bookkeeping is only known after the placeholder exists, so persist it
    // now — a reload has to be able to write the reply back into the right slot.
    await postBatch('annotate', {
        jobId: job.jobId,
        mode: job.mode,
        swipeId: job.swipeId,
        originalMes: job.originalMes,
    }).catch(error => console.error('Failed to persist batch delivery info.', error));
    track(job);

    toastr.info(t`Reply will arrive here when it's done — keep using other chats meanwhile.`, t`Batch queued`, { timeOut: 8000 });
    return 'queued';
}

/**
 * Parks a placeholder in the chat so the pending reply has a visible, on-disk home
 * that survives navigating away or reloading. Records on the job whatever delivery
 * will need to write the real reply back into the right place.
 * `fromStreaming` suppresses the MESSAGE_RECEIVED/CHARACTER_MESSAGE_RENDERED emits
 * so extensions (TTS, translate, …) don't act on placeholder text — they fire once
 * for real at delivery.
 * @param {BatchJob} job Job being submitted (mutated with delivery bookkeeping)
 * @returns {Promise<void>}
 */
async function placeholderFor(job) {
    if (job.mode === 'continue') {
        // Continue appends to the existing message, so there's no new message to own
        // the placeholder. Remember the text as it stands and tack the marker on.
        const target = chat[chat.length - 1];
        job.originalMes = target.mes;
        await saveReply({ type: 'continue', getMessage: `\n\n${PLACEHOLDER}`, fromStreaming: true });
    } else {
        // 'swipe' fills the slot the swipe machinery just opened; 'normal' pushes a
        // new message. Either way saveReply leaves it as the last message in chat.
        await saveReply({ type: job.mode, getMessage: PLACEHOLDER, fromStreaming: true });
    }

    const message = chat[chat.length - 1];
    message.extra = message.extra ?? {};
    message.extra.claude_batch_job_id = job.jobId;
    message.extra.claude_batch_pending = true;

    if (job.mode === 'swipe') {
        // Deliver into this exact slot even if the user swipes around while waiting.
        job.swipeId = message.swipe_id ?? 0;
    }

    await saveChatConditional();
}

/**
 * Reads the batch settings sent by the backend (sourced from config.yaml): the
 * master switch plus the poll cadence.
 * @param {object} data Response payload from submit/list
 */
function applyServerSettings(data) {
    if (typeof data?.batchEnabled === 'boolean') {
        setClaudeBatchServerEnabled(data.batchEnabled);
    }
    if (Number.isFinite(data?.pollIntervalMs)) {
        pollIntervalMs = Math.max(5000, data.pollIntervalMs);
    }
    if (Number.isFinite(data?.maxWaitMinutes)) {
        maxWaitMinutes = Math.max(1, data.maxWaitMinutes);
    }
}

/**
 * Registers a job and starts its poller.
 * @param {BatchJob} job Job to track
 */
function track(job) {
    if (activeJobs.has(job.jobId)) {
        return;
    }
    activeJobs.set(job.jobId, job);
    job.timer = setInterval(() => pollJob(job.jobId), pollIntervalMs);
    // Don't make the user wait a full interval for the first check on resume.
    setTimeout(() => pollJob(job.jobId), 2000);
}

/**
 * Stops polling a job and forgets it locally.
 * @param {string} jobId Job identifier
 */
function untrack(jobId) {
    const job = activeJobs.get(jobId);
    if (job?.timer) {
        clearInterval(job.timer);
    }
    activeJobs.delete(jobId);
}

/**
 * Tells the server the job is done with, so it stops being resumed on reload.
 * @param {string} jobId Job identifier
 */
async function ackJob(jobId) {
    try {
        await postBatch('ack', { jobId });
    } catch (error) {
        console.error('Failed to acknowledge Claude batch job.', error);
    }
}

/**
 * Checks a job's status and, once it ends, fetches and delivers its result.
 * @param {string} jobId Job identifier
 * @returns {Promise<void>}
 */
async function pollJob(jobId) {
    const job = activeJobs.get(jobId);
    if (!job || job.delivering || job.polling) {
        return;
    }

    if (job.reply) {
        await deliver(job);
        return;
    }

    if (Date.now() - job.createdAt > maxWaitMinutes * 60 * 1000) {
        // Anthropic allows up to 24h, so the batch may well still be running.
        // Stop nagging the API but keep the job on the server: reloading resumes it.
        untrack(jobId);
        toastr.warning(
            t`Still not done after ${String(maxWaitMinutes)} minutes — no longer polling. Reload SillyTavern to resume waiting.`,
            t`Claude Flex batch`,
            { timeOut: 20000 },
        );
        return;
    }

    job.polling = true;
    try {
        const response = await postBatch('status', { jobId, batchId: job.batchId });
        if (!response.ok) {
            console.warn('Claude batch status check failed.', response.status);
            return;
        }

        const status = await response.json();
        if (status?.processing_status !== 'ended') {
            return;
        }

        const resultResponse = await postBatch('result', { jobId, batchId: job.batchId, customId: job.customId });
        if (!resultResponse.ok) {
            console.warn('Claude batch result fetch failed.', resultResponse.status);
            return;
        }

        const result = await resultResponse.json();
        if (result?.resultType !== 'succeeded' || !result?.reply) {
            untrack(jobId);
            // A refusal carries a reason worth showing verbatim (stop_details.category).
            const reason = result?.error?.message;
            await failJob(job, reason || t`Batch ${result?.resultType ?? 'failed'}: no reply was produced.`);
            return;
        }

        untrack(jobId);
        job.reply = result.reply;
        await deliver(job);
    } catch (error) {
        console.error('Claude batch poll error.', error);
    } finally {
        job.polling = false;
    }
}

/**
 * Finds the placeholder message for a job in the currently loaded chat.
 * @param {string} jobId Job identifier
 * @returns {number} Message index, or -1 if not found
 */
function findPlaceholderIndex(jobId) {
    return chat.findIndex(message => message?.extra?.claude_batch_job_id === jobId);
}

/**
 * Writes text into a job's placeholder message, or appends it if the placeholder
 * is gone (deleted by the user, most likely).
 * @param {BatchJob} job Job being delivered
 * @param {string} text Message text
 * @param {string} reasoning Reasoning text
 * @returns {Promise<boolean>} Whether the message was written
 */
async function writeIntoChat(job, text, reasoning) {
    const index = findPlaceholderIndex(job.jobId);

    if (power_user.trim_spaces) {
        text = text.trim();
    }

    if (index === -1) {
        // Placeholder is gone (deleted, or the chat was rolled back). Append instead
        // of guessing where it belonged.
        await saveReply({ type: 'normal', getMessage: text, reasoning });
        await saveChatConditional();
        return true;
    }

    // Continue appends to what was already there rather than replacing it.
    const finalText = job.mode === 'continue' ? `${job.originalMes ?? ''}${text}` : text;

    const message = chat[index];
    message.extra = message.extra ?? {};
    message.extra.reasoning = reasoning || '';
    delete message.extra.claude_batch_pending;
    delete message.extra.claude_batch_job_id;
    message.gen_finished = new Date();

    // Fill the slot this job reserved. For a swipe that's the slot captured at submit,
    // which may not be the one on screen if the user swiped around while waiting.
    const slotId = job.mode === 'swipe' && typeof job.swipeId === 'number'
        ? job.swipeId
        : message.swipe_id;

    if (Array.isArray(message.swipes) && typeof slotId === 'number') {
        message.swipes[slotId] = finalText;
        if (Array.isArray(message.swipe_info) && message.swipe_info[slotId]) {
            message.swipe_info[slotId].extra = structuredClone(message.extra);
        }
    }

    const isSlotOnScreen = typeof slotId !== 'number' || message.swipe_id === slotId;
    if (isSlotOnScreen) {
        message.mes = finalText;
        updateMessageBlock(index, message);
    }

    await eventSource.emit(event_types.MESSAGE_RECEIVED, index, job.mode);
    await eventSource.emit(event_types.CHARACTER_MESSAGE_RENDERED, index, job.mode);
    await saveChatConditional();
    return true;
}

/**
 * Delivers a completed job into its origin chat. If that chat isn't open, the job
 * is held until the user opens it — the on-disk placeholder keeps its place.
 * @param {BatchJob} job Job to deliver
 * @returns {Promise<void>}
 */
async function deliver(job) {
    const chatName = job.characterName || t`another chat`;

    if (job.chatId && getCurrentChatId() !== job.chatId) {
        // Hold it: park the job so CHAT_CHANGED can deliver it later.
        if (!activeJobs.has(job.jobId)) {
            activeJobs.set(job.jobId, job);
        }
        if (!job.notified) {
            job.notified = true;
            toastr.success(t`Flex reply is ready in ${chatName}. Open that chat to see it.`, t`Claude Flex batch`, { timeOut: 15000 });
        }
        return;
    }

    job.delivering = true;
    try {
        // Same post-processing the synchronous path applies in Generate's onSuccess.
        const text = cleanUpMessage({
            getMessage: extractMessageFromData(job.reply, 'openai') || '',
            isImpersonate: false,
            isContinue: false,
            displayIncompleteSentences: false,
        });
        let reasoning = getRegexedString(extractReasoningFromData(job.reply, { mainApi: 'openai', chatCompletionSource: 'claude' }) || '', regex_placement.REASONING);
        if (power_user.trim_spaces) {
            reasoning = reasoning.trim();
        }
        await writeIntoChat(job, text, reasoning);
        toastr.success(t`Flex reply delivered.`, t`Claude Flex batch`, { timeOut: 6000 });
    } catch (error) {
        console.error('Failed to deliver Claude batch reply.', error);
    } finally {
        job.delivering = false;
        untrack(job.jobId);
        await ackJob(job.jobId);
    }
}

/**
 * Replaces a job's placeholder with an error note.
 * @param {BatchJob} job Job that failed
 * @param {string} reason Human-readable failure reason
 * @returns {Promise<void>}
 */
async function failJob(job, reason) {
    toastr.error(reason, t`Batch Processing`, { timeOut: 15000 });

    if (!job.chatId || getCurrentChatId() === job.chatId) {
        const index = findPlaceholderIndex(job.jobId);
        if (index !== -1) {
            const message = chat[index];
            // A failed continue should leave the message as it was, not blow it away.
            message.mes = job.mode === 'continue'
                ? `${job.originalMes ?? ''}\n\n*⚠️ ${reason}*`
                : `*⚠️ ${reason}*`;
            delete message.extra.claude_batch_pending;
            delete message.extra.claude_batch_job_id;
            if (Array.isArray(message.swipes) && typeof message.swipe_id === 'number') {
                message.swipes[message.swipe_id] = message.mes;
            }
            updateMessageBlock(index, message);
            await saveChatConditional();
        }
    }

    await ackJob(job.jobId);
}

/**
 * Restores pending batch jobs after a page reload or server restart, and hooks up
 * deferred delivery for jobs whose origin chat isn't currently open.
 * @returns {Promise<void>}
 */
export async function initClaudeBatchTracker() {
    eventSource.on(event_types.CHAT_CHANGED, async () => {
        const currentChatId = getCurrentChatId();
        for (const job of [...activeJobs.values()]) {
            if (job.reply && job.chatId === currentChatId) {
                await deliver(job);
            }
        }
    });

    try {
        const response = await fetch(`${API_BASE}/list`, {
            method: 'GET',
            headers: getRequestHeaders(),
        });

        if (!response.ok) {
            return;
        }

        const data = await response.json();
        applyServerSettings(data);

        for (const stored of data?.jobs ?? []) {
            if (stored.status === 'ready' && stored.resultReply) {
                // Finished while ST was down — deliver as soon as we can.
                const job = { ...stored, reply: stored.resultReply };
                activeJobs.set(job.jobId, job);
                await deliver(job);
                continue;
            }

            track({
                jobId: stored.jobId,
                batchId: stored.batchId,
                customId: stored.customId,
                chatId: stored.chatId ?? null,
                characterName: stored.characterName ?? null,
                createdAt: stored.createdAt ?? Date.now(),
                mode: stored.mode ?? 'normal',
                swipeId: stored.swipeId,
                originalMes: stored.originalMes,
            });
        }

        if (activeJobs.size) {
            console.info(`Resumed ${activeJobs.size} Claude batch job(s).`);
        }
    } catch (error) {
        console.error('Failed to restore Claude batch jobs.', error);
    }
}

/**
 * Whether any batch job is currently pending. Used for UI affordances.
 * @returns {boolean}
 */
export function hasPendingClaudeBatches() {
    return activeJobs.size > 0;
}

/**
 * Whether a chat has a batch in flight. Generating in such a chat would feed the
 * placeholder into the prompt as an assistant turn (and a swipe would swipe the
 * placeholder itself), so the caller blocks generation until it lands.
 * @param {string} [chatId] Chat id (defaults to the open chat)
 * @returns {boolean}
 */
export function hasPendingClaudeBatchForChat(chatId = getCurrentChatId()) {
    if (!chatId) {
        return false;
    }
    return [...activeJobs.values()].some(job => job.chatId === chatId);
}
