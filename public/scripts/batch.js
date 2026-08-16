import {
    chat,
    cleanUpMessage,
    event_types,
    eventSource,
    extractMessageFromData,
    formatGenerationTimer,
    formatTokenCounter,
    getCurrentChatId,
    getRequestHeaders,
    main_api,
    name2,
    saveChatConditional,
    saveReply,
    updateMessageBlock,
    updateMessageTokenCount,
} from '../script.js';
import { extractReasoningFromData } from './reasoning.js';
import { createGenerationParameters, getBatchRequestExtras, getChatCompletionModel, getOnDemandBatchProvider, hasBatchApiKey, oai_settings, resolveBatchPlan, setBatchServerEnabled } from './openai.js';
import { getRegexedString, regex_placement } from './extensions/regex/engine.js';
import { power_user } from './power-user.js';
import { callGenericPopup, POPUP_RESULT, POPUP_TYPE } from './popup.js';
import { t } from './i18n.js';

const API_BASE = '/api/backends/chat-completions/batch';

/** Text shown in the chat while the batch is cooking. */
const PLACEHOLDER = '*⏳ Waiting for a batched reply…*';

/**
 * How each provider's batch reply differs once it's back. Everything else about a
 * job — placeholder, polling, delivery, cancellation — is provider-neutral.
 * @type {Record<string, { label: string, source: string, usage: (reply: object) => { total: any, thinking: any } }>}
 */
const PROVIDERS = {
    claude: {
        label: 'Claude Flex batch',
        source: 'claude',
        // The API's own usage block (the same one behind the server's `out:` line):
        // `output_tokens` totals the reply, and the thinking half is only summarized
        // by the visible reasoning, so it has to come from the counter.
        usage: reply => ({
            total: reply?.usage?.output_tokens,
            thinking: reply?.usage?.output_tokens_details?.thinking_tokens,
        }),
    },
    openrouter: {
        label: 'OpenRouter batch',
        source: 'openrouter',
        usage: reply => ({
            total: reply?.usage?.completion_tokens,
            thinking: reply?.usage?.completion_tokens_details?.reasoning_tokens,
        }),
    },
};

/** @param {string} [provider] Provider name */
function providerInfo(provider) {
    return PROVIDERS[provider] ?? PROVIDERS.claude;
}

/** Active jobs, keyed by jobId. @type {Map<string, BatchJob>} */
const activeJobs = new Map();

let pollIntervalMs = 20000;
let maxWaitMinutes = 90;

/**
 * @typedef {object} BatchJob
 * @property {string} jobId Server-side job identifier
 * @property {string} provider Batch provider ('claude' | 'openrouter')
 * @property {string} batchId Provider-side batch identifier
 * @property {string} customId custom_id of the single request in the batch
 * @property {string|null} chatId Chat the reply belongs to
 * @property {string|null} characterName Character name for toasts
 * @property {number} createdAt Timestamp of submission
 * @property {'normal'|'swipe'|'continue'} mode How the reply gets written back
 * @property {number} [swipeId] Swipe slot to fill (mode 'swipe')
 * @property {string} [originalMes] Message text before the continuation (mode 'continue')
 * @property {number} [timer] setInterval handle
 * @property {number} [pollingSince] When this session started watching the job
 * @property {boolean} [pollingStopped] Whether the wait window ran out and polling gave up
 * @property {object} [reply] Completed reply payload, awaiting delivery
 * @property {boolean} [delivering] Guard against re-entrant delivery
 * @property {boolean} [polling] Guard against overlapping polls
 * @property {boolean} [cancelling] Guard while a cancel is in flight
 * @property {boolean} [notified] Whether the "ready elsewhere" toast has been shown
 */

/**
 * Credentials/settings the backend needs on every batch call. `generate_data` is
 * long gone by the time a resumed poller runs, so these are re-read from settings.
 * The provider comes from the job where there is one: a resumed job may belong to
 * an API the user has since switched away from.
 * @param {BatchJob} [job] Job the call is about
 * @returns {object}
 */
function batchExtras(job) {
    const extras = getBatchRequestExtras();
    return job?.provider ? { ...extras, batch_provider: job.provider } : extras;
}

/**
 * @param {string} path Endpoint path under the batch API base
 * @param {object} body Request payload
 * @param {BatchJob} [job] Job the call is about
 * @returns {Promise<Response>}
 */
function postBatch(path, body, job) {
    return fetch(`${API_BASE}/${path}`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ ...batchExtras(job), ...body }),
    });
}

/**
 * Submits a generation as a batch and returns immediately, leaving a placeholder
 * message in the chat that gets filled in when the batch finishes.
 * @param {string} type Generation type
 * @param {object} generateData Generation payload from Generate() — carries the prompt, not the API body
 * @param {import('../script.js').AdditionalRequestOptions} [options] Additional request options
 * @param {AbortSignal} [signal] Generation abort signal
 * @returns {Promise<'queued'|'sync'|'refused'>} 'sync' = run the normal request; 'refused' = abort, don't bill
 */
export async function startBatch(type, generateData, options = {}, signal = null) {
    const plan = resolveBatchPlan(type);
    if (plan.mode !== 'batch') {
        // Defensive only: Generate() has already shown the reason and, where the
        // answer is "refuse", stopped before anything was committed to the chat.
        return plan.mode === 'refuse' ? 'refused' : 'sync';
    }

    // A batch detaches the instant it's submitted, so an abort raised while the prompt
    // was still being assembled — the stop button, or Prompt Inspector's "Cancel
    // generation" — has to be caught here. The synchronous path gets this for free by
    // handing the signal to fetch; there's no in-flight request for it to cancel here.
    if (signal?.aborted) {
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
        console.error('Batch parameters could not be built.', error);
        toastr.error(t`Couldn't build the batch request. Nothing was sent.`, t`Batch Processing`, { timeOut: 15000 });
        return 'refused';
    }

    // Building the parameters awaits CHAT_COMPLETION_SETTINGS_READY listeners, so the
    // user has had another window in which to cancel. Last check before it's billable.
    if (signal?.aborted) {
        return 'refused';
    }

    // A failed submit has sent nothing, so the model decides what happens next: for a
    // batch-only Claude model there's no affordable synchronous form to fall back to,
    // while an OpenRouter batch is an optimization over a request that is perfectly
    // sendable at standard price. Either way the reason is shown, never swallowed.
    const onSubmitFailure = plan.provider === 'claude' ? 'refused' : 'sync';

    try {
        response = await fetch(`${API_BASE}/submit`, {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                ...generate_data,
                stream: false,
                batch_provider: plan.provider,
                batch_chat_id: chatId ?? null,
                batch_character_name: name2,
            }),
        });
    } catch (error) {
        console.error('Batch submit failed.', error);
        toastr.error(t`Couldn't reach the batch endpoint. Nothing was sent.`, t`Batch Processing`, { timeOut: 15000 });
        return onSubmitFailure;
    }

    if (!response.ok) {
        const detail = await response.json().catch(() => null);
        console.error('Batch submit error.', response.status, detail);
        applyServerSettings(plan.provider, detail);
        const fallbackNotice = onSubmitFailure === 'sync'
            ? t`Batch submission failed — sending this one normally, at standard price.`
            : t`Batch submission failed. Nothing was sent — switch to another Claude model to generate normally.`;
        toastr.error(
            detail?.reason || fallbackNotice,
            t`Batch Processing`,
            { timeOut: 15000, extendedTimeOut: 25000 },
        );
        return onSubmitFailure;
    }

    const data = await response.json();
    applyServerSettings(plan.provider, data);

    const job = {
        jobId: data.jobId,
        provider: data.provider ?? plan.provider,
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
    }, job).catch(error => console.error('Failed to persist batch delivery info.', error));
    track(job);
    // After track(): the cancel button only shows for a job that's actually tracked.
    refreshAllCancelButtons();

    toastr.info(t`Reply will arrive here when it's done — cancel it from the message, or keep using other chats meanwhile.`, t`Batch queued`, { timeOut: 8000 });
    return 'queued';
}

/**
 * Why an explicit `batch=true` generation can't run, if it can't. A chat generation
 * falls back to a normal send when it can't be batched; this one is an opt-in on a
 * single command, so a refusal has to name what's missing instead of quietly sending
 * the request at full price. The group-chat and impersonate blockers don't apply
 * here — nothing is delivered into a chat, the caller just awaits the text.
 * @returns {string|null} Human-readable reason, or null when the batch can go ahead
 */
export function getBatchOnDemandBlocker() {
    if (main_api !== 'openai') {
        return t`batch=true needs the Chat Completion API.`;
    }

    if (!getOnDemandBatchProvider()) {
        return t`batch=true needs a batch-capable API: a batch-enabled Claude model, or OpenRouter (with the server's batch support left on).`;
    }

    if (!hasBatchApiKey()) {
        return t`batch=true needs an sk-ant API key as the proxy password — subscription auth is billed at full price and gets no batch discount.`;
    }

    return null;
}

/**
 * Delay between polls that the stop button can cut short. `delay()` can't be
 * interrupted, and an awaited batch shouldn't sit out a full poll interval after
 * the user has already cancelled.
 * @param {number} ms Milliseconds to wait
 * @param {AbortSignal} [signal] Abort signal
 * @returns {Promise<void>}
 */
function interruptibleDelay(ms, signal) {
    return new Promise((resolve, reject) => {
        const cleanup = () => {
            clearTimeout(timer);
            signal?.removeEventListener('abort', onAbort);
        };
        const onAbort = () => {
            cleanup();
            reject(new Error('Batch generation was aborted.'));
        };
        const timer = setTimeout(() => {
            cleanup();
            resolve();
        }, ms);

        if (signal?.aborted) {
            onAbort();
            return;
        }
        signal?.addEventListener('abort', onAbort, { once: true });
    });
}

/**
 * Best-effort cancel of a batch nothing is waiting on any more. Providers drop
 * requests that haven't started yet; the endpoint drops the persisted job whatever
 * the answer, so this doubles as the acknowledgement.
 * @param {{ jobId: string, batchId: string, provider?: string }} job Job to abandon
 * @returns {Promise<void>}
 */
async function abandonBatch(job) {
    try {
        await postBatch('cancel', { jobId: job.jobId, batchId: job.batchId }, job);
    } catch (error) {
        console.error('Failed to cancel abandoned batch.', error);
    }
}

/**
 * Runs one generation through the batch API and waits for it instead of detaching
 * it into the chat. This is the `/gen batch=true` path: the caller is a slash
 * command that awaits a string, so there's no message to park a placeholder in and
 * nothing to deliver later — the reply *is* the return value.
 * @param {object} generateData Fully built generation parameters, as sendOpenAIRequest assembles them
 * @param {AbortSignal} [signal] Abort signal — cancels the batch and rejects
 * @returns {Promise<object>} Reply payload in the same shape the synchronous path returns
 * @throws {Error} If the batch can't be submitted, fails, is refused, or outlives the wait window
 */
export async function requestBatchReply(generateData, signal = null) {
    const blocker = getBatchOnDemandBlocker();
    if (blocker) {
        throw new Error(blocker);
    }

    if (signal?.aborted) {
        throw new Error('Batch generation was aborted.');
    }

    const provider = getOnDemandBatchProvider();
    let response;
    try {
        // `batch_mode` marks the job as awaited: it has no home in any chat, so a
        // reload must drop it rather than resume it into whatever chat is open.
        response = await fetch(`${API_BASE}/submit`, {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({ ...generateData, stream: false, batch_provider: provider, batch_mode: 'slash' }),
        });
    } catch (error) {
        console.error('Batch submit failed.', error);
        throw new Error('Couldn\'t reach the batch endpoint. Nothing was sent.');
    }

    if (!response.ok) {
        const detail = await response.json().catch(() => null);
        console.error('Batch submit error.', response.status, detail);
        applyServerSettings(provider, detail);
        throw new Error(detail?.reason || 'Batch submission failed. Nothing was sent.');
    }

    const job = await response.json();
    applyServerSettings(job.provider ?? provider, job);
    console.info(`Awaiting ${providerInfo(job.provider).label} ${job.batchId} (job ${job.jobId}).`);
    toastr.info(t`Waiting for the batched reply — this can take several minutes.`, t`Batch Processing`, { timeOut: 10000 });

    const deadline = Date.now() + maxWaitMinutes * 60 * 1000;
    try {
        while (true) {
            await interruptibleDelay(pollIntervalMs, signal);

            if (Date.now() > deadline) {
                throw new Error(`Batch didn't finish within ${maxWaitMinutes} minutes. Nothing was returned.`);
            }

            // A blip on a poll isn't a failed generation: keep asking until the
            // deadline rather than throwing away a batch that's still cooking.
            const statusResponse = await postBatch('status', { jobId: job.jobId, batchId: job.batchId }, job);
            if (!statusResponse.ok) {
                console.warn('Batch status check failed.', statusResponse.status);
                continue;
            }

            const status = await statusResponse.json();
            if (status?.state !== 'ended') {
                continue;
            }

            const resultResponse = await postBatch('result', { jobId: job.jobId, batchId: job.batchId, customId: job.customId }, job);
            if (!resultResponse.ok) {
                console.warn('Batch result fetch failed.', resultResponse.status);
                continue;
            }

            const result = await resultResponse.json();
            if (result?.resultType !== 'succeeded' || !result?.reply) {
                // A refusal carries a reason worth showing verbatim (stop_details.category).
                throw new Error(result?.error?.message || `Batch ${result?.resultType ?? 'failed'}: no reply was produced.`);
            }

            await ackJob(job.jobId, job);
            return result.reply;
        }
    } catch (error) {
        await abandonBatch(job);
        throw error;
    }
}

/**
 * The job a message is a pending placeholder for, if it is one. The `claude_`-prefixed
 * pair is what placeholders written before OpenRouter batching carried, so both are
 * read — a batch that was in flight across the upgrade still finds its home.
 * @param {object} message Chat message
 * @returns {string|null} Job identifier, or null when the message isn't pending
 */
function pendingJobIdOf(message) {
    const extra = message?.extra;
    if (!extra?.batch_pending && !extra?.claude_batch_pending) {
        return null;
    }
    return extra.batch_job_id ?? extra.claude_batch_job_id ?? null;
}

/**
 * Clears the pending markers once a job is settled, in both spellings.
 * @param {object} extra Message `extra` object (mutated in place)
 */
function clearPendingMarkers(extra) {
    delete extra.batch_pending;
    delete extra.batch_job_id;
    delete extra.claude_batch_pending;
    delete extra.claude_batch_job_id;
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
    message.extra.batch_job_id = job.jobId;
    message.extra.batch_pending = true;

    // saveReply timed and counted the placeholder text itself ("0.0s", a handful of
    // tokens). Blank both — delivery fills them with the real wait and output size.
    // gen_started stays: it's the clock the elapsed time is measured from.
    message.gen_finished = undefined;
    delete message.extra.token_count;
    delete message.extra.reasoning_token_count;
    delete message.extra.reported_reasoning_tokens;
    refreshTimerAndTokenDom(chat.length - 1, message);

    if (job.mode === 'swipe') {
        // Deliver into this exact slot even if the user swipes around while waiting.
        job.swipeId = message.swipe_id ?? 0;
    }

    await saveChatConditional();
}

/**
 * Repaints a message's generation timer and token counter from the message object.
 * `updateMessageBlock` doesn't touch either — the synchronous paths own them through
 * `addOneMessage` or the streaming processor's cached DOM refs, neither of which runs
 * for a batch delivery.
 * @param {number} index Message index
 * @param {object} message Chat message
 */
function refreshTimerAndTokenDom(index, message) {
    const element = document.querySelector(`#chat .mes[mesid="${index}"]`);
    if (!element) {
        return;
    }

    const { timerValue, timerTitle } = formatGenerationTimer(
        message.gen_started,
        message.gen_finished,
        message.extra?.token_count,
        message.extra?.reasoning_duration,
        message.extra?.time_to_first_token,
    );

    const timerDom = element.querySelector('.mes_timer');
    if (timerDom) {
        timerDom.textContent = timerValue ?? '';
        timerDom.title = timerTitle ?? '';
    }

    const { counterValue, counterTitle } = formatTokenCounter(message.extra);
    const counterDom = element.querySelector('.tokenCounterDisplay');
    if (counterDom) {
        counterDom.textContent = counterValue;
        counterDom.title = counterTitle;
    }
}

/**
 * Shows the per-message cancel button on a pending placeholder and hides it once the
 * job is resolved. A batch detaches on submit — the generation is over as far as the
 * rest of the UI is concerned, so the placeholder itself has to carry the only handle
 * on the job. `#mes_stop` is long gone by then and `stopGeneration()` never saw it.
 * @param {number} index Message index
 * @param {object} message Chat message
 */
function refreshCancelButton(index, message) {
    const button = document.querySelector(`#chat .mes[mesid="${index}"] .mes_batch_cancel`);
    if (!(button instanceof HTMLElement)) {
        return;
    }

    // Being flagged pending isn't enough: a delivered or dropped job can leave its
    // markers behind on disk, and cancelling needs the batchId only a tracked job
    // has. Show the button only where it can actually do something — which includes
    // a job we've stopped polling, since that's exactly when the user wants it gone.
    const jobId = pendingJobIdOf(message);
    button.style.display = jobId && activeJobs.has(jobId) ? '' : 'none';
}

/**
 * Repaints every cancel button in the open chat. Needed after any render that rebuilds
 * message elements from scratch — chat load, lazy-loading older messages — since a
 * resumed job's placeholder comes back from disk with a fresh, buttonless DOM node.
 */
function refreshAllCancelButtons() {
    chat.forEach((message, index) => {
        if (pendingJobIdOf(message)) {
            refreshCancelButton(index, message);
        }
    });
}

/**
 * Stores a delivered reply's token counts from the API's own usage block — the same
 * one behind the server's usage line — since the visible reasoning only summarizes
 * the thinking the model was actually billed for. A continuation is the exception:
 * those counts cover just the appended part, but the counter describes the whole
 * message, so it falls back to counting locally.
 * @param {BatchJob} job Job being delivered
 * @param {object} extra Message `extra` object (mutated in place)
 * @param {string} text Final message text
 * @param {string} reasoning Reasoning text
 * @returns {Promise<number>} Token count, or 0 when the counter is disabled
 */
async function countReplyTokens(job, extra, text, reasoning) {
    if (!power_user.message_token_count_enabled) {
        return 0;
    }

    const reportedUsage = job.mode === 'continue' ? null : providerInfo(job.provider).usage(job.reply);
    return await updateMessageTokenCount(extra, text, reasoning, reportedUsage);
}

/**
 * Reads the batch settings sent by the backend (sourced from config.yaml): the
 * provider's master switch, plus the poll cadence when they belong to the API
 * currently in use. The cadence is a single global pair — only one provider can be
 * selected at a time, and a resumed job of the other kind is happy to be polled on
 * either schedule, so taking the inactive provider's numbers would be arbitrary.
 * @param {string} provider Provider the settings belong to
 * @param {object} data Response payload from submit/list
 * @param {boolean} [applyCadence] Whether to adopt this provider's poll timings
 */
function applyServerSettings(provider, data, applyCadence = true) {
    if (typeof data?.batchEnabled === 'boolean') {
        setBatchServerEnabled(provider, data.batchEnabled);
    }
    if (!applyCadence) {
        return;
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
    // The give-up window runs from here, not from submission. Measured from
    // `createdAt` — which is persisted — a job restored past the window would hit the
    // give-up branch on its very first poll, in this session and every session after,
    // and so could never be checked on again; "reload to resume waiting" was advice
    // that by construction could not work.
    job.pollingSince = Date.now();
    job.pollingStopped = false;
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
 * Stops polling a job but keeps it tracked. A job we've given up chasing still has a
 * live batch behind it, and cancelling needs the batchId only a tracked job carries —
 * so dropping it entirely would leave the user no way to be rid of it short of editing
 * the server's job store by hand.
 * @param {string} jobId Job identifier
 */
function stopPolling(jobId) {
    const job = activeJobs.get(jobId);
    if (!job) {
        return;
    }
    if (job.timer) {
        clearInterval(job.timer);
        job.timer = undefined;
    }
    job.pollingStopped = true;
}

/**
 * Tells the server the job is done with, so it stops being resumed on reload.
 * @param {string} jobId Job identifier
 * @param {BatchJob} [job] Job being acknowledged
 */
async function ackJob(jobId, job) {
    try {
        await postBatch('ack', { jobId }, job);
    } catch (error) {
        console.error('Failed to acknowledge batch job.', error);
    }
}

/**
 * Checks a job's status and, once it ends, fetches and delivers its result.
 * @param {string} jobId Job identifier
 * @returns {Promise<void>}
 */
async function pollJob(jobId) {
    const job = activeJobs.get(jobId);
    if (!job || job.delivering || job.polling || job.cancelling || job.pollingStopped) {
        return;
    }

    if (job.reply) {
        await deliver(job);
        return;
    }

    if (Date.now() - (job.pollingSince ?? job.createdAt) > maxWaitMinutes * 60 * 1000) {
        // Both APIs allow up to 24h, so the batch may well still be running. Stop
        // nagging the API but keep the job on the server: reloading starts a fresh
        // window. The toast is clickable because otherwise a batch that never lands
        // has no end state — it would just be re-resumed and re-warned about forever.
        stopPolling(jobId);
        toastr.warning(
            t`Still not done after ${String(maxWaitMinutes)} minutes — no longer polling. Reload SillyTavern to keep waiting, or click here to drop it.`,
            providerInfo(job.provider).label,
            { timeOut: 20000, extendedTimeOut: 30000, onclick: () => cancelJob(jobId) },
        );
        return;
    }

    job.polling = true;
    try {
        const response = await postBatch('status', { jobId, batchId: job.batchId }, job);
        if (!response.ok) {
            // Batches are deleted once they age out (29 days at Anthropic, 30 at
            // OpenRouter), and a 404 is the one answer no amount of retrying will
            // improve. Settle the placeholder instead of polling a batch that no
            // longer exists until the window runs out.
            if (response.status === 404) {
                untrack(jobId);
                await failJob(job, t`This batch is no longer available from the provider.`);
                return;
            }
            console.warn('Batch status check failed.', response.status);
            return;
        }

        const status = await response.json();
        if (status?.state !== 'ended') {
            return;
        }

        const resultResponse = await postBatch('result', { jobId, batchId: job.batchId, customId: job.customId }, job);
        if (!resultResponse.ok) {
            // The batch ended but its results are gone. Nothing to wait for — settle
            // it. See above.
            if (resultResponse.status === 404) {
                untrack(jobId);
                await failJob(job, t`This batch's results are no longer available from the provider.`);
                return;
            }
            console.warn('Batch result fetch failed.', resultResponse.status);
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
        console.error('Batch poll error.', error);
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
    return chat.findIndex(message => pendingJobIdOf(message) === jobId);
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
        const appended = chat[chat.length - 1];
        // saveReply timed this against the generation that produced the placeholder,
        // which for a resumed job may be from a previous session entirely.
        appended.gen_started = new Date(job.createdAt);
        appended.gen_finished = new Date();
        await countReplyTokens(job, appended.extra, appended.mes, reasoning);
        refreshTimerAndTokenDom(chat.length - 1, appended);
        await saveChatConditional();
        return true;
    }

    // Continue appends to what was already there rather than replacing it.
    const finalText = job.mode === 'continue' ? `${job.originalMes ?? ''}${text}` : text;

    const message = chat[index];
    message.extra = message.extra ?? {};
    message.extra.reasoning = reasoning || '';
    clearPendingMarkers(message.extra);
    message.gen_started = message.gen_started ?? new Date(job.createdAt);
    message.gen_finished = new Date();

    await countReplyTokens(job, message.extra, finalText, reasoning);

    // Fill the slot this job reserved. For a swipe that's the slot captured at submit,
    // which may not be the one on screen if the user swiped around while waiting.
    const slotId = job.mode === 'swipe' && typeof job.swipeId === 'number'
        ? job.swipeId
        : message.swipe_id;

    if (Array.isArray(message.swipes) && typeof slotId === 'number') {
        message.swipes[slotId] = finalText;
        if (Array.isArray(message.swipe_info) && message.swipe_info[slotId]) {
            message.swipe_info[slotId].extra = structuredClone(message.extra);
            message.swipe_info[slotId].gen_started = message.gen_started;
            message.swipe_info[slotId].gen_finished = message.gen_finished;
        }
    }

    const isSlotOnScreen = typeof slotId !== 'number' || message.swipe_id === slotId;
    if (isSlotOnScreen) {
        message.mes = finalText;
        updateMessageBlock(index, message);
        refreshTimerAndTokenDom(index, message);
    }
    refreshCancelButton(index, message);

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
    const provider = providerInfo(job.provider);

    if (job.chatId && getCurrentChatId() !== job.chatId) {
        // Hold it: park the job so CHAT_CHANGED can deliver it later.
        if (!activeJobs.has(job.jobId)) {
            activeJobs.set(job.jobId, job);
        }
        if (!job.notified) {
            job.notified = true;
            toastr.success(t`Batched reply is ready in ${chatName}. Open that chat to see it.`, provider.label, { timeOut: 15000 });
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
        let reasoning = getRegexedString(extractReasoningFromData(job.reply, { mainApi: 'openai', chatCompletionSource: provider.source }) || '', regex_placement.REASONING);
        if (power_user.trim_spaces) {
            reasoning = reasoning.trim();
        }
        await writeIntoChat(job, text, reasoning);
        toastr.success(t`Batched reply delivered.`, provider.label, { timeOut: 6000 });
    } catch (error) {
        console.error('Failed to deliver batch reply.', error);
    } finally {
        job.delivering = false;
        untrack(job.jobId);
        await ackJob(job.jobId, job);
    }
}

/**
 * Settles a job's placeholder to a final text, clearing the pending markers so both
 * the cancel button and the per-chat generation block lift. No-op when the origin
 * chat isn't open — the on-disk placeholder keeps its place until it is.
 * @param {BatchJob} job Job whose placeholder is being settled
 * @param {string} text Final message text
 * @returns {Promise<void>}
 */
async function resolvePlaceholder(job, text) {
    if (job.chatId && getCurrentChatId() !== job.chatId) {
        return;
    }

    const index = findPlaceholderIndex(job.jobId);
    if (index === -1) {
        return;
    }

    const message = chat[index];
    message.mes = text;
    clearPendingMarkers(message.extra);
    if (Array.isArray(message.swipes) && typeof message.swipe_id === 'number') {
        message.swipes[message.swipe_id] = message.mes;
    }
    updateMessageBlock(index, message);
    refreshCancelButton(index, message);
    await saveChatConditional();
}

/**
 * Replaces a job's placeholder with an error note.
 * @param {BatchJob} job Job that failed
 * @param {string} reason Human-readable failure reason
 * @returns {Promise<void>}
 */
async function failJob(job, reason) {
    toastr.error(reason, t`Batch Processing`, { timeOut: 15000 });

    // A failed continue should leave the message as it was, not blow it away.
    await resolvePlaceholder(job, job.mode === 'continue'
        ? `${job.originalMes ?? ''}\n\n*⚠️ ${reason}*`
        : `*⚠️ ${reason}*`);

    await ackJob(job.jobId, job);
}

/**
 * Cancels a pending batch from its placeholder's cancel button. This is the only stop
 * control a batch has: submission detaches the generation, so by the time the job
 * exists `#mes_stop` is hidden and `stopGeneration()` has no handle on it.
 * @param {string} jobId Job identifier
 * @returns {Promise<void>}
 */
async function cancelJob(jobId) {
    const job = activeJobs.get(jobId);
    if (!job || job.delivering || job.cancelling) {
        return;
    }

    const confirmed = await callGenericPopup(
        t`Cancel this batched reply? Requests that haven't started yet are dropped, but one already in progress may still finish and be billed.`,
        POPUP_TYPE.CONFIRM,
    );

    // The reply may have landed while the popup sat open.
    if (confirmed !== POPUP_RESULT.AFFIRMATIVE || !activeJobs.has(jobId)) {
        return;
    }

    // Hold the poller off rather than untracking: a cancel that doesn't reach the
    // server has to leave the job exactly as it found it, still being waited on.
    job.cancelling = true;
    let cancelled = false;
    try {
        const response = await postBatch('cancel', { jobId, batchId: job.batchId }, job);
        // The endpoint drops the persisted job for any answer it gets back from the
        // provider, refusals included; only its own 500 leaves the job on the server.
        cancelled = response.status !== 500;
        if (!response.ok) {
            console.warn('Batch cancel was refused upstream.', response.status);
        }
    } catch (error) {
        console.error('Batch cancel error.', error);
    } finally {
        job.cancelling = false;
    }

    if (!cancelled) {
        toastr.error(t`Couldn't reach the batch endpoint — the reply is still on its way.`, t`Batch Processing`, { timeOut: 12000 });
        return;
    }

    untrack(jobId);
    // A cancelled continue reverts to exactly what it was — the user knows why, so
    // there's nothing to annotate, and the message stays immediately re-continuable.
    await resolvePlaceholder(job, job.mode === 'continue' ? (job.originalMes ?? '') : '*🚫 Batch cancelled.*');
    toastr.info(t`Batch cancelled.`, t`Batch Processing`, { timeOut: 6000 });
}

/**
 * Restores pending batch jobs after a page reload or server restart, and hooks up
 * deferred delivery for jobs whose origin chat isn't currently open.
 * @returns {Promise<void>}
 */
export async function initBatchTracker() {
    eventSource.on(event_types.CHAT_CHANGED, async () => {
        const currentChatId = getCurrentChatId();
        for (const job of [...activeJobs.values()]) {
            if (job.reply && job.chatId === currentChatId) {
                await deliver(job);
            }
        }
        // After the deliveries, so a placeholder that just got filled isn't handed a
        // cancel button for a job that no longer exists.
        refreshAllCancelButtons();
    });

    // Lazy-loading rebuilds message elements from the template, losing button state;
    // swiping left off a placeholder swaps `extra` out from under it and back again.
    eventSource.on(event_types.MORE_MESSAGES_LOADED, () => refreshAllCancelButtons());
    eventSource.on(event_types.MESSAGE_SWIPED, () => refreshAllCancelButtons());

    // Delegated: placeholders come and go, and a resumed one is rendered by the chat
    // loader long before this module knows the job exists.
    document.addEventListener('click', event => {
        const button = event.target instanceof Element ? event.target.closest('.mes_batch_cancel') : null;
        if (!button) {
            return;
        }
        const index = Number(button.closest('.mes')?.getAttribute('mesid'));
        const jobId = pendingJobIdOf(chat[index]);
        if (jobId) {
            cancelJob(jobId);
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
        const activeProvider = getOnDemandBatchProvider() ?? 'claude';
        for (const [provider, settings] of Object.entries(data?.settings ?? {})) {
            applyServerSettings(provider, settings, provider === activeProvider);
        }

        for (const stored of data?.jobs ?? []) {
            // An awaited batch (`/gen batch=true`) belonged to a slash command that
            // died with the page. There's nothing left to hand the reply to, and it
            // was never headed for a chat, so drop it instead of resuming it.
            if (stored.mode === 'slash') {
                console.warn(`Dropping awaited batch job ${stored.jobId} — whatever was waiting on it is gone.`);
                await abandonBatch(stored);
                toastr.warning(t`A batched slash command was still waiting when the page reloaded — it's been dropped.`, t`Batch Processing`, { timeOut: 12000 });
                continue;
            }

            if (stored.status === 'ready' && stored.resultReply) {
                // Finished while ST was down — deliver as soon as we can.
                const job = { ...stored, reply: stored.resultReply };
                activeJobs.set(job.jobId, job);
                await deliver(job);
                continue;
            }

            track({
                jobId: stored.jobId,
                provider: stored.provider ?? 'claude',
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

        // The open chat may have been rendered before this ran, so its restored
        // placeholders are still sitting there without a cancel button.
        refreshAllCancelButtons();

        if (activeJobs.size) {
            console.info(`Resumed ${activeJobs.size} batch job(s).`);
        }
    } catch (error) {
        console.error('Failed to restore batch jobs.', error);
    }
}

/**
 * Whether any batch job is currently pending. Used for UI affordances.
 * A job we've stopped polling is still tracked so it stays cancellable, but nothing
 * is waiting on it any more — it doesn't count as pending.
 * @returns {boolean}
 */
export function hasPendingBatches() {
    return [...activeJobs.values()].some(job => !job.pollingStopped);
}

/**
 * Whether a chat has a batch in flight. Generating in such a chat would feed the
 * placeholder into the prompt as an assistant turn (and a swipe would swipe the
 * placeholder itself), so the caller blocks generation until it lands.
 * @param {string} [chatId] Chat id (defaults to the open chat)
 * @returns {boolean}
 */
export function hasPendingBatchForChat(chatId = getCurrentChatId()) {
    if (!chatId) {
        return false;
    }
    // A job we've given up polling doesn't hold the chat hostage: its placeholder is
    // still there to be swiped away or cancelled, and blocking on it would leave the
    // chat unusable until the user noticed why.
    return [...activeJobs.values()].some(job => job.chatId === chatId && !job.pollingStopped);
}
