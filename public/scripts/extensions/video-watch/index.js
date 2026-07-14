import { getBase64Async, isTrueBoolean, saveBase64AsFile } from '../../utils.js';
import { getContext, extension_settings, renderExtensionTemplateAsync } from '../../extensions.js';
import { chat_metadata, eventSource, event_types, getRequestHeaders, saveSettingsDebounced } from '../../../script.js';
import { getMessageTimeStamp } from '../../RossAscends-mods.js';
import { isImageInliningSupported } from '../../openai.js';
import { SlashCommandParser } from '../../slash-commands/SlashCommandParser.js';
import { SlashCommand } from '../../slash-commands/SlashCommand.js';
import { ARGUMENT_TYPE, SlashCommandArgument, SlashCommandNamedArgument } from '../../slash-commands/SlashCommandArgument.js';
import { callGenericPopup, POPUP_RESULT, POPUP_TYPE } from '../../popup.js';
import { debounce_timeout, MEDIA_DISPLAY, MEDIA_SOURCE, MEDIA_TYPE } from '../../constants.js';

export { MODULE_NAME };

const MODULE_NAME = 'video-watch';
const PLUGIN_BASE = '/api/plugins/video-watch';

// Hosts yt-dlp reliably supports and that we're willing to auto-offer on paste.
const VIDEO_URL_REGEX = /https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?[\w=&%-]*v=|youtu\.be\/|youtube\.com\/shorts\/|vimeo\.com\/|(?:www\.)?tiktok\.com\/|(?:mobile\.)?(?:twitter|x)\.com\/\S+\/status\/|twitch\.tv\/\S+\/clip\/|clips\.twitch\.tv\/|dailymotion\.com\/video\/)\S+/i;

/** @type {Record<string, any>} */
const defaultSettings = {
    detail: 'balanced',        // transcript | efficient | balanced | token-burner
    resolution: 512,           // frame width in px
    maxFrames: '',             // '' => use the detail-mode cap
    fps: '',                   // '' => auto (clamped to 2fps server-side)
    whisper: 'auto',           // auto | groq | openai | none
    noDedup: false,            // keep near-duplicate frames
    showTranscriptInChat: true,// render the transcript text in the chat message
    autoReply: true,           // trigger a model reply after inserting the message
    autoDetectUrls: false,     // offer to watch pasted/sent video URLs
};

function getSettings() {
    if (!extension_settings[MODULE_NAME]) {
        extension_settings[MODULE_NAME] = structuredClone(defaultSettings);
    }
    for (const [key, value] of Object.entries(defaultSettings)) {
        if (extension_settings[MODULE_NAME][key] === undefined) {
            extension_settings[MODULE_NAME][key] = value;
        }
    }
    return extension_settings[MODULE_NAME];
}

/**
 * Convert the UI/settings-level whisper choice into request flags.
 * @param {string} choice
 * @returns {{ whisper?: string, noWhisper?: boolean }}
 */
function whisperToFlags(choice) {
    if (choice === 'groq' || choice === 'openai') return { whisper: choice };
    if (choice === 'none') return { noWhisper: true };
    return {};
}

/**
 * Call the server plugin to extract frames + transcript from a video.
 * @param {object} payload Request body for POST /extract
 * @returns {Promise<object>} Parsed response
 */
async function requestExtraction(payload) {
    const response = await fetch(`${PLUGIN_BASE}/extract`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data?.error || `Extraction failed (HTTP ${response.status}).`);
    }
    return data;
}

/**
 * Query the server plugin's dependency/key preflight (setup.py --json).
 * @returns {Promise<object|null>}
 */
async function requestStatus() {
    try {
        const response = await fetch(`${PLUGIN_BASE}/status`, { headers: getRequestHeaders() });
        return await response.json().catch(() => null);
    } catch {
        return null;
    }
}

/**
 * Compose the chat message body from the extraction result.
 * @param {{ source: string, question: string, transcript: string, transcriptSource: string, frameCount: number }} args
 * @returns {string}
 */
function buildMessageText({ source, question, transcript, transcriptSource, frameCount }) {
    const settings = getSettings();
    const lines = [];
    const q = String(question || '').trim();
    lines.push(q || 'Watch this video and describe what happens, citing timestamps.');
    lines.push('');

    const meta = [`source: ${source || 'uploaded file'}`];
    meta.push(frameCount ? `${frameCount} frame${frameCount === 1 ? '' : 's'} attached (chronological, titled with t=MM:SS)` : 'no frames (transcript only)');
    if (transcript) meta.push(`transcript via ${transcriptSource || 'captions'}`);
    lines.push(`[video-watch — ${meta.join(', ')}]`);

    if (transcript && settings.showTranscriptInChat) {
        lines.push('');
        lines.push('Transcript:');
        lines.push(transcript);
    }
    return lines.join('\n');
}

/**
 * Insert a user message carrying the extracted frames + transcript, then
 * optionally trigger a model reply. Mirrors caption's sendCaptionedMessage but
 * with a LIST media display so every frame is inlined.
 * @param {{ mesText: string, media: Array<object> }} args
 */
async function sendWatchMessage({ mesText, media }) {
    const settings = getSettings();
    const context = getContext();

    const message = {
        name: context.name1,
        is_user: true,
        send_date: getMessageTimeStamp(),
        mes: mesText,
        extra: {
            media,
            media_display: MEDIA_DISPLAY.LIST,
            media_index: 0,
            inline_image: media.length > 0,
            video_watch: true, // marks our own output so auto-detect skips it
        },
    };

    chat_metadata.tainted = true;
    context.chat.push(message);
    const messageId = context.chat.length - 1;
    await eventSource.emit(event_types.MESSAGE_SENT, messageId);
    context.addOneMessage(message);
    await eventSource.emit(event_types.USER_MESSAGE_RENDERED, messageId);
    await context.saveChat();
    setTimeout(() => context.scrollOnMediaLoad(), debounce_timeout.short);

    if (settings.autoReply) {
        try {
            await context.generate('normal');
        } catch (error) {
            console.error('[video-watch] auto-reply generation failed', error);
        }
    }
}

/**
 * Full watch flow: extract -> upload frames -> insert message (-> reply).
 * @param {object} opts
 * @param {string} [opts.source] Video URL or server-side path
 * @param {string} [opts.videoDataUrl] Base64 data URL of an uploaded video file
 * @param {string} [opts.question] Question to ask about the video
 * @param {boolean} [opts.quiet] Suppress the "watching" toast
 * @returns {Promise<string>} Empty string (slash-command return value)
 */
async function runWatch(opts = {}) {
    const settings = getSettings();
    const source = String(opts.source || '').trim();
    const hasUpload = typeof opts.videoDataUrl === 'string' && opts.videoDataUrl.startsWith('data:');

    if (!source && !hasUpload) {
        toastr.error('Provide a video URL, a local server path, or pick a file.', 'Video Watch');
        return '';
    }

    const effectiveDetail = opts.detail || settings.detail;
    if (effectiveDetail !== 'transcript' && !isImageInliningSupported()) {
        toastr.warning('Media inlining is off or unsupported by the current model — frames will be attached but not sent. Enable it in the Chat Completion settings and use a vision model.', 'Video Watch', { timeOut: 8000 });
    }

    const payload = {
        source: source || undefined,
        videoDataUrl: hasUpload ? opts.videoDataUrl : undefined,
        detail: effectiveDetail,
        start: opts.start,
        end: opts.end,
        maxFrames: opts.maxFrames ?? (settings.maxFrames === '' ? undefined : settings.maxFrames),
        resolution: opts.resolution ?? settings.resolution,
        fps: opts.fps ?? (settings.fps === '' ? undefined : settings.fps),
        timestamps: opts.timestamps,
        noDedup: opts.noDedup ?? settings.noDedup,
        ...whisperToFlags(opts.whisper || settings.whisper),
    };

    const toast = opts.quiet ? null : toastr.info('Downloading and analyzing the video…', 'Video Watch', { timeOut: 0, extendedTimeOut: 0 });
    try {
        const result = await requestExtraction(payload);
        const context = getContext();

        // Persist each returned frame to a servable URL for the media array.
        const media = [];
        for (let i = 0; i < (result.frames?.length || 0); i++) {
            const frame = result.frames[i];
            const base64 = String(frame.dataUrl).split(',')[1];
            if (!base64) continue;
            const url = await saveBase64AsFile(base64, context.name2, `watch_${Date.now()}_${i}`, 'jpg');
            media.push({
                url,
                type: MEDIA_TYPE.IMAGE,
                title: `t=${frame.timestamp}${frame.reason ? ` (${frame.reason})` : ''}`,
                source: MEDIA_SOURCE.UPLOAD,
            });
        }

        if (media.length === 0 && !result.transcript) {
            toastr.warning('No frames or transcript could be produced for this video.', 'Video Watch');
            return '';
        }

        const mesText = buildMessageText({
            source,
            question: opts.question,
            transcript: result.transcript,
            transcriptSource: result.transcriptSource,
            frameCount: media.length,
        });

        await sendWatchMessage({ mesText, media });
        return '';
    } catch (error) {
        console.error('[video-watch] extraction failed', error);
        toastr.error(error.message || 'Unknown error', 'Video Watch');
        return '';
    } finally {
        if (toast) toastr.clear(toast);
    }
}

/**
 * Open the "watch a video" dialog with the full control surface.
 */
async function openWatchDialog(prefillSource = '') {
    const settings = getSettings();
    const $dlg = $(await renderExtensionTemplateAsync(MODULE_NAME, 'dialog', {
        source: prefillSource,
        detail: settings.detail,
        resolution: settings.resolution,
        maxFrames: settings.maxFrames,
        fps: settings.fps,
        whisper: settings.whisper,
        noDedup: settings.noDedup,
    }));

    // Reflect current settings into the controls.
    $dlg.find('#vw_dlg_detail').val(settings.detail);
    $dlg.find('#vw_dlg_whisper').val(settings.whisper);
    $dlg.find('#vw_dlg_nodedup').prop('checked', !!settings.noDedup);

    /** @type {File|null} */
    let pickedFile = null;
    $dlg.find('#vw_dlg_file').on('change', function () {
        pickedFile = this.files && this.files[0] ? this.files[0] : null;
        $dlg.find('#vw_dlg_file_name').text(pickedFile ? pickedFile.name : '');
    });

    const result = await callGenericPopup($dlg, POPUP_TYPE.CONFIRM, '', {
        okButton: 'Watch',
        cancelButton: 'Cancel',
        wide: true,
        allowVerticalScrolling: true,
    });

    if (result !== POPUP_RESULT.AFFIRMATIVE) return;

    const numOrUndef = (v) => {
        const n = Number(String(v).trim());
        return Number.isFinite(n) && n > 0 ? n : undefined;
    };

    let videoDataUrl;
    if (pickedFile) {
        videoDataUrl = await getBase64Async(pickedFile);
    }

    await runWatch({
        source: String($dlg.find('#vw_dlg_source').val() || '').trim(),
        videoDataUrl,
        question: String($dlg.find('#vw_dlg_question').val() || '').trim(),
        detail: String($dlg.find('#vw_dlg_detail').val()),
        start: String($dlg.find('#vw_dlg_start').val() || '').trim() || undefined,
        end: String($dlg.find('#vw_dlg_end').val() || '').trim() || undefined,
        maxFrames: numOrUndef($dlg.find('#vw_dlg_maxframes').val()),
        resolution: numOrUndef($dlg.find('#vw_dlg_resolution').val()),
        fps: numOrUndef($dlg.find('#vw_dlg_fps').val()),
        timestamps: String($dlg.find('#vw_dlg_timestamps').val() || '').trim() || undefined,
        whisper: String($dlg.find('#vw_dlg_whisper').val()),
        noDedup: $dlg.find('#vw_dlg_nodedup').prop('checked'),
    });
}

/**
 * Slash command callback: /watch <url-or-path> [question]
 * @param {object} args Named args (detail, start, end, frames, resolution, fps, timestamps, whisper, dedup, quiet)
 * @param {string} unnamed "<source> [question]"
 */
async function watchCommandCallback(args, unnamed) {
    const raw = String(unnamed || '').trim();
    if (!raw) {
        // No source given -> open the dialog for a guided run.
        await openWatchDialog();
        return '';
    }

    // First whitespace-delimited token is the source; the rest is the question.
    const firstSpace = raw.search(/\s/);
    const source = firstSpace === -1 ? raw : raw.slice(0, firstSpace);
    const question = firstSpace === -1 ? '' : raw.slice(firstSpace + 1).trim();

    await runWatch({
        source,
        question,
        detail: args?.detail,
        start: args?.start,
        end: args?.end,
        maxFrames: args?.frames,
        resolution: args?.resolution,
        fps: args?.fps,
        timestamps: args?.timestamps,
        whisper: args?.whisper,
        noDedup: args?.dedup !== undefined ? !isTrueBoolean(args.dedup) : undefined,
        quiet: isTrueBoolean(args?.quiet),
    });
    return '';
}

/**
 * Render the setup preflight banner from setup.py --json into the settings drawer.
 */
async function refreshStatusBanner() {
    const $banner = $('#vw_status');
    if ($banner.length === 0) return;
    $banner.removeClass('vw-status-ok vw-status-warn').text('Checking setup…');

    const status = await requestStatus();
    if (!status) {
        $banner.addClass('vw-status-warn').text('Could not reach the video-watch server plugin. Is "enableServerPlugins: true" set in config.yaml and the server restarted?');
        return;
    }
    if (status.error) {
        $banner.addClass('vw-status-warn').text(status.error);
        return;
    }
    if (status.can_proceed) {
        const key = status.has_api_key ? `Whisper: ${status.whisper_backend}` : 'no Whisper key (caption-less videos come back frames-only)';
        $banner.addClass('vw-status-ok').text(`Ready. ${key}.`);
        return;
    }

    const parts = [];
    if (Array.isArray(status.missing_binaries) && status.missing_binaries.length) {
        parts.push(`missing: ${status.missing_binaries.join(', ')}`);
    }
    if (!status.has_api_key) {
        parts.push('no Whisper API key');
    }
    const hint = status.platform === 'Windows'
        ? ' Install with: winget install Gyan.FFmpeg ; winget install yt-dlp.yt-dlp'
        : '';
    const keyNote = !status.has_api_key
        ? ' Transcription reuses your OpenAI/Groq key saved in SillyTavern (or one in ~/.config/watch/.env); without it, caption-less videos come back frames-only.'
        : '';
    $banner.addClass('vw-status-warn').text(`Setup incomplete (${parts.join('; ')}).${hint}${keyNote}`);
}

export async function init() {
    getSettings();

    // Settings drawer.
    const settingsHtml = await renderExtensionTemplateAsync(MODULE_NAME, 'settings', {});
    $('#extensions_settings2').append(settingsHtml);

    const settings = getSettings();
    $('#vw_detail').val(settings.detail).on('change', function () {
        settings.detail = String($(this).val());
        saveSettingsDebounced();
    });
    $('#vw_resolution').val(settings.resolution).on('input', function () {
        settings.resolution = Number($(this).val()) || defaultSettings.resolution;
        saveSettingsDebounced();
    });
    $('#vw_whisper').val(settings.whisper).on('change', function () {
        settings.whisper = String($(this).val());
        saveSettingsDebounced();
    });
    $('#vw_nodedup').prop('checked', !!settings.noDedup).on('input', function () {
        settings.noDedup = !!$(this).prop('checked');
        saveSettingsDebounced();
    });
    $('#vw_show_transcript').prop('checked', !!settings.showTranscriptInChat).on('input', function () {
        settings.showTranscriptInChat = !!$(this).prop('checked');
        saveSettingsDebounced();
    });
    $('#vw_auto_reply').prop('checked', !!settings.autoReply).on('input', function () {
        settings.autoReply = !!$(this).prop('checked');
        saveSettingsDebounced();
    });
    $('#vw_auto_detect').prop('checked', !!settings.autoDetectUrls).on('input', function () {
        settings.autoDetectUrls = !!$(this).prop('checked');
        saveSettingsDebounced();
    });
    $('#vw_open_dialog').on('click', () => openWatchDialog());
    $('#vw_refresh_status').on('click', () => refreshStatusBanner());

    // Wand-menu quick button (fall back to the shared menu — we don't reserve a slot).
    const wandContainer = document.getElementById('video_watch_wand_container') || document.getElementById('extensionsMenu');
    if (wandContainer) {
        const button = $(`
        <div id="video_watch_button" class="list-group-item flex-container flexGap5" title="Watch a video">
            <div class="fa-solid fa-film extensionsMenuExtensionButton"></div>
            <span data-i18n="Watch Video">Watch Video</span>
        </div>`);
        button.on('click', () => openWatchDialog());
        $(wandContainer).append(button);
    }

    // Auto-detect: offer to watch a video URL in a freshly sent user message.
    eventSource.on(event_types.MESSAGE_SENT, (messageId) => {
        if (!getSettings().autoDetectUrls) return;
        const message = getContext().chat[messageId];
        if (!message?.is_user || message?.extra?.video_watch) return;
        if (Array.isArray(message?.extra?.media) && message.extra.media.length) return;
        const match = VIDEO_URL_REGEX.exec(String(message.mes || ''));
        if (!match) return;
        const url = match[0];
        const toast = toastr.info(`Detected a video URL. Click to watch it.`, 'Video Watch', { timeOut: 12000, onclick: () => openWatchDialog(url) });
        void toast;
    });

    // Slash command.
    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'watch',
        callback: watchCommandCallback,
        returns: 'nothing',
        namedArgumentList: [
            SlashCommandNamedArgument.fromProps({
                name: 'detail',
                description: 'frame fidelity/speed dial',
                typeList: [ARGUMENT_TYPE.STRING],
                enumList: ['transcript', 'efficient', 'balanced', 'token-burner'],
            }),
            SlashCommandNamedArgument.fromProps({ name: 'start', description: 'focus range start (SS, MM:SS, HH:MM:SS)', typeList: [ARGUMENT_TYPE.STRING] }),
            SlashCommandNamedArgument.fromProps({ name: 'end', description: 'focus range end (SS, MM:SS, HH:MM:SS)', typeList: [ARGUMENT_TYPE.STRING] }),
            SlashCommandNamedArgument.fromProps({ name: 'frames', description: 'max frames (overrides the detail cap)', typeList: [ARGUMENT_TYPE.NUMBER] }),
            SlashCommandNamedArgument.fromProps({ name: 'resolution', description: 'frame width in px (default 512)', typeList: [ARGUMENT_TYPE.NUMBER] }),
            SlashCommandNamedArgument.fromProps({ name: 'fps', description: 'override auto-fps (max 2)', typeList: [ARGUMENT_TYPE.NUMBER] }),
            SlashCommandNamedArgument.fromProps({ name: 'timestamps', description: 'comma-separated timestamps to grab a frame at', typeList: [ARGUMENT_TYPE.STRING] }),
            SlashCommandNamedArgument.fromProps({
                name: 'whisper',
                description: 'transcription backend',
                typeList: [ARGUMENT_TYPE.STRING],
                enumList: ['auto', 'groq', 'openai', 'none'],
            }),
            SlashCommandNamedArgument.fromProps({ name: 'dedup', description: 'drop near-duplicate frames (default true)', typeList: [ARGUMENT_TYPE.BOOLEAN] }),
            SlashCommandNamedArgument.fromProps({ name: 'quiet', description: 'suppress the progress toast', typeList: [ARGUMENT_TYPE.BOOLEAN] }),
        ],
        unnamedArgumentList: [
            new SlashCommandArgument('video URL or path, then an optional question', [ARGUMENT_TYPE.STRING], false),
        ],
        helpString: `
            <div>
                Watch a video and drop its frames + transcript into the chat as multimodal input.
            </div>
            <div>
                <strong>Usage:</strong> <code>/watch https://youtu.be/ID what happens at 0:30?</code>
            </div>
            <div>
                Run <code>/watch</code> with no argument to open the full dialog. Needs a vision model with media inlining enabled.
            </div>
        `,
    }));

    void refreshStatusBanner();
    document.body.classList.add('video-watch');
}
