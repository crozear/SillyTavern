import { getRequestHeaders, saveSettingsDebounced } from '../script.js';
import { extension_settings } from './extensions.js';
import { POPUP_RESULT, POPUP_TYPE, callGenericPopup } from './popup.js';
import { t } from './i18n.js';

const BASE = '/api/openai-storage';

function ensureSettings() {
    if (!extension_settings.openai_storage || typeof extension_settings.openai_storage !== 'object') {
        extension_settings.openai_storage = {};
    }
    const s = extension_settings.openai_storage;
    if (typeof s.chunking_auto !== 'boolean') s.chunking_auto = true;
    if (!Number.isFinite(s.max_chunk_size_tokens)) s.max_chunk_size_tokens = 800;
    if (!Number.isFinite(s.chunk_overlap_tokens)) s.chunk_overlap_tokens = 400;
    if (!Number.isFinite(s.file_search_max_num_results)) s.file_search_max_num_results = 20;
    if (!Number.isFinite(s.file_search_score_threshold)) s.file_search_score_threshold = 0;
    if (!s.file_search_ranker) s.file_search_ranker = 'auto';
    if (!Array.isArray(s.default_vector_store_ids)) s.default_vector_store_ids = [];
    return s;
}

async function jsonOrThrow(res) {
    if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return res.json();
}

// ---- Files API ----

export async function uploadOpenAIFile(filename, data_b64) {
    const res = await fetch(`${BASE}/files/upload`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ filename, data_b64, purpose: 'assistants' }),
    });
    return jsonOrThrow(res);
}

export async function deleteOpenAIFile(file_id) {
    const res = await fetch(`${BASE}/files/delete`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ file_id }),
    });
    return jsonOrThrow(res);
}

export async function listOpenAIFiles() {
    const res = await fetch(`${BASE}/files/list`, { headers: getRequestHeaders() });
    return jsonOrThrow(res);
}

// ---- Vector Stores API ----

export async function createVectorStore(name) {
    const res = await fetch(`${BASE}/vector-stores/create`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ name }),
    });
    return jsonOrThrow(res);
}

export async function listVectorStores() {
    const res = await fetch(`${BASE}/vector-stores/list`, { headers: getRequestHeaders() });
    return jsonOrThrow(res);
}

export async function deleteVectorStore(vector_store_id) {
    const res = await fetch(`${BASE}/vector-stores/delete`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ vector_store_id }),
    });
    return jsonOrThrow(res);
}

export async function listVectorStoreFiles(vector_store_id) {
    const res = await fetch(`${BASE}/vector-stores/${encodeURIComponent(vector_store_id)}/files`, {
        headers: getRequestHeaders(),
    });
    return jsonOrThrow(res);
}

export async function attachFilesToStore(vector_store_id, file_ids, chunking_strategy = null) {
    const body = { file_ids };
    if (chunking_strategy) body.chunking_strategy = chunking_strategy;
    const res = await fetch(`${BASE}/vector-stores/${encodeURIComponent(vector_store_id)}/files/attach`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify(body),
    });
    return jsonOrThrow(res);
}

export async function detachFileFromStore(vector_store_id, file_id) {
    const res = await fetch(`${BASE}/vector-stores/${encodeURIComponent(vector_store_id)}/files/detach`, {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ file_id }),
    });
    return jsonOrThrow(res);
}

export async function getFileBatch(vector_store_id, batch_id) {
    const res = await fetch(`${BASE}/vector-stores/${encodeURIComponent(vector_store_id)}/file-batches/${encodeURIComponent(batch_id)}`, {
        headers: getRequestHeaders(),
    });
    return jsonOrThrow(res);
}

export async function pollBatchUntilDone(vector_store_id, batch_id, { intervalMs = 1500, timeoutMs = 120000 } = {}) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        const b = await getFileBatch(vector_store_id, batch_id);
        if (['completed', 'failed', 'cancelled'].includes(b?.status)) return b;
        await new Promise(r => setTimeout(r, intervalMs));
    }
    throw new Error('Timed out waiting for vector store file batch');
}

export function buildChunkingStrategy() {
    const s = ensureSettings();
    if (s.chunking_auto) return { type: 'auto' };
    return {
        type: 'static',
        static: {
            max_chunk_size_tokens: s.max_chunk_size_tokens,
            chunk_overlap_tokens: s.chunk_overlap_tokens,
        },
    };
}

export function getDefaultVectorStoreIds() {
    const s = ensureSettings();
    return Array.isArray(s.default_vector_store_ids) ? s.default_vector_store_ids : [];
}

export function getFileSearchOptions() {
    const s = ensureSettings();
    return {
        max_num_results: s.file_search_max_num_results,
        score_threshold: s.file_search_score_threshold,
        ranker: s.file_search_ranker,
    };
}

// ---- Data Bank panel UI ----

let lastStoreList = [];

async function refreshStoreList(rootEl) {
    const select = rootEl.querySelector('#openai_vector_store_select');
    if (!select) return;
    try {
        const data = await listVectorStores();
        lastStoreList = Array.isArray(data?.data) ? data.data : [];
    } catch (e) {
        console.warn('Failed to list vector stores', e);
        lastStoreList = [];
    }
    const s = ensureSettings();
    select.innerHTML = '';
    if (!lastStoreList.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '(no vector stores)';
        select.appendChild(opt);
    }
    for (const vs of lastStoreList) {
        const opt = document.createElement('option');
        opt.value = vs.id;
        const counts = vs.file_counts || {};
        opt.textContent = `${vs.name || '(unnamed)'} — ${vs.id} (${counts.total ?? 0} files)`;
        select.appendChild(opt);
    }
    if (s.default_vector_store_ids?.length) {
        select.value = s.default_vector_store_ids[0];
    }
    updateDefaultCheckbox(rootEl);
}

function updateDefaultCheckbox(rootEl) {
    const select = rootEl.querySelector('#openai_vector_store_select');
    const cb = rootEl.querySelector('#openai_vector_store_default');
    if (!select || !cb) return;
    const s = ensureSettings();
    cb.checked = !!select.value && s.default_vector_store_ids?.includes(select.value);
}

/**
 * Wire up the OpenAI vector store panel inside the Data Bank manager.
 * Called once after the manager template is rendered.
 * @param {HTMLElement|JQuery<HTMLElement>} root
 */
export function initOpenAIStorePanel(root) {
    const rootEl = root instanceof Element ? root : (root?.[0] ?? document.body);
    const s = ensureSettings();

    // Hydrate inputs from settings
    /** @type {HTMLInputElement | null} */
    const auto = rootEl.querySelector('#openai_chunking_auto');
    /** @type {HTMLInputElement | null} */
    const maxTok = rootEl.querySelector('#openai_chunking_max_tokens');
    /** @type {HTMLInputElement | null} */
    const overlap = rootEl.querySelector('#openai_chunking_overlap');
    /** @type {HTMLInputElement | null} */
    const fsMax = rootEl.querySelector('#openai_file_search_max');
    /** @type {HTMLInputElement | null} */
    const fsThr = rootEl.querySelector('#openai_file_search_threshold');
    /** @type {HTMLSelectElement | null} */
    const fsRank = rootEl.querySelector('#openai_file_search_ranker');

    if (auto) auto.checked = !!s.chunking_auto;
    if (maxTok) maxTok.value = String(s.max_chunk_size_tokens);
    if (overlap) overlap.value = String(s.chunk_overlap_tokens);
    if (fsMax) fsMax.value = String(s.file_search_max_num_results);
    if (fsThr) fsThr.value = String(s.file_search_score_threshold);
    if (fsRank) fsRank.value = String(s.file_search_ranker);

    auto?.addEventListener('change', () => {
        s.chunking_auto = auto.checked;
        saveSettingsDebounced();
    });
    maxTok?.addEventListener('change', () => {
        const v = Number(maxTok.value);
        if (v < 100 || v > 4096) {
            toastr.warning(t`max_chunk_size_tokens must be between 100 and 4096`);
            maxTok.value = String(s.max_chunk_size_tokens);
            return;
        }
        s.max_chunk_size_tokens = v;
        if (s.chunk_overlap_tokens > v / 2) {
            s.chunk_overlap_tokens = Math.floor(v / 2);
            if (overlap) overlap.value = String(s.chunk_overlap_tokens);
        }
        saveSettingsDebounced();
    });
    overlap?.addEventListener('change', () => {
        const v = Number(overlap.value);
        const max = s.max_chunk_size_tokens;
        if (v < 0 || v > max / 2) {
            toastr.warning(t`chunk_overlap_tokens must be between 0 and max/2 (${Math.floor(max / 2)})`);
            overlap.value = String(s.chunk_overlap_tokens);
            return;
        }
        s.chunk_overlap_tokens = v;
        saveSettingsDebounced();
    });
    fsMax?.addEventListener('change', () => {
        const v = Math.max(1, Math.min(50, Number(fsMax.value)));
        s.file_search_max_num_results = v;
        fsMax.value = String(v);
        saveSettingsDebounced();
    });
    fsThr?.addEventListener('change', () => {
        const v = Math.max(0, Math.min(1, Number(fsThr.value)));
        s.file_search_score_threshold = v;
        fsThr.value = String(v);
        saveSettingsDebounced();
    });
    fsRank?.addEventListener('change', () => {
        s.file_search_ranker = String(fsRank.value);
        saveSettingsDebounced();
    });

    /** @type {HTMLSelectElement | null} */
    const select = rootEl.querySelector('#openai_vector_store_select');
    const refreshBtn = rootEl.querySelector('#openai_vector_store_refresh');
    const createBtn = rootEl.querySelector('#openai_vector_store_create');
    const deleteBtn = rootEl.querySelector('#openai_vector_store_delete');
    /** @type {HTMLInputElement | null} */
    const defaultCb = rootEl.querySelector('#openai_vector_store_default');

    select?.addEventListener('change', () => updateDefaultCheckbox(rootEl));

    refreshBtn?.addEventListener('click', () => refreshStoreList(rootEl));

    createBtn?.addEventListener('click', async () => {
        const name = await callGenericPopup(t`Vector store name:`, POPUP_TYPE.INPUT, '', { okButton: 'Create' });
        if (!name) return;
        try {
            const vs = await createVectorStore(String(name));
            toastr.success(t`Created vector store: ${vs.id}`);
            await refreshStoreList(rootEl);
            if (select) select.value = vs.id;
            updateDefaultCheckbox(rootEl);
        } catch (e) {
            toastr.error(String(e?.message || e));
        }
    });

    deleteBtn?.addEventListener('click', async () => {
        if (!select?.value) return;
        const ok = await callGenericPopup(t`Delete vector store ${select.value}? This cannot be undone.`, POPUP_TYPE.CONFIRM);
        if (ok !== POPUP_RESULT.AFFIRMATIVE) return;
        try {
            await deleteVectorStore(select.value);
            // Remove from defaults
            s.default_vector_store_ids = (s.default_vector_store_ids ?? []).filter(id => id !== select.value);
            saveSettingsDebounced();
            await refreshStoreList(rootEl);
        } catch (e) {
            toastr.error(String(e?.message || e));
        }
    });

    defaultCb?.addEventListener('change', () => {
        if (!select?.value) return;
        const ids = new Set(s.default_vector_store_ids ?? []);
        if (defaultCb.checked) ids.add(select.value);
        else ids.delete(select.value);
        s.default_vector_store_ids = [...ids];
        saveSettingsDebounced();
    });

    refreshStoreList(rootEl).catch(() => { /* swallow */ });
}
