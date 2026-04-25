import { Buffer } from 'node:buffer';

import fetch from 'node-fetch';
import FormData from 'form-data';
import express from 'express';

import { readSecret, SECRET_KEYS } from './secrets.js';

export const router = express.Router();

const OPENAI_BASE = 'https://api.openai.com/v1';

function getKey(req) {
    return readSecret(req.user.directories, SECRET_KEYS.OPENAI);
}

function authHeaders(key, extra = {}) {
    return {
        'Authorization': `Bearer ${key}`,
        'OpenAI-Beta': 'assistants=v2',
        ...extra,
    };
}

async function passthrough(res, openaiRes) {
    const text = await openaiRes.text();
    let body;
    try { body = JSON.parse(text); } catch { body = text; }
    return res.status(openaiRes.status).send(body);
}

function validateChunkingStrategy(strategy) {
    if (!strategy || strategy.type === 'auto') return { ok: true };
    if (strategy.type !== 'static') return { ok: false, message: 'chunking_strategy.type must be "auto" or "static"' };
    const s = strategy.static;
    if (!s) return { ok: false, message: 'chunking_strategy.static is required when type=static' };
    const max = Number(s.max_chunk_size_tokens);
    const overlap = Number(s.chunk_overlap_tokens);
    if (!Number.isFinite(max) || max < 100 || max > 4096) {
        return { ok: false, message: 'max_chunk_size_tokens must be between 100 and 4096' };
    }
    if (!Number.isFinite(overlap) || overlap < 0 || overlap > max / 2) {
        return { ok: false, message: 'chunk_overlap_tokens must be >= 0 and <= max_chunk_size_tokens / 2' };
    }
    return { ok: true };
}

// ---- Files ----

router.post('/files/upload', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');

        const { filename, data_b64, purpose } = request.body || {};
        if (!filename || !data_b64) return response.status(400).send('filename and data_b64 are required');

        const buffer = Buffer.from(data_b64, 'base64');
        const form = new FormData();
        form.append('purpose', purpose || 'assistants');
        form.append('file', buffer, { filename });

        const r = await fetch(`${OPENAI_BASE}/files`, {
            method: 'POST',
            headers: { ...form.getHeaders(), ...authHeaders(key) },
            body: form,
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /files/upload error:', error);
        return response.sendStatus(500);
    }
});

router.post('/files/delete', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const { file_id } = request.body || {};
        if (!file_id) return response.status(400).send('file_id is required');

        const r = await fetch(`${OPENAI_BASE}/files/${encodeURIComponent(file_id)}`, {
            method: 'DELETE',
            headers: authHeaders(key),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /files/delete error:', error);
        return response.sendStatus(500);
    }
});

router.get('/files/list', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const purpose = request.query.purpose || 'assistants';
        const r = await fetch(`${OPENAI_BASE}/files?purpose=${encodeURIComponent(String(purpose))}`, {
            headers: authHeaders(key),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /files/list error:', error);
        return response.sendStatus(500);
    }
});

// ---- Vector stores ----

router.post('/vector-stores/create', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const { name, expires_after, metadata } = request.body || {};
        const body = {};
        if (name) body.name = name;
        if (expires_after) body.expires_after = expires_after;
        if (metadata) body.metadata = metadata;

        const r = await fetch(`${OPENAI_BASE}/vector_stores`, {
            method: 'POST',
            headers: authHeaders(key, { 'Content-Type': 'application/json' }),
            body: JSON.stringify(body),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /vector-stores/create error:', error);
        return response.sendStatus(500);
    }
});

router.get('/vector-stores/list', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const r = await fetch(`${OPENAI_BASE}/vector_stores?limit=100`, {
            headers: authHeaders(key),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /vector-stores/list error:', error);
        return response.sendStatus(500);
    }
});

router.post('/vector-stores/delete', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const { vector_store_id } = request.body || {};
        if (!vector_store_id) return response.status(400).send('vector_store_id is required');

        const r = await fetch(`${OPENAI_BASE}/vector_stores/${encodeURIComponent(vector_store_id)}`, {
            method: 'DELETE',
            headers: authHeaders(key),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /vector-stores/delete error:', error);
        return response.sendStatus(500);
    }
});

router.get('/vector-stores/:id/files', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const id = encodeURIComponent(request.params.id);
        const r = await fetch(`${OPENAI_BASE}/vector_stores/${id}/files?limit=100`, {
            headers: authHeaders(key),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /vector-stores/:id/files error:', error);
        return response.sendStatus(500);
    }
});

router.post('/vector-stores/:id/files/attach', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const id = encodeURIComponent(request.params.id);
        const { file_ids, chunking_strategy, attributes } = request.body || {};
        if (!Array.isArray(file_ids) || !file_ids.length) return response.status(400).send('file_ids[] required');

        const v = validateChunkingStrategy(chunking_strategy);
        if (!v.ok) return response.status(400).send(v.message);

        const body = { file_ids };
        if (chunking_strategy) body.chunking_strategy = chunking_strategy;
        if (attributes) body.attributes = attributes;

        const r = await fetch(`${OPENAI_BASE}/vector_stores/${id}/file_batches`, {
            method: 'POST',
            headers: authHeaders(key, { 'Content-Type': 'application/json' }),
            body: JSON.stringify(body),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /vector-stores/:id/files/attach error:', error);
        return response.sendStatus(500);
    }
});

router.post('/vector-stores/:id/files/detach', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const id = encodeURIComponent(request.params.id);
        const { file_id } = request.body || {};
        if (!file_id) return response.status(400).send('file_id is required');

        const r = await fetch(`${OPENAI_BASE}/vector_stores/${id}/files/${encodeURIComponent(file_id)}`, {
            method: 'DELETE',
            headers: authHeaders(key),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /vector-stores/:id/files/detach error:', error);
        return response.sendStatus(500);
    }
});

router.get('/vector-stores/:id/file-batches/:batch_id', async (request, response) => {
    try {
        const key = getKey(request);
        if (!key) return response.status(400).send('No OpenAI API key configured');
        const id = encodeURIComponent(request.params.id);
        const batchId = encodeURIComponent(request.params.batch_id);
        const r = await fetch(`${OPENAI_BASE}/vector_stores/${id}/file_batches/${batchId}`, {
            headers: authHeaders(key),
        });
        return passthrough(response, r);
    } catch (error) {
        console.error('[openai-storage] /vector-stores/:id/file-batches/:batch_id error:', error);
        return response.sendStatus(500);
    }
});
