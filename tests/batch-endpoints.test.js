import { jest } from '@jest/globals';
import { fileURLToPath } from 'url';

const calls = [];
let queue = [];

jest.unstable_mockModule('node-fetch', () => ({
    default: jest.fn(async (url, init = {}) => {
        calls.push({ url: String(url), method: init.method ?? 'GET', headers: init.headers ?? {}, body: init.body });
        const next = queue.shift() ?? { status: 200, body: {} };
        return {
            ok: next.status < 400,
            status: next.status,
            headers: { get: () => null },
            json: async () => next.body,
            text: async () => typeof next.body === 'string' ? next.body : JSON.stringify(next.body),
        };
    }),
}));

const util = await import('../src/util.js');
// Resolved from this file, not the cwd: jest runs from tests/ but a bare
// `npx jest --rootDir .` runs from the repo root, and util exits the process
// outright when the config path doesn't resolve.
util.setConfigFilePath(fileURLToPath(new URL('../default/config.yaml', import.meta.url)));

const { router } = await import('../src/endpoints/backends/chat-completions.js');

/** Finds a mounted route handler by path + method. */
function handlerFor(path, method) {
    for (const layer of router.stack) {
        if (!layer.route) continue;
        const paths = Array.isArray(layer.route.path) ? layer.route.path : [layer.route.path];
        if (paths.includes(path) && layer.route.methods[method]) {
            return layer.route.stack[layer.route.stack.length - 1].handle;
        }
    }
    throw new Error(`No handler for ${method.toUpperCase()} ${path}`);
}

function mockResponse() {
    const res = {
        statusCode: 200,
        payload: undefined,
        status(code) { res.statusCode = code; return res; },
        send(payload) { res.payload = payload; return res; },
        setHeader() { return res; },
        headersSent: false,
    };
    return res;
}

const os = await import('os');
const USER = { directories: { root: (await import('path')).join(os.tmpdir(), 'st-batch-endpoint-test') } };

function makeRequest(body) {
    return { body, user: USER, socket: { removeAllListeners() {}, on() {} } };
}

beforeEach(async () => {
    calls.length = 0;
    queue = [];
    const fs = await import('fs');
    const path = await import('path');
    fs.rmSync(USER.directories.root, { recursive: true, force: true });
    fs.mkdirSync(USER.directories.root, { recursive: true });
    // The OpenRouter key is a server-side secret, so give this user one.
    fs.writeFileSync(
        path.join(USER.directories.root, 'secrets.json'),
        JSON.stringify({ api_key_openrouter: [{ id: 'test', label: 'test', value: 'sk-or-v1-testkey', active: true }] }),
        'utf-8',
    );
});

describe('OpenRouter batch submit', () => {
    const baseBody = {
        chat_completion_source: 'openrouter',
        batch_provider: 'openrouter',
        model: 'anthropic/claude-sonnet-4.5',
        messages: [{ role: 'user', content: 'Say hello.' }],
        temperature: 1,
        max_tokens: 300,
        stream: false,
        include_reasoning: true,
        middleout: 'on',
    };

    test('posts to /api/beta/batches with endpoint and model serialized before requests', async () => {
        queue.push({ status: 200, body: { id: 'batch_abc', status: 'validating' } });

        const res = mockResponse();
        await handlerFor('/batch/submit', 'post')(makeRequest({ ...baseBody }), res);

        expect(res.statusCode).toBe(200);
        expect(res.payload.batchId).toBe('batch_abc');
        expect(res.payload.provider).toBe('openrouter');

        const call = calls[0];
        expect(call.url).toBe('https://openrouter.ai/api/beta/batches');
        expect(call.method).toBe('POST');
        expect(String(call.headers.Authorization ?? '')).toMatch(/^Bearer /);

        // OpenRouter stream-parses the body and 400s if `requests` comes first.
        const keys = Object.keys(JSON.parse(call.body));
        expect(keys).toEqual(['endpoint', 'model', 'requests']);

        const payload = JSON.parse(call.body);
        expect(payload.endpoint).toBe('/v1/chat/completions');
        expect(payload.model).toBe('anthropic/claude-sonnet-4.5');
        expect(payload.requests).toHaveLength(1);
        expect(payload.requests[0].custom_id).toMatch(/^st-/);
        // The batch-level model governs; repeating it in the body risks a mismatch.
        expect(payload.requests[0].body.model).toBeUndefined();
        expect(payload.requests[0].body.stream).toBeUndefined();
        expect(payload.requests[0].body.messages).toEqual([{ role: 'user', content: 'Say hello.' }]);
        expect(payload.requests[0].body.max_tokens).toBe(300);
        expect(payload.requests[0].body.transforms).toEqual(['middle-out']);
        expect(payload.requests[0].body.reasoning).toEqual({ exclude: false });
    });

    test('refuses a prompt with non-text content instead of letting validation reject it', async () => {
        const res = mockResponse();
        await handlerFor('/batch/submit', 'post')(makeRequest({
            ...baseBody,
            messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }, { type: 'image_url', image_url: { url: 'data:...' } }] }],
        }), res);

        expect(res.statusCode).toBe(409);
        expect(res.payload.ineligible).toBe(true);
        expect(res.payload.reason).toMatch(/text-only/);
        expect(calls).toHaveLength(0); // nothing was sent
    });
});

describe('OpenRouter batch status', () => {
    test('normalizes in-flight statuses to pending', async () => {
        queue.push({ status: 200, body: { id: 'batch_abc', status: 'in_progress', request_counts: { total: 1, completed: 0, failed: 0 } } });
        const res = mockResponse();
        await handlerFor('/batch/status', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc' }), res);

        expect(calls[0].url).toBe('https://openrouter.ai/api/beta/batches/batch_abc');
        expect(res.payload.state).toBe('pending');
        expect(res.payload.providerStatus).toBe('in_progress');
    });

    test('normalizes every terminal status to ended', async () => {
        const states = {};
        for (const status of ['completed', 'failed', 'expired', 'cancelled']) {
            queue.push({ status: 200, body: { id: 'batch_abc', status } });
            const res = mockResponse();
            await handlerFor('/batch/status', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc' }), res);
            states[status] = res.payload.state;
        }
        expect(states).toEqual({ completed: 'ended', failed: 'ended', expired: 'ended', cancelled: 'ended' });
    });
});

describe('OpenRouter batch result', () => {
    const completed = (choices, extra = {}) => ({
        id: 'batch_abc',
        status: 'completed',
        usage: { prompt_tokens: 20, completion_tokens: 40, cost: 0.000225, is_byok: false },
        results: [{
            id: 'batch_req_1',
            custom_id: 'st-xyz',
            response: { status_code: 200, body: { id: 'gen-batch-1', model: 'anthropic/claude-sonnet-4.5', choices, usage: { prompt_tokens: 20, completion_tokens: 40 } } },
            error: null,
            ...extra,
        }],
    });

    test('returns the chat completion body for a successful result', async () => {
        queue.push({ status: 200, body: completed([{ index: 0, message: { role: 'assistant', content: 'Hello there.', reasoning: 'thinking...' }, finish_reason: 'stop' }]) });

        const res = mockResponse();
        await handlerFor('/batch/result', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc', customId: 'st-xyz' }), res);

        expect(res.payload.resultType).toBe('succeeded');
        expect(res.payload.reply.choices[0].message.content).toBe('Hello there.');
        expect(res.payload.reply.choices[0].message.reasoning).toBe('thinking...');
    });

    test('reports a filtered response as refused', async () => {
        queue.push({ status: 200, body: completed([{ index: 0, message: { role: 'assistant', content: '' }, finish_reason: 'content_filter' }]) });

        const res = mockResponse();
        await handlerFor('/batch/result', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc', customId: 'st-xyz' }), res);

        expect(res.payload.resultType).toBe('refused');
        expect(res.payload.error.message).toMatch(/content_filter/);
    });

    test('reports an empty reply as refused rather than delivering nothing', async () => {
        queue.push({ status: 200, body: completed([{ index: 0, message: { role: 'assistant', content: '' }, finish_reason: 'length' }]) });

        const res = mockResponse();
        await handlerFor('/batch/result', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc', customId: 'st-xyz' }), res);

        expect(res.payload.resultType).toBe('refused');
        expect(res.payload.error.message).toMatch(/empty reply/);
    });

    test('surfaces a non-completed batch as an error', async () => {
        queue.push({ status: 200, body: { id: 'batch_abc', status: 'expired', results: null, error: { message: 'Batch expired before completion.' } } });

        const res = mockResponse();
        await handlerFor('/batch/result', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc', customId: 'st-xyz' }), res);

        expect(res.payload.resultType).toBe('errored');
        expect(res.payload.error.message).toMatch(/expired/);
    });

    test('surfaces a per-request error', async () => {
        queue.push({ status: 200, body: {
            id: 'batch_abc', status: 'completed',
            results: [{ id: 'r1', custom_id: 'st-xyz', response: null, error: { message: 'context length exceeded' } }],
        } });

        const res = mockResponse();
        await handlerFor('/batch/result', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc', customId: 'st-xyz' }), res);

        expect(res.payload.resultType).toBe('errored');
        expect(res.payload.error.message).toMatch(/context length/);
    });
});

describe('provider dispatch', () => {
    test('routes to Claude by default and to OpenRouter when the job says so', async () => {
        queue.push({ status: 200, body: { processing_status: 'in_progress' } });
        const claudeRes = mockResponse();
        await handlerFor('/batch/status', 'post')(makeRequest({ batchId: 'msgbatch_1', reverse_proxy: 'https://proxy.example/v1' }), claudeRes);
        expect(calls[0].url).toBe('https://proxy.example/v1/messages/batches/msgbatch_1');
        expect(claudeRes.payload.state).toBe('pending');

        calls.length = 0;
        queue.push({ status: 200, body: { status: 'in_progress' } });
        const orRes = mockResponse();
        await handlerFor('/batch/status', 'post')(makeRequest({ batch_provider: 'openrouter', batchId: 'batch_abc' }), orRes);
        expect(calls[0].url).toBe('https://openrouter.ai/api/beta/batches/batch_abc');
    });

    test('cancels through the provider the persisted job belongs to', async () => {
        // Submit an OpenRouter job so it lands in the store, then cancel it by jobId
        // alone — the stored provider has to be what decides where the cancel goes.
        queue.push({ status: 200, body: { id: 'batch_abc', status: 'validating' } });
        const submitRes = mockResponse();
        await handlerFor('/batch/submit', 'post')(makeRequest({
            chat_completion_source: 'openrouter',
            batch_provider: 'openrouter',
            model: 'openai/gpt-4o',
            messages: [{ role: 'user', content: 'hi' }],
        }), submitRes);
        const jobId = submitRes.payload.jobId;

        calls.length = 0;
        queue.push({ status: 200, body: { id: 'batch_abc', status: 'cancelling' } });
        const cancelRes = mockResponse();
        await handlerFor('/batch/cancel', 'post')(makeRequest({ jobId, batchId: 'batch_abc' }), cancelRes);

        expect(calls[0].url).toBe('https://openrouter.ai/api/beta/batches/batch_abc/cancel');
        expect(calls[0].method).toBe('POST');

        // Cancel drops the persisted job.
        const listRes = mockResponse();
        await handlerFor('/batch/list', 'get')(makeRequest({}), listRes);
        expect(listRes.payload.jobs).toHaveLength(0);
    });

    test('reports per-provider settings on list', async () => {
        const res = mockResponse();
        await handlerFor('/batch/list', 'get')(makeRequest({}), res);
        expect(res.payload.settings.claude.batchEnabled).toBe(true);
        expect(res.payload.settings.openrouter.batchEnabled).toBe(true);
        expect(res.payload.settings.openrouter.pollIntervalMs).toBe(15000);
    });
});

describe('Claude batch (regression — shares the generalized routes)', () => {
    const claudeBody = {
        chat_completion_source: 'claude',
        model: 'claude-fable-5',
        messages: [{ role: 'user', content: 'Say hello.' }],
        max_tokens: 300,
        reverse_proxy: 'https://proxy.example/v1',
        proxy_password: 'sk-ant-test',
    };

    test('submits a batch of one to the Messages Batches API', async () => {
        queue.push({ status: 200, body: { id: 'msgbatch_1', processing_status: 'in_progress' } });

        const res = mockResponse();
        await handlerFor('/batch/submit', 'post')(makeRequest({ ...claudeBody }), res);

        expect(res.statusCode).toBe(200);
        expect(res.payload.provider).toBe('claude');
        expect(calls[0].url).toBe('https://proxy.example/v1/messages/batches');

        const payload = JSON.parse(calls[0].body);
        expect(payload.requests).toHaveLength(1);
        expect(payload.requests[0].custom_id).toMatch(/^st-/);
        expect(payload.requests[0].params.model).toBe('claude-fable-5');
        expect(payload.requests[0].params.stream).toBeUndefined();
    });

    test('refuses without an sk-ant key rather than billing a sync request', async () => {
        const res = mockResponse();
        await handlerFor('/batch/submit', 'post')(makeRequest({ ...claudeBody, proxy_password: 'oauth-token' }), res);

        expect(res.statusCode).toBe(409);
        expect(res.payload.reason).toMatch(/sk-ant/);
        expect(calls).toHaveLength(0);
    });

    test('refuses a model that is not batch-only', async () => {
        const res = mockResponse();
        await handlerFor('/batch/submit', 'post')(makeRequest({ ...claudeBody, model: 'claude-sonnet-4-6' }), res);

        expect(res.statusCode).toBe(409);
        expect(res.payload.reason).toMatch(/not enabled/);
        expect(calls).toHaveLength(0);
    });

    test('parses the JSONL result into the hybrid reply shape', async () => {
        const line = JSON.stringify({
            custom_id: 'st-xyz',
            result: {
                type: 'succeeded',
                message: {
                    id: 'msg_1', model: 'claude-fable-5', stop_reason: 'end_turn',
                    content: [{ type: 'thinking', thinking: 'hmm' }, { type: 'text', text: 'Hello there.' }],
                    usage: { input_tokens: 20, output_tokens: 40 },
                },
            },
        });
        queue.push({ status: 200, body: line });

        const res = mockResponse();
        await handlerFor('/batch/result', 'post')(makeRequest({
            batchId: 'msgbatch_1', customId: 'st-xyz', reverse_proxy: 'https://proxy.example/v1', proxy_password: 'sk-ant-test',
        }), res);

        expect(calls[0].url).toBe('https://proxy.example/v1/messages/batches/msgbatch_1/results');
        expect(res.payload.resultType).toBe('succeeded');
        expect(res.payload.reply.choices[0].message.content).toBe('Hello there.');
        expect(res.payload.reply.content).toHaveLength(2);
    });

    test('accepts the pre-rename /claude-batch/* alias', async () => {
        queue.push({ status: 200, body: { processing_status: 'ended' } });
        const res = mockResponse();
        await handlerFor('/claude-batch/status', 'post')(makeRequest({
            batchId: 'msgbatch_1', reverse_proxy: 'https://proxy.example/v1', proxy_password: 'sk-ant-test',
        }), res);
        expect(res.payload.state).toBe('ended');
    });
});
