const express = require('express');
const axios   = require('axios');
const cors    = require('cors');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ─── Config ───────────────────────────────────────────────────────────────────
const PORT               = process.env.PORT || 3000;
const NIM_API_BASE       = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY        = process.env.NIM_API_KEY;
const ENABLE_THINKING    = process.env.ENABLE_THINKING_MODE === 'true';
const SHOW_REASONING     = process.env.SHOW_REASONING !== 'false';

// ─── Model Mapping ────────────────────────────────────────────────────────────
// Janitor AI / WyvernChat model name  →  NVIDIA NIM model string
const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'nvidia/llama-3.1-nemotron-ultra-253b-v1', // ✅ confirmed
  'gpt-4':          'qwen/qwen3-coder-480b-a35b-instruct',      // ✅ confirmed
  'gpt-4-turbo':    'moonshotai/kimi-k2-instruct',              // ✅ confirmed (streaming-safe)
  'gpt-4o':         'deepseek-ai/deepseek-v4-flash',            // ✅ confirmed (v3.1 doesn't exist)
  'claude-3-opus':  'openai/gpt-oss-120b',                      // ✅ confirmed
  'claude-3-sonnet':'openai/gpt-oss-20b',                       // ✅ confirmed
  'gemini-pro':     'deepseek-ai/deepseek-v4-pro',              // ✅ confirmed
  'minimax':        'minimaxai/minimax-m2.7'                    // ✅ confirmed
};
const DEFAULT_MODEL = 'nvidia/llama-3.1-nemotron-ultra-253b-v1'; // fallback

// ─── /v1/models ───────────────────────────────────────────────────────────────
app.get('/v1/models', (_req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim'
    }))
  });
});

// ─── /v1/chat/completions ─────────────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: { message: 'NIM_API_KEY is not set', type: 'server_error', code: 500 }
      });
    }

    const {
      model, messages, temperature, max_tokens, stream,
      frequency_penalty, presence_penalty, top_p, repetition_penalty
    } = req.body;

    // Resolve model — log incoming name so you can see what clients send
    console.log(`[PROXY] Incoming model: "${model}"`);
    const nimModel = MODEL_MAPPING[model] || (() => {
      console.warn(`[PROXY] Unknown model "${model}" — falling back to Nemotron 253B`);
      return DEFAULT_MODEL;
    })();
    console.log(`[PROXY] Routing to: "${nimModel}"`);

    // Build NIM request — always stream to keep Render connection alive (prevents 504)
    const nimRequest = {
      model:             nimModel,
      messages,
      temperature:       temperature       ?? 0.6,
      max_tokens:        max_tokens        ?? 9024,
      frequency_penalty: frequency_penalty ?? 0.4,
      presence_penalty:  presence_penalty  ?? 0.4,
      top_p:             top_p             ?? 0.9,
      stream:            true,
      ...(repetition_penalty || ENABLE_THINKING ? {
        extra_body: {
          ...(repetition_penalty ? { repetition_penalty } : {}),
          ...(ENABLE_THINKING    ? { chat_template_kwargs: { thinking: true } } : {})
        }
      } : {})
    };

    // ── Retry helper (429 + 504 with exponential backoff) ──────────────────────
    const nimFetch = async (payload, retries = 6, delay = 3000) => {
      for (let i = 0; i <= retries; i++) {
        try {
          return await axios.post(`${NIM_API_BASE}/chat/completions`, payload, {
            headers: {
              'Authorization': `Bearer ${NIM_API_KEY}`,
              'Content-Type':  'application/json'
            },
            responseType: 'stream',
            timeout: 0   // no axios timeout — streaming keeps Render alive
          });
        } catch (err) {
          const status = err.response?.status;
          if ((status === 429 || status === 504) && i < retries) {
            const retryAfter = parseInt(err.response?.headers?.['retry-after'] || 0) * 1000;
            const wait = retryAfter || delay * Math.pow(2, i);
            console.warn(`[PROXY] ${status} — retrying in ${wait}ms (attempt ${i + 1}/${retries})`);
            await new Promise(r => setTimeout(r, wait));
          } else {
            throw err;
          }
        }
      }
    };

    const response = await nimFetch(nimRequest);

    // ── Stream handler ─────────────────────────────────────────────────────────
    const processLine = (line, reasoningStarted) => {
      if (!line.startsWith('data: ') || line.includes('[DONE]')) return { line, reasoningStarted };
      try {
        const data = JSON.parse(line.slice(6));
        if (data.choices?.[0]?.delta) {
          const reasoning = data.choices[0].delta.reasoning_content;
          const content   = data.choices[0].delta.content;
          let combined = '';
          if (SHOW_REASONING) {
            if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
            else if (reasoning)                 { combined = reasoning; }
            if (content && reasoningStarted)    { combined += '</think>\n\n' + content; reasoningStarted = false; }
            else if (content)                   { combined += content; }
          } else {
            combined = content || '';
          }
          data.choices[0].delta.content = combined;
          delete data.choices[0].delta.reasoning_content;
        }
        return { line: `data: ${JSON.stringify(data)}\n\n`, reasoningStarted };
      } catch { return { line: line + '\n', reasoningStarted }; }
    };

    if (stream) {
      // ── Streaming client (WyvernChat etc.) ──────────────────────────────────
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');

      let buf = '', reasoningStarted = false;
      response.data.on('data', chunk => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        lines.forEach(line => {
          const result = processLine(line, reasoningStarted);
          reasoningStarted = result.reasoningStarted;
          res.write(result.line);
        });
      });
      response.data.on('end',   ()    => res.end());
      response.data.on('error', err   => { console.error('[PROXY] Stream error:', err); res.end(); });

    } else {
      // ── Non-streaming client (JanitorAI etc.) — collect stream → JSON ───────
      let buf = '', fullContent = '', fullReasoning = '', finishReason = '';
      let promptTokens = 0, completionTokens = 0;

      await new Promise((resolve, reject) => {
        response.data.on('data',  chunk => { buf += chunk.toString(); });
        response.data.on('end',   ()    => {
          buf.split('\n').forEach(line => {
            if (!line.startsWith('data: ') || line.includes('[DONE]')) return;
            try {
              const data = JSON.parse(line.slice(6));
              fullContent   += data.choices?.[0]?.delta?.content           || '';
              fullReasoning += data.choices?.[0]?.delta?.reasoning_content || '';
              if (data.choices?.[0]?.finish_reason) finishReason = data.choices[0].finish_reason;
              if (data.usage) {
                promptTokens     = data.usage.prompt_tokens     || 0;
                completionTokens = data.usage.completion_tokens || 0;
              }
            } catch {}
          });
          resolve();
        });
        response.data.on('error', reject);
      });

      if (SHOW_REASONING && fullReasoning)
        fullContent = '<think>\n' + fullReasoning + '\n</think>\n\n' + fullContent;

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: fullContent }, finish_reason: finishReason }],
        usage:   { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens }
      });
    }

  } catch (err) {
    const status  = err.response?.status || 500;
    const message = err.message || 'Unknown proxy error';
    console.error(`[PROXY ERROR ${status}]:`, message);
    res.status(status).json({
      error: { message: `Request failed with status code ${status}`, type: 'invalid_request_error', code: status }
    });
  }
});

app.listen(PORT, () => console.log(`[PROXY] NIM Proxy running on port ${PORT}`));

/*
  MODEL MAPPING TABLE
  ┌─────────────────┬──────────────────────────────────────────────┐
  │ Client model    │ NIM model                                    │
  ├─────────────────┼──────────────────────────────────────────────┤
  │ gpt-3.5-turbo   │ nvidia/llama-3.1-nemotron-ultra-253b-v1      │
  │ gpt-4           │ qwen/qwen3-coder-480b-a35b-instruct          │
  │ gpt-4-turbo     │ moonshotai/kimi-k2.6                         │
  │ gpt-4o          │ deepseek-ai/deepseek-v3.1                    │
  │ claude-3-opus   │ nvidia/llama-3.3-nemotron-super-49b-v1       │
  │ claude-3-sonnet │ nvidia/mistral-nemo-minitron-8b-8k-instruct  │
  │ gemini-pro      │ deepseek-ai/deepseek-v4-pro                  │
  │ minimax         │ minimaxai/minimax-m2.7                        │
  └─────────────────┴──────────────────────────────────────────────┘

  RENDER ENV VARS NEEDED
  NIM_API_KEY         = your NVIDIA NIM API key
  ENABLE_THINKING_MODE= true  (optional, for reasoning models)
  SHOW_REASONING      = false (optional, hide <think> blocks)
*/