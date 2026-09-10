// server.js - OpenAI to OpenRouter Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// OpenRouter API configuration
const OR_BASE = 'https://openrouter.ai/api/v1';
const OR_KEY  = process.env.OPENROUTER_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false; // Set to true to show <think> tags in output

// Model mapping
// Free models (no credits needed): deepseek/deepseek-v4-flash:free — 200 req/day
// Paid models draw from your credit balance
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek/deepseek-v4-flash:free',    // Free — use for casual chats
  'gpt-4':         'deepseek/deepseek-v4-flash-0731',     // $0.05/M input — cheapest paid
  'gpt-4-turbo':   'deepseek/deepseek-v4-flash',          // $0.07/M input — #1 RP model
  'gpt-4o':        'deepseek/deepseek-v4-flash-0731',     // $0.05/M input — great value
  'claude-3-opus': 'deepseek/deepseek-v4-pro',            // Premium — 1.65T params
  'claude-3-sonnet':'deepseek/deepseek-r1',                // Reasoning model
  'gemini-pro':    'deepseek/deepseek-v4-flash:free',     // Free fallback
  'minimax':       'deepseek/deepseek-v4-flash:free'      // Free fallback
};

// Trim old messages — keeps system prompt + recent history
const trimMessages = (messages, maxTokens = 16000) => {
  const estimate = msgs => msgs.reduce((sum, m) => sum + Math.ceil((m.content || '').length / 4), 0);
  if (estimate(messages) <= maxTokens) return messages;
  const system = messages.filter(m => m.role === 'system');
  const rest   = messages.filter(m => m.role !== 'system');
  while (rest.length > 1 && estimate([...system, ...rest]) > maxTokens) rest.shift();
  console.warn(`Trimmed to ~${estimate([...system, ...rest])} tokens`);
  return [...system, ...rest];
};

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to OpenRouter Proxy', reasoning: SHOW_REASONING });
});

// Test all mapped models
app.get('/test-models', async (req, res) => {
  const results = {};
  for (const [alias, orModel] of Object.entries(MODEL_MAPPING)) {
    try {
      const r = await axios.post(`${OR_BASE}/chat/completions`, {
        model: orModel,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        stream: false
      }, {
        headers: {
          'Authorization': `Bearer ${OR_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://janitor-proxy.onrender.com',
          'X-Title': 'Janitor AI Proxy'
        },
        timeout: 20000,
        validateStatus: () => true
      });
      results[alias] = { model: orModel, status: r.status, ok: r.status < 400 };
    } catch (err) {
      results[alias] = { model: orModel, status: 'timeout', ok: false };
    }
  }
  res.json(results);
});

// Models list
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id, object: 'model', created: 1700000000, owned_by: 'openrouter-proxy'
    }))
  });
});

// Individual model lookup
app.get('/v1/models/:modelId', (req, res) => {
  res.json({ id: req.params.modelId, object: 'model', created: 1700000000, owned_by: 'openrouter-proxy' });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream,
            frequency_penalty, presence_penalty, top_p } = req.body;

    // Model selection with fallback
    let orModel = MODEL_MAPPING[model];
    if (!orModel) {
      const m = model.toLowerCase();
      orModel = (m.includes('gpt-4') || m.includes('opus') || m.includes('large'))
        ? 'deepseek/deepseek-v4-flash-0731'
        : 'deepseek/deepseek-v4-flash:free';
    }

    const useStream = true;

    const orRequest = {
      model: orModel,
      messages: trimMessages(messages),
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 2048,
      stream: useStream
    };

    // Only forward params that are explicitly set by the client
    if (frequency_penalty != null) orRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty  != null) orRequest.presence_penalty  = presence_penalty;
    if (top_p             != null) orRequest.top_p             = top_p;

    // Retry with exponential backoff for 429 and 503
    const orFetch = async (retries = 6, delay = 3000) => {
      for (let i = 0; i <= retries; i++) {
        try {
          return await axios.post(`${OR_BASE}/chat/completions`, orRequest, {
            headers: {
              'Authorization': `Bearer ${OR_KEY}`,
              'Content-Type': 'application/json',
              'HTTP-Referer': 'https://janitor-proxy.onrender.com',
              'X-Title': 'Janitor AI Proxy'
            },
            responseType: 'stream',
            timeout: 0
          });
        } catch (err) {
          const status = err.response?.status;
          if ((status === 429 || status === 503) && i < retries) {
            const retryAfter = parseInt(err.response?.headers?.['retry-after'] || 0) * 1000;
            const wait = retryAfter || delay * Math.pow(2, i);
            console.warn(`${status} — retrying in ${wait}ms (attempt ${i + 1}/${retries})`);
            await new Promise(r => setTimeout(r, wait));
          } else throw err;
        }
      }
    };

    const response = await orFetch();

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '', reasoningStarted = false, charCount = 0, streamDone = false;
      const MAX_CHARS = 4000; // ~1000 tokens — cuts off before any runaway builds

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) { if (!res.writableEnded) res.write(line + '\n'); return; }
          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content;
              const content   = data.choices[0].delta.content;
              if (SHOW_REASONING) {
                let combined = '';
                if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
                else if (reasoning) { combined = reasoning; }
                if (content && reasoningStarted) { combined += '</think>\n\n' + content; reasoningStarted = false; }
                else if (content) { combined += content; }
                if (combined) { data.choices[0].delta.content = combined; delete data.choices[0].delta.reasoning_content; }
              } else {
                data.choices[0].delta.content = content || '';
                delete data.choices[0].delta.reasoning_content;
              }
            }
            if (streamDone) return;
            if (!res.writableEnded) res.write(`data: ${JSON.stringify(data)}\n\n`);
            charCount += (data.choices?.[0]?.delta?.content || '').length;
            charCount += (data.choices?.[0]?.delta?.reasoning_content || '').length;
            if (charCount > MAX_CHARS) {
              streamDone = true;
              if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
              response.data.destroy();
            }
          } catch (e) { if (!res.writableEnded) res.write(line + '\n'); }
        });
      });

      response.data.on('end',  () => { if (!res.writableEnded) res.end(); });
      response.data.on('error', (err) => { console.error('Stream error:', err); if (!res.writableEnded) res.end(); });

    } else {
      let buffer = '', fullContent = '', fullReasoning = '', finishReason = '';

      await new Promise((resolve, reject) => {
        response.data.on('data', (chunk) => { buffer += chunk.toString(); });
        response.data.on('end', () => {
          buffer.split('\n').forEach(line => {
            if (!line.startsWith('data: ') || line.includes('[DONE]')) return;
            try {
              const data = JSON.parse(line.slice(6));
              fullContent   += data.choices?.[0]?.delta?.content           || '';
              fullReasoning += data.choices?.[0]?.delta?.reasoning_content || '';
              if (data.choices?.[0]?.finish_reason) finishReason = data.choices[0].finish_reason;
            } catch (e) {}
          });
          resolve();
        });
        response.data.on('error', reject);
      });

      if (SHOW_REASONING && fullReasoning) fullContent = '<think>\n' + fullReasoning + '\n</think>\n\n' + fullContent;

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: fullContent }, finish_reason: finishReason }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    if (res.headersSent) {
      if (!res.writableEnded) res.end();
    } else {
      res.status(error.response?.status || 500).json({
        error: { message: error.message || 'Internal server error', type: 'invalid_request_error', code: error.response?.status || 500 }
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenRouter Proxy running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
});