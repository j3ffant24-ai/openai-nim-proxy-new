// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': deepseek-ai/deepseek-v4-pro-0813',
  'gpt-4':         'deepseek-ai/deepseek-v4-flash-0731',
  'gpt-4-turbo':   'moonshotai/kimi-k3',
  'gpt-4o':        'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-opus': 'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-sonnet':'nvidia/nemotron-3-super-120b-a12b',
  'gemini-pro':    'nvidia/nemotron-3-super-120b-a12b',
  'minimax':       'nvidia/nemotron-3-super-120b-a12b'
};

// Trim old messages to avoid NIM 413 — keeps system prompt + recent history
const trimMessages = (messages, maxTokens = 24000) => {
  const estimate = msgs => msgs.reduce((sum, m) => sum + Math.ceil((m.content || '').length / 4), 0);
  if (estimate(messages) <= maxTokens) return messages;
  const system = messages.filter(m => m.role === 'system');
  const rest   = messages.filter(m => m.role !== 'system');
  while (rest.length > 1 && estimate([...system, ...rest]) > maxTokens) rest.shift();
  console.warn(`Trimmed messages to ${estimate([...system, ...rest])} estimated tokens`);
  return [...system, ...rest];
};

// Test all mapped models — visit /test-models to see which ones work with your key
app.get('/test-models', async (req, res) => {
  const results = {};
  for (const [alias, nimModel] of Object.entries(MODEL_MAPPING)) {
    try {
      const r = await axios.post(`${NIM_API_BASE}/chat/completions`, {
        model: nimModel,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        stream: false
      }, {
        headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        timeout: 15000,
        validateStatus: () => true
      });
      results[alias] = { nim_model: nimModel, status: r.status, ok: r.status < 400 };
    } catch (err) {
      results[alias] = { nim_model: nimModel, status: 'timeout', ok: false };
    }
  }
  res.json(results);
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Models endpoint — required by WyvernChat and most OpenAI-compatible frontends
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: 1700000000,
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, frequency_penalty, presence_penalty, top_p, repetition_penalty } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b') || modelLower.includes('large')) {
          nimModel = 'nvidia/nemotron-3-ultra-550b-a55b';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b') || modelLower.includes('medium')) {
          nimModel = 'deepseek-ai/deepseek-v4-flash';
        } else {
          nimModel = 'nvidia/nemotron-3-super-120b-a12b';
        }
      }
    }
    
    // Force streaming always — keeps Render connection alive, prevents 504
    const useStream = true;

    // Build NIM request — only include optional params if actually set
    const nimRequest = {
      model: nimModel,
      messages: trimMessages(messages),
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 2048,
      stream: useStream
    };

    // Only add penalty params if the client sent them (avoids 400 on strict models)
    if (frequency_penalty != null) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty  != null) nimRequest.presence_penalty  = presence_penalty;
    if (top_p             != null) nimRequest.top_p             = top_p;
    if (repetition_penalty != null) nimRequest.repetition_penalty = repetition_penalty;

    // Only add extra_body if thinking mode is actually on
    if (ENABLE_THINKING_MODE) nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    
    // Retry helper with exponential backoff for 429s
    const nimFetch = async (retries = 6, delay = 3000) => {
      for (let i = 0; i <= retries; i++) {
        try {
          return await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers: {
              'Authorization': `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            responseType: 'stream',
            timeout: 0 // no timeout — streaming keeps Render alive
          });
        } catch (err) {
          const status = err.response?.status;
          if ((status === 429 || status === 504) && i < retries) {
            const retryAfter = parseInt(err.response?.headers?.['retry-after'] || 0) * 1000;
            const wait = retryAfter || delay * Math.pow(2, i);
            console.warn(`${status} error. Retrying in ${wait}ms... (attempt ${i + 1}/${retries})`);
            await new Promise(r => setTimeout(r, wait));
          } else {
            throw err;
          }
        }
      }
    };

    // Make request to NVIDIA NIM API
    const response = await nimFetch();
    
    if (stream) {
      // Pass stream directly to client
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;
      let tokenCount = 0;
      const MAX_STREAM_TOKENS = 2500;
      let streamDone = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) { res.write(line + '\n'); return; }
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
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
              res.write(`data: ${JSON.stringify(data)}\n\n`);
              tokenCount += (data.choices?.[0]?.delta?.content || '').length / 4;
              if (tokenCount > MAX_STREAM_TOKENS) {
                streamDone = true;
                if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
                response.data.destroy(); // stop NIM from sending more
              }
            } catch (e) { res.write(line + '\n'); }
          }
        });
      });
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => { console.error('Stream error:', err); res.end(); });

    } else {
      // Client wants JSON — collect the stream and assemble it
      let buffer = '', fullContent = '', fullReasoning = '', finishReason = '', promptTokens = 0, completionTokens = 0;

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
              if (data.usage) { promptTokens = data.usage.prompt_tokens || 0; completionTokens = data.usage.completion_tokens || 0; }
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
        usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens }
      });
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    if (res.headersSent) {
      // Stream already started — just close it cleanly
      if (!res.writableEnded) res.end();
    } else {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message || 'Internal server error',
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    }
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});