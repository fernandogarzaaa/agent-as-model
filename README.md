# agent-as-model

A true MCP server where the agent **is** the model and the API key.

Install it on any MCP host (Claude Code, Cursor, Windsurf, any spec-compliant
client). It exposes product-style completion tools. Every call is fulfilled by
sampling the host agent's own LLM. No API keys exist anywhere in this system.

## The idea

AI products usually demand a model API key before they do anything useful.
This server flips that around. The tools speak the request shapes products
already use, and the intelligence comes from the agent on the other end of
the MCP connection. The agent is the model. The agent is the key.

## Install

Requires the `mcp` Python package v2 (`pip install "mcp>=2,<3"`) and a host
that declares the MCP `sampling` capability.

Claude Code:

```bash
claude mcp add agent-as-model -- python3 /path/to/agent-as-model/server.py
```

Cursor, Windsurf, or any client that takes MCP JSON:

```json
{
  "mcpServers": {
    "agent-as-model": {
      "command": "python3",
      "args": ["/path/to/agent-as-model/server.py"]
    }
  }
}
```

## Tools

**chat_complete** — OpenAI chat-completions-shaped request, completion text back.

```json
{
  "messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Explain MCP sampling."}
  ],
  "model": "agent-default",
  "temperature": 0.7,
  "max_tokens": 500
}
```

**messages_create** — Anthropic messages-shaped request, completion text back.
`system` accepts a string or a list of text blocks. Message content accepts a
string or a list of text blocks.

**list_models** — returns the one model this server offers: `agent-default`,
an alias for the host agent's own model.

## How it works

1. A product (or agent) calls `chat_complete` with ordinary chat messages.
2. The server translates them into MCP sampling messages and asks the host
   to complete them. On pre-2026 connections this is a direct
   `sampling/createMessage` request. On 2026-07-28+ it rides the
   multi-round-trip pattern. One tool body serves both eras.
3. The host agent generates the completion with its own model and the server
   returns the text. The `model` argument is passed to the host as a model
   hint only; `agent-default` means "your own model".

## Honest caveats

- Sampling is deprecated upstream (SEP-2577, spec 2026-07-28, earliest
  removal 2027-07-28). The official migration path is provider API keys.
  This project exists because the agent-as-model architecture is the point,
  not the keys. The era-portable form used here is the sanctioned way to
  keep it working across both spec eras.
- The host is in charge. Sampling runs under the host's consent and policy.
  A host without the sampling capability gets a clear
  `MISSING_REQUIRED_CLIENT_CAPABILITY` error, not a silent failure.
- Latency is one host model call per tool call. This is a thinking
  endpoint, not a fast one.
- v1 serves text only. Tool calls, images, and audio in either direction
  are rejected with explicit errors, not silently dropped.

## Roadmap

- Provider-key fallback for hosts without sampling (opt-in, per-call choice
  stays with the user).
- Structured output passthrough where hosts support it.
- More product shapes as tools (embeddings-shaped requests cannot be
  sampled meaningfully; out of scope by design).

## Development

```bash
python3 -m pytest tests/ -q
python3 -m ruff format . && python3 -m ruff check .
```

14 tests, all green: message translation unit tests plus end-to-end runs of
the full MCP loop (tool call to sampled completion) against a scripted host
on both the modern multi-round-trip era and the legacy direct-sampling era.
