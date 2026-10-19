# SPEC: agent-as-model

## Goal

One installable MCP server that lets agents and AI products get completions
without any model API key, because the serving agent's own inference is the
model.

## Non-goals

- No HTTP frontend. The dropped stdlib HTTP shim design is superseded by
  this spec; products that only speak HTTP are out of scope for v1.
- No provider API keys in v1. A fallback is roadmap-only and stays opt-in.
- No embeddings, no image/audio generation. Sampling produces text.
- No multi-turn session state. Each tool call is self-contained.

## Architecture

```
product/agent --tools/call--> agent-as-model --sampling--> host agent's LLM
                                       |
                                   no keys here
```

- `protocol.py`: pure translation. OpenAI chat messages and Anthropic
  messages into `SamplingMessage` lists plus a system prompt. Text
  extraction from sampling results. No I/O, no keys.
- `server.py`: `MCPServer` (official Python SDK v2) over stdio with four
  tools: `chat_complete` and `messages_create` (chat-shaped completion), the
  raw `generate` tool (a plain prompt sent as a single user sampling message
  with no system prompt, no chat formatting overhead), and `list_models`.
  Sampling goes through `Resolve`/`Sample` resolver injection, which
  is the era-portable form: direct `sampling/createMessage` before
  2026-07-28, `InputRequiredResult` multi-round-trip on 2026-07-28+.
- The `model` tool argument becomes an MCP model hint. `agent-default`
  means the host's own model.

## Decisions

- Official SDK v2, not hand-rolled JSON-RPC. The SDK owns version
  negotiation and the MRTR machinery.
- Sampling despite deprecation (SEP-2577). Rationale recorded in README:
  the architecture is the product, the deprecation floor is 2027-07-28 at
  the earliest, and the resolver form migrates cleanly to whatever
  succeeds sampling.
- `generate` is the thin path: raw prompt in, completion text out, same
  host model, no chat formatting. The prompt becomes a single user sampling
  message with no system prompt; `model`, `temperature`, and `max_tokens`
  behave exactly as on the chat tools. No `tools`, no `output_schema`, no
  `tool_choice`: those are chat-tool features, and generate stays zero
  overhead by leaving them out.
- Capability failures are loud. A host without `sampling` gets
  `MISSING_REQUIRED_CLIENT_CAPABILITY`, never an empty completion.
- Non-text sampling results raise instead of being stringified.
- Tool-use passthrough: `tools`/`tool_choice` on both completion tools are
  forwarded to the host's model (legacy sampling params and modern MRTR
  payload alike), fenced by the host's `sampling.tools` capability. Without
  it the tools are dropped before sending, the degradation is logged
  server-side as a warning, and the call returns plain text: explicit, never
  a silent half-state. The return shape stays a text completion; the caller
  gets no in-band degradation flag, because adding one would corrupt the
  shape for every caller. A plain-text answer to a tools request simply
  returns text.
- Host tool calls return as JSON text in the stable shape
  `{"text": "...", "tool_calls": [{"name": ..., "id": ..., "input": ...}]}`
  (keys sorted). The server never executes them.
- `tools` and `output_schema` conflict by design: output_schema is already
  implemented as exactly one required tool, so combining them leaves the
  required-tool constraint ambiguous. The call fails loud naming both,
  rather than silently preferring one.
- Caller-supplied tool definitions are validated up front (non-empty unique
  name, valid object input schema) and never reach the host malformed.

## Security

- The server holds no credentials and makes no network calls.
- Sampling consent and policy belong to the host; the server cannot
  escalate what the host refuses.
- stdio transport only in v1. No socket, no auth surface.

## Verification

- `tests/test_protocol.py`: 8 unit tests over translation and extraction.
- `tests/test_server.py`: end-to-end tests with the in-process SDK client
  and a scripted sampling host: tool listing (Resolve param hidden from
  schemas), `chat_complete` on both protocol eras with wire-level
  assertions (system prompt, messages, temperature, max tokens, model
  hints), `messages_create`, `list_models`, the no-sampling error path, and
  `generate` (prompt as a single user message with no system prompt on both
  eras, params pass-through, tool listing, description).
- `ruff format` and `ruff check` clean.
- Standing implementation gate: testteeth mutation run on touched code
  (see commit log for the score).
