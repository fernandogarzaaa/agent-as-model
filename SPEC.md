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
- No server-side session state. The server keeps no sessions and no history;
  each tool call is self-contained. Multi-turn continuations are host-owned
  via the cursor protocol (`previous_response_id` / `response_id`), which is
  stateless by construction: the caller passes the accumulated history back
  in, and the cursor is a tamper-evident commitment over that history, not a
  server-side lookup.

## Architecture

```
product/agent --tools/call--> agent-as-model --sampling--> host agent's LLM
                                       |
                                   no keys here
```

- `protocol.py`: pure translation. OpenAI chat messages, Anthropic
  messages, and Responses API input items into `SamplingMessage` lists plus
  a system prompt. Text extraction from sampling results. No I/O, no keys.
- `server.py`: `MCPServer` (official Python SDK v2) over stdio with five
  tools: `chat_complete` and `messages_create` (chat-shaped completion), the
  Responses-shaped `responses_create` (Responses request in, Responses
  response object out), the raw `generate` tool (a plain prompt sent as a
  single user sampling message with no system prompt, no chat formatting
  overhead), and `list_models`.
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
  overhead by leaving them out. It does take `persona`: validation-only,
  never a sampling feature.
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
- Personas are stateless per-call contracts: the host supplies `persona`
  (`{name, input_schema, output_schema}`) on each completion call; the
  server keeps no registry. `name` is required and non-empty, and both
  schemas, when given, must be valid JSON Schema objects with root type
  `object`, validated before anything is sent.
- A persona's `input_schema` constrains the tool's request arguments (the
  parameter set with defaults applied, minus `persona` itself: messages,
  model, temperature, max_tokens, output_schema, tools, tool_choice,
  previous_response_id on the chat tools, plus system on messages_create;
  input, instructions, model, temperature, max_tokens, output_schema, tools,
  tool_choice, previous_response_id on responses_create; prompt, model,
  temperature, max_tokens, previous_response_id on generate). Its `output_schema` constrains the host's
  completion text, which must parse as JSON validating against it, reusing
  the output_schema validation machinery. Both gates sit around the era
  dispatch, so the legacy and modern paths share them identically.
- `persona.output_schema` and `output_schema` conflict by design, on the
  same loud-failure precedent as `tools`/`output_schema`: both declare the
  structured-output contract, and `output_schema` additionally changes the
  request into the toolChoice-required shape. A persona's `output_schema`
  never injects a required tool into the sampling request; the host answers
  in plain text and the server parses it.
- `generate` accepts `persona`: it is a completion tool on the same host
  model, and persona is request/response validation only, never a sampling
  feature, so the thin path stays zero overhead.
- `responses_create` speaks the Responses API request/response shape mapped
  onto the same sampling machinery. Request: `input` (string or list of input
  items), `instructions` (optional system prompt), `model` (default
  "agent-default"), `temperature`, `max_tokens` (named for uniformity with
  the other tools, not `max_output_tokens`), plus the same optional
  `output_schema`, `tools`, `tool_choice`, and `persona` as the other
  completion tools: they ride the shared `_complete` era dispatch, so this
  is wiring, not new machinery. Input translation lives in `protocol.py`:
  a string `input` becomes one user message; an item list maps roles to
  messages, with system/developer items folding into the system prompt
  alongside `instructions` (which leads), and content as a string or a list
  of `input_text`/`text` parts; non-text parts and unknown roles are
  rejected at parse time. Response: `{"id": "resp_<uuid4 hex>", "object":
  "response", "model": <the model alias passed in>, "output": [...]}` with
  a unique id per call. A plain-text completion becomes one `message` item
  with a single `output_text` block; a tool-calling host answer becomes the
  accompanying text as a message item (omitted when empty) followed by one
  `function_call` item per call: `{"type": "function_call", "id":
  "fc_<call id>", "call_id": "<host tool-call id>", "name": ...,
  "arguments": "<JSON string>"}`. The host's tool-call id is the only id the
  server sees, so it maps to the Responses `call_id` (the identifier the
  client echoes when submitting tool outputs) and the object id is derived
  from it with the `fc_` prefix; `arguments` is the JSON string of the tool
  input, keys sorted per the server's canonical JSON convention. The
  tool-call split reads the documented passthrough shape only when tools
  were actually forwarded (the same predicate as the `_complete` capability
  fence), so a plain-text answer to a degraded tools request is never
  misread as tool calls. The persona output gate validates the completion
  text before the response object is built, identically on both eras.
- Host-owned session cursors: multi-turn continuations need no server state.
  All four completion tools accept an optional `previous_response_id`, and
  every completion response carries a cursor the caller passes back to
  continue the conversation. The cursor is `cur_` + the sha256 hex of the
  canonical JSON `{"model","tool","transcript"}` (sorted keys, compact
  separators: `json.dumps(..., sort_keys=True, separators=(",", ":"))`
  encoded as UTF-8), where the transcript is the conversation through the
  assistant turn being committed, as `{"role","content"}` entries. Per-tool
  canonicalization: `chat_complete` uses the caller's message list verbatim
  (system messages stay in the list); `messages_create` puts the flattened
  system prompt first (when given), then the messages with block content
  flattened; `responses_create` puts `instructions` first as a system entry
  (when given), then the input items in order with text flattened (a string
  input is one user turn; system/developer items keep their own roles);
  `generate` is `[{"role":"user","content":<prompt>}]`. The assistant entry
  carries the completion text exactly as the sampling layer returned it (for
  `responses_create`, the output_text string before function_call splitting).
  The commitment covers the tool name (cursors do not replay across tools)
  and the model alias (a model change breaks the chain loudly), and it is
  deterministic: any caller can recompute a cursor from the transcript it
  holds, which is also how the server verifies without keeping state.
  `previous_response_id` has four forms: omitted (None) means the cursor
  protocol is off and the tool returns exactly its legacy shape; `""` starts
  a cursor chain (nothing verified); `cur_<64 hex>` continues a verified
  chain, recomputed over the history passed back cut at its last assistant
  message, failing loud naming expected vs received on mismatch (modified
  history, a different model or tool, or a foreign cursor); `resp_<32 hex>`
  is an opaque `responses_create` response id, which commits to nothing and
  therefore anchors a new verifiable chain at the passed history instead of
  failing. Anything else fails loud as an unrecognized cursor format. To
  branch from an earlier turn, truncate the history to that turn and pass its
  cursor: the server verifies exactly what it is given. Return shape when the
  protocol is active: the text tools return JSON text
  `{"completion":"<the exact legacy text>","response_id":"cur_<...>"}` (keys
  sorted); callers that omit `previous_response_id` never see the envelope.
  For `responses_create` the `id` field IS the cursor: on the cursor path it
  carries the `cur_` commitment, on the legacy path it stays the opaque
  `resp_` unique id. (One `resp_` id cannot be both unique-per-call and a
  deterministic commitment without server state, hence the two id shapes.)
  `generate` is the documented exception: the thin path has no structured
  history channel (the prompt is the only input), so a `cur_` cursor can
  never be verified there; passing one fails loud saying so instead of
  silently skipping the check, and `""` mints a cursor over the current
  prompt and completion. The protocol lives at the tool layer, above the era
  dispatch: verification runs before `_complete` on both eras (on the modern
  era it fails fast on round one, before any InputRequiredResult), the cursor
  is minted when the completion arrives, and the host re-supplies
  `previous_response_id` on the MRTR retry round, like `persona`.
  `previous_response_id` is part of the persona-validated request arguments.

## Security

- The server holds no credentials and makes no network calls.
- Sampling consent and policy belong to the host; the server cannot
  escalate what the host refuses.
- stdio transport only in v1. No socket, no auth surface.

## Verification

- `tests/test_protocol.py`: 19 unit tests over translation and extraction,
  including the Responses translator (string input; item lists with
  system/developer folding alongside instructions; input_text/text part
  aliases; loud rejection of non-text parts, unknown roles, empty lists,
  and parts with no text body).
- `tests/test_server.py`: end-to-end tests with the in-process SDK client
  and a scripted sampling host: tool listing (Resolve param hidden from
  schemas), `chat_complete` on both protocol eras with wire-level
  assertions (system prompt, messages, temperature, max tokens, model
  hints), `messages_create`, `list_models`, the no-sampling error path, and
  `generate` (prompt as a single user message with no system prompt on both
  eras, params pass-through, tool listing, description), and the persona
  paths (advertised on all four completion tools, request-side validation
  success and rejection-before-sending, response-side validation success and
  loud failure, malformed-persona rejection, the
  output_schema/persona.output_schema conflict, identical on both eras), and
  `responses_create` (tool listing and schema, string input and item-list
  input with system/developer folding and instructions, the Responses
  response object shape with unique resp_-prefixed ids, model echo,
  output_text blocks, function_call items on both eras including the
  empty-text case, the capability-degraded tools path, max_tokens passthrough
  and default, output_schema/tools/persona acceptance, the
  tools/output_schema conflict, and loud rejection of non-text input parts,
  identical on both eras where they overlap), and the cursor protocol
  (previous_response_id advertised on all four completion tools; the pinned
  canonical cursor computation; chain start, verified continuation, tamper
  mismatch failing loud naming expected vs received, malformed cursors,
  opaque resp_ anchoring, and the legacy shape untouched when the parameter
  is omitted; the multi-turn happy path on both eras with cross-era cursor
  agreement; responses_create id-as-cursor on the cursor path and the opaque
  resp_ id on the legacy path; the generate thin-path verification limit;
  branch-by-truncation; MRTR round-one verification before any
  InputRequiredResult).
- `ruff format` and `ruff check` clean.
- Standing implementation gate: testteeth mutation run on touched code
  (see commit log for the score).
