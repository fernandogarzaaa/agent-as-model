# agent-as-model

A true MCP server where the agent **is** the model and the API key.

mcp-name: io.github.fernandogarzaaa/agent-as-model

## Install

```bash
uvx agent-as-model
```

Claude Code:

```bash
claude mcp add agent-as-model -- uvx agent-as-model
```

Any client that takes MCP JSON, paste this:

```json
{
  "mcpServers": {
    "agent-as-model": {
      "command": "uvx",
      "args": ["agent-as-model"]
    }
  }
}
```

Requires Python 3.10+ and a host that declares the MCP `sampling`
capability. No API keys, ever.

From a local checkout: `python -m agent_as_model`. From the repo
without installing: `uvx --from git+https://github.com/fernandogarzaaa/agent-as-model agent-as-model`.

## The idea

AI products usually demand a model API key before they do anything useful.
This server flips that around. The tools speak the request shapes products
already use, and the intelligence comes from the agent on the other end of
the MCP connection. The agent is the model. The agent is the key.

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

**responses_create** — OpenAI Responses API-shaped request, Responses API
response object back. `input` is a string (one user message) or a list of
input items `{role, content}`: `role` is `user`, `assistant`, `system`, or
`developer`; `content` is a string or a list of text parts
(`{type: "input_text"|"text", input_text: "..."}`). System and developer
items fold into the system prompt alongside `instructions`, which leads.
Non-text parts are rejected. `max_tokens` keeps the name from the other
tools (the Responses API calls it `max_output_tokens`). `output_schema`,
`tools`, `tool_choice`, and `persona` behave exactly as on the other
completion tools.

The returned object is `{"id": "resp_<unique>", "object": "response",
"model": <the model alias passed in>, "output": [...]}`. A plain-text host
answer becomes a single `message` item with one `output_text` block. When
the host answers by calling tools, the accompanying text (when non-empty)
becomes a message item and each tool call becomes a `function_call` item:
`{"type": "function_call", "id": "fc_<call id>", "call_id": "<host
tool-call id>", "name": "...", "arguments": "<JSON>"}`; the host's
tool-call id maps to `call_id` and the object id is derived from it.

**generate** — raw prompt string in, completion text out, same host model, no
chat formatting. The thin path for callers that do not need chat shapes: the
prompt is sent as a single user sampling message with no system prompt and no
role translation. `model`, `temperature`, and `max_tokens` behave exactly as
on the other tools; there are no `tools`, `output_schema`, or `tool_choice`
parameters, because those are chat-tool features and generate is zero
overhead. It does accept the optional `persona` contract, which is
request/response validation only and adds no sampling overhead.

**list_models** — returns the one model this server offers: `agent-default`,
an alias for the host agent's own model.

**Structured output** — the three chat-shaped completion tools accept an optional
`output_schema` (a JSON Schema object with root type `object`). When given,
the sampling request carries exactly one tool whose input schema is that
schema, marked as the required tool choice: the host must answer by invoking
the tool, so its tool-use input is the structured completion. The server
validates that input against the schema and returns it as JSON text. A host
answer that violates the schema fails loudly; bad output is never returned
silently. The schema itself is validated up front, so a malformed schema is
the caller's error, not a mid-flow surprise.

**Tool-use passthrough** — the three chat-shaped completion tools also accept `tools`, a list
of tool definitions (`{name, description, input_schema}` with `input_schema`
a JSON Schema object with root type `object`), and `tool_choice`
(`auto`/`required`/`none`, default `auto`). The definitions are validated up
front (non-empty unique names, valid object schemas) and forwarded to the
host's model in the sampling request on both protocol eras. Forwarding is
fenced by the host's `sampling.tools` capability: a host that does not
advertise it gets a tool-free request instead, the degradation is logged
server-side as a warning, and the call returns plain text. When the host
answers by calling tools, the tool returns JSON text in this stable shape
(keys sorted):
`{"text": "...", "tool_calls": [{"id": "...", "input": {...}, "name": "..."}]}`.
A plain-text host answer returns text as before. Passing both `tools` and
`output_schema` is an error: `output_schema` is already implemented as one
required tool, so combining them is contradictory.

**Personas** — all four completion tools (`chat_complete`,
`messages_create`, `responses_create`, and `generate`) accept an optional `persona`
(`{name, input_schema, output_schema}`). The server is stateless: there is
no persona registry, the host supplies the persona per call, and the same
validation runs on both protocol eras. Before anything is sent, the persona
itself is validated (`name` required and non-empty; `input_schema` and
`output_schema`, when given, must be valid JSON Schema objects with root
type `object`), and the tool's request arguments are validated against
`input_schema` when given. The validated request object is the tool's
parameter set with defaults applied, minus `persona` itself: `messages`,
`model`, `temperature`, `max_tokens`, `output_schema`, `tools`,
`tool_choice` for `chat_complete` (plus `system` for `messages_create`), and
`prompt`, `model`, `temperature`, `max_tokens` for `generate`. After the
host answers, the completion text must parse as JSON validating against
`output_schema` when given; invalid output fails loudly, naming the
persona. Unlike `output_schema`, a persona's `output_schema` does not change
the sampling request (no required tool is injected): the host answers in
plain text and the server parses it. Passing both `output_schema` and
`persona.output_schema` is an error: the two are contradictory sources of
the structured-output contract.

**Continuing a conversation** — the server keeps no session state, so
multi-turn continuations are host-owned. All four completion tools accept an
optional `previous_response_id`, and every completion response carries a
cursor you pass back to continue. The cursor is `cur_` + sha256 over the
canonical JSON `{"model","tool","transcript"}` (sorted keys, compact
separators), the transcript being the conversation through the assistant
turn: a tamper-evident commitment, deterministic, so you can recompute it
yourself. Pass `""` to start a chain. To continue, pass the accumulated
history back in (including the previous assistant turn, exactly as received)
with the previous turn's cursor: the server recomputes the cursor over your
history and fails loud, naming expected vs received, if it does not match.
A `responses_create` response id (`resp_...`) anchors a new chain instead of
failing, since opaque ids commit to nothing. When `previous_response_id` is
given, the text tools return `{"completion": "<text>", "response_id":
"cur_<...>"}` instead of plain text; `responses_create` carries the cursor
in its `id` field. Omit `previous_response_id` and everything behaves
exactly as before. To branch from an earlier turn, truncate the history to
that turn and pass its cursor. On `generate` (no structured history
channel) a `cur_` cursor fails loud rather than silently skipping
verification; pass `""` to mint a cursor over the current prompt and
completion.

## How it works

1. A product (or agent) calls `chat_complete` with ordinary chat messages.
2. The server translates them into a completion payload and asks the host
   to complete it. On pre-2026 connections this is a direct
   `sampling/createMessage` request. On 2026-07-28+ the server returns a
   multi-round-trip `InputRequiredResult` carrying the full payload instead
   of calling the sampling API, and the completion arrives on the retry
   round. One tool body serves both eras.
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
  On pre-2026-07-28 connections a host without the sampling capability gets
  a clear `MISSING_REQUIRED_CLIENT_CAPABILITY` error, not a silent failure.
  The 2026-07-28+ MRTR path needs no sampling capability at all.
- Latency is one host model call per tool call. This is a thinking
  endpoint, not a fast one.
- Tool calls are passed through, never executed. The server forwards tool
  definitions to the host's model and returns the host's tool calls as JSON;
  it never runs them. Images and audio in either direction are still rejected
  with explicit errors, not silently dropped.

## Roadmap

- Provider-key fallback for hosts without sampling (opt-in, per-call choice
  stays with the user).
- Structured output via the toolChoice-required trick (shipped: `output_schema`
  on both completion tools, server-side validation with the `jsonschema` package).
- Tool-use passthrough (shipped: `tools`/`tool_choice` on both completion
  tools, fenced by the host's `sampling.tools` capability with explicit
  plain-text degradation, documented JSON result shape).
- More product shapes as tools (embeddings-shaped requests cannot be
  sampled meaningfully; out of scope by design).

## Development

```bash
python3 -m pytest tests/ -q
python3 -m ruff format . && python3 -m ruff check .
```

125 tests, all green: message translation unit tests plus end-to-end runs of
the full MCP loop (tool call to sampled completion) against a scripted host
on both the modern multi-round-trip era and the legacy direct-sampling era,
including the structured-output paths (single required tool in the sampling
params and MRTR payload, validation success, and loud validation failure),
the tool-use passthrough paths (tools and tool_choice forwarded on both
eras, capability-absent degradation to plain text with an explicit logged
warning, documented JSON result shape, and the tools/output_schema conflict),
the generate tool (single user message with no system prompt and no
tools on both eras, param pass-through, tool listing, description), the
persona paths (persona advertised on all three completion tools,
request-side validation success and rejection-before-sending, response-side
validation success and loud failure, malformed-persona rejection, the
output_schema/persona.output_schema conflict, identical behavior on both
eras), and the host-owned session cursor protocol (pinned canonical cursor
computation; chain start; verified multi-turn continuation happy path on
both eras with cross-era cursor agreement; tampered history failing loud
naming expected vs received across edited user turn, edited assistant turn,
and changed model; malformed cursor formats; the opaque `resp_` id
anchoring a new chain; legacy return shapes untouched when
`previous_response_id` is omitted; `responses_create` id-as-cursor on the
cursor path; the `generate` thin-path verification limit;
branch-by-truncation; MRTR round-one verification before any
InputRequiredResult is returned).
