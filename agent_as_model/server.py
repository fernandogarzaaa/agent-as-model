"""agent-as-model: a true MCP server where the agent IS the model and the API key.

Install on any MCP host (Claude Code, Cursor, Windsurf, ...). The server
exposes product-style completion tools; every call is fulfilled by sampling
the host agent's own LLM. No API keys exist anywhere in this system.

Tools:
  chat_complete   OpenAI chat-completions-shaped request -> completion text
  messages_create Anthropic messages-shaped request -> completion text
  responses_create OpenAI Responses-shaped request -> Responses response object
  generate        raw prompt string -> completion text (same host model, no chat formatting)
  list_models     the model this server offers (the host's own model, aliased)

The chat-shaped completion tools (chat_complete, messages_create,
responses_create) accept an optional output_schema (a JSON Schema object).
When given, the sampling request carries exactly one tool whose input schema
is that schema, marked as the required tool choice (the toolChoice-required
trick): the host must answer by invoking the tool, and the server validates
the tool-use input against the schema before returning it as JSON text.
Invalid host output fails loudly instead of being returned silently.

Function calling passthrough: both completion tools also accept `tools` (a
list of tool definitions) and `tool_choice`. They are forwarded into the
sampling request on both eras (legacy sampling params, modern MRTR payload),
fenced by the host's sampling.tools capability: a host that does not
advertise tools support gets a tool-free request and answers in plain text,
and the degradation is logged server-side. When the host answers with tool
calls, the tool returns JSON text of the form
{"text": ..., "tool_calls": [{"name": ..., "id": ..., "input": ...}]}; a
plain-text answer still returns text as before. `tools` and `output_schema`
cannot be combined: output_schema is the single-required-tool convenience
for structured output, and combining them raises a clear error naming both.

Personas (item 10): every completion tool also accepts an optional `persona`
`{"name", "input_schema", "output_schema"}`. The server is stateless: there is
no persona registry, the host supplies the persona per call, and the server
validates around `_complete`, identically on both eras. Before anything is
sent, the persona itself is validated (non-empty name, valid object schemas)
and the tool's request arguments are validated against `input_schema` when
given. After the host answers, the completion text must parse as JSON
validating against `output_schema` when given, reusing the item-3 machinery.
`persona` rides `generate` too: it is request/response validation only, never
a sampling feature, so generate stays the zero-overhead thin path.

Era handling is explicit. The tool body reads the negotiated protocol
version from the request Context and dispatches:

- pre-2026-07-28 (legacy): direct `sampling/createMessage` server-to-client
  request, exactly as before. A host that did not declare the sampling
  capability gets a loud MISSING_REQUIRED_CLIENT_CAPABILITY failure.
- 2026-07-28+ (modern, post-sampling): the SDK sampling API is never
  touched. The tool returns an MRTR `InputRequiredResult` carrying the full
  request payload (translated sampling messages, system prompt, temperature,
  max_tokens, model hint); the host fulfills the completion and retries the
  call with its response, and the tool body returns the completion text.
  No sampling capability is required on this path, so the server keeps
  working after sampling itself is removed.

Upstream note: sampling is deprecated as of the 2026-07-28 spec (SEP-2577,
earliest removal 2027-07-28). The MRTR branch is the post-sampling survival
path: it needs nothing from the host except the InputRequiredResult round
trip.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Annotated, Literal

import jsonschema
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.shared.message import ServerMessageMetadata
from mcp.types import (
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    ClientCapabilities,
    CreateMessageRequest,
    CreateMessageRequestParams,
    CreateMessageResult,
    CreateMessageResultWithTools,
    InputRequiredResult,
    MissingRequiredClientCapabilityErrorData,
    ModelHint,
    ModelPreferences,
    SamplingCapability,
    SamplingMessage,
    TextContent,
    Tool,
    ToolChoice,
    ToolUseContent,
)
from mcp_types.version import is_version_at_least
from pydantic import BaseModel, Field

from agent_as_model import __version__
from agent_as_model.protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    ChatMessage,
    ResponseInputItem,
    _response_item_text,
    _text_of,
    anthropic_to_sampling,
    extract_text,
    openai_to_sampling,
    responses_to_sampling,
)

mcp = MCPServer(
    name="agent-as-model",
    version=__version__,
    instructions=(
        "Completion tools backed by the host agent's own model. "
        "No API keys are used or needed: the connected agent is the model."
    ),
)

# First protocol revision whose tools/call completes through the
# multi-round-trip instead of a standalone sampling/createMessage request.
# Pinned (not a moving LATEST), matching the SDK's own pin.
_MODERN_ERA = "2026-07-28"

# Server-assigned key under which the MRTR input request asks the host for
# the completion. The host retries the tool call with its response for this key.
_SAMPLE_INPUT_KEY = "sample"

# Name of the single sampling tool the toolChoice-required trick constrains
# the host to. SEP-1577's required tool choice has no per-name selector in the
# SDK (ToolChoice carries only mode); pairing mode="required" with exactly one
# tool forces the host to answer by invoking that tool, so its input IS the
# structured completion.
_STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"

logger = logging.getLogger("agent_as_model")


class ToolDefinition(BaseModel):
    """One caller-defined tool to forward to the host's model.

    Callers pass these as plain dicts; the server validates them before they
    travel anywhere. `input_schema` follows the MCP Tool inputSchema shape: a
    JSON Schema object with root type "object".
    """

    name: str
    description: str | None = None
    input_schema: dict


class Persona(BaseModel):
    """A host-registered persona contract for one completion call.

    The server is stateless: it keeps no persona registry. The host supplies
    the persona per call as a plain dict, and the server validates the call
    against it. `input_schema` constrains the tool's request arguments (see
    `_validate_persona_request` for the exact per-tool shape); `output_schema`
    constrains the host's completion text, which must parse as JSON validating
    against it. Both schemas follow the same shape as `output_schema`: a JSON
    Schema object with root type "object".
    """

    name: str
    input_schema: dict | None = None
    output_schema: dict | None = None


def _is_modern_era(protocol_version: str | None) -> bool:
    """True on 2026-07-28+ connections, where sampling rides the multi-round-trip.

    A missing version (outside an active request) means the synchronous,
    pre-2026-07-28 transport, so it counts as legacy.
    """
    return protocol_version is not None and is_version_at_least(protocol_version, _MODERN_ERA)


def _assert_legacy_sampling_allowed(protocol_version: str | None) -> None:
    """Gate: direct sampling/createMessage exists only on pre-2026-07-28 connections.

    Raises a diagnostic naming the era and the MRTR alternative instead of
    letting a sampling call fail somewhere opaque.
    """
    if _is_modern_era(protocol_version):
        raise ToolError(
            f"sampling/createMessage is unavailable on protocol {protocol_version}: "
            "direct sampling was removed in the 2026-07-28 era. Fulfill the completion "
            "through the MRTR branch instead: the server returns an InputRequiredResult "
            "carrying the full request payload, no sampling API involved."
        )


def _require_sampling_capability(ctx: Context) -> None:
    """Loud failure when a legacy-era host did not declare the sampling capability."""
    capabilities = ctx.client_capabilities
    sampling = capabilities.sampling if capabilities is not None else None
    if sampling is not None:
        return
    required = ClientCapabilities(sampling=SamplingCapability())
    data = MissingRequiredClientCapabilityErrorData(required_capabilities=required)
    raise MCPError(
        code=MISSING_REQUIRED_CLIENT_CAPABILITY,
        message=(
            "Client did not declare the sampling capability required by this tool call "
            "on the pre-2026-07-28 direct-sampling path; declare sampling, or connect "
            "over the 2026-07-28+ era, where the completion rides the MRTR "
            "InputRequiredResult instead"
        ),
        data=data.model_dump(by_alias=True, mode="json", exclude_none=True),
    )


def _require_object_schema(schema: object, param_name: str = "output_schema") -> dict:
    """Validate a caller-supplied schema before it travels anywhere.

    It must be a JSON Schema object with root type "object", and it must
    itself be a valid schema. A bad schema is the caller's bug: fail here,
    loudly, not mid-flow.
    """
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ToolError(
            f"{param_name} must be a JSON Schema object with root type 'object' "
            f"(got {type(schema).__name__}); a sampling tool carries it as its "
            "input schema"
        )
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ToolError(f"{param_name} is not a valid JSON Schema: {exc.message}") from exc
    return schema


def _validate_tool_definitions(definitions: list[ToolDefinition] | None) -> list[Tool] | None:
    """Validate caller-supplied tool definitions and convert them to MCP tools.

    Every definition needs a non-empty unique name and a valid object input
    schema. Malformed tools are the caller's bug: they are rejected here,
    never forwarded to the host.
    """
    if definitions is None:
        return None
    if not definitions:
        raise ToolError("tools must be a non-empty list when given")
    tools: list[Tool] = []
    seen: set[str] = set()
    for index, definition in enumerate(definitions):
        name = definition.name.strip()
        if not name:
            raise ToolError(f"tools[{index}]: 'name' is required and must be a non-empty string")
        if name in seen:
            raise ToolError(f"tools[{index}]: duplicate tool name {name!r}")
        seen.add(name)
        input_schema = _require_object_schema(
            definition.input_schema, param_name=f"tools[{index}].input_schema"
        )
        tools.append(Tool(name=name, description=definition.description, input_schema=input_schema))
    return tools


def _check_output_schema_tools_conflict(
    output_schema: dict | None, tools: list[ToolDefinition] | list[Tool] | None
) -> None:
    """`tools` and `output_schema` are contradictory: output_schema is already
    implemented as exactly one required tool, so asking for both leaves the
    required-tool constraint ambiguous. Fail loud, naming both, instead of
    silently preferring one."""
    if output_schema is not None and tools is not None:
        raise ToolError(
            "tools and output_schema cannot be combined: output_schema is the "
            "single-required-tool convenience for structured output; pass tools "
            "or output_schema, not both"
        )


def _check_output_schema_persona_conflict(
    output_schema: dict | None, persona: Persona | None
) -> None:
    """`output_schema` and `persona.output_schema` are contradictory sources of
    the structured-output contract: output_schema additionally drives the
    toolChoice-required request mechanism, so combining them leaves the
    constraint ambiguous. Fail loud naming both, like the tools/output_schema
    conflict."""
    if output_schema is not None and persona is not None and persona.output_schema is not None:
        raise ToolError(
            "output_schema and persona.output_schema cannot be combined: both declare "
            "the structured-output contract, and output_schema is the "
            "single-required-tool convenience that also changes the sampling "
            "request; pass output_schema or persona.output_schema, not both"
        )


def _validate_persona(persona: Persona | None) -> Persona | None:
    """Validate the host-supplied persona before anything travels.

    name is required and non-empty; when given, input_schema and
    output_schema must each be a valid JSON Schema object with root type
    "object" (the same gate as output_schema). A malformed persona is the
    caller's bug: fail here, loudly, not mid-flow.
    """
    if persona is None:
        return None
    if not persona.name.strip():
        raise ToolError("persona.name is required and must be a non-empty string")
    if persona.input_schema is not None:
        _require_object_schema(persona.input_schema, param_name="persona.input_schema")
    if persona.output_schema is not None:
        _require_object_schema(persona.output_schema, param_name="persona.output_schema")
    return persona


def _validation_location(exc: jsonschema.ValidationError) -> str:
    """JSON-pointer-style location of a schema violation, for loud errors."""
    return (
        "/" + "/".join(str(part) for part in exc.absolute_path) if exc.absolute_path else "(root)"
    )


def _validate_persona_request(persona: Persona | None, request_arguments: dict) -> None:
    """Request-side persona gate: the tool's request arguments must satisfy the
    persona's input_schema.

    The validated object is exactly the tool's parameter set with server
    defaults applied, minus the persona itself:
    - chat_complete: {"messages", "model", "temperature", "max_tokens",
      "output_schema", "tools", "tool_choice"}
    - messages_create: the same plus "system"
    - responses_create: {"input", "instructions", "model", "temperature",
      "max_tokens", "output_schema", "tools", "tool_choice"}
    - generate: {"prompt", "model", "temperature", "max_tokens"}
    `messages` are the parsed message dicts (role/content), `prompt` the raw
    string. Runs before anything is sent; a violation is the caller's bug and
    fails loud here.
    """
    if persona is None or persona.input_schema is None:
        return
    try:
        jsonschema.validate(instance=request_arguments, schema=persona.input_schema)
    except jsonschema.ValidationError as exc:
        raise ToolError(
            f"persona {persona.name!r}: request failed input-schema validation at "
            f"{_validation_location(exc)}: {exc.message}; the request was rejected "
            "before anything was sent"
        ) from exc


def _validate_persona_output(persona: Persona | None, completion: str) -> str:
    """Response-side persona gate: when the persona declares an output_schema,
    the host's completion text must parse as JSON validating against it.

    Reuses the item-3 machinery (`_validate_structured_output`); the error
    names the persona so the host knows which contract rejected the output.
    """
    if persona is None or persona.output_schema is None:
        return completion
    try:
        output = json.loads(completion)
    except json.JSONDecodeError as exc:
        raise ToolError(
            f"persona {persona.name!r} requires structured output, but the host's "
            f"completion is not valid JSON ({exc.msg}); the completion was rejected "
            "rather than returned as-is"
        ) from exc
    _validate_structured_output(
        output, persona.output_schema, source=f"persona {persona.name!r} output-schema"
    )
    return completion


def _finish_persona(
    persona: Persona | None, result: str | InputRequiredResult
) -> str | InputRequiredResult:
    """Response-side persona gate applied around `_complete`, so both eras share it.

    The MRTR first round returns an InputRequiredResult (no completion yet):
    it passes through untouched, and the gate runs on the retry round when the
    host's completion arrives as text. The host re-supplies the persona on
    every round: the server is stateless.
    """
    if isinstance(result, InputRequiredResult):
        return result
    return _validate_persona_output(persona, result)


# ---------------------------------------------------------------------------
# Item 6: host-owned session cursors (previous_response_id / response_id)
# ---------------------------------------------------------------------------
#
# The server is stateless: it keeps no sessions and no history. Multi-turn
# continuations are host-owned: the caller passes the accumulated history back
# in on each call, and the previous turn's cursor in previous_response_id.
#
# A cursor is "cur_" + the sha256 hex of the canonical JSON
# {"model", "tool", "transcript"} (sorted keys, compact separators), where the
# transcript is the conversation through the assistant turn being committed,
# as {"role", "content"} entries. It is a tamper-evident commitment: anyone
# holding the transcript can recompute it, and any modification of the history
# changes it. No server state is involved.
#
# previous_response_id has four accepted forms:
# - None (omitted): the cursor protocol is off. The tool returns exactly what
#   it returned before this item: plain text for the text tools, the response
#   object with a "resp_"-prefixed unique id for responses_create.
# - "": start a cursor chain. Nothing is verified; the response carries the
#   newly minted cursor.
# - "cur_<64 hex>": continue a verified chain. The server recomputes the
#   cursor over the history passed back (cut at the last assistant message)
#   and fails loud, naming expected vs received, on mismatch.
# - "resp_<32 hex>": an opaque responses_create response id. It commits to
#   nothing (uuid4), so it cannot be verified; it anchors a new verifiable
#   chain at the history passed back instead of failing.
# Anything else fails loud as an unrecognized cursor format.
#
# generate is the exception: the thin path has no structured history channel
# (the prompt is the only input), so a "cur_" cursor can never be verified
# there. Passing one fails loud and says so, instead of silently skipping the
# check. Pass "" to mint a cursor over the current prompt and completion;
# cursors are deterministic, so the caller can always re-derive and check them
# against the documented canonical form.

_CURSOR_PREFIX = "cur_"
_OPAQUE_ID_PREFIX = "resp_"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _canonical_json(obj: object) -> str:
    """Canonical JSON for cursor commitments: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _mint_cursor(tool: str, model: str, transcript: list[dict[str, str]]) -> str:
    """Mint the tamper-evident session cursor for one completed turn.

    The commitment covers the tool name (cursors do not replay across tools),
    the model alias (a model change breaks the chain loudly), and the
    transcript through the assistant turn being committed. Deterministic: the
    caller can recompute any cursor from the transcript it holds.
    """
    canonical = _canonical_json({"model": model, "tool": tool, "transcript": transcript})
    return _CURSOR_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_cursor(value: str) -> bool:
    """A verifiable cursor: "cur_" + 64 lowercase hex chars (a sha256 hex digest)."""
    return (
        value.startswith(_CURSOR_PREFIX)
        and len(value) == len(_CURSOR_PREFIX) + 64
        and all(char in _HEX_DIGITS for char in value[len(_CURSOR_PREFIX) :])
    )


def _is_opaque_response_id(value: str) -> bool:
    """An opaque responses_create response id: "resp_" + 32 hex chars (uuid4 hex).

    Opaque by construction: it identifies the response but commits to nothing,
    so it can anchor a new chain but never continue a verified one.
    """
    return (
        value.startswith(_OPAQUE_ID_PREFIX)
        and len(value) == len(_OPAQUE_ID_PREFIX) + 32
        and all(char in _HEX_DIGITS for char in value[len(_OPAQUE_ID_PREFIX) :])
    )


def _cursor_history_chat(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Canonical transcript entries for the caller's chat messages.

    The caller's message list verbatim, as {"role", "content"} entries: system
    messages stay in the list, because this is the caller's OpenAI transcript,
    not the folded sampling translation. The new assistant turn is appended by
    the caller of this helper.
    """
    return [{"role": message.role, "content": message.content} for message in messages]


def _cursor_history_anthropic(
    messages: list[AnthropicMessage], system: str | list[AnthropicContentBlock] | None
) -> list[dict[str, str]]:
    """Canonical transcript entries: the flattened system prompt first when the
    caller gave one, then the messages in order with block content flattened."""
    entries: list[dict[str, str]] = []
    if system is not None:
        entries.append({"role": "system", "content": _text_of(system)})
    entries.extend(
        {"role": message.role, "content": _text_of(message.content)} for message in messages
    )
    return entries


def _cursor_history_responses(
    input: str | list[ResponseInputItem], instructions: str | None
) -> list[dict[str, str]]:
    """Canonical transcript entries: instructions as the leading system entry when
    given, then the input items in order with text flattened. System and
    developer items keep their own roles: this is the caller's transcript, not
    the folded sampling translation."""
    entries: list[dict[str, str]] = []
    if instructions is not None:
        entries.append({"role": "system", "content": instructions})
    if isinstance(input, str):
        entries.append({"role": "user", "content": input})
    else:
        for item in input:
            entries.append({"role": item.role, "content": _response_item_text(item)})
    return entries


def _check_previous_response_id(
    tool: str,
    model: str,
    history: list[dict[str, str]],
    previous_response_id: str | None,
    *,
    verifiable: bool = True,
) -> None:
    """Request gate: a supplied previous_response_id must match the history.

    "" starts a chain (nothing to verify); a "resp_" id anchors a new chain at
    the passed history (opaque ids commit to nothing); a "cur_" cursor is
    recomputed over the history cut at its last assistant message and must
    match exactly, failing loud naming expected vs received. Anything else is
    an unrecognized cursor format and fails loud. `verifiable=False` is the
    generate thin path: it has no structured history channel, so a "cur_"
    cursor can never be checked there and fails loud saying so instead of
    silently skipping verification.
    """
    if previous_response_id is None or previous_response_id == "":
        return
    if _is_opaque_response_id(previous_response_id):
        return
    if not _is_cursor(previous_response_id):
        raise ToolError(
            f"previous_response_id has an unrecognized format ({previous_response_id!r}): "
            "expected '' to start a cursor chain, 'cur_<64 hex chars>' to continue a "
            "verified chain, or 'resp_<32 hex chars>' (a responses_create response id) "
            "to anchor a new chain at the history passed back"
        )
    if not verifiable:
        raise ToolError(
            f"{tool} cannot verify previous_response_id against history: the thin path "
            "has no structured history channel (the prompt is the only input), so a "
            "'cur_' cursor can never be checked here. Pass previous_response_id='' to "
            "mint a cursor over the current prompt and completion instead; cursors are "
            "deterministic, so the caller can re-derive and check them against the "
            "documented canonical form."
        )
    last_assistant = max(
        (index for index, entry in enumerate(history) if entry["role"] == "assistant"),
        default=None,
    )
    if last_assistant is None:
        raise ToolError(
            "previous_response_id was supplied but the history passed back contains no "
            "assistant message: the cursor commits to the transcript through the "
            "previous assistant turn, so pass the accumulated history including the "
            "previous assistant message(s), or '' to start a new chain"
        )
    expected = _mint_cursor(tool, model, history[: last_assistant + 1])
    if expected != previous_response_id:
        raise ToolError(
            f"previous_response_id mismatch: expected {expected} for the history passed "
            f"back, received {previous_response_id}; the history does not match the "
            "conversation the cursor was minted for (modified history, a different "
            "model or tool, or a cursor from another conversation)"
        )


def _cursor_envelope(completion: str, cursor: str) -> str:
    """The cursor-active return shape for the text tools: JSON text carrying the
    completion exactly as the legacy path would have returned it, plus the
    response cursor. Keys are sorted for a deterministic wire shape (the
    insertion order below is deliberately unsorted, so a sort_keys flip
    changes the string). Callers that omit previous_response_id never see this."""
    return json.dumps({"response_id": cursor, "completion": completion}, sort_keys=True)


def _sampling_tools_supported(ctx: Context) -> bool:
    """True when the host advertised the sampling.tools capability.

    Presence of sampling.tools is what the spec uses to signal full tool
    calling support in sampling requests. Absence (or no sampling capability
    at all) means tools must not be sent.
    """
    capabilities = getattr(ctx, "client_capabilities", None)
    sampling = getattr(capabilities, "sampling", None) if capabilities is not None else None
    return sampling is not None and sampling.tools is not None


def _warn_degraded_tools() -> None:
    """Explicit, server-side record of the tools-to-plain-text degradation.

    The tool result shape stays a plain text completion, so the warning goes
    to the server log (and is documented in the tool descriptions) rather
    than corrupting the return shape.
    """
    logger.warning(
        "Host did not advertise the sampling.tools capability; the requested "
        "tools were dropped and the call degrades to a plain-text completion"
    )


def _sampling_params(
    messages: list[SamplingMessage],
    system_prompt: str | None,
    temperature: float | None,
    max_tokens: int,
    model: str,
    output_schema: dict | None = None,
    *,
    tools: list[Tool] | None = None,
    tool_choice: Literal["auto", "required", "none"] | None = None,
) -> CreateMessageRequestParams:
    """Build the request payload every era path carries: messages, system
    prompt, temperature, max_tokens, and the model hint.

    When output_schema is given, the request carries exactly one tool whose
    input schema is that schema, marked as the required tool choice: the
    toolChoice-required trick for structured output. The host must answer by
    invoking the tool, so its tool-use input is the structured completion.

    When caller tools are given (validated by _validate_tool_definitions),
    they are forwarded as-is with the caller's tool choice, defaulting to
    "auto". The sampling.tools capability fence is applied by _complete
    before sending, not here.
    """
    out_tools: list[Tool] | None = None
    out_tool_choice: ToolChoice | None = None
    if output_schema is not None:
        out_tools = [
            Tool(
                name=_STRUCTURED_OUTPUT_TOOL_NAME,
                description=(
                    "Return the completion as structured data matching this "
                    "tool's input schema. You must call this tool."
                ),
                input_schema=_require_object_schema(output_schema),
            )
        ]
        out_tool_choice = ToolChoice(mode="required")
    elif tools is not None:
        out_tools = tools
        out_tool_choice = ToolChoice(mode=tool_choice if tool_choice is not None else "auto")
    return CreateMessageRequestParams(
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model_preferences=ModelPreferences(hints=[ModelHint(name=model)]),
        tools=out_tools,
        tool_choice=out_tool_choice,
    )


def _format_tool_calls(result: CreateMessageResultWithTools) -> str:
    """Format a tool-enabled host answer as the documented passthrough shape.

    When the host invoked tools, the result is JSON text:
    {"text": "...", "tool_calls": [{"name": ..., "id": ..., "input": ...}]},
    with keys sorted for determinism. "text" carries any accompanying text
    blocks (possibly ""). When the host answered in plain text despite being
    offered tools, the text is returned as before: no JSON wrapper, the same
    shape a tool-free call returns. Unknown content block types are a host
    contract violation and fail loud.
    """
    blocks = result.content_as_list
    texts: list[str] = []
    calls: list[dict] = []
    for block in blocks:
        if isinstance(block, TextContent):
            texts.append(block.text)
        elif isinstance(block, ToolUseContent):
            calls.append({"name": block.name, "id": block.id, "input": block.input})
        else:
            raise ToolError(
                "Tool-enabled completion returned an unsupported content block "
                f"{type(block).__name__}; the completion was rejected rather "
                "than returned as-is"
            )
    text = "\n".join(texts)
    if not calls:
        return text
    return json.dumps({"text": text, "tool_calls": calls}, sort_keys=True)


def _parse_tool_calls_json(completion: str) -> tuple[str, list[dict]] | None:
    """Split the documented tool-use passthrough JSON into (text, calls).

    Returns None when the completion is not the passthrough shape; only then
    does the completion stay a single message item. Never raises: a host that
    answered in plain prose (even with tools offered) keeps its text untouched.
    """
    try:
        parsed = json.loads(completion)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    calls = parsed.get("tool_calls")
    if not isinstance(calls, list) or "text" not in parsed:
        return None
    text = parsed["text"]
    return (text if isinstance(text, str) else "", calls)


def _responses_object(completion: str, model: str, *, used_tools: bool) -> dict:
    """Wrap a completion in the OpenAI Responses API response object.

    {"id": "resp_<unique>", "object": "response", "model": <model alias>,
     "output": [{"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": <completion>}]}]}

    The id is unique per call (uuid4 hex). When tools were actually forwarded
    to the host (the caller passed tools AND the host advertised
    sampling.tools) and the host answered with tool calls, the completion
    text carries the documented passthrough shape {"text": ..., "tool_calls":
    [...]}; it is split into output items in the Responses shape: the
    accompanying text becomes a message item (omitted when empty), and each
    tool call becomes a function_call item
    {"type": "function_call", "id": "fc_<call id>", "call_id": "<call id>",
     "name": <name>, "arguments": "<JSON>"}.

    The id/call_id split mirrors the Responses API: the host's tool-call id
    is the only id the server sees, so it maps to call_id (the identifier the
    client echoes when submitting tool outputs), and the object id is derived
    from it with the fc_ prefix. arguments is the JSON string of the tool
    input, keys sorted, matching the server's canonical JSON convention. The
    server never executes the calls.
    """
    text = completion
    calls: list[dict] = []
    if used_tools:
        parsed = _parse_tool_calls_json(completion)
        if parsed is not None:
            text, calls = parsed
    output: list[dict] = []
    if text or not calls:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    for call in calls:
        call_id = str(call.get("id", ""))
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": call.get("name"),
                "arguments": json.dumps(call.get("input", {}), sort_keys=True),
            }
        )
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "model": model,
        "output": output,
    }


async def _legacy_sample(
    ctx: Context,
    params: CreateMessageRequestParams,
    output_schema: dict | None = None,
    *,
    tools: list[Tool] | None = None,
) -> str:
    """Direct sampling/createMessage for pre-2026-07-28 connections.

    The era gate forbids this on modern connections; the sampling capability
    gate fails loud when the host did not declare it. With an output_schema
    the required-tool response is extracted and validated against it. With
    caller tools (already fenced by _complete), a tool-calling answer comes
    back in the documented passthrough JSON shape, and a plain-text answer
    returns text as before.
    """
    _assert_legacy_sampling_allowed(ctx.protocol_version)
    if ctx.session.can_send_request:
        _require_sampling_capability(ctx)
    if output_schema is not None:
        with_tools = await ctx.session.send_request(
            CreateMessageRequest(params=params),
            CreateMessageResultWithTools,
            metadata=ServerMessageMetadata(related_request_id=ctx.request_id),
        )
        output = _extract_structured_output(with_tools)
        _validate_structured_output(output, output_schema)
        return json.dumps(output, sort_keys=True)
    if tools is not None:
        tool_answer = await ctx.session.send_request(
            CreateMessageRequest(params=params),
            CreateMessageResultWithTools,
            metadata=ServerMessageMetadata(related_request_id=ctx.request_id),
        )
        return _format_tool_calls(tool_answer)
    result = await ctx.session.send_request(
        CreateMessageRequest(params=params),
        CreateMessageResult,
        metadata=ServerMessageMetadata(related_request_id=ctx.request_id),
    )
    return extract_text(result)


def _extract_structured_output(result: CreateMessageResultWithTools) -> dict:
    """Pull the structured completion out of the host's required-tool response.

    The toolChoice-required trick constrains the host to a single tool, so it
    answers by invoking that tool; the tool-use input dict IS the structured
    completion. A response with no such tool use is a host contract violation:
    reject it, never silently stringify it.
    """
    uses = [
        block
        for block in result.content_as_list
        if isinstance(block, ToolUseContent) and block.name == _STRUCTURED_OUTPUT_TOOL_NAME
    ]
    if not uses:
        raise ToolError(
            f"output_schema requested a structured tool-use response, but the host's "
            f"completion contains no {_STRUCTURED_OUTPUT_TOOL_NAME!r} tool use; "
            "the completion was rejected rather than returned as-is"
        )
    return uses[0].input


def _validate_structured_output(
    output: dict, output_schema: dict, *, source: str = "output-schema"
) -> None:
    """Gate: the host's structured completion must satisfy the caller's schema.

    Raises ToolError naming the exact mismatch. Bad output is never returned
    silently, and the schema is never dropped. `source` names the schema's
    origin in the error: the output_schema parameter, or a persona's.
    """
    try:
        jsonschema.validate(instance=output, schema=output_schema)
    except jsonschema.ValidationError as exc:
        raise ToolError(
            f"Host completion failed {source} validation at {_validation_location(exc)}: "
            f"{exc.message}; the completion was rejected rather than returned as-is"
        ) from exc


def _mrtr_completion_request(params: CreateMessageRequestParams) -> InputRequiredResult:
    """First MRTR round: hand the host the full request payload, no sampling API.

    The input request carries the completion the host must produce; the
    opaque request_state echoes the same payload so the retry round can be
    matched to what was asked.
    """
    return InputRequiredResult(
        input_requests={_SAMPLE_INPUT_KEY: CreateMessageRequest(params=params)},
        request_state=params.model_dump_json(by_alias=True, exclude_none=True),
    )


def _mrtr_completion_result(
    ctx: Context, output_schema: dict | None = None, *, tools: list[Tool] | None = None
) -> str | None:
    """Second MRTR round: the host's fulfilled completion, or None on round one.

    With an output_schema the host must answer through the required tool: the
    tool-use input is extracted and validated, and a plain-text answer is a
    contract violation, not a completion. With caller tools (already fenced by
    _complete), a tool-calling answer comes back in the documented passthrough
    JSON shape, and a plain-text answer returns text as before.
    """
    responses = ctx.input_responses
    if not responses:
        return None
    response = responses.get(_SAMPLE_INPUT_KEY)
    if response is None:
        return None
    if output_schema is not None:
        if not isinstance(response, CreateMessageResultWithTools):
            raise ToolError(
                f"output_schema requested a structured tool-use response for "
                f"{_SAMPLE_INPUT_KEY!r}, but the host returned "
                f"{type(response).__name__}; the required-tool constraint was "
                "not honored, so the completion was rejected"
            )
        output = _extract_structured_output(response)
        _validate_structured_output(output, output_schema)
        return json.dumps(output, sort_keys=True)
    if tools is not None and isinstance(response, CreateMessageResultWithTools):
        return _format_tool_calls(response)
    if isinstance(response, CreateMessageResult | CreateMessageResultWithTools):
        return extract_text(response)
    raise ToolError(
        f"MRTR completion round received a response of the wrong kind for "
        f"{_SAMPLE_INPUT_KEY!r}: {type(response).__name__}; "
        "expected the host's CreateMessageResult"
    )


async def _complete(
    ctx: Context,
    params: CreateMessageRequestParams,
    output_schema: dict | None = None,
    *,
    tools: list[Tool] | None = None,
    tool_choice: Literal["auto", "required", "none"] | None = None,
) -> str | InputRequiredResult:
    """Era dispatch: MRTR InputRequiredResult on 2026-07-28+, direct sampling before.

    Applies the sampling.tools capability fence: when caller tools were
    requested but the host did not advertise tools support, the tools are
    dropped here (on both eras), the degradation is logged, and the call
    proceeds as a plain-text completion. The conflict between tools and
    output_schema fails loud before anything travels.
    """
    _check_output_schema_tools_conflict(output_schema, tools)
    if tools is not None and not _sampling_tools_supported(ctx):
        _warn_degraded_tools()
        params = params.model_copy(update={"tools": None, "tool_choice": None})
        tools = None
    if _is_modern_era(ctx.protocol_version):
        completed = _mrtr_completion_result(ctx, output_schema, tools=tools)
        if completed is None:
            return _mrtr_completion_request(params)
        return completed
    return await _legacy_sample(ctx, params, output_schema, tools=tools)


@mcp.tool(
    description=(
        "Generate a chat completion from an OpenAI chat-completions-shaped request. "
        "The completion is produced by the host agent's own model: direct MCP sampling "
        "on pre-2026-07-28 connections, an MRTR InputRequiredResult carrying the "
        "request payload on 2026-07-28+. "
        "When output_schema is given, the request carries exactly one required tool "
        "whose input schema is that schema, and the validated structured completion "
        "is returned as JSON text. "
        "When tools is given, the tool definitions are forwarded to the host's model "
        "with tool_choice, but only if the host advertised the sampling.tools "
        "capability: otherwise the tools are dropped (a warning is logged server-side) "
        "and the call degrades to a plain-text completion. "
        "When the host answers by calling tools, the returned text is JSON: "
        '{"text": "...", "tool_calls": [{"id": "...", "input": {...}, "name": "..."}]} '
        "(keys sorted). A plain-text host answer returns text as before. "
        "tools and output_schema cannot be combined. "
        "An optional persona {name, input_schema, output_schema} registers a "
        "per-call contract: the request arguments are validated against "
        "input_schema before anything is sent, and the host's completion text "
        "must parse as JSON validating against output_schema. The server is "
        "stateless and keeps no persona registry. persona.output_schema and "
        "output_schema cannot be combined. "
        "Session cursors: the server keeps no session state. To continue a "
        "conversation, pass the accumulated history back in and the previous "
        "turn's cursor as previous_response_id: '' starts a cursor chain "
        "(nothing verified), 'cur_<64 hex>' continues a verified chain "
        "(checked against the history passed back; mismatch fails loud naming "
        "expected vs received), 'resp_<32 hex>' (a responses_create response id) "
        "anchors a new chain at the passed history. When previous_response_id is "
        "given, the completion returns as JSON "
        '\'{"completion": "<text>", "response_id": "cur_<...>"}\' instead of '
        "plain text; when omitted the tool returns plain text exactly as before. "
        "The cursor is 'cur_' + sha256 over canonical JSON "
        '\'{"model","tool","transcript"}\' (sorted keys, compact separators), '
        "the transcript being the conversation through the assistant turn, so the "
        "caller can recompute it. Re-supply previous_response_id on the MRTR retry "
        "round on 2026-07-28+ connections, like persona. "
        "No API key is used or needed: the connected agent is the model."
    )
)
async def chat_complete(
    ctx: Context,
    messages: Annotated[
        list[ChatMessage],
        Field(description="Chat messages; system roles become the system prompt."),
    ],
    model: Annotated[
        str, Field(description="Model alias; agent-default means the host's own model.")
    ] = "agent-default",
    temperature: Annotated[
        float | None, Field(description="Sampling temperature, if the host honors it.")
    ] = None,
    max_tokens: Annotated[int, Field(description="Maximum tokens to generate.")] = 4096,
    output_schema: Annotated[
        dict | None,
        Field(
            description=(
                "Optional JSON Schema (root type object). When given, the sampling "
                "request carries exactly one required tool with this input schema "
                "and the completion is validated against it before returning."
            )
        ),
    ] = None,
    tools: Annotated[
        list[ToolDefinition] | None,
        Field(
            description=(
                "Optional tool definitions to forward to the host's model. Each is "
                "{name, description, input_schema} where input_schema is a JSON "
                "Schema object with root type 'object'. Forwarded only when the "
                "host advertised the sampling.tools capability; otherwise dropped "
                "with a server-side warning and the call degrades to plain text."
            )
        ),
    ] = None,
    tool_choice: Annotated[
        Literal["auto", "required", "none"] | None,
        Field(
            description=(
                "Optional tool-choice mode, forwarded with tools; defaults to "
                "'auto' when tools are given."
            )
        ),
    ] = None,
    persona: Annotated[
        Persona | None,
        Field(
            description=(
                "Optional persona contract for this call: {name, input_schema, "
                "output_schema}. name is required and non-empty; the server is "
                "stateless and keeps no persona registry. When input_schema is "
                "given, the request arguments (messages, model, temperature, "
                "max_tokens, output_schema, tools, tool_choice, "
                "previous_response_id) are validated "
                "against it before anything is sent. When output_schema is "
                "given, the host's completion text must parse as JSON "
                "validating against it. persona.output_schema and output_schema "
                "cannot be combined."
            )
        ),
    ] = None,
    previous_response_id: Annotated[
        str | None,
        Field(
            description=(
                "Optional session cursor for host-owned multi-turn continuations. "
                "The server keeps no session state: pass the accumulated history "
                "back in messages and the previous turn's cursor here. '' starts "
                "a cursor chain (nothing verified); 'cur_<64 hex>' continues a "
                "verified chain, checked against the history passed back "
                "(mismatch fails loud, naming expected vs received); "
                "'resp_<32 hex>' (a responses_create response id) anchors a new "
                "chain at the passed history. When given, the completion returns "
                'as JSON \'{"completion": "<text>", "response_id": '
                '"cur_<...>"}\' instead of plain text; when omitted the tool '
                "returns plain text exactly as before."
            )
        ),
    ] = None,
) -> str | InputRequiredResult:
    """Complete a chat with the host agent's model (OpenAI request shape)."""
    persona = _validate_persona(persona)
    _check_output_schema_persona_conflict(output_schema, persona)
    validated_tools = _validate_tool_definitions(tools)
    system_prompt, sampling_messages = openai_to_sampling(messages)
    history = _cursor_history_chat(messages)
    _check_previous_response_id("chat_complete", model, history, previous_response_id)
    _validate_persona_request(
        persona,
        {
            "messages": [message.model_dump() for message in messages],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output_schema": output_schema,
            "tools": [tool.model_dump() for tool in validated_tools] if validated_tools else None,
            "tool_choice": tool_choice,
            "previous_response_id": previous_response_id,
        },
    )
    params = _sampling_params(
        sampling_messages,
        system_prompt,
        temperature,
        max_tokens,
        model,
        output_schema,
        tools=validated_tools,
        tool_choice=tool_choice,
    )
    completed = _finish_persona(
        persona, await _complete(ctx, params, output_schema, tools=validated_tools)
    )
    if isinstance(completed, InputRequiredResult):
        return completed
    if previous_response_id is None:
        return completed
    cursor = _mint_cursor(
        "chat_complete", model, [*history, {"role": "assistant", "content": completed}]
    )
    return _cursor_envelope(completed, cursor)


@mcp.tool(
    description=(
        "Generate a completion from an Anthropic messages-shaped request. "
        "The completion is produced by the host agent's own model: direct MCP sampling "
        "on pre-2026-07-28 connections, an MRTR InputRequiredResult carrying the "
        "request payload on 2026-07-28+. "
        "When output_schema is given, the request carries exactly one required tool "
        "whose input schema is that schema, and the validated structured completion "
        "is returned as JSON text. "
        "When tools is given, the tool definitions are forwarded to the host's model "
        "with tool_choice, but only if the host advertised the sampling.tools "
        "capability: otherwise the tools are dropped (a warning is logged server-side) "
        "and the call degrades to a plain-text completion. "
        "When the host answers by calling tools, the returned text is JSON: "
        '{"text": "...", "tool_calls": [{"id": "...", "input": {...}, "name": "..."}]} '
        "(keys sorted). A plain-text host answer returns text as before. "
        "tools and output_schema cannot be combined. "
        "An optional persona {name, input_schema, output_schema} registers a "
        "per-call contract: the request arguments are validated against "
        "input_schema before anything is sent, and the host's completion text "
        "must parse as JSON validating against output_schema. The server is "
        "stateless and keeps no persona registry. persona.output_schema and "
        "output_schema cannot be combined. "
        "Session cursors: the server keeps no session state. To continue a "
        "conversation, pass the accumulated history back in and the previous "
        "turn's cursor as previous_response_id: '' starts a cursor chain "
        "(nothing verified), 'cur_<64 hex>' continues a verified chain "
        "(checked against the history passed back; mismatch fails loud naming "
        "expected vs received), 'resp_<32 hex>' (a responses_create response id) "
        "anchors a new chain at the passed history. When previous_response_id is "
        "given, the completion returns as JSON "
        '\'{"completion": "<text>", "response_id": "cur_<...>"}\' instead of '
        "plain text; when omitted the tool returns plain text exactly as before. "
        "The cursor is 'cur_' + sha256 over canonical JSON "
        '\'{"model","tool","transcript"}\' (sorted keys, compact separators), '
        "the transcript being the conversation through the assistant turn, so the "
        "caller can recompute it. Re-supply previous_response_id on the MRTR retry "
        "round on 2026-07-28+ connections, like persona. "
        "No API key is used or needed: the connected agent is the model."
    )
)
async def messages_create(
    ctx: Context,
    messages: Annotated[
        list[AnthropicMessage], Field(description="User/assistant messages in order.")
    ],
    model: Annotated[
        str, Field(description="Model alias; agent-default means the host's own model.")
    ] = "agent-default",
    system: Annotated[
        str | list[AnthropicContentBlock] | None,
        Field(description="System prompt as text or text blocks."),
    ] = None,
    temperature: Annotated[
        float | None, Field(description="Sampling temperature, if the host honors it.")
    ] = None,
    max_tokens: Annotated[int, Field(description="Maximum tokens to generate.")] = 4096,
    output_schema: Annotated[
        dict | None,
        Field(
            description=(
                "Optional JSON Schema (root type object). When given, the sampling "
                "request carries exactly one required tool with this input schema "
                "and the completion is validated against it before returning."
            )
        ),
    ] = None,
    tools: Annotated[
        list[ToolDefinition] | None,
        Field(
            description=(
                "Optional tool definitions to forward to the host's model. Each is "
                "{name, description, input_schema} where input_schema is a JSON "
                "Schema object with root type 'object'. Forwarded only when the "
                "host advertised the sampling.tools capability; otherwise dropped "
                "with a server-side warning and the call degrades to plain text."
            )
        ),
    ] = None,
    tool_choice: Annotated[
        Literal["auto", "required", "none"] | None,
        Field(
            description=(
                "Optional tool-choice mode, forwarded with tools; defaults to "
                "'auto' when tools are given."
            )
        ),
    ] = None,
    persona: Annotated[
        Persona | None,
        Field(
            description=(
                "Optional persona contract for this call: {name, input_schema, "
                "output_schema}. name is required and non-empty; the server is "
                "stateless and keeps no persona registry. When input_schema is "
                "given, the request arguments (messages, model, system, "
                "temperature, max_tokens, output_schema, tools, tool_choice, "
                "previous_response_id) "
                "are validated against it before anything is sent. When "
                "output_schema is given, the host's completion text must parse "
                "as JSON validating against it. persona.output_schema and "
                "output_schema cannot be combined."
            )
        ),
    ] = None,
    previous_response_id: Annotated[
        str | None,
        Field(
            description=(
                "Optional session cursor for host-owned multi-turn continuations. "
                "The server keeps no session state: pass the accumulated history "
                "back in messages and the previous turn's cursor here. '' starts "
                "a cursor chain (nothing verified); 'cur_<64 hex>' continues a "
                "verified chain, checked against the history passed back "
                "(mismatch fails loud, naming expected vs received); "
                "'resp_<32 hex>' (a responses_create response id) anchors a new "
                "chain at the passed history. When given, the completion returns "
                'as JSON \'{"completion": "<text>", "response_id": '
                '"cur_<...>"}\' instead of plain text; when omitted the tool '
                "returns plain text exactly as before."
            )
        ),
    ] = None,
) -> str | InputRequiredResult:
    """Complete a conversation with the host agent's model (Anthropic request shape)."""
    persona = _validate_persona(persona)
    _check_output_schema_persona_conflict(output_schema, persona)
    validated_tools = _validate_tool_definitions(tools)
    system_prompt, sampling_messages = anthropic_to_sampling(messages, system)
    history = _cursor_history_anthropic(messages, system)
    _check_previous_response_id("messages_create", model, history, previous_response_id)
    _validate_persona_request(
        persona,
        {
            "messages": [message.model_dump() for message in messages],
            "model": model,
            "system": [block.model_dump() for block in system]
            if isinstance(system, list)
            else system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output_schema": output_schema,
            "tools": [tool.model_dump() for tool in validated_tools] if validated_tools else None,
            "tool_choice": tool_choice,
            "previous_response_id": previous_response_id,
        },
    )
    params = _sampling_params(
        sampling_messages,
        system_prompt,
        temperature,
        max_tokens,
        model,
        output_schema,
        tools=validated_tools,
        tool_choice=tool_choice,
    )
    completed = _finish_persona(
        persona, await _complete(ctx, params, output_schema, tools=validated_tools)
    )
    if isinstance(completed, InputRequiredResult):
        return completed
    if previous_response_id is None:
        return completed
    cursor = _mint_cursor(
        "messages_create", model, [*history, {"role": "assistant", "content": completed}]
    )
    return _cursor_envelope(completed, cursor)


@mcp.tool(
    description=(
        "Generate a completion from an OpenAI Responses-shaped request, returning "
        "the Responses API response object. The completion is produced by the host "
        "agent's own model: direct MCP sampling on pre-2026-07-28 connections, an "
        "MRTR InputRequiredResult carrying the request payload on 2026-07-28+. "
        "`input` is a string or a list of input items; each item is {role, content} "
        "with role user, assistant, system, or developer, and content a string or a "
        "list of text parts ({type: 'input_text'|'text', input_text: '...'}). A string "
        "input becomes one user message. System and developer items fold into the "
        "system prompt alongside `instructions`, which leads. Non-text parts are "
        "rejected. `max_tokens` is named for uniformity with the other tools (the "
        "Responses API calls it max_output_tokens); same meaning. "
        "When output_schema is given, the request carries exactly one required tool "
        "whose input schema is that schema, and the validated structured completion "
        "is returned as JSON text inside the output_text. "
        "When tools is given, the tool definitions are forwarded to the host's model "
        "with tool_choice, but only if the host advertised the sampling.tools "
        "capability: otherwise the tools are dropped (a warning is logged server-side) "
        "and the call degrades to a plain-text completion. "
        "When the host answers by calling tools, the output array carries the "
        "accompanying text as a message item (omitted when empty) followed by one "
        "function_call item per call: {type: 'function_call', id: 'fc_<call id>', "
        "call_id: '<host tool-call id>', name: '...', arguments: '<JSON>'}; the host's "
        "tool-call id maps to call_id, the object id is derived from it. A plain-text "
        "host answer returns a single message item. tools and output_schema cannot "
        "be combined. "
        "An optional persona {name, input_schema, output_schema} registers a "
        "per-call contract: the request arguments are validated against "
        "input_schema before anything is sent, and the host's completion text "
        "must parse as JSON validating against output_schema. The server is "
        "stateless and keeps no persona registry. persona.output_schema and "
        "output_schema cannot be combined. "
        "Session cursors: the server keeps no session state. To continue a "
        "conversation, pass the accumulated history back in input and the previous "
        "turn's response id as previous_response_id: '' starts a cursor chain "
        "(nothing verified), 'cur_<64 hex>' continues a verified chain "
        "(checked against the history passed back; mismatch fails loud naming "
        "expected vs received), 'resp_<32 hex>' (a previous response id) anchors "
        "a new chain at the passed history. The response object's id field IS "
        "the cursor: when previous_response_id is given it carries the 'cur_' "
        "tamper-evident commitment, when omitted it stays the opaque 'resp_' "
        "unique id exactly as before. The cursor is 'cur_' + sha256 over "
        'canonical JSON \'{"model","tool","transcript"}\' (sorted keys, '
        "compact separators), the transcript being the conversation through the "
        "assistant turn, so the caller can recompute it. Re-supply "
        "previous_response_id on the MRTR retry round on 2026-07-28+ "
        "connections, like persona. "
        "No API key is used or needed: the connected agent is the model."
    )
)
async def responses_create(
    ctx: Context,
    input: Annotated[
        str | list[ResponseInputItem],
        Field(
            description=(
                "The Responses API input: a string, which becomes one user "
                "message, or a list of input items {role, content}. role is "
                "user, assistant, system, or developer; content is a string or "
                "a list of text parts ({type: 'input_text'|'text', input_text: "
                "'...'}). System and developer items fold into the system "
                "prompt alongside instructions. Non-text parts are rejected."
            )
        ),
    ],
    instructions: Annotated[
        str | None,
        Field(description="Optional system prompt; leads the system/developer items."),
    ] = None,
    model: Annotated[
        str, Field(description="Model alias; agent-default means the host's own model.")
    ] = "agent-default",
    temperature: Annotated[
        float | None, Field(description="Sampling temperature, if the host honors it.")
    ] = None,
    max_tokens: Annotated[
        int,
        Field(
            description=(
                "Maximum tokens to generate. Named max_tokens for uniformity "
                "with the other tools; the Responses API calls it "
                "max_output_tokens."
            )
        ),
    ] = 4096,
    output_schema: Annotated[
        dict | None,
        Field(
            description=(
                "Optional JSON Schema (root type object). When given, the sampling "
                "request carries exactly one required tool with this input schema "
                "and the completion is validated against it before returning."
            )
        ),
    ] = None,
    tools: Annotated[
        list[ToolDefinition] | None,
        Field(
            description=(
                "Optional tool definitions to forward to the host's model. Each is "
                "{name, description, input_schema} where input_schema is a JSON "
                "Schema object with root type 'object'. Forwarded only when the "
                "host advertised the sampling.tools capability; otherwise dropped "
                "with a server-side warning and the call degrades to plain text."
            )
        ),
    ] = None,
    tool_choice: Annotated[
        Literal["auto", "required", "none"] | None,
        Field(
            description=(
                "Optional tool-choice mode, forwarded with tools; defaults to "
                "'auto' when tools are given."
            )
        ),
    ] = None,
    persona: Annotated[
        Persona | None,
        Field(
            description=(
                "Optional persona contract for this call: {name, input_schema, "
                "output_schema}. name is required and non-empty; the server is "
                "stateless and keeps no persona registry. When input_schema is "
                "given, the request arguments (input, instructions, model, "
                "temperature, max_tokens, output_schema, tools, tool_choice, "
                "previous_response_id) "
                "are validated against it before anything is sent. When "
                "output_schema is given, the host's completion text must parse "
                "as JSON validating against it. persona.output_schema and "
                "output_schema cannot be combined."
            )
        ),
    ] = None,
    previous_response_id: Annotated[
        str | None,
        Field(
            description=(
                "Optional session cursor for host-owned multi-turn continuations. "
                "The server keeps no session state: pass the accumulated history "
                "back in input and the previous turn's response id here. '' "
                "starts a cursor chain (nothing verified); 'cur_<64 hex>' "
                "continues a verified chain, checked against the history passed "
                "back (mismatch fails loud, naming expected vs received); "
                "'resp_<32 hex>' (a previous response id) anchors a new chain "
                "at the passed history. When given, the response object's id "
                "field carries the 'cur_' cursor; when omitted the id stays the "
                "opaque 'resp_' unique id exactly as before."
            )
        ),
    ] = None,
) -> dict | InputRequiredResult:
    """Complete a Responses API request with the host agent's model."""
    persona = _validate_persona(persona)
    _check_output_schema_persona_conflict(output_schema, persona)
    validated_tools = _validate_tool_definitions(tools)
    system_prompt, sampling_messages = responses_to_sampling(input, instructions)
    history = _cursor_history_responses(input, instructions)
    _check_previous_response_id("responses_create", model, history, previous_response_id)
    _validate_persona_request(
        persona,
        {
            "input": [item.model_dump() for item in input] if isinstance(input, list) else input,
            "instructions": instructions,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output_schema": output_schema,
            "tools": [tool.model_dump() for tool in validated_tools] if validated_tools else None,
            "tool_choice": tool_choice,
            "previous_response_id": previous_response_id,
        },
    )
    params = _sampling_params(
        sampling_messages,
        system_prompt,
        temperature,
        max_tokens,
        model,
        output_schema,
        tools=validated_tools,
        tool_choice=tool_choice,
    )
    completed = _finish_persona(
        persona, await _complete(ctx, params, output_schema, tools=validated_tools)
    )
    if isinstance(completed, InputRequiredResult):
        return completed
    # The predicate mirrors the capability fence inside _complete: tools count
    # as used exactly when they were forwarded to the host, so a plain-text
    # answer to a degraded tools request is never misread as tool calls.
    response_object = _responses_object(
        completed,
        model,
        used_tools=validated_tools is not None and _sampling_tools_supported(ctx),
    )
    if previous_response_id is not None:
        # The id field IS the cursor: on the cursor path it carries the
        # tamper-evident commitment instead of the opaque unique id. The
        # assistant entry commits the completion text exactly as the sampling
        # layer returned it (the output_text string, before function_call
        # splitting), so the caller passes that text back as the assistant
        # content when continuing.
        response_object["id"] = _mint_cursor(
            "responses_create",
            model,
            [*history, {"role": "assistant", "content": completed}],
        )
    return response_object


@mcp.tool(
    description=(
        "Generate a raw text completion from a plain prompt string. "
        "The prompt is sent as a single user sampling message with no system prompt "
        "and no chat formatting: raw prompt in, completion text out, produced by "
        "the host agent's own model. Same model hint and parameters as the other "
        "completion tools, minus any chat or tool features: no tools, no "
        "output_schema, no tool_choice. It does accept the optional persona "
        "contract ({name, input_schema, output_schema}): request/response "
        "validation only, never a sampling feature, so the thin path stays "
        "zero overhead. "
        "Session cursors: the server keeps no session state. previous_response_id "
        "'' mints a cursor over the current prompt and completion, returned as "
        'JSON \'{"completion": "<text>", "response_id": "cur_<...>"}\' '
        "instead of plain text; when omitted the tool returns plain text exactly "
        "as before. generate has no structured history channel (the prompt is the "
        "only input), so a 'cur_' cursor can never be verified here: passing one "
        "fails loud instead of silently skipping the check. The cursor is 'cur_' "
        '+ sha256 over canonical JSON \'{"model","tool","transcript"}\' '
        "(sorted keys, compact separators) with the transcript "
        '[{"role": "user", "content": <prompt>}, {"role": "assistant", '
        '"content": <completion>}], so the caller can recompute it. '
        "No API key is used or needed: the connected agent is the model."
    )
)
async def generate(
    ctx: Context,
    prompt: Annotated[
        str, Field(description="The raw prompt text to complete. No chat formatting applied.")
    ],
    model: Annotated[
        str, Field(description="Model alias; agent-default means the host's own model.")
    ] = "agent-default",
    temperature: Annotated[
        float | None, Field(description="Sampling temperature, if the host honors it.")
    ] = None,
    max_tokens: Annotated[int, Field(description="Maximum tokens to generate.")] = 4096,
    persona: Annotated[
        Persona | None,
        Field(
            description=(
                "Optional persona contract for this call: {name, input_schema, "
                "output_schema}. name is required and non-empty; the server is "
                "stateless and keeps no persona registry. When input_schema is "
                "given, the request arguments (prompt, model, temperature, "
                "max_tokens, previous_response_id) are validated against it "
                "before anything is sent. "
                "When output_schema is given, the host's completion text must "
                "parse as JSON validating against it."
            )
        ),
    ] = None,
    previous_response_id: Annotated[
        str | None,
        Field(
            description=(
                "Optional session cursor for host-owned continuations. '' mints "
                "a cursor over the current prompt and completion, returned as "
                'JSON \'{"completion": "<text>", "response_id": '
                '"cur_<...>"}\' instead of plain text; when omitted the tool '
                "returns plain text exactly as before. generate has no "
                "structured history channel, so a 'cur_' cursor can never be "
                "verified here and fails loud rather than silently skipping "
                "the check."
            )
        ),
    ] = None,
) -> str | InputRequiredResult:
    """Generate a completion from a raw prompt with the host agent's model."""
    persona = _validate_persona(persona)
    history = [{"role": "user", "content": prompt}]
    _check_previous_response_id("generate", model, history, previous_response_id, verifiable=False)
    _validate_persona_request(
        persona,
        {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "previous_response_id": previous_response_id,
        },
    )
    params = _sampling_params(
        [SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
        None,
        temperature,
        max_tokens,
        model,
    )
    completed = _finish_persona(persona, await _complete(ctx, params))
    if isinstance(completed, InputRequiredResult):
        return completed
    if previous_response_id is None:
        return completed
    cursor = _mint_cursor(
        "generate", model, [*history, {"role": "assistant", "content": completed}]
    )
    return _cursor_envelope(completed, cursor)


@mcp.tool(
    description="List the models this server offers. There is exactly one: agent-default, an alias for the host agent's own model."
)
async def list_models() -> str:
    """List available models (the host's own model, aliased)."""
    return json.dumps(
        [
            {
                "id": "agent-default",
                "owned_by": "host-agent",
                "description": (
                    "Alias for the MCP host's own model, reached via sampling. No API key required."
                ),
            }
        ]
    )


def main() -> None:
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
