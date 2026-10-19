"""agent-as-model: a true MCP server where the agent IS the model and the API key.

Install on any MCP host (Claude Code, Cursor, Windsurf, ...). The server
exposes product-style completion tools; every call is fulfilled by sampling
the host agent's own LLM. No API keys exist anywhere in this system.

Tools:
  chat_complete   OpenAI chat-completions-shaped request -> completion text
  messages_create Anthropic messages-shaped request -> completion text
  list_models     the model this server offers (the host's own model, aliased)

Both completion tools accept an optional output_schema (a JSON Schema object).
When given, the sampling request carries exactly one tool whose input schema
is that schema, marked as the required tool choice (the toolChoice-required
trick): the host must answer by invoking the tool, and the server validates
the tool-use input against the schema before returning it as JSON text.
Invalid host output fails loudly instead of being returned silently.

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
import json
from typing import Annotated

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
    Tool,
    ToolChoice,
    ToolUseContent,
)
from mcp_types.version import is_version_at_least
from pydantic import Field

from agent_as_model import __version__
from agent_as_model.protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    ChatMessage,
    anthropic_to_sampling,
    extract_text,
    openai_to_sampling,
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


def _require_object_schema(output_schema: object) -> dict:
    """Validate the caller's output_schema before it travels anywhere.

    The single required tool carries it as its input schema, so it must be a
    JSON Schema object with root type "object", and it must itself be a valid
    schema. A bad schema is the caller's bug: fail here, loudly, not mid-flow.
    """
    if not isinstance(output_schema, dict) or output_schema.get("type") != "object":
        raise ToolError(
            "output_schema must be a JSON Schema object with root type 'object' "
            f"(got {type(output_schema).__name__}); the single required sampling tool "
            "carries it as its input schema"
        )
    try:
        jsonschema.Draft202012Validator.check_schema(output_schema)
    except jsonschema.SchemaError as exc:
        raise ToolError(f"output_schema is not a valid JSON Schema: {exc.message}") from exc
    return output_schema


def _sampling_params(
    messages: list[SamplingMessage],
    system_prompt: str | None,
    temperature: float | None,
    max_tokens: int,
    model: str,
    output_schema: dict | None = None,
) -> CreateMessageRequestParams:
    """Build the request payload every era path carries: messages, system
    prompt, temperature, max_tokens, and the model hint.

    When output_schema is given, the request carries exactly one tool whose
    input schema is that schema, marked as the required tool choice: the
    toolChoice-required trick for structured output. The host must answer by
    invoking the tool, so its tool-use input is the structured completion.
    """
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    if output_schema is not None:
        tools = [
            Tool(
                name=_STRUCTURED_OUTPUT_TOOL_NAME,
                description=(
                    "Return the completion as structured data matching this "
                    "tool's input schema. You must call this tool."
                ),
                input_schema=_require_object_schema(output_schema),
            )
        ]
        tool_choice = ToolChoice(mode="required")
    return CreateMessageRequestParams(
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model_preferences=ModelPreferences(hints=[ModelHint(name=model)]),
        tools=tools,
        tool_choice=tool_choice,
    )


async def _legacy_sample(
    ctx: Context, params: CreateMessageRequestParams, output_schema: dict | None = None
) -> str:
    """Direct sampling/createMessage for pre-2026-07-28 connections.

    The era gate forbids this on modern connections; the sampling capability
    gate fails loud when the host did not declare it. With an output_schema
    the required-tool response is extracted and validated against it.
    """
    _assert_legacy_sampling_allowed(ctx.protocol_version)
    if ctx.session.can_send_request:
        _require_sampling_capability(ctx)
    if output_schema is None:
        result = await ctx.session.send_request(
            CreateMessageRequest(params=params),
            CreateMessageResult,
            metadata=ServerMessageMetadata(related_request_id=ctx.request_id),
        )
        return extract_text(result)
    with_tools = await ctx.session.send_request(
        CreateMessageRequest(params=params),
        CreateMessageResultWithTools,
        metadata=ServerMessageMetadata(related_request_id=ctx.request_id),
    )
    output = _extract_structured_output(with_tools)
    _validate_structured_output(output, output_schema)
    return json.dumps(output, sort_keys=True)


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


def _validate_structured_output(output: dict, output_schema: dict) -> None:
    """Gate: the host's structured completion must satisfy the caller's schema.

    Raises ToolError naming the exact mismatch. Bad output is never returned
    silently, and the schema is never dropped.
    """
    try:
        jsonschema.validate(instance=output, schema=output_schema)
    except jsonschema.ValidationError as exc:
        location = (
            "/" + "/".join(str(part) for part in exc.absolute_path)
            if exc.absolute_path
            else "(root)"
        )
        raise ToolError(
            f"Host completion failed output-schema validation at {location}: {exc.message}; "
            "the completion was rejected rather than returned as-is"
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


def _mrtr_completion_result(ctx: Context, output_schema: dict | None = None) -> str | None:
    """Second MRTR round: the host's fulfilled completion, or None on round one.

    With an output_schema the host must answer through the required tool: the
    tool-use input is extracted and validated, and a plain-text answer is a
    contract violation, not a completion.
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
    if isinstance(response, CreateMessageResult | CreateMessageResultWithTools):
        return extract_text(response)
    raise ToolError(
        f"MRTR completion round received a response of the wrong kind for "
        f"{_SAMPLE_INPUT_KEY!r}: {type(response).__name__}; "
        "expected the host's CreateMessageResult"
    )


async def _complete(
    ctx: Context, params: CreateMessageRequestParams, output_schema: dict | None = None
) -> str | InputRequiredResult:
    """Era dispatch: MRTR InputRequiredResult on 2026-07-28+, direct sampling before."""
    if _is_modern_era(ctx.protocol_version):
        completed = _mrtr_completion_result(ctx, output_schema)
        if completed is None:
            return _mrtr_completion_request(params)
        return completed
    return await _legacy_sample(ctx, params, output_schema)


@mcp.tool(
    description=(
        "Generate a chat completion from an OpenAI chat-completions-shaped request. "
        "The completion is produced by the host agent's own model: direct MCP sampling "
        "on pre-2026-07-28 connections, an MRTR InputRequiredResult carrying the "
        "request payload on 2026-07-28+. "
        "When output_schema is given, the request carries exactly one required tool "
        "whose input schema is that schema, and the validated structured completion "
        "is returned as JSON text. "
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
) -> str | InputRequiredResult:
    """Complete a chat with the host agent's model (OpenAI request shape)."""
    system_prompt, sampling_messages = openai_to_sampling(messages)
    params = _sampling_params(
        sampling_messages, system_prompt, temperature, max_tokens, model, output_schema
    )
    return await _complete(ctx, params, output_schema)


@mcp.tool(
    description=(
        "Generate a completion from an Anthropic messages-shaped request. "
        "The completion is produced by the host agent's own model: direct MCP sampling "
        "on pre-2026-07-28 connections, an MRTR InputRequiredResult carrying the "
        "request payload on 2026-07-28+. "
        "When output_schema is given, the request carries exactly one required tool "
        "whose input schema is that schema, and the validated structured completion "
        "is returned as JSON text. "
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
) -> str | InputRequiredResult:
    """Complete a conversation with the host agent's model (Anthropic request shape)."""
    system_prompt, sampling_messages = anthropic_to_sampling(messages, system)
    params = _sampling_params(
        sampling_messages, system_prompt, temperature, max_tokens, model, output_schema
    )
    return await _complete(ctx, params, output_schema)


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
