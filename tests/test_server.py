"""End-to-end tests: an in-process MCP client drives the real server tools.

The scripted sampling callback stands in for the host agent's own model, so
these tests exercise the full loop a real platform install would run:
tools/call -> completion request -> canned host completion -> tool result.

Both protocol eras are covered. On the legacy era (direct sampling/
createMessage) the tool round-trips through the SDK sampling API. On the
modern era (2026-07-28+) the tool never touches the sampling API: it returns
an MRTR InputRequiredResult carrying the full request payload, and the
completion comes back on the retry round.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anyio
import pytest
from mcp import types
from mcp.client import Client
from mcp.client.session import ClientRequestContext
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError

import agent_as_model.server as agent_server

SERVER = agent_server.mcp


def _run(fn):
    return anyio.run(fn)


async def _canned_sampler(
    context: ClientRequestContext, params: types.CreateMessageRequestParams
) -> types.CreateMessageResult:
    """Stand in for the host agent: echo the wire params as the completion."""
    seen = {
        "system_prompt": params.system_prompt,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "messages": [(m.role, m.content.text) for m in params.messages],
        "hints": [h.name for h in params.model_preferences.hints]
        if params.model_preferences and params.model_preferences.hints
        else [],
    }
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="CANNED:" + json.dumps(seen, sort_keys=True)),
        model="test-host-model",
        stop_reason="endTurn",
    )


def _tool_text(result) -> str:
    assert len(result.content) == 1
    block = result.content[0]
    assert block.type == "text"
    return block.text


def test_tools_listed_and_context_param_hidden():
    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert {"chat_complete", "messages_create", "list_models"} <= names
            for name in ("chat_complete", "messages_create"):
                tool = next(t for t in tools.tools if t.name == name)
                props = tool.input_schema.get("properties", {})
                assert "ctx" not in props
                assert "completion" not in props

    _run(go)


def test_chat_complete_modern_era():
    """Modern era: the client drives the MRTR loop and the tool returns the completion."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "chat_complete",
                {
                    "messages": [
                        {"role": "system", "content": "You are terse."},
                        {"role": "user", "content": "Say hi."},
                    ],
                    "model": "agent-default",
                    "temperature": 0.5,
                    "max_tokens": 100,
                },
            )
            seen = json.loads(_tool_text(result).removeprefix("CANNED:"))
            assert seen["system_prompt"] == "You are terse."
            assert seen["messages"] == [["user", "Say hi."]]
            assert seen["max_tokens"] == 100
            assert seen["temperature"] == 0.5
            assert seen["hints"] == ["agent-default"]

    _run(go)


def test_chat_complete_modern_era_input_required_payload():
    """Modern era, round one: the tool returns an InputRequiredResult whose
    payload carries everything the host needs: the translated messages, the
    system prompt, temperature, max_tokens, and the model hint."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            first = await client.session.call_tool(
                "chat_complete",
                {
                    "messages": [
                        {"role": "system", "content": "You are terse."},
                        {"role": "user", "content": "Say hi."},
                    ],
                    "model": "agent-default",
                    "temperature": 0.5,
                    "max_tokens": 100,
                },
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)
            assert set(first.input_requests) == {"sample"}
            request = first.input_requests["sample"]
            assert isinstance(request, types.CreateMessageRequest)
            params = request.params
            assert params.system_prompt == "You are terse."
            assert [(m.role, m.content.text) for m in params.messages] == [("user", "Say hi.")]
            assert params.temperature == 0.5
            assert params.max_tokens == 100
            assert [h.name for h in params.model_preferences.hints] == ["agent-default"]

            # request_state is sealed by the server kernel on the wire (opaque to
            # the client by spec); it echoes the same payload on the retry round.
            assert first.request_state

    _run(go)


def test_messages_create_modern_era_input_required_payload():
    """Modern era, round one for the Anthropic shape: same payload contract."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            first = await client.session.call_tool(
                "messages_create",
                {
                    "model": "agent-default",
                    "system": "Be brief.",
                    "messages": [{"role": "user", "content": "Summarize this."}],
                    "max_tokens": 50,
                },
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)
            params = first.input_requests["sample"].params
            assert params.system_prompt == "Be brief."
            assert [(m.role, m.content.text) for m in params.messages] == [
                ("user", "Summarize this.")
            ]
            assert params.max_tokens == 50

    _run(go)


def test_chat_complete_legacy_era():
    """Legacy era: direct sampling/createMessage through the SDK sampling API."""

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "chat_complete",
                {"messages": [{"role": "user", "content": "Ping"}]},
            )
            seen = json.loads(_tool_text(result).removeprefix("CANNED:"))
            assert seen["messages"] == [["user", "Ping"]]
            assert seen["system_prompt"] is None

    _run(go)


def test_messages_create_anthropic_shape():
    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "messages_create",
                {
                    "model": "agent-default",
                    "system": "Be brief.",
                    "messages": [{"role": "user", "content": "Summarize this."}],
                    "max_tokens": 50,
                },
            )
            seen = json.loads(_tool_text(result).removeprefix("CANNED:"))
            assert seen["system_prompt"] == "Be brief."
            assert seen["messages"] == [["user", "Summarize this."]]

    _run(go)


def test_list_models():
    async def go():
        async with Client(SERVER) as client:
            result = await client.call_tool("list_models", {})
            models = json.loads(_tool_text(result))
            assert models[0]["id"] == "agent-default"

    _run(go)


def test_default_max_tokens_reaches_host():
    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            for tool_name, args in (
                ("chat_complete", {"messages": [{"role": "user", "content": "Hi"}]}),
                ("messages_create", {"messages": [{"role": "user", "content": "Hi"}]}),
            ):
                result = await client.call_tool(tool_name, args)
                seen = json.loads(_tool_text(result).removeprefix("CANNED:"))
                assert seen["max_tokens"] == 4096

    _run(go)


def test_main_runs_stdio(monkeypatch):
    calls = []

    async def fake_run_stdio_async():
        calls.append(True)

    monkeypatch.setattr(SERVER, "run_stdio_async", fake_run_stdio_async)
    agent_server.main()
    assert calls == [True]


def test_missing_sampling_capability_fails_clearly():
    """Legacy path: a host without the sampling capability fails loud with
    MISSING_REQUIRED_CLIENT_CAPABILITY, not a generic sampling refusal."""

    async def go():
        async with Client(SERVER, mode="legacy") as client:  # no sampling_callback
            with pytest.raises(MCPError, match="(?i)sampling") as excinfo:
                await client.call_tool(
                    "chat_complete", {"messages": [{"role": "user", "content": "Hi"}]}
                )
            assert excinfo.value.code == types.MISSING_REQUIRED_CLIENT_CAPABILITY

    _run(go)


def test_require_sampling_capability_unit():
    """The capability gate raises MCPError when sampling is undeclared and
    passes silently when it is declared."""
    bare = SimpleNamespace(client_capabilities=types.ClientCapabilities())
    with pytest.raises(MCPError) as excinfo:
        agent_server._require_sampling_capability(bare)
    assert excinfo.value.code == types.MISSING_REQUIRED_CLIENT_CAPABILITY

    declared = SimpleNamespace(
        client_capabilities=types.ClientCapabilities(sampling=types.SamplingCapability())
    )
    assert agent_server._require_sampling_capability(declared) is None


def test_legacy_sample_refuses_modern_era():
    """_legacy_sample itself enforces the era gate before touching the session."""
    params = agent_server._sampling_params([], None, None, 10, "agent-default")

    async def go():
        with pytest.raises(ToolError, match="MRTR"):
            await agent_server._legacy_sample(
                SimpleNamespace(protocol_version="2026-07-28"), params
            )

    _run(go)


def test_mrtr_completion_request_state_payload():
    """request_state carries the full payload as wire-aliased JSON with no nulls."""
    params = agent_server._sampling_params(
        [types.SamplingMessage(role="user", content=types.TextContent(type="text", text="Hi"))],
        "Be terse.",
        0.5,
        100,
        "agent-default",
    )
    result = agent_server._mrtr_completion_request(params)
    state = json.loads(result.request_state)
    assert state["systemPrompt"] == "Be terse."
    assert state["messages"] == [{"role": "user", "content": {"type": "text", "text": "Hi"}}]
    assert state["temperature"] == 0.5
    assert state["maxTokens"] == 100
    assert state["modelPreferences"]["hints"] == [{"name": "agent-default"}]

    bare = agent_server._mrtr_completion_request(
        agent_server._sampling_params([], None, None, 10, "agent-default")
    )
    bare_state = json.loads(bare.request_state)
    assert "systemPrompt" not in bare_state
    assert "temperature" not in bare_state


def test_mrtr_completion_result_round_one_and_wrong_kind():
    """Round one (no responses) yields None; a wrong-kind response raises."""
    assert agent_server._mrtr_completion_result(SimpleNamespace(input_responses=None)) is None
    assert agent_server._mrtr_completion_result(SimpleNamespace(input_responses={})) is None
    wrong = SimpleNamespace(input_responses={"sample": types.TextContent(type="text", text="nope")})
    with pytest.raises(ToolError, match="wrong kind"):
        agent_server._mrtr_completion_result(wrong)


def test_era_gate():
    assert agent_server._is_modern_era("2026-07-28")
    assert not agent_server._is_modern_era("2025-11-25")
    assert not agent_server._is_modern_era(None)
    # Unknown versions are treated conservatively as legacy, matching the SDK.
    assert not agent_server._is_modern_era("2027-01-01")


def test_legacy_sampling_gate_forbids_modern_era():
    """The gate diagnostic names the era and the MRTR alternative."""
    with pytest.raises(ToolError, match="2026-07-28") as excinfo:
        agent_server._assert_legacy_sampling_allowed("2026-07-28")
    message = str(excinfo.value)
    assert "MRTR" in message
    assert "InputRequiredResult" in message
    # Legacy versions pass the gate without raising.
    agent_server._assert_legacy_sampling_allowed("2025-11-25")
    agent_server._assert_legacy_sampling_allowed(None)


_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def _structured_sampler_factory(output, schema=_OUTPUT_SCHEMA):
    """Stand in for a host that honors the required tool: assert the single
    required tool is in the sampling params, then answer through it."""

    async def sampler(context, params):
        assert params.tools is not None and len(params.tools) == 1
        tool = params.tools[0]
        assert tool.name == "structured_output"
        assert tool.input_schema == schema
        assert params.tool_choice is not None
        assert params.tool_choice.mode == "required"
        return types.CreateMessageResultWithTools(
            role="assistant",
            content=[
                types.ToolUseContent(
                    type="tool_use",
                    name="structured_output",
                    id="tool-1",
                    input=output,
                )
            ],
            model="test-host-model",
            stop_reason="toolUse",
        )

    return sampler


def _tool_use_result(output):
    return types.CreateMessageResultWithTools(
        role="assistant",
        content=[
            types.ToolUseContent(
                type="tool_use", name="structured_output", id="tool-1", input=output
            )
        ],
        model="test-host-model",
        stop_reason="toolUse",
    )


def test_output_schema_exposed_in_tool_schemas():
    """Both tools advertise output_schema; the ctx param stays hidden."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            for name in ("chat_complete", "messages_create"):
                tool = next(t for t in tools.tools if t.name == name)
                props = tool.input_schema.get("properties", {})
                assert "output_schema" in props
                assert "ctx" not in props

    _run(go)


def test_output_schema_legacy_era_single_required_tool():
    """Legacy era: with output_schema the sampling params carry exactly one
    required tool whose input schema is the schema; the validated structured
    completion returns as JSON text. Both tools."""

    async def go():
        sampler = _structured_sampler_factory({"answer": "yes", "confidence": 0.9})
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            for tool_name, args in (
                (
                    "chat_complete",
                    {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "output_schema": _OUTPUT_SCHEMA,
                    },
                ),
                (
                    "messages_create",
                    {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "output_schema": _OUTPUT_SCHEMA,
                    },
                ),
            ):
                result = await client.call_tool(tool_name, args)
                assert json.loads(_tool_text(result)) == {"answer": "yes", "confidence": 0.9}

    _run(go)


def test_output_schema_modern_era_payload_carries_required_tool():
    """Modern era, round one: the InputRequiredResult payload carries the
    single required tool in both the input request and the request_state."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            first = await client.session.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "output_schema": _OUTPUT_SCHEMA,
                },
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)
            params = first.input_requests["sample"].params
            assert len(params.tools) == 1
            assert params.tools[0].name == "structured_output"
            assert params.tools[0].input_schema == _OUTPUT_SCHEMA
            assert params.tool_choice.mode == "required"

            # request_state is sealed by the server kernel on the wire (opaque to
            # the client by spec); its raw shape is pinned at the unit level below.
            assert first.request_state

    _run(go)


def test_output_schema_modern_era_request_state_carries_required_tool():
    """Unit level: the MRTR request_state JSON carries the single required tool."""
    params = agent_server._sampling_params(
        [types.SamplingMessage(role="user", content=types.TextContent(type="text", text="Hi"))],
        None,
        None,
        100,
        "agent-default",
        _OUTPUT_SCHEMA,
    )
    result = agent_server._mrtr_completion_request(params)
    state = json.loads(result.request_state)
    assert len(state["tools"]) == 1
    assert state["tools"][0]["name"] == "structured_output"
    assert state["tools"][0]["inputSchema"] == _OUTPUT_SCHEMA
    assert state["toolChoice"] == {"mode": "required"}


def test_output_schema_modern_era_full_loop():
    """Modern era, full MRTR loop: the host's tool-use answer is extracted,
    validated, and returned as JSON text."""

    async def go():
        sampler = _structured_sampler_factory({"answer": "structured hi"})
        async with Client(SERVER, sampling_callback=sampler) as client:
            result = await client.call_tool(
                "messages_create",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "output_schema": _OUTPUT_SCHEMA,
                },
            )
            assert json.loads(_tool_text(result)) == {"answer": "structured hi"}

    _run(go)


def test_output_schema_validation_failure_raises_legacy():
    """Legacy era: host output violating the schema fails loud, naming the
    mismatch, instead of being returned silently."""

    async def go():
        sampler = _structured_sampler_factory({"answer": "ok", "confidence": "not-a-number"})
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "output_schema": _OUTPUT_SCHEMA,
                },
            )
            assert result.is_error
            text = _tool_text(result)
            assert "output-schema validation" in text
            assert "confidence" in text

    _run(go)


def test_mrtr_completion_result_schema_success_and_failure():
    """Modern round two: valid structured output returns as JSON; invalid
    output raises naming the mismatch."""
    good = SimpleNamespace(input_responses={"sample": _tool_use_result({"answer": "ok"})})
    assert json.loads(agent_server._mrtr_completion_result(good, _OUTPUT_SCHEMA)) == {
        "answer": "ok"
    }

    bad = SimpleNamespace(
        input_responses={"sample": _tool_use_result({"answer": "ok", "confidence": "x"})}
    )
    with pytest.raises(ToolError, match="output-schema validation") as excinfo:
        agent_server._mrtr_completion_result(bad, _OUTPUT_SCHEMA)
    assert "confidence" in str(excinfo.value)


def test_mrtr_completion_result_schema_rejects_plain_text():
    """Modern round two with a schema: a plain-text host answer is a contract
    violation, not a completion."""
    plain = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="hi"),
        model="test-host-model",
        stop_reason="endTurn",
    )
    ctx = SimpleNamespace(input_responses={"sample": plain})
    with pytest.raises(ToolError, match="not honored"):
        agent_server._mrtr_completion_result(ctx, _OUTPUT_SCHEMA)


def test_extract_structured_output_requires_tool_use():
    """A WithTools response with no structured_output tool use is rejected."""
    text_only = types.CreateMessageResultWithTools(
        role="assistant",
        content=[types.TextContent(type="text", text="no tools here")],
        model="test-host-model",
        stop_reason="endTurn",
    )
    with pytest.raises(ToolError, match="no 'structured_output' tool use"):
        agent_server._extract_structured_output(text_only)


def test_output_schema_returns_canonical_sorted_json():
    """The validated completion is returned with deterministically sorted keys,
    whatever order the host's tool-use input used. Both era paths."""
    schema = {
        "type": "object",
        "properties": {"zebra": {"type": "integer"}, "apple": {"type": "integer"}},
    }
    sampler = _structured_sampler_factory({"zebra": 1, "apple": 2}, schema)

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "chat_complete",
                {"messages": [{"role": "user", "content": "Hi"}], "output_schema": schema},
            )
            assert _tool_text(result) == '{"apple": 2, "zebra": 1}'

    _run(go)

    ctx = SimpleNamespace(input_responses={"sample": _tool_use_result({"zebra": 1, "apple": 2})})
    assert agent_server._mrtr_completion_result(ctx, schema) == '{"apple": 2, "zebra": 1}'


def test_extract_structured_output_returns_input():
    result = _tool_use_result({"answer": "x", "confidence": 1.0})
    assert agent_server._extract_structured_output(result) == {"answer": "x", "confidence": 1.0}


def test_require_object_schema_rejects_bad_schemas():
    """Non-object roots, non-dicts, and invalid JSON Schemas fail at the gate."""
    with pytest.raises(ToolError, match="root type 'object'"):
        agent_server._require_object_schema({"type": "array"})
    with pytest.raises(ToolError, match="root type 'object'"):
        agent_server._require_object_schema(["not", "a", "dict"])
    with pytest.raises(ToolError, match="not a valid JSON Schema"):
        agent_server._require_object_schema(
            {"type": "object", "properties": {"x": {"type": "bogus"}}}
        )
    assert agent_server._require_object_schema(_OUTPUT_SCHEMA) == _OUTPUT_SCHEMA


def test_sampling_params_without_schema_carries_no_tools():
    """The default path is untouched: no tools, no tool choice, exactly the
    pinned payload shape."""
    params = agent_server._sampling_params([], None, None, 10, "agent-default")
    assert params.tools is None
    assert params.tool_choice is None


def test_sampling_params_with_schema_carries_single_required_tool():
    """_sampling_params builds the exact toolChoice-required shape."""
    params = agent_server._sampling_params([], None, None, 10, "agent-default", _OUTPUT_SCHEMA)
    assert params.tools is not None and len(params.tools) == 1
    assert params.tools[0].name == "structured_output"
    assert params.tools[0].input_schema == _OUTPUT_SCHEMA
    assert params.tool_choice is not None and params.tool_choice.mode == "required"


# ---------------------------------------------------------------------------
# Item 4: function-calling / tool-use passthrough
# ---------------------------------------------------------------------------

_TOOL_DEFS = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "search",
        "description": "Search the web.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
]

_TOOLS_CAPABILITY = types.SamplingCapability(tools=types.SamplingToolsCapability())
_NO_TOOLS_CAPABILITY = types.SamplingCapability()
_MINIMAL_TOOLS = [
    types.Tool(name="t", input_schema={"type": "object"}),
]


def _tool_call_sampler_factory(tool_calls, text="", expected_choice="auto"):
    """Stand in for a tools-capable host: assert the forwarded tools and
    tool choice are in the sampling params, then answer with tool calls."""

    async def sampler(context, params):
        assert params.tools is not None
        assert [t.name for t in params.tools] == ["get_weather", "search"]
        assert params.tools[0].description == "Get the weather for a city."
        assert params.tools[0].input_schema["properties"]["city"] == {"type": "string"}
        assert params.tool_choice is not None
        assert params.tool_choice.mode == expected_choice
        content = [types.TextContent(type="text", text=text)] if text else []
        for call in tool_calls:
            content.append(
                types.ToolUseContent(
                    type="tool_use", name=call["name"], id=call["id"], input=call["input"]
                )
            )
        return types.CreateMessageResultWithTools(
            role="assistant",
            content=content,
            model="test-host-model",
            stop_reason="toolUse",
        )

    return sampler


async def _no_tools_sampler(context, params):
    """Stand in for a host without tools support: the request must arrive
    tool-free, and the answer is plain text."""
    assert params.tools is None
    assert params.tool_choice is None
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="plain answer"),
        model="test-host-model",
        stop_reason="endTurn",
    )


def test_tools_and_tool_choice_exposed_in_tool_schemas():
    """Both tools advertise tools and tool_choice; the ctx param stays hidden."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            for name in ("chat_complete", "messages_create"):
                tool = next(t for t in tools.tools if t.name == name)
                props = tool.input_schema.get("properties", {})
                assert "tools" in props
                assert "tool_choice" in props
                assert "ctx" not in props
                defs = tool.input_schema.get("$defs", {})
                item_props = defs.get("ToolDefinition", {}).get("properties", {})
                assert {"name", "description", "input_schema"} <= set(item_props)

    _run(go)


def test_tools_forwarded_legacy_params_and_result_shape():
    """Legacy era: tools and the tool choice reach the sampling params, and the
    host's tool calls come back in the documented JSON shape. Both tools."""

    tool_calls = [
        {"name": "get_weather", "id": "call-1", "input": {"city": "Naga"}},
        {"name": "search", "id": "call-2", "input": {"query": "mcp sampling"}},
    ]

    async def go():
        sampler = _tool_call_sampler_factory(tool_calls, text="thinking out loud")
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=sampler,
            sampling_capabilities=_TOOLS_CAPABILITY,
        ) as client:
            for tool_name, args in (
                ("chat_complete", {"messages": [{"role": "user", "content": "Hi"}]}),
                ("messages_create", {"messages": [{"role": "user", "content": "Hi"}]}),
            ):
                result = await client.call_tool(
                    tool_name, {**args, "tools": _TOOL_DEFS, "tool_choice": "auto"}
                )
                assert not result.is_error
                payload = json.loads(_tool_text(result))
                assert payload["text"] == "thinking out loud"
                assert payload["tool_calls"] == [
                    {"id": "call-1", "input": {"city": "Naga"}, "name": "get_weather"},
                    {
                        "id": "call-2",
                        "input": {"query": "mcp sampling"},
                        "name": "search",
                    },
                ]

    _run(go)


def test_tool_choice_forwarding():
    """Legacy era: explicit 'required' and 'none' tool choices reach the host."""

    async def go():
        for choice in ("required", "none"):
            sampler = _tool_call_sampler_factory([], text="just text", expected_choice=choice)
            async with Client(
                SERVER,
                mode="legacy",
                sampling_callback=sampler,
                sampling_capabilities=_TOOLS_CAPABILITY,
            ) as client:
                result = await client.call_tool(
                    "chat_complete",
                    {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "tools": _TOOL_DEFS,
                        "tool_choice": choice,
                    },
                )
                assert _tool_text(result) == "just text"

    _run(go)


def test_tools_modern_era_payload_carries_tools():
    """Modern era, round one: the InputRequiredResult payload carries the
    forwarded tools and the tool choice, defaulting to auto."""

    async def go():
        async with Client(
            SERVER,
            sampling_callback=_canned_sampler,
            sampling_capabilities=_TOOLS_CAPABILITY,
        ) as client:
            first = await client.session.call_tool(
                "chat_complete",
                {"messages": [{"role": "user", "content": "Hi"}], "tools": _TOOL_DEFS},
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)
            params = first.input_requests["sample"].params
            assert [t.name for t in params.tools] == ["get_weather", "search"]
            assert params.tool_choice.mode == "auto"

            # request_state is sealed by the server kernel on the wire (opaque to
            # the client by spec); its raw shape is pinned at the unit level below.
            assert first.request_state

    _run(go)


def test_tools_modern_era_request_state_carries_tools():
    """Unit level: the MRTR request_state JSON carries the forwarded tools and
    the tool choice."""
    tools = agent_server._validate_tool_definitions(
        [agent_server.ToolDefinition(**d) for d in _TOOL_DEFS]
    )
    params = agent_server._sampling_params(
        [types.SamplingMessage(role="user", content=types.TextContent(type="text", text="Hi"))],
        None,
        None,
        100,
        "agent-default",
        tools=tools,
        tool_choice="required",
    )
    result = agent_server._mrtr_completion_request(params)
    state = json.loads(result.request_state)
    assert [t["name"] for t in state["tools"]] == ["get_weather", "search"]
    assert state["tools"][0]["inputSchema"]["properties"] == {"city": {"type": "string"}}
    assert state["toolChoice"] == {"mode": "required"}


def test_tools_modern_era_full_loop():
    """Modern era, full MRTR loop: the host's tool calls return in the
    documented JSON shape."""

    tool_calls = [{"name": "get_weather", "id": "call-9", "input": {"city": "Manila"}}]

    async def go():
        sampler = _tool_call_sampler_factory(tool_calls)
        async with Client(
            SERVER,
            sampling_callback=sampler,
            sampling_capabilities=_TOOLS_CAPABILITY,
        ) as client:
            result = await client.call_tool(
                "messages_create",
                {"messages": [{"role": "user", "content": "Hi"}], "tools": _TOOL_DEFS},
            )
            payload = json.loads(_tool_text(result))
            assert payload["tool_calls"] == [
                {"id": "call-9", "input": {"city": "Manila"}, "name": "get_weather"}
            ]
            assert payload["text"] == ""

    _run(go)


def test_capability_absent_degrades_to_plain_text_legacy(caplog):
    """Legacy era: a host without sampling.tools gets a tool-free request, the
    degradation is logged explicitly, and the caller gets plain text."""

    async def go():
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=_no_tools_sampler,
            sampling_capabilities=_NO_TOOLS_CAPABILITY,
        ) as client:
            with caplog.at_level("WARNING", logger="agent_as_model"):
                result = await client.call_tool(
                    "chat_complete",
                    {"messages": [{"role": "user", "content": "Hi"}], "tools": _TOOL_DEFS},
                )
            assert _tool_text(result) == "plain answer"

    _run(go)
    assert any("sampling.tools" in record.message for record in caplog.records)


def test_capability_absent_degrades_to_plain_text_modern(caplog):
    """Modern era, round one: without sampling.tools the MRTR payload carries
    no tools, the degradation is logged, and the shape stays intact."""

    async def go():
        async with Client(
            SERVER,
            sampling_callback=_no_tools_sampler,
            sampling_capabilities=_NO_TOOLS_CAPABILITY,
        ) as client:
            with caplog.at_level("WARNING", logger="agent_as_model"):
                first = await client.session.call_tool(
                    "chat_complete",
                    {"messages": [{"role": "user", "content": "Hi"}], "tools": _TOOL_DEFS},
                    allow_input_required=True,
                )
            assert isinstance(first, types.InputRequiredResult)
            params = first.input_requests["sample"].params
            assert params.tools is None
            assert params.tool_choice is None

    _run(go)
    assert any("sampling.tools" in record.message for record in caplog.records)


def test_tools_and_output_schema_conflict():
    """Providing both tools and output_schema fails loud, naming both. Both tools."""

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            for tool_name, args in (
                ("chat_complete", {"messages": [{"role": "user", "content": "Hi"}]}),
                ("messages_create", {"messages": [{"role": "user", "content": "Hi"}]}),
            ):
                result = await client.call_tool(
                    tool_name,
                    {**args, "tools": _TOOL_DEFS, "output_schema": _OUTPUT_SCHEMA},
                )
                assert result.is_error
                text = _tool_text(result)
                assert "tools" in text and "output_schema" in text

    _run(go)


def test_validate_tool_definitions():
    """Malformed tool definitions are rejected with clear errors before
    reaching the host; valid ones convert to MCP tools."""
    good = agent_server._validate_tool_definitions(
        [agent_server.ToolDefinition(**d) for d in _TOOL_DEFS]
    )
    assert [t.name for t in good] == ["get_weather", "search"]
    assert good[0].input_schema == _TOOL_DEFS[0]["input_schema"]
    assert agent_server._validate_tool_definitions(None) is None

    def make(**overrides):
        base = dict(_TOOL_DEFS[0])
        base.update(overrides)
        return agent_server.ToolDefinition(**base)

    with pytest.raises(ToolError, match="non-empty string"):
        agent_server._validate_tool_definitions([make(name="")])
    with pytest.raises(ToolError, match="non-empty string"):
        agent_server._validate_tool_definitions([make(name="   ")])
    with pytest.raises(ToolError, match="duplicate tool name"):
        agent_server._validate_tool_definitions([make(), make()])
    with pytest.raises(ToolError, match=r"tools\[0\]\.input_schema.*root type 'object'"):
        agent_server._validate_tool_definitions([make(input_schema={"type": "array"})])
    with pytest.raises(ToolError, match="not a valid JSON Schema"):
        agent_server._validate_tool_definitions(
            [make(input_schema={"type": "object", "properties": {"x": {"type": "bogus"}}})]
        )
    with pytest.raises(ToolError, match="non-empty list"):
        agent_server._validate_tool_definitions([])


def test_format_tool_calls_shape():
    """The passthrough shape: mixed text plus tool calls become sorted-key
    JSON; a plain-text answer returns text; unknown blocks fail loud."""
    mixed = types.CreateMessageResultWithTools(
        role="assistant",
        content=[
            types.TextContent(type="text", text="here you go"),
            types.ToolUseContent(type="tool_use", name="search", id="c-1", input={"query": "x"}),
        ],
        model="m",
        stop_reason="toolUse",
    )
    # Keys are sorted for a deterministic wire shape (sort_keys=True); the raw
    # string pins the order so a sort_keys flip is caught.
    assert agent_server._format_tool_calls(mixed) == (
        '{"text": "here you go", '
        '"tool_calls": [{"id": "c-1", "input": {"query": "x"}, "name": "search"}]}'
    )

    text_only = types.CreateMessageResultWithTools(
        role="assistant",
        content=[types.TextContent(type="text", text="just words")],
        model="m",
        stop_reason="endTurn",
    )
    assert agent_server._format_tool_calls(text_only) == "just words"

    weird = types.CreateMessageResultWithTools(
        role="assistant",
        content=[
            types.ImageContent(
                type="image",
                data="AAAA",
                mimeType="image/png",
            )
        ],
        model="m",
        stop_reason="endTurn",
    )
    with pytest.raises(ToolError, match="unsupported content block"):
        agent_server._format_tool_calls(weird)


def test_mrtr_completion_result_tool_calls():
    """Modern round two with tools: tool calls return the JSON shape; a
    plain-text host answer returns text as before."""
    with_tools = SimpleNamespace(
        input_responses={
            "sample": types.CreateMessageResultWithTools(
                role="assistant",
                content=[
                    types.ToolUseContent(
                        type="tool_use", name="search", id="c-2", input={"query": "y"}
                    )
                ],
                model="m",
                stop_reason="toolUse",
            )
        }
    )
    payload = json.loads(agent_server._mrtr_completion_result(with_tools, tools=_MINIMAL_TOOLS))
    assert payload["tool_calls"] == [{"id": "c-2", "input": {"query": "y"}, "name": "search"}]

    plain = SimpleNamespace(
        input_responses={
            "sample": types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text="plain"),
                model="m",
                stop_reason="endTurn",
            )
        }
    )
    assert agent_server._mrtr_completion_result(plain, tools=_MINIMAL_TOOLS) == "plain"


def test_sampling_tools_supported():
    """The fence reads sampling.tools presence, never assuming support."""
    assert agent_server._sampling_tools_supported(
        SimpleNamespace(client_capabilities=types.ClientCapabilities(sampling=_TOOLS_CAPABILITY))
    )
    assert not agent_server._sampling_tools_supported(
        SimpleNamespace(client_capabilities=types.ClientCapabilities(sampling=_NO_TOOLS_CAPABILITY))
    )
    assert not agent_server._sampling_tools_supported(
        SimpleNamespace(client_capabilities=types.ClientCapabilities())
    )
    assert not agent_server._sampling_tools_supported(SimpleNamespace(client_capabilities=None))
    assert not agent_server._sampling_tools_supported(SimpleNamespace())


def test_sampling_params_with_tools():
    """_sampling_params forwards validated tools with the chosen choice,
    defaulting to auto; the default path stays tool-free."""
    tools = agent_server._validate_tool_definitions(
        [agent_server.ToolDefinition(**d) for d in _TOOL_DEFS]
    )
    params = agent_server._sampling_params([], None, None, 10, "agent-default", tools=tools)
    assert [t.name for t in params.tools] == ["get_weather", "search"]
    assert params.tool_choice.mode == "auto"

    params = agent_server._sampling_params(
        [], None, None, 10, "agent-default", tools=tools, tool_choice="none"
    )
    assert params.tool_choice.mode == "none"

    bare = agent_server._sampling_params([], None, None, 10, "agent-default")
    assert bare.tools is None
    assert bare.tool_choice is None


# ---------------------------------------------------------------------------
# Item 9: generate-shaped raw completion tool
# ---------------------------------------------------------------------------


def test_generate_listed_with_others():
    """generate is listed alongside the other tools; the ctx param stays
    hidden; the schema exposes prompt/model/temperature/max_tokens plus the
    persona contract: no chat-tool surface (tools, output_schema,
    tool_choice), since persona is validation-only, never a sampling
    feature."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert {"chat_complete", "messages_create", "list_models", "generate"} <= names
            tool = next(t for t in tools.tools if t.name == "generate")
            props = tool.input_schema.get("properties", {})
            assert "ctx" not in props
            assert "prompt" in props
            assert "persona" in props
            assert "tools" not in props
            assert "output_schema" not in props
            assert "tool_choice" not in props
            required = tool.input_schema.get("required", [])
            assert "prompt" in required

    _run(go)


def test_generate_modern_era_full_loop():
    """Modern era: the full MRTR loop returns the completion; params pass
    through (temperature, max_tokens, model hint); the prompt arrives as a
    single user message with no system prompt."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "generate",
                {
                    "prompt": "Write a haiku about sampling.",
                    "model": "custom-model",
                    "temperature": 0.7,
                    "max_tokens": 200,
                },
            )
            seen = json.loads(_tool_text(result).removeprefix("CANNED:"))
            assert seen["messages"] == [["user", "Write a haiku about sampling."]]
            assert seen["system_prompt"] is None
            assert seen["temperature"] == 0.7
            assert seen["max_tokens"] == 200
            assert seen["hints"] == ["custom-model"]

    _run(go)


def test_generate_modern_era_input_required_payload():
    """Modern era, round one: the InputRequiredResult payload carries the
    prompt as a single user message with no system prompt, and carries no
    tools, no tool choice, no output schema: the zero-overhead thin path."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            first = await client.session.call_tool(
                "generate",
                {"prompt": "Say the word hello."},
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)
            params = first.input_requests["sample"].params
            assert [(m.role, m.content.text) for m in params.messages] == [
                ("user", "Say the word hello.")
            ]
            assert params.system_prompt is None
            assert params.tools is None
            assert params.tool_choice is None
            assert params.temperature is None
            assert params.max_tokens == 4096
            assert [h.name for h in params.model_preferences.hints] == ["agent-default"]
            assert first.request_state

    _run(go)


def test_generate_legacy_era():
    """Legacy era: direct sampling/createMessage with the prompt as the only
    message, no system prompt, default params."""

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            result = await client.call_tool("generate", {"prompt": "Ping"})
            seen = json.loads(_tool_text(result).removeprefix("CANNED:"))
            assert seen["messages"] == [["user", "Ping"]]
            assert seen["system_prompt"] is None
            assert seen["temperature"] is None
            assert seen["max_tokens"] == 4096

    _run(go)


def test_generate_description_states_raw_in_text_out():
    """The tool description says plainly what it is: raw prompt in,
    completion text out, same host model, no chat formatting."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            tool = next(t for t in tools.tools if t.name == "generate")
            description = tool.description.lower()
            assert "raw" in description
            assert "no chat formatting" in description
            assert "host agent's own model" in description

    _run(go)


# ---------------------------------------------------------------------------
# Item 10: typed per-tool schemas for host-registered personas
# ---------------------------------------------------------------------------

_PERSONA_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {"type": "array", "minItems": 1},
        "max_tokens": {"type": "integer", "minimum": 1},
    },
    "required": ["messages"],
}


def _json_text_sampler_factory(text):
    """Stand in for a host answering in plain text: the completion is raw text,
    which persona.output_schema validation parses as JSON."""

    async def sampler(context, params):
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text=text),
            model="test-host-model",
            stop_reason="endTurn",
        )

    return sampler


def test_persona_exposed_in_tool_schemas():
    """All three completion tools advertise persona; the Persona shape carries
    name/input_schema/output_schema; the ctx param stays hidden."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            for name in ("chat_complete", "messages_create", "generate"):
                tool = next(t for t in tools.tools if t.name == name)
                props = tool.input_schema.get("properties", {})
                assert "persona" in props
                assert "ctx" not in props
            persona_def = None
            for tool in tools.tools:
                persona_def = tool.input_schema.get("$defs", {}).get("Persona")
                if persona_def is not None:
                    break
            assert persona_def is not None
            assert {"name", "input_schema", "output_schema"} <= set(
                persona_def.get("properties", {})
            )
            assert "name" in persona_def.get("required", [])

    _run(go)


def test_persona_request_validation_success_both_eras():
    """Request-side gate, both eras: a persona whose input_schema the request
    satisfies lets the call through; the completion comes back untouched."""

    async def go():
        persona = {"name": "terse", "input_schema": _PERSONA_INPUT_SCHEMA}
        # Legacy era.
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "chat_complete",
                {"messages": [{"role": "user", "content": "Hi"}], "persona": persona},
            )
            assert not result.is_error
            assert _tool_text(result).startswith("CANNED:")
        # Modern era, full MRTR loop.
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "messages_create",
                {"messages": [{"role": "user", "content": "Hi"}], "persona": persona},
            )
            assert not result.is_error
            assert _tool_text(result).startswith("CANNED:")

    _run(go)


def test_persona_request_validation_failure_rejects_before_sending():
    """Request-side gate: a request violating the persona's input_schema fails
    loud, naming the persona, and nothing is sent to the host. Both eras."""

    async def go():
        calls = []

        async def recording_sampler(context, params):
            calls.append(params)
            return await _canned_sampler(context, params)

        persona = {"name": "pairwise", "input_schema": _PERSONA_INPUT_SCHEMA}
        args = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 0,  # violates minimum: 1
            "persona": persona,
        }
        # Legacy era.
        async with Client(SERVER, mode="legacy", sampling_callback=recording_sampler) as client:
            result = await client.call_tool("chat_complete", args)
            assert result.is_error
            text = _tool_text(result)
            assert "pairwise" in text
            assert "input-schema validation" in text
            assert "before anything was sent" in text
        # Modern era: the tool raises before returning an InputRequiredResult.
        async with Client(SERVER, sampling_callback=recording_sampler) as client:
            result = await client.call_tool("chat_complete", args)
            assert result.is_error
            assert "pairwise" in _tool_text(result)
        # The host was never asked: validation runs before anything is sent.
        assert calls == []

    _run(go)


def test_persona_request_validation_generate():
    """generate takes persona too: the input_schema constrains the prompt, and
    a violating prompt is rejected before anything is sent."""

    async def go():
        calls = []

        async def recording_sampler(context, params):
            calls.append(params)
            return await _canned_sampler(context, params)

        schema = {
            "type": "object",
            "properties": {"prompt": {"type": "string", "minLength": 5}},
            "required": ["prompt"],
        }
        async with Client(SERVER, mode="legacy", sampling_callback=recording_sampler) as client:
            ok = await client.call_tool(
                "generate",
                {
                    "prompt": "Say the word hello.",
                    "persona": {"name": "haiku", "input_schema": schema},
                },
            )
            assert not ok.is_error
            assert _tool_text(ok).startswith("CANNED:")

            bad = await client.call_tool(
                "generate",
                {"prompt": "Hi", "persona": {"name": "haiku", "input_schema": schema}},
            )
            assert bad.is_error
            text = _tool_text(bad)
            assert "haiku" in text
            assert "input-schema validation" in text
        # Exactly one host call: the rejected prompt never traveled.
        assert len(calls) == 1

    _run(go)


def test_persona_request_validation_failure_messages_create():
    """messages_create's request-side gate: a request violating the persona's
    input_schema fails loud before anything is sent."""

    async def go():
        calls = []

        async def recording_sampler(context, params):
            calls.append(params)
            return await _canned_sampler(context, params)

        schema = {
            "type": "object",
            "properties": {"system": {"type": "string", "minLength": 1}},
        }
        async with Client(SERVER, mode="legacy", sampling_callback=recording_sampler) as client:
            result = await client.call_tool(
                "messages_create",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "persona": {"name": "sysonly", "input_schema": schema},
                },
            )
            assert result.is_error
            text = _tool_text(result)
            assert "sysonly" in text
            assert "input-schema validation" in text
            assert "/system" in text
        assert calls == []

    _run(go)


def test_persona_output_validation_success_legacy():
    """Response-side gate: a host completion that parses as JSON validating
    against persona.output_schema is returned as-is."""

    async def go():
        sampler = _json_text_sampler_factory('{"answer": "yes", "confidence": 0.9}')
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "persona": {"name": "answerer", "output_schema": _OUTPUT_SCHEMA},
                },
            )
            assert not result.is_error
            assert _tool_text(result) == '{"answer": "yes", "confidence": 0.9}'

    _run(go)


def test_persona_output_validation_success_modern_full_loop():
    """Response-side gate, modern era full MRTR loop: valid JSON completion
    returns as-is."""

    async def go():
        sampler = _json_text_sampler_factory('{"answer": "structured hi"}')
        async with Client(SERVER, sampling_callback=sampler) as client:
            result = await client.call_tool(
                "messages_create",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "persona": {"name": "answerer", "output_schema": _OUTPUT_SCHEMA},
                },
            )
            assert not result.is_error
            assert _tool_text(result) == '{"answer": "structured hi"}'

    _run(go)


def test_persona_output_validation_failure():
    """Response-side gate: non-JSON text and schema-violating JSON both fail
    loud, naming the persona. Legacy and modern eras."""

    async def go():
        persona = {"name": "answerer", "output_schema": _OUTPUT_SCHEMA}
        args = {"messages": [{"role": "user", "content": "Hi"}], "persona": persona}
        # Legacy era: prose is not valid JSON for the persona.
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=_json_text_sampler_factory("just prose, no json"),
        ) as client:
            result = await client.call_tool("chat_complete", args)
            assert result.is_error
            text = _tool_text(result)
            assert "answerer" in text
            assert "not valid JSON" in text
        # Legacy era: JSON violating the schema names the mismatch.
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=_json_text_sampler_factory('{"answer": 42}'),
        ) as client:
            result = await client.call_tool("chat_complete", args)
            assert result.is_error
            text = _tool_text(result)
            assert "answerer" in text
            assert "output-schema validation" in text
            assert "/answer" in text
        # Modern era, full MRTR loop: schema violation fails the same way.
        async with Client(
            SERVER, sampling_callback=_json_text_sampler_factory('{"answer": 42}')
        ) as client:
            result = await client.call_tool("messages_create", args)
            assert result.is_error
            text = _tool_text(result)
            assert "answerer" in text
            assert "output-schema validation" in text

    _run(go)


def test_persona_modern_era_round_one_defers_response_validation():
    """Modern era, round one: a persona.output_schema does NOT inject a
    required tool (unlike output_schema); the host answers in plain text and
    the response-side gate runs on the retry round."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            first = await client.session.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "persona": {"name": "answerer", "output_schema": _OUTPUT_SCHEMA},
                },
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)
            params = first.input_requests["sample"].params
            assert params.tools is None
            assert params.tool_choice is None
            assert first.request_state

    _run(go)


def test_output_schema_and_persona_output_schema_conflict():
    """output_schema and persona.output_schema together fail loud, naming
    both. Both chat tools, before anything is sent."""

    async def go():
        calls = []

        async def recording_sampler(context, params):
            calls.append(params)
            return await _canned_sampler(context, params)

        async with Client(SERVER, mode="legacy", sampling_callback=recording_sampler) as client:
            for tool_name, args in (
                ("chat_complete", {"messages": [{"role": "user", "content": "Hi"}]}),
                ("messages_create", {"messages": [{"role": "user", "content": "Hi"}]}),
            ):
                result = await client.call_tool(
                    tool_name,
                    {
                        **args,
                        "output_schema": _OUTPUT_SCHEMA,
                        "persona": {"name": "answerer", "output_schema": _OUTPUT_SCHEMA},
                    },
                )
                assert result.is_error
                text = _tool_text(result)
                assert "output_schema" in text
                assert "persona.output_schema" in text
        assert calls == []

    _run(go)


def test_malformed_persona_rejected_e2e():
    """End to end: an empty or blank persona name fails loud before the host
    is involved."""

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            for bad_name in ("", "   "):
                result = await client.call_tool(
                    "chat_complete",
                    {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "persona": {"name": bad_name},
                    },
                )
                assert result.is_error
                assert "persona.name" in _tool_text(result)

    _run(go)


def test_validate_persona():
    """Unit: None passes through; empty/blank names and bad schemas fail loud
    with the offending field named."""
    assert agent_server._validate_persona(None) is None
    persona = agent_server._validate_persona(agent_server.Persona(name="terse"))
    assert persona is not None and persona.name == "terse"
    assert persona.input_schema is None and persona.output_schema is None

    with pytest.raises(ToolError, match="persona.name.*non-empty string"):
        agent_server._validate_persona(agent_server.Persona(name=""))
    with pytest.raises(ToolError, match="persona.name.*non-empty string"):
        agent_server._validate_persona(agent_server.Persona(name="   "))
    with pytest.raises(ToolError, match=r"persona\.input_schema.*root type 'object'"):
        agent_server._validate_persona(
            agent_server.Persona(name="x", input_schema={"type": "array"})
        )
    with pytest.raises(ToolError, match=r"persona\.output_schema.*not a valid JSON Schema"):
        agent_server._validate_persona(
            agent_server.Persona(
                name="x",
                output_schema={"type": "object", "properties": {"a": {"type": "bogus"}}},
            )
        )
    assert (
        agent_server._validate_persona(
            agent_server.Persona(
                name="x",
                input_schema=_PERSONA_INPUT_SCHEMA,
                output_schema=_OUTPUT_SCHEMA,
            )
        )
        is not None
    )


def test_check_output_schema_persona_conflict_unit():
    """Unit: the conflict fires only when both schema sources are present."""
    persona = agent_server.Persona(name="x", output_schema=_OUTPUT_SCHEMA)
    bare = agent_server.Persona(name="x")
    agent_server._check_output_schema_persona_conflict(None, persona)
    agent_server._check_output_schema_persona_conflict(_OUTPUT_SCHEMA, None)
    agent_server._check_output_schema_persona_conflict(_OUTPUT_SCHEMA, bare)
    with pytest.raises(ToolError, match="persona.output_schema"):
        agent_server._check_output_schema_persona_conflict(_OUTPUT_SCHEMA, persona)
    assert "output_schema" in str(
        pytest.raises(
            ToolError, agent_server._check_output_schema_persona_conflict, _OUTPUT_SCHEMA, persona
        ).value
    )


def test_validate_persona_request_unit():
    """Unit: the request gate no-ops without a persona or input_schema, and
    reports the exact location of a violation."""
    agent_server._validate_persona_request(None, {"messages": []})
    agent_server._validate_persona_request(agent_server.Persona(name="x"), {"messages": []})
    agent_server._validate_persona_request(
        agent_server.Persona(name="terse", input_schema=_PERSONA_INPUT_SCHEMA),
        {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
    )
    with pytest.raises(ToolError, match=r"persona 'terse'.*input-schema validation.*\(root\)"):
        agent_server._validate_persona_request(
            agent_server.Persona(name="terse", input_schema=_PERSONA_INPUT_SCHEMA),
            {"max_tokens": 10},  # "messages" is required
        )


def test_validate_persona_output_unit():
    """Unit: no-op without a persona or output_schema; non-JSON and
    schema-violating completions fail loud."""
    persona = agent_server.Persona(name="answerer", output_schema=_OUTPUT_SCHEMA)
    assert agent_server._validate_persona_output(None, "anything") == "anything"
    assert (
        agent_server._validate_persona_output(agent_server.Persona(name="x"), "anything")
        == "anything"
    )
    assert agent_server._validate_persona_output(persona, '{"answer": "ok"}') == '{"answer": "ok"}'
    with pytest.raises(ToolError, match="not valid JSON"):
        agent_server._validate_persona_output(persona, "not json at all")
    with pytest.raises(ToolError, match=r"persona 'answerer'.*output-schema validation"):
        agent_server._validate_persona_output(persona, '{"answer": 42}')


def test_finish_persona():
    """Unit: the MRTR first round passes through untouched; otherwise the
    response-side gate runs."""
    params = agent_server._sampling_params([], None, None, 10, "agent-default")
    first_round = agent_server._mrtr_completion_request(params)
    persona = agent_server.Persona(name="answerer", output_schema=_OUTPUT_SCHEMA)
    assert agent_server._finish_persona(persona, first_round) is first_round
    assert agent_server._finish_persona(None, "plain") == "plain"
    assert agent_server._finish_persona(persona, '{"answer": "ok"}') == '{"answer": "ok"}'
    with pytest.raises(ToolError, match="not valid JSON"):
        agent_server._finish_persona(persona, "plain")


def test_generate_lists_persona():
    """generate's schema exposes persona alongside the thin-path params."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            tool = next(t for t in tools.tools if t.name == "generate")
            props = tool.input_schema.get("properties", {})
            assert "persona" in props
            assert "prompt" in props
            # Still the thin path: no chat or tool features.
            assert "tools" not in props
            assert "output_schema" not in props
            assert "tool_choice" not in props

    _run(go)


# ---------------------------------------------------------------------------
# Item 5: responses_create (OpenAI Responses API shape)
# ---------------------------------------------------------------------------


def _response_payload(result):
    """The tool returns a dict; the server serializes it as a JSON text block."""
    assert not result.is_error
    return json.loads(_tool_text(result))


def test_responses_create_listed_and_schema():
    """responses_create is listed with the Responses request surface: input,
    instructions, model, temperature, max_tokens, output_schema, tools,
    tool_choice, persona; the ctx param stays hidden."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert "responses_create" in names
            tool = next(t for t in tools.tools if t.name == "responses_create")
            props = tool.input_schema.get("properties", {})
            assert "ctx" not in props
            assert {
                "input",
                "instructions",
                "model",
                "temperature",
                "max_tokens",
                "output_schema",
                "tools",
                "tool_choice",
                "persona",
            } <= set(props)
            assert "input" in tool.input_schema.get("required", [])

    _run(go)


def test_responses_create_modern_era_string_input_shape():
    """Modern era, full MRTR loop with a string input: the tool returns the
    Responses response object with a resp_-prefixed unique id, object
    'response', the model alias echoed, and a single message item whose
    output_text carries the completion."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {
                    "input": "Say hi.",
                    "model": "agent-default",
                    "temperature": 0.5,
                    "max_tokens": 100,
                },
            )
            payload = _response_payload(result)
            assert payload["id"].startswith("resp_")
            assert len(payload["id"]) == len("resp_") + 32
            assert payload["object"] == "response"
            assert payload["model"] == "agent-default"
            text = payload["output"][0]["content"][0]["text"]
            seen = json.loads(text.removeprefix("CANNED:"))
            assert payload["output"] == [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
            assert seen["messages"] == [["user", "Say hi."]]
            assert seen["system_prompt"] is None
            assert seen["temperature"] == 0.5
            assert seen["max_tokens"] == 100
            assert seen["hints"] == ["agent-default"]

    _run(go)


def test_responses_create_ids_unique_per_call():
    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            ids = []
            for _ in range(3):
                result = await client.call_tool("responses_create", {"input": "Hi"})
                ids.append(_response_payload(result)["id"])
            assert all(i.startswith("resp_") for i in ids)
            assert len(set(ids)) == 3

    _run(go)


def test_responses_create_input_items_fold_system_and_instructions():
    """An item list: user/assistant items become messages, system and
    developer items fold into the system prompt alongside instructions,
    and multi-part content joins its text."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {
                    "input": [
                        {"role": "system", "content": "System A."},
                        {"role": "developer", "content": "Developer B."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "input_text": "First."},
                                {"type": "text", "text": "Second."},
                            ],
                        },
                    ],
                    "instructions": "Instructions first.",
                },
            )
            payload = _response_payload(result)
            text = payload["output"][0]["content"][0]["text"]
            seen = json.loads(text.removeprefix("CANNED:"))
            assert seen["system_prompt"] == "Instructions first.\n\nSystem A.\n\nDeveloper B."
            assert seen["messages"] == [["user", "First.\nSecond."]]

    _run(go)


def test_responses_create_legacy_era():
    """Legacy era: direct sampling/createMessage; the Responses object still
    echoes the model alias and carries the completion as output_text."""

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {"input": "Ping", "instructions": "Be terse.", "model": "custom-model"},
            )
            payload = _response_payload(result)
            assert payload["object"] == "response"
            assert payload["model"] == "custom-model"
            text = payload["output"][0]["content"][0]["text"]
            seen = json.loads(text.removeprefix("CANNED:"))
            assert seen["system_prompt"] == "Be terse."
            assert seen["messages"] == [["user", "Ping"]]
            assert seen["hints"] == ["custom-model"]

    _run(go)


def test_responses_create_modern_era_input_required_payload():
    """Modern era, round one: the InputRequiredResult payload carries the
    translated messages, the system prompt, and the sampling params."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            first = await client.session.call_tool(
                "responses_create",
                {"input": "Say hi.", "instructions": "Be terse.", "max_tokens": 60},
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)
            params = first.input_requests["sample"].params
            assert params.system_prompt == "Be terse."
            assert [(m.role, m.content.text) for m in params.messages] == [("user", "Say hi.")]
            assert params.max_tokens == 60
            assert first.request_state

    _run(go)


def test_responses_create_max_tokens_passthrough():
    """max_tokens reaches the host; the default is 4096 like the other tools."""

    async def go():
        async with Client(SERVER, sampling_callback=_canned_sampler) as client:
            result = await client.call_tool("responses_create", {"input": "Hi", "max_tokens": 77})
            text = _response_payload(result)["output"][0]["content"][0]["text"]
            assert json.loads(text.removeprefix("CANNED:"))["max_tokens"] == 77

            result = await client.call_tool("responses_create", {"input": "Hi"})
            text = _response_payload(result)["output"][0]["content"][0]["text"]
            assert json.loads(text.removeprefix("CANNED:"))["max_tokens"] == 4096

    _run(go)


def test_responses_create_output_schema():
    """output_schema is accepted: the structured completion returns as JSON
    text inside the output_text block. Legacy and modern eras."""

    async def go():
        sampler = _structured_sampler_factory({"answer": "yes", "confidence": 0.9})
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {"input": "Hi", "output_schema": _OUTPUT_SCHEMA},
            )
            payload = _response_payload(result)
            text = payload["output"][0]["content"][0]["text"]
            assert json.loads(text) == {"answer": "yes", "confidence": 0.9}

        modern_sampler = _structured_sampler_factory({"answer": "modern"})
        async with Client(SERVER, sampling_callback=modern_sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {"input": "Hi", "output_schema": _OUTPUT_SCHEMA},
            )
            payload = _response_payload(result)
            text = payload["output"][0]["content"][0]["text"]
            assert json.loads(text) == {"answer": "modern"}

    _run(go)


def test_responses_create_tools_become_function_call_items():
    """Legacy era: tools are accepted; when the host answers with tool calls,
    the output array carries the accompanying text as a message item followed
    by function_call items in the documented shape."""

    tool_calls = [
        {"name": "get_weather", "id": "call-1", "input": {"unit": "metric", "city": "Naga"}}
    ]

    async def go():
        sampler = _tool_call_sampler_factory(tool_calls, text="checking")
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=sampler,
            sampling_capabilities=_TOOLS_CAPABILITY,
        ) as client:
            result = await client.call_tool(
                "responses_create",
                {"input": "Hi", "tools": _TOOL_DEFS, "tool_choice": "auto"},
            )
            payload = _response_payload(result)
            assert payload["output"] == [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "checking"}],
                },
                {
                    "type": "function_call",
                    "id": "fc_call-1",
                    "call_id": "call-1",
                    "name": "get_weather",
                    # Insertion order was unit, city: arguments are canonical
                    # sorted JSON, so a sort_keys flip changes this string.
                    "arguments": '{"city": "Naga", "unit": "metric"}',
                },
            ]

    _run(go)


def test_responses_create_function_calls_omit_empty_message_item():
    """When the host's tool-call answer has no accompanying text, the output
    array carries only the function_call items."""

    tool_calls = [{"name": "search", "id": "c-2", "input": {"query": "y"}}]

    async def go():
        sampler = _tool_call_sampler_factory(tool_calls)
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=sampler,
            sampling_capabilities=_TOOLS_CAPABILITY,
        ) as client:
            result = await client.call_tool(
                "responses_create", {"input": "Hi", "tools": _TOOL_DEFS}
            )
            payload = _response_payload(result)
            assert payload["output"] == [
                {
                    "type": "function_call",
                    "id": "fc_c-2",
                    "call_id": "c-2",
                    "name": "search",
                    "arguments": '{"query": "y"}',
                }
            ]

    _run(go)


def test_responses_create_modern_era_full_loop_tools():
    """Modern era, full MRTR loop: the host's tool calls come back as
    function_call items in the Responses object."""

    tool_calls = [{"name": "get_weather", "id": "call-9", "input": {"city": "Manila"}}]

    async def go():
        sampler = _tool_call_sampler_factory(tool_calls)
        async with Client(
            SERVER, sampling_callback=sampler, sampling_capabilities=_TOOLS_CAPABILITY
        ) as client:
            result = await client.call_tool(
                "responses_create", {"input": "Hi", "tools": _TOOL_DEFS}
            )
            payload = _response_payload(result)
            assert payload["object"] == "response"
            assert payload["output"] == [
                {
                    "type": "function_call",
                    "id": "fc_call-9",
                    "call_id": "call-9",
                    "name": "get_weather",
                    "arguments": '{"city": "Manila"}',
                }
            ]

    _run(go)


def test_responses_create_degraded_tools_stay_message_item(caplog):
    """Legacy era: without sampling.tools the tools are dropped with a
    server-side warning and the plain-text answer stays a single message item:
    it is never misread as tool calls."""

    async def go():
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=_no_tools_sampler,
            sampling_capabilities=_NO_TOOLS_CAPABILITY,
        ) as client:
            with caplog.at_level("WARNING", logger="agent_as_model"):
                result = await client.call_tool(
                    "responses_create", {"input": "Hi", "tools": _TOOL_DEFS}
                )
            payload = _response_payload(result)
            assert payload["output"] == [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "plain answer"}],
                }
            ]

    _run(go)
    assert any("sampling.tools" in record.message for record in caplog.records)


def test_responses_create_tools_output_schema_conflict():
    """tools and output_schema together fail loud, naming both."""

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {"input": "Hi", "tools": _TOOL_DEFS, "output_schema": _OUTPUT_SCHEMA},
            )
            assert result.is_error
            text = _tool_text(result)
            assert "tools" in text and "output_schema" in text

    _run(go)


def test_responses_create_persona_request_validation():
    """persona is accepted: the input_schema gate constrains the request
    (input, instructions, model, ...) and rejects before anything is sent."""

    async def go():
        calls = []

        async def recording_sampler(context, params):
            calls.append(params)
            return await _canned_sampler(context, params)

        schema = {
            "type": "object",
            "properties": {"input": {"type": "string", "minLength": 5}},
            "required": ["input"],
        }
        async with Client(SERVER, mode="legacy", sampling_callback=recording_sampler) as client:
            ok = await client.call_tool(
                "responses_create",
                {
                    "input": "Hello world",
                    "persona": {"name": "verbose", "input_schema": schema},
                },
            )
            assert not ok.is_error
            assert _response_payload(ok)["object"] == "response"

            bad = await client.call_tool(
                "responses_create",
                {"input": "Hi", "persona": {"name": "verbose", "input_schema": schema}},
            )
            assert bad.is_error
            text = _tool_text(bad)
            assert "verbose" in text
            assert "input-schema validation" in text
            assert "before anything was sent" in text
        # Exactly one host call: the rejected input never traveled.
        assert len(calls) == 1

    _run(go)


def test_responses_create_persona_output_validation():
    """persona.output_schema gates the completion text: valid JSON returns
    inside the response object, prose fails loud naming the persona."""

    async def go():
        sampler = _json_text_sampler_factory('{"answer": "yes"}')
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {
                    "input": "Hi",
                    "persona": {"name": "answerer", "output_schema": _OUTPUT_SCHEMA},
                },
            )
            payload = _response_payload(result)
            assert payload["output"][0]["content"][0]["text"] == '{"answer": "yes"}'

        async with Client(
            SERVER, sampling_callback=_json_text_sampler_factory("not json")
        ) as client:
            result = await client.call_tool(
                "responses_create",
                {
                    "input": "Hi",
                    "persona": {"name": "answerer", "output_schema": _OUTPUT_SCHEMA},
                },
            )
            assert result.is_error
            text = _tool_text(result)
            assert "answerer" in text
            assert "not valid JSON" in text

    _run(go)


def test_responses_create_rejects_non_text_input_part():
    """A non-text input part fails loud end to end."""

    async def go():
        async with Client(SERVER, mode="legacy", sampling_callback=_canned_sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": "http://x/y.png"}],
                        }
                    ]
                },
            )
            assert result.is_error

    _run(go)


def test_responses_object_plain_text_shape_unit():
    """Unit: the response object shape for a plain-text completion."""
    obj = agent_server._responses_object("hello", "agent-default", used_tools=False)
    assert obj["id"].startswith("resp_")
    assert obj["object"] == "response"
    assert obj["model"] == "agent-default"
    assert obj["output"] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }
    ]


def test_responses_object_model_echoes_alias():
    obj = agent_server._responses_object("x", "custom-model", used_tools=False)
    assert obj["model"] == "custom-model"


def test_responses_object_ids_unique_unit():
    first = agent_server._responses_object("x", "agent-default", used_tools=False)["id"]
    second = agent_server._responses_object("x", "agent-default", used_tools=False)["id"]
    assert first != second


def test_parse_tool_calls_json():
    """Unit: only the exact passthrough shape splits into (text, calls);
    anything else stays a single message item and never raises."""
    assert agent_server._parse_tool_calls_json("plain prose") is None
    assert agent_server._parse_tool_calls_json('{"nope": 1}') is None
    assert agent_server._parse_tool_calls_json('{"text": "t"}') is None
    assert agent_server._parse_tool_calls_json('["text", "tool_calls"]') is None
    text, calls = agent_server._parse_tool_calls_json(
        '{"text": "t", "tool_calls": [{"id": "c", "name": "n", "input": {}}]}'
    )
    assert text == "t"
    assert calls == [{"id": "c", "name": "n", "input": {}}]


def test_responses_create_output_schema_persona_output_schema_conflict():
    """output_schema and persona.output_schema together fail loud on
    responses_create too, naming both, before anything is sent."""

    async def go():
        calls = []

        async def recording_sampler(context, params):
            calls.append(params)
            return await _canned_sampler(context, params)

        async with Client(SERVER, mode="legacy", sampling_callback=recording_sampler) as client:
            result = await client.call_tool(
                "responses_create",
                {
                    "input": "Hi",
                    "output_schema": _OUTPUT_SCHEMA,
                    "persona": {"name": "answerer", "output_schema": _OUTPUT_SCHEMA},
                },
            )
            assert result.is_error
            text = _tool_text(result)
            assert "output_schema" in text
            assert "persona.output_schema" in text
        assert calls == []

    _run(go)


def test_responses_create_degraded_tools_never_misreads_prose_as_calls():
    """The used_tools predicate mirrors the _complete capability fence: when
    tools were dropped for a host without sampling.tools, even a plain-text
    answer shaped like the passthrough JSON stays a single message item."""

    async def prose_sampler(context, params):
        assert params.tools is None
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(
                type="text",
                text='{"text": "looks like tool calls", '
                '"tool_calls": [{"id": "c", "name": "n", "input": {}}]}',
            ),
            model="test-host-model",
            stop_reason="endTurn",
        )

    async def go():
        async with Client(
            SERVER,
            mode="legacy",
            sampling_callback=prose_sampler,
            sampling_capabilities=_NO_TOOLS_CAPABILITY,
        ) as client:
            result = await client.call_tool(
                "responses_create", {"input": "Hi", "tools": _TOOL_DEFS}
            )
            payload = _response_payload(result)
            assert payload["output"] == [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"text": "looks like tool calls", '
                            '"tool_calls": [{"id": "c", "name": "n", "input": {}}]}',
                        }
                    ],
                }
            ]

    _run(go)


def test_responses_object_arguments_sorted_keys_unit():
    """Unit: function_call arguments are canonical sorted JSON even when the
    passthrough text carried insertion-ordered input keys."""
    completion = json.dumps(
        {
            "text": "",
            "tool_calls": [{"id": "c-1", "name": "n", "input": {"unit": "m", "city": "x"}}],
        }
    )
    obj = agent_server._responses_object(completion, "agent-default", used_tools=True)
    assert obj["output"] == [
        {
            "type": "function_call",
            "id": "fc_c-1",
            "call_id": "c-1",
            "name": "n",
            "arguments": '{"city": "x", "unit": "m"}',
        }
    ]


# ---------------------------------------------------------------------------
# Item 6: host-owned session cursors (previous_response_id / response_id)
# ---------------------------------------------------------------------------


def _counting_sampler_factory():
    """Stand in for a host whose reply differs per call, so consecutive turns
    are distinguishable in the transcript."""

    calls = []

    async def sampler(context, params):
        calls.append(params)
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text=f"reply-{len(calls)}"),
            model="test-host-model",
            stop_reason="endTurn",
        )

    return sampler, calls


def _envelope_payload(result):
    """The cursor-active return shape for the text tools: JSON text carrying
    the completion and the response cursor."""
    assert not result.is_error
    payload = json.loads(_tool_text(result))
    assert set(payload) == {"completion", "response_id"}
    return payload


def test_previous_response_id_exposed_in_tool_schemas():
    """All four completion tools advertise previous_response_id; the ctx param
    stays hidden."""

    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            for name in ("chat_complete", "messages_create", "responses_create", "generate"):
                tool = next(t for t in tools.tools if t.name == name)
                props = tool.input_schema.get("properties", {})
                assert "previous_response_id" in props
                assert "ctx" not in props

    _run(go)


def test_cursor_minting_unit():
    """The cursor is 'cur_' + sha256 over the exact canonical JSON
    {"model","tool","transcript"} with sorted keys and compact separators: the
    pinned vector below is the contract any caller reimplementation must match."""
    transcript = [{"role": "user", "content": "Hi"}]
    cursor = agent_server._mint_cursor("chat_complete", "agent-default", transcript)
    assert cursor == "cur_109fa07a1114ce07cc8e28e24342ad0b16c67397e1fa998f26c505dee92adbb7"
    # Deterministic across calls.
    assert agent_server._mint_cursor("chat_complete", "agent-default", transcript) == cursor
    # Sensitive to every committed field.
    assert agent_server._mint_cursor("messages_create", "agent-default", transcript) != cursor
    assert agent_server._mint_cursor("chat_complete", "other-model", transcript) != cursor
    assert (
        agent_server._mint_cursor(
            "chat_complete",
            "agent-default",
            [*transcript, {"role": "assistant", "content": "yo"}],
        )
        != cursor
    )


def test_cursor_predicates_unit():
    """The format gates accept exactly the documented shapes."""
    assert agent_server._is_cursor("cur_" + "a" * 64)
    assert not agent_server._is_cursor("cur_" + "a" * 63)
    assert not agent_server._is_cursor("cur_" + "a" * 65)
    assert not agent_server._is_cursor("cur_" + "z" * 64)  # non-hex
    assert not agent_server._is_cursor("CUR_" + "a" * 64)
    assert not agent_server._is_cursor("resp_" + "a" * 32)
    assert not agent_server._is_cursor("")
    assert agent_server._is_opaque_response_id("resp_" + "b" * 32)
    assert not agent_server._is_opaque_response_id("resp_" + "b" * 31)
    assert not agent_server._is_opaque_response_id("resp_" + "z" * 32)
    assert not agent_server._is_opaque_response_id("cur_" + "b" * 64)
    assert not agent_server._is_opaque_response_id("")


def test_cursor_envelope_legacy_era():
    """Legacy era, cursor chain start (''): the completion returns in the
    envelope {"completion", "response_id"}, with the completion text exactly
    as the legacy path returns it and the cursor recomputable by the caller
    from the documented canonical form."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "previous_response_id": "",
                },
            )
            payload = _envelope_payload(result)
            assert payload["completion"] == "reply-1"
            assert payload["response_id"] == agent_server._mint_cursor(
                "chat_complete",
                "agent-default",
                [
                    {"role": "user", "content": "Say hi."},
                    {"role": "assistant", "content": "reply-1"},
                ],
            )
            # The envelope wire shape is pinned: keys sorted, so a sort_keys
            # flip changes this string.
            assert (
                _tool_text(result)
                == '{"completion": "reply-1", "response_id": "' + payload["response_id"] + '"}'
            )

    _run(go)


def test_cursor_envelope_modern_era_full_loop():
    """Modern era, full MRTR loop: the same envelope contract. Round one still
    returns the InputRequiredResult untouched (no completion yet, so no cursor)."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, sampling_callback=sampler) as client:
            first = await client.session.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "previous_response_id": "",
                },
                allow_input_required=True,
            )
            assert isinstance(first, types.InputRequiredResult)

            result = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "previous_response_id": "",
                },
            )
            payload = _envelope_payload(result)
            assert payload["completion"] == "reply-1"
            assert payload["response_id"].startswith("cur_")

    _run(go)


def test_multiturn_continuation_happy_path_both_eras():
    """The multi-turn flow: call 1 mints cursor A; call 2 passes A back with
    the accumulated history and gets cursor B over the longer transcript.
    Both eras, and the cursors agree across eras: the protocol lives at the
    tool layer, above the era dispatch."""

    async def run_turns(mode_kwargs):
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, sampling_callback=sampler, **mode_kwargs) as client:
            first = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Who are you?"}],
                    "previous_response_id": "",
                },
            )
            cursor_a = _envelope_payload(first)["response_id"]

            history = [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "reply-1"},
                {"role": "user", "content": "And your mission?"},
            ]
            second = await client.call_tool(
                "chat_complete",
                {"messages": history, "previous_response_id": cursor_a},
            )
            payload = _envelope_payload(second)
            assert payload["completion"] == "reply-2"
            cursor_b = payload["response_id"]
            assert cursor_b != cursor_a
            assert cursor_b == agent_server._mint_cursor(
                "chat_complete",
                "agent-default",
                [*history, {"role": "assistant", "content": "reply-2"}],
            )
            return cursor_a, cursor_b

    async def go():
        legacy_a, legacy_b = await run_turns({"mode": "legacy"})
        modern_a, modern_b = await run_turns({})
        assert legacy_a == modern_a
        assert legacy_b == modern_b

    _run(go)


def test_tampered_history_fails_loud():
    """Tampered history with a valid cursor fails loud, naming expected vs
    received. Variants: a user turn edited, an assistant turn edited, and the
    model changed between turns."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            first = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Who are you?"}],
                    "previous_response_id": "",
                },
            )
            cursor_a = _envelope_payload(first)["response_id"]

            good_history = [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "reply-1"},
                {"role": "user", "content": "More?"},
            ]
            tampered_user = [
                {"role": "user", "content": "Who are you, REALLY?"},
                {"role": "assistant", "content": "reply-1"},
                {"role": "user", "content": "More?"},
            ]
            tampered_assistant = [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "reply-1, altered"},
                {"role": "user", "content": "More?"},
            ]
            for bad_history in (tampered_user, tampered_assistant):
                result = await client.call_tool(
                    "chat_complete",
                    {"messages": bad_history, "previous_response_id": cursor_a},
                )
                assert result.is_error
                text = _tool_text(result)
                assert "previous_response_id mismatch" in text
                assert "expected" in text and "received" in text
                assert f"received {cursor_a}" in text
                # The expected cursor is named too, and it differs.
                assert f"expected {cursor_a}" not in text

            wrong_model = await client.call_tool(
                "chat_complete",
                {
                    "messages": good_history,
                    "model": "other-model",
                    "previous_response_id": cursor_a,
                },
            )
            assert wrong_model.is_error
            assert "previous_response_id mismatch" in _tool_text(wrong_model)

    _run(go)


def test_previous_response_id_malformed_and_missing_assistant():
    """Unrecognized cursor formats fail loud; a cur_ cursor with no assistant
    message in the history fails loud instead of verifying against nothing."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            bad_format = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "previous_response_id": "bogus",
                },
            )
            assert bad_format.is_error
            assert "unrecognized format" in _tool_text(bad_format)

            no_assistant = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "previous_response_id": "cur_" + "a" * 64,
                },
            )
            assert no_assistant.is_error
            assert "no assistant message" in _tool_text(no_assistant)

    _run(go)


def test_omitted_previous_response_id_keeps_legacy_shape():
    """Regression guard: without previous_response_id the tool returns plain
    text exactly as before, no envelope."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "chat_complete", {"messages": [{"role": "user", "content": "Hi"}]}
            )
            assert _tool_text(result) == "reply-1"

    _run(go)


def test_messages_create_multiturn_with_system():
    """messages_create continuation: the system prompt is part of the committed
    transcript, so changing it between turns breaks the chain loudly."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, sampling_callback=sampler) as client:
            first = await client.call_tool(
                "messages_create",
                {
                    "system": "Be terse.",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "previous_response_id": "",
                },
            )
            payload = _envelope_payload(first)
            assert payload["response_id"] == agent_server._mint_cursor(
                "messages_create",
                "agent-default",
                [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "reply-1"},
                ],
            )

            changed_system = await client.call_tool(
                "messages_create",
                {
                    "system": "Be verbose.",
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "More"},
                    ],
                    "previous_response_id": payload["response_id"],
                },
            )
            assert changed_system.is_error
            assert "previous_response_id mismatch" in _tool_text(changed_system)

    _run(go)


def test_generate_cursor_mints_and_thin_path_limit():
    """generate mints a cursor over (prompt, completion) when asked with '';
    a cur_ cursor fails loud: the thin path has no structured history channel
    to verify it against, and the server will not silently skip the check."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            result = await client.call_tool(
                "generate", {"prompt": "Say hi.", "previous_response_id": ""}
            )
            payload = _envelope_payload(result)
            assert payload["completion"] == "reply-1"
            assert payload["response_id"] == agent_server._mint_cursor(
                "generate",
                "agent-default",
                [
                    {"role": "user", "content": "Say hi."},
                    {"role": "assistant", "content": "reply-1"},
                ],
            )

            no_verify = await client.call_tool(
                "generate",
                {"prompt": "Say hi again.", "previous_response_id": payload["response_id"]},
            )
            assert no_verify.is_error
            assert "no structured history channel" in _tool_text(no_verify)

    _run(go)


def test_responses_create_id_is_cursor():
    """The id field IS the cursor on responses_create. Legacy calls (no
    previous_response_id) keep the opaque 'resp_' unique id; cursor-active
    calls carry the tamper-evident 'cur_' commitment in id, verified against
    the history passed back."""

    async def go():
        legacy_sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=legacy_sampler) as client:
            legacy = await client.call_tool("responses_create", {"input": "Hi"})
            assert _response_payload(legacy)["id"].startswith("resp_")

        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            first = await client.call_tool(
                "responses_create", {"input": "Hi", "previous_response_id": ""}
            )
            payload = _response_payload(first)
            cursor_a = payload["id"]
            assert cursor_a.startswith("cur_")
            text_a = payload["output"][0]["content"][0]["text"]
            assert text_a == "reply-1"
            assert cursor_a == agent_server._mint_cursor(
                "responses_create",
                "agent-default",
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "reply-1"},
                ],
            )

            second = await client.call_tool(
                "responses_create",
                {
                    "input": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "More?"},
                    ],
                    "previous_response_id": cursor_a,
                },
            )
            payload_b = _response_payload(second)
            cursor_b = payload_b["id"]
            assert cursor_b.startswith("cur_") and cursor_b != cursor_a

            tampered = await client.call_tool(
                "responses_create",
                {
                    "input": [
                        {"role": "user", "content": "Hi, altered"},
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "More?"},
                    ],
                    "previous_response_id": cursor_a,
                },
            )
            assert tampered.is_error
            assert "previous_response_id mismatch" in _tool_text(tampered)

    _run(go)


def test_responses_create_opaque_id_anchors_new_chain():
    """A legacy 'resp_' response id passed as previous_response_id cannot be
    verified (opaque by construction), so it anchors a new verifiable chain
    at the history passed back instead of failing: the next response carries
    a 'cur_' cursor, and the chain verifies from there."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            legacy = await client.call_tool("responses_create", {"input": "Hi"})
            opaque = _response_payload(legacy)["id"]
            assert opaque.startswith("resp_")

            continued = await client.call_tool(
                "responses_create",
                {
                    "input": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "More?"},
                    ],
                    "previous_response_id": opaque,
                },
            )
            payload = _response_payload(continued)
            assert payload["id"].startswith("cur_")

            again = await client.call_tool(
                "responses_create",
                {
                    "input": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "More?"},
                        {"role": "assistant", "content": "reply-2"},
                        {"role": "user", "content": "Even more?"},
                    ],
                    "previous_response_id": payload["id"],
                },
            )
            assert not again.is_error
            assert _response_payload(again)["id"].startswith("cur_")

    _run(go)


def test_mrtr_round_one_verifies_before_input_required():
    """Modern era: a tampered history fails on round one, before any
    InputRequiredResult is returned."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, sampling_callback=sampler) as client:
            first = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "previous_response_id": "",
                },
            )
            cursor_a = _envelope_payload(first)["response_id"]
            bad = await client.call_tool(
                "chat_complete",
                {
                    "messages": [
                        {"role": "user", "content": "Hi, tampered"},
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "More?"},
                    ],
                    "previous_response_id": cursor_a,
                },
            )
            assert bad.is_error
            assert "previous_response_id mismatch" in _tool_text(bad)

    _run(go)


def test_branch_by_truncation():
    """Continuing from an earlier turn: truncate the history to that turn and
    pass its cursor; the server verifies exactly what it is given."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            first = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Q1"}],
                    "previous_response_id": "",
                },
            )
            cursor_a = _envelope_payload(first)["response_id"]
            branched = await client.call_tool(
                "chat_complete",
                {
                    "messages": [
                        {"role": "user", "content": "Q1"},
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "Different Q2"},
                    ],
                    "previous_response_id": cursor_a,
                },
            )
            payload = _envelope_payload(branched)
            assert payload["completion"] == "reply-2"
            assert payload["response_id"].startswith("cur_")

    _run(go)


def test_cursor_history_canonicalization_unit():
    """Per-tool transcript canonicalization: chat keeps the caller's roles
    verbatim; Anthropic flattens the system prompt first; Responses keeps
    instructions leading and item roles verbatim; string input is one user turn."""
    from agent_as_model.protocol import (
        AnthropicContentBlock,
        AnthropicMessage,
        ChatMessage,
        ResponseInputItem,
    )

    assert agent_server._cursor_history_chat(
        [
            ChatMessage(role="system", content="Sys."),
            ChatMessage(role="user", content="Hi"),
        ]
    ) == [
        {"role": "system", "content": "Sys."},
        {"role": "user", "content": "Hi"},
    ]

    assert agent_server._cursor_history_anthropic(
        [AnthropicMessage(role="user", content="Hi")],
        [
            AnthropicContentBlock(type="text", text="A."),
            AnthropicContentBlock(type="text", text="B."),
        ],
    ) == [
        {"role": "system", "content": "A.\nB."},
        {"role": "user", "content": "Hi"},
    ]
    assert agent_server._cursor_history_anthropic(
        [AnthropicMessage(role="user", content="Hi")], None
    ) == [{"role": "user", "content": "Hi"}]

    assert agent_server._cursor_history_responses(
        [
            ResponseInputItem(role="developer", content="Dev."),
            ResponseInputItem(role="user", content="Hi"),
        ],
        "Lead.",
    ) == [
        {"role": "system", "content": "Lead."},
        {"role": "developer", "content": "Dev."},
        {"role": "user", "content": "Hi"},
    ]
    assert agent_server._cursor_history_responses("Just hi.", None) == [
        {"role": "user", "content": "Just hi."}
    ]


def test_persona_request_validation_sees_previous_response_id():
    """The persona input_schema gate sees previous_response_id as part of the
    request arguments."""

    async def go():
        sampler, _ = _counting_sampler_factory()
        schema = {
            "type": "object",
            "properties": {"previous_response_id": {"type": "string"}},
            "required": ["previous_response_id"],
        }
        persona = {"name": "chained", "input_schema": schema}
        async with Client(SERVER, mode="legacy", sampling_callback=sampler) as client:
            missing = await client.call_tool(
                "chat_complete",
                {"messages": [{"role": "user", "content": "Hi"}], "persona": persona},
            )
            assert missing.is_error
            assert "input-schema validation" in _tool_text(missing)

            present = await client.call_tool(
                "chat_complete",
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "persona": persona,
                    "previous_response_id": "",
                },
            )
            assert not present.is_error
            assert _envelope_payload(present)["response_id"].startswith("cur_")

    _run(go)
