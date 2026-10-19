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
