"""End-to-end tests: an in-process MCP client drives the real server tools.

The scripted sampling callback stands in for the host agent's own model, so
these tests exercise the full loop a real platform install would run:
tools/call -> sampling request -> canned host completion -> tool result.
Both protocol eras are covered: the default (2026-07-28, multi-round-trip)
and legacy (direct sampling/createMessage).
"""

from __future__ import annotations

import json

import anyio
import pytest
from mcp import types
from mcp.client import Client
from mcp.client.session import ClientRequestContext

import server as agent_server

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


def test_tools_listed_and_resolve_param_hidden():
    async def go():
        async with Client(SERVER) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert {"chat_complete", "messages_create", "list_models"} <= names
            for name in ("chat_complete", "messages_create"):
                tool = next(t for t in tools.tools if t.name == name)
                assert "completion" not in tool.input_schema.get("properties", {})

    _run(go)


def test_chat_complete_modern_era():
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


def test_chat_complete_legacy_era():
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


def test_missing_sampling_capability_fails_clearly():
    async def go():
        async with Client(SERVER) as client:  # no sampling_callback
            with pytest.raises(Exception, match="(?i)sampling"):
                await client.call_tool(
                    "chat_complete", {"messages": [{"role": "user", "content": "Hi"}]}
                )

    _run(go)
