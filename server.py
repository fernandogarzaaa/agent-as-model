"""agent-as-model: a true MCP server where the agent IS the model and the API key.

Install on any MCP host (Claude Code, Cursor, Windsurf, ...). The server
exposes product-style completion tools; every call is fulfilled by sampling
the host agent's own LLM. No API keys exist anywhere in this system.

Tools:
  chat_complete   OpenAI chat-completions-shaped request -> completion text
  messages_create Anthropic messages-shaped request -> completion text
  list_models     the model this server offers (the host's own model, aliased)

Upstream note: sampling is deprecated as of the 2026-07-28 spec (SEP-2577,
earliest removal 2027-07-28). This server uses the era-portable
Resolve/Sample form: a direct sampling request on pre-2026 connections and
the multi-round-trip pattern on 2026-07-28+. One tool body serves both eras.
The host must declare the sampling capability or calls fail with a clear
MISSING_REQUIRED_CLIENT_CAPABILITY error.
"""

from __future__ import annotations

import json
from typing import Annotated

from mcp.server.mcpserver import MCPServer, Resolve, Sample
from mcp.types import (
    CreateMessageResult,
    ModelHint,
    ModelPreferences,
)
from pydantic import Field

from protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    ChatMessage,
    anthropic_to_sampling,
    extract_text,
    openai_to_sampling,
)

mcp = MCPServer(
    name="agent-as-model",
    version="0.1.0",
    instructions=(
        "Completion tools backed by the host agent's own model. "
        "No API keys are used or needed: the connected agent is the model."
    ),
)


async def _sample_chat(
    messages: list[ChatMessage],
    model: str,
    temperature: float | None,
    max_tokens: int,
) -> Sample:
    system_prompt, sampling_messages = openai_to_sampling(messages)
    return Sample(
        messages=sampling_messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model_preferences=ModelPreferences(hints=[ModelHint(name=model)]),
    )


@mcp.tool(
    description=(
        "Generate a chat completion from an OpenAI chat-completions-shaped request. "
        "The completion is produced by the host agent's own model via MCP sampling. "
        "No API key is used or needed: the connected agent is the model."
    )
)
async def chat_complete(
    completion: Annotated[CreateMessageResult, Resolve(_sample_chat)],
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
) -> str:
    """Complete a chat with the host agent's model (OpenAI request shape)."""
    return extract_text(completion)


async def _sample_messages(
    messages: list[AnthropicMessage],
    system: str | list[AnthropicContentBlock] | None,
    model: str,
    temperature: float | None,
    max_tokens: int,
) -> Sample:
    system_prompt, sampling_messages = anthropic_to_sampling(messages, system)
    return Sample(
        messages=sampling_messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model_preferences=ModelPreferences(hints=[ModelHint(name=model)]),
    )


@mcp.tool(
    description=(
        "Generate a completion from an Anthropic messages-shaped request. "
        "The completion is produced by the host agent's own model via MCP sampling. "
        "No API key is used or needed: the connected agent is the model."
    )
)
async def messages_create(
    completion: Annotated[CreateMessageResult, Resolve(_sample_messages)],
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
) -> str:
    """Complete a conversation with the host agent's model (Anthropic request shape)."""
    return extract_text(completion)


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
        ],
        indent=2,
    )


def main() -> None:
    mcp.run_stdio_async()


if __name__ == "__main__":
    main()
