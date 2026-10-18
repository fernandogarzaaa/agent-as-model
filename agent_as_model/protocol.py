"""Message translation for agent-as-model.

The server never touches a model API key. Every completion is produced by the
host agent's own LLM through MCP sampling (or the multi-round-trip successor
on 2026-07-28+ connections). This module translates the two most common
product API shapes into MCP sampling messages and extracts plain text from
sampling results.
"""

from __future__ import annotations

from typing import Literal

from mcp.types import CreateMessageResult, SamplingMessage, TextContent
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """One OpenAI chat-completions message (text content only, v1)."""

    role: Literal["user", "assistant", "system"]
    content: str


class AnthropicContentBlock(BaseModel):
    """One Anthropic text content block (v1 supports text only)."""

    type: Literal["text"]
    text: str


class AnthropicMessage(BaseModel):
    """One Anthropic messages API message (text only, v1)."""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


def _text_of(content: str | list[AnthropicContentBlock]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content)


def openai_to_sampling(
    messages: list[ChatMessage],
) -> tuple[str | None, list[SamplingMessage]]:
    """Split OpenAI chat messages into a system prompt and sampling messages."""
    if not messages:
        raise ValueError("messages must be a non-empty list")
    system_parts: list[str] = []
    sampling: list[SamplingMessage] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        else:
            sampling.append(
                SamplingMessage(
                    role=message.role,
                    content=TextContent(type="text", text=message.content),
                )
            )
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, sampling


def anthropic_to_sampling(
    messages: list[AnthropicMessage],
    system: str | list[AnthropicContentBlock] | None,
) -> tuple[str | None, list[SamplingMessage]]:
    """Split an Anthropic messages request into a system prompt and sampling messages."""
    if not messages:
        raise ValueError("messages must be a non-empty list")
    system_prompt = _text_of(system) if system is not None else None
    sampling = [
        SamplingMessage(
            role=message.role,
            content=TextContent(type="text", text=_text_of(message.content)),
        )
        for message in messages
    ]
    return system_prompt, sampling


def extract_text(result: CreateMessageResult) -> str:
    """Pull plain text out of a sampling result.

    Raises ValueError on non-text content instead of silently stringifying it.
    """
    content = result.content
    if isinstance(content, TextContent):
        return content.text
    raise ValueError(
        f"agent-as-model serves text completions only, the host returned {type(content).__name__}"
    )
