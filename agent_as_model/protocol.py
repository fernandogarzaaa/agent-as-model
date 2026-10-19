"""Message translation for agent-as-model.

The server never touches a model API key. Every completion is produced by the
host agent's own LLM through MCP sampling (or the multi-round-trip successor
on 2026-07-28+ connections). This module translates the three most common
product API shapes into MCP sampling messages and extracts plain text from
sampling results.
"""

from __future__ import annotations

from typing import Literal

from mcp.types import CreateMessageResult, SamplingMessage, TextContent
from pydantic import BaseModel, ConfigDict, Field


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


class ResponseContentPart(BaseModel):
    """One text part inside a Responses API input item's content list.

    The Responses API names the text key `input_text`; `text` is accepted as
    the chat-completions alias. A part of any other type is rejected at parse
    time: this server completes text only.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["input_text", "text"]
    text: str | None = Field(default=None, alias="input_text")


class ResponseInputItem(BaseModel):
    """One Responses API input item: a message with a role and text content.

    Roles beyond user/assistant/system/developer (function_call items, for
    example) are rejected at parse time: this server has no tool-output
    round-trip to map them onto.
    """

    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ResponseContentPart]


def _response_item_text(item: ResponseInputItem) -> str:
    """Flatten one input item's content to plain text, failing loud on a text
    part with no text body."""
    content = item.content
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for part in content:
        if part.text is None:
            raise ValueError(
                f"Responses input item with role {item.role!r} has a {part.type!r} "
                "part with no text body; text parts must carry their text"
            )
        texts.append(part.text)
    return "\n".join(texts)


def responses_to_sampling(
    input: str | list[ResponseInputItem], instructions: str | None
) -> tuple[str | None, list[SamplingMessage]]:
    """Split a Responses API request into a system prompt and sampling messages.

    A string input becomes a single user message. A list of input items maps
    to messages: user/assistant items become sampling messages; system and
    developer items fold into the system prompt alongside `instructions`,
    which leads (the API-level system message frames the call; per-item
    system/developer content refines it), joined with blank lines.

    Content may be a string or a list of text parts (`input_text`/`text`);
    non-text parts are rejected loudly at parse time, as is an empty input
    list: there is nothing to complete.
    """
    if isinstance(input, str):
        return instructions, [
            SamplingMessage(role="user", content=TextContent(type="text", text=input))
        ]
    if not input:
        raise ValueError("input must be a non-empty list when given as items")
    system_parts: list[str] = []
    if instructions is not None:
        system_parts.append(instructions)
    sampling: list[SamplingMessage] = []
    for item in input:
        text = _response_item_text(item)
        if item.role in ("system", "developer"):
            system_parts.append(text)
        else:
            sampling.append(
                SamplingMessage(
                    role=item.role,
                    content=TextContent(type="text", text=text),
                )
            )
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, sampling
