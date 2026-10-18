"""Unit tests for the message-translation layer."""

import pytest
from mcp.types import CreateMessageResult, ImageContent, TextContent

from protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    ChatMessage,
    anthropic_to_sampling,
    extract_text,
    openai_to_sampling,
)


def test_openai_system_messages_become_system_prompt():
    system_prompt, sampling = openai_to_sampling(
        [
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="system", content="No emojis."),
            ChatMessage(role="assistant", content="Hello"),
        ]
    )
    assert system_prompt == "You are terse.\n\nNo emojis."
    assert [(m.role, m.content.text) for m in sampling] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
    ]


def test_openai_without_system_gives_none_prompt():
    system_prompt, sampling = openai_to_sampling([ChatMessage(role="user", content="Hi")])
    assert system_prompt is None
    assert len(sampling) == 1


def test_openai_empty_messages_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        openai_to_sampling([])


def test_anthropic_text_blocks_and_system_blocks():
    system_prompt, sampling = anthropic_to_sampling(
        [
            AnthropicMessage(
                role="user",
                content=[
                    AnthropicContentBlock(type="text", text="First."),
                    AnthropicContentBlock(type="text", text="Second."),
                ],
            )
        ],
        system=[
            AnthropicContentBlock(type="text", text="Be brief."),
            AnthropicContentBlock(type="text", text="Be kind."),
        ],
    )
    assert system_prompt == "Be brief.\nBe kind."
    assert sampling[0].content.text == "First.\nSecond."


def test_anthropic_plain_strings():
    system_prompt, sampling = anthropic_to_sampling(
        [AnthropicMessage(role="user", content="Hello")], system="Shout."
    )
    assert system_prompt == "Shout."
    assert sampling[0].content.text == "Hello"


def test_anthropic_no_system():
    system_prompt, _ = anthropic_to_sampling([AnthropicMessage(role="user", content="Hello")], None)
    assert system_prompt is None


def test_anthropic_empty_messages_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        anthropic_to_sampling([], None)


def test_extract_text_returns_text():
    result = CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text="done"),
        model="m",
        stop_reason="endTurn",
    )
    assert extract_text(result) == "done"


def test_extract_text_rejects_non_text():
    result = CreateMessageResult(
        role="assistant",
        content=ImageContent(type="image", data="eA==", mime_type="image/png"),
        model="m",
        stop_reason="endTurn",
    )
    with pytest.raises(ValueError, match="text completions only"):
        extract_text(result)
