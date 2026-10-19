"""Unit tests for the message-translation layer."""

import pytest
from mcp.types import CreateMessageResult, ImageContent, TextContent

from agent_as_model.protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    ChatMessage,
    ResponseContentPart,
    ResponseInputItem,
    anthropic_to_sampling,
    extract_text,
    openai_to_sampling,
    responses_to_sampling,
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


# ---------------------------------------------------------------------------
# Item 5: Responses API input translation
# ---------------------------------------------------------------------------


def test_responses_string_input_becomes_single_user_message():
    system_prompt, sampling = responses_to_sampling("Hello there.", "Be terse.")
    assert system_prompt == "Be terse."
    assert [(m.role, m.content.text) for m in sampling] == [("user", "Hello there.")]


def test_responses_string_input_without_instructions():
    system_prompt, sampling = responses_to_sampling("Hi", None)
    assert system_prompt is None
    assert len(sampling) == 1


def test_responses_item_list_maps_roles_and_folds_system():
    system_prompt, sampling = responses_to_sampling(
        [
            ResponseInputItem(role="system", content="System A."),
            ResponseInputItem(role="developer", content="Developer B."),
            ResponseInputItem(role="user", content="Question."),
            ResponseInputItem(role="assistant", content="Draft."),
            ResponseInputItem(role="user", content="Follow-up."),
        ],
        "Instructions first.",
    )
    # instructions lead; per-item system/developer content refines them.
    assert system_prompt == "Instructions first.\n\nSystem A.\n\nDeveloper B."
    assert [(m.role, m.content.text) for m in sampling] == [
        ("user", "Question."),
        ("assistant", "Draft."),
        ("user", "Follow-up."),
    ]


def test_responses_item_list_without_instructions():
    system_prompt, sampling = responses_to_sampling(
        [ResponseInputItem(role="user", content="Hi")], None
    )
    assert system_prompt is None
    assert [(m.role, m.content.text) for m in sampling] == [("user", "Hi")]


def test_responses_content_parts_join_text():
    item = ResponseInputItem(
        role="user",
        content=[
            ResponseContentPart(type="input_text", input_text="First."),
            ResponseContentPart(type="text", text="Second."),
        ],
    )
    _, sampling = responses_to_sampling([item], None)
    assert sampling[0].content.text == "First.\nSecond."


def test_responses_part_alias_accepts_both_keys():
    by_alias = ResponseContentPart(type="input_text", input_text="via alias")
    by_name = ResponseContentPart(type="text", text="via name")
    assert by_alias.text == "via alias"
    assert by_name.text == "via name"


def test_responses_non_text_part_rejected_at_parse():
    with pytest.raises(Exception, match="input_text"):
        ResponseInputItem(role="user", content=[{"type": "image_url", "image_url": "x"}])


def test_responses_unknown_role_rejected_at_parse():
    with pytest.raises(Exception, match="role"):
        ResponseInputItem(role="function_call", content="nope")


def test_responses_text_part_without_body_rejected():
    item = ResponseInputItem(role="user", content=[ResponseContentPart(type="input_text")])
    with pytest.raises(ValueError, match="no text body"):
        responses_to_sampling([item], None)


def test_responses_empty_item_list_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        responses_to_sampling([], None)
