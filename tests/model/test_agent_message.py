import pytest

from inspect_ai.model._agent_message import validate_agent_message


def test_validate_agent_message_accepts_string_id() -> None:
    item = {
        "type": "agent_message",
        "id": "msg_123",
        "author": "agent-1",
        "recipient": "agent-2",
        "role": "assistant",
        "content": [{"type": "input_text", "text": "hello"}],
    }

    assert validate_agent_message(item) is item


def test_validate_agent_message_rejects_non_string_id() -> None:
    with pytest.raises(
        ValueError, match="agent_message field 'id' must be a string"
    ):
        validate_agent_message({"type": "agent_message", "id": 123, "content": []})


def test_validate_agent_message_rejects_null_id() -> None:
    with pytest.raises(ValueError, match="agent_message field 'id' must be a string"):
        validate_agent_message({"type": "agent_message", "id": None, "content": []})


def test_validate_agent_message_still_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unsupported fields: unexpected"):
        validate_agent_message(
            {"type": "agent_message", "id": "msg_123", "unexpected": True}
        )
