"""Tests for multi-part tool results in the Anthropic protocol."""

from __future__ import annotations

from ai.providers.anthropic import protocol
from ai.types import messages


class TestToolResultToAnthropic:
    def test_text_output(self) -> None:
        result = protocol._tool_result_to_anthropic(
            messages.TextOutput(value="hello")
        )
        assert result == "hello"

    def test_json_output_none(self) -> None:
        result = protocol._tool_result_to_anthropic(messages.JsonOutput())
        assert result == ""

    def test_json_output_dict(self) -> None:
        result = protocol._tool_result_to_anthropic(
            messages.JsonOutput(value={"key": "value"})
        )
        assert result == '{"key":"value"}'

    def test_json_output_list(self) -> None:
        result = protocol._tool_result_to_anthropic(
            messages.JsonOutput(value=[1, 2, 3])
        )
        assert result == "[1,2,3]"

    def test_error_text_output(self) -> None:
        result = protocol._tool_result_to_anthropic(
            messages.ErrorTextOutput(value="boom")
        )
        assert result == "boom"

    def test_execution_denied(self) -> None:
        result = protocol._tool_result_to_anthropic(
            messages.ExecutionDeniedOutput(reason="user said no")
        )
        assert result == "Tool execution denied: user said no"

    def test_content_text_and_file(self) -> None:
        fp = messages.FilePart(data="b64data", media_type="image/png")
        result = protocol._tool_result_to_anthropic(
            messages.ContentOutput(
                value=[messages.TextPart(text="Image loaded"), fp]
            )
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Image loaded"}
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "base64"
        assert result[1]["source"]["media_type"] == "image/png"
        assert result[1]["source"]["data"] == "b64data"

    def test_content_file_only(self) -> None:
        fp = messages.FilePart(data="b64data", media_type="image/jpeg")
        result = protocol._tool_result_to_anthropic(
            messages.ContentOutput(value=[fp])
        )
        assert isinstance(result, list)
        assert result[0]["type"] == "image"
        assert result[0]["source"]["media_type"] == "image/jpeg"

    def test_content_bytes_file(self) -> None:
        fp = messages.FilePart(data=b"\x89PNG", media_type="image/png")
        result = protocol._tool_result_to_anthropic(
            messages.ContentOutput(value=[messages.TextPart(text="desc"), fp])
        )
        assert isinstance(result, list)
        assert result[1]["type"] == "image"
        assert result[1]["source"]["data"] != ""


class TestMessagesToAnthropicMultipart:
    async def test_tool_result_with_file_part(self) -> None:
        """FilePart in tool results produces image content blocks."""
        fp = messages.FilePart(data="iVBOR", media_type="image/png")
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Read image")],
            ),
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolCallPart(
                        tool_call_id="tc-1",
                        tool_name="read",
                        tool_args='{"path": "test.png"}',
                    )
                ],
            ),
            messages.Message(
                role="tool",
                parts=[
                    messages.ToolResultPart(
                        tool_call_id="tc-1",
                        tool_name="read",
                        result=messages.ContentOutput(
                            value=[
                                messages.TextPart(text="Image loaded"),
                                fp,
                            ]
                        ),
                    )
                ],
            ),
        ]
        _, result = await protocol._messages_to_anthropic(msgs)
        tool_msg = result[-1]
        assert tool_msg["role"] == "user"
        tr = tool_msg["content"][0]
        assert tr["type"] == "tool_result"
        content = tr["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Image loaded"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/png"
