"""Tests for multi-part tool results in the OpenAI protocol."""

from __future__ import annotations

from ai.providers.openai import protocol
from ai.types import messages


class TestToolResultToOpenai:
    def test_text_output(self) -> None:
        result = protocol._tool_result_to_openai(
            messages.TextOutput(value="hello")
        )
        assert result == "hello"

    def test_json_output_none(self) -> None:
        result = protocol._tool_result_to_openai(messages.JsonOutput())
        assert result == ""

    def test_json_output_dict(self) -> None:
        result = protocol._tool_result_to_openai(
            messages.JsonOutput(value={"key": "value"})
        )
        assert result == '{"key":"value"}'

    def test_json_output_list(self) -> None:
        result = protocol._tool_result_to_openai(
            messages.JsonOutput(value=[1, 2, 3])
        )
        assert result == "[1,2,3]"

    def test_error_text_output(self) -> None:
        result = protocol._tool_result_to_openai(
            messages.ErrorTextOutput(value="boom")
        )
        assert result == "boom"

    def test_execution_denied(self) -> None:
        result = protocol._tool_result_to_openai(
            messages.ExecutionDeniedOutput(reason="user said no")
        )
        assert result == "Tool execution denied: user said no"

    def test_content_text_and_image(self) -> None:
        fp = messages.FilePart(data="b64data", media_type="image/png")
        result = protocol._tool_result_to_openai(
            messages.ContentOutput(
                value=[messages.TextPart(text="Image loaded"), fp]
            )
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Image loaded"}
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        assert "b64data" in result[1]["image_url"]["url"]

    def test_content_image_only(self) -> None:
        fp = messages.FilePart(data="b64data", media_type="image/jpeg")
        result = protocol._tool_result_to_openai(
            messages.ContentOutput(value=[fp])
        )
        assert isinstance(result, list)
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )

    def test_content_non_image_file(self) -> None:
        fp = messages.FilePart(data="pdfdata", media_type="application/pdf")
        result = protocol._tool_result_to_openai(
            messages.ContentOutput(value=[messages.TextPart(text="desc"), fp])
        )
        assert isinstance(result, list)
        assert result[1] == {"type": "text", "text": "[file: application/pdf]"}


class TestMessagesToOpenaiMultipart:
    async def test_tool_result_with_file_part(self) -> None:
        """ContentOutput with a FilePart produces image_url parts."""
        fp = messages.FilePart(data="iVBOR", media_type="image/png")
        msgs = [
            messages.Message(
                role="system",
                parts=[messages.TextPart(text="System")],
            ),
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
        result = await protocol._messages_to_openai(msgs)
        tool_msg = result[-1]
        assert tool_msg["role"] == "tool"
        content = tool_msg["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Image loaded"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
